"""GPU rehearsal: prove the CUDA path and predict the H200 run time.

Runs on anything with a GPU — Kaggle T4/P100, Colab, whatever is free. Two jobs:

  1. **Prove the path.** Mixed precision selects correctly (bf16 on Ampere+,
     fp16 with a GradScaler otherwise), the model fits, forward/backward/step
     all work on real audio. These are the failures that would otherwise eat the
     first hour of the H200 window.

  2. **Predict Thursday.** Measure throughput as *seconds of audio processed per
     second of wall clock*, and find the largest batch that fits. Both scale to
     the H200 far better than raw step counts, because they are independent of
     how long the clips happen to be.

The extrapolation is deliberately crude and stated as such: a T4 is roughly an
order of magnitude off an H200 for this workload, and memory scaling is not
linear in practice. The point is to distinguish "about 5 hours" from "about
50 hours" before committing, not to predict to the minute.

    python tests/bench_gpu.py --minutes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DONDO = "KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en"
REPO = "ghananlpcommunity/ghana-speech"
SHARD = "Kusaal_kus/train-00000-of-00022.parquet"

# Rough single-GPU bf16 dense throughput, TFLOPS. Only used for a ballpark
# scale factor; treat the resulting estimate as an order of magnitude.
TFLOPS = {"T4": 65, "P100": 19, "V100": 125, "A100": 312, "L4": 121,
          "H100": 989, "H200": 989, "A10G": 125, "RTX": 165}


def gpu_class(name: str) -> tuple[str, float]:
    for k, v in TFLOPS.items():
        if k.lower() in name.lower().replace(" ", ""):
            return k, v
    return "unknown", 65.0


def fetch_rows(n: int):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(f"datasets/{REPO}/{SHARD}", "rb") as fh:
        batch = next(pq.ParquetFile(fh).iter_batches(batch_size=n))
    rows = batch.to_pylist()
    for r in rows:
        r["config"] = "Kusaal_kus"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3.0, help="How long to benchmark.")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch-durations", type=float, nargs="*",
                    default=[60, 120, 240, 320])
    ap.add_argument("--target-hours", type=float, default=700.0,
                    help="Training hours planned for the H200 run.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--out", default="results/bench_gpu.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No GPU. This script is for Kaggle/Colab; use tests/test_pipeline.py on CPU.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    klass, tflops = gpu_class(name)

    from kasa42.asr.train import has_native_bf16

    native = has_native_bf16()
    cc = torch.cuda.get_device_capability()
    amp_dtype = torch.bfloat16 if native else torch.float16

    print(f"GPU        : {name}  ({vram:.0f} GB, sm_{cc[0]}{cc[1]}, "
          f"class={klass}, ~{tflops} TFLOPS)")
    print(f"bf16       : native={native}  "
          f"(torch.cuda.is_bf16_supported()={torch.cuda.is_bf16_supported()}, "
          f"which counts emulation)")
    print(f"amp dtype  : {str(amp_dtype).split('.')[-1]}")
    print(f"torch      : {torch.__version__}\n")
    if not native:
        print("Pre-Ampere: bf16 would be emulated in software, so fp16 is used.\n")

    from transformers import SeamlessM4TFeatureExtractor

    from kasa42.asr.dataset import CharTokenizer, Collator
    from kasa42.asr.model import Kasa42ForCTC
    from kasa42.data.text import normalize
    from kasa42.data.vocab import load_dondo_vocab

    rows = fetch_rows(args.rows)
    print(f"fetched {len(rows)} real Kusaal clips, "
          f"{sum(float(r['duration']) for r in rows)/60:.1f} min of audio\n")

    dondo = load_dondo_vocab()
    vocab = dict(dondo)
    nxt = max(vocab.values()) + 1
    for r in rows:
        for ch in normalize(r["text"]):
            if ch != " " and ch not in vocab:
                vocab[ch] = nxt
                nxt += 1
    vp = Path("results/_bench_vocab.json")
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(json.dumps({"vocab": vocab}, ensure_ascii=False), encoding="utf-8")
    tok = CharTokenizer.from_json(str(vp))

    model = Kasa42ForCTC(DONDO, vocab_size=len(vocab), n_languages=42,
                         blank_id=tok.blank).cuda()
    model.extend_ctc_head(dondo, vocab, DONDO)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))

    params = sum(p.numel() for p in model.parameters()) / 1e6
    # AdamW keeps fp32 weights, grads, and two moments: 16 bytes per parameter.
    # Worth stating, because it is usually the reason a batch will not fit and
    # it is invisible in a bare OOM message.
    fixed_gb = params * 1e6 * 16 / 1e9
    print(f"model      : {params:.0f}M params, vocab {len(vocab)}")
    print(f"optimiser  : ~{fixed_gb:.1f} GB fixed (weights+grads+AdamW moments) "
          f"of {vram:.0f} GB")
    print(f"activations have ~{max(vram - fixed_gb, 0):.1f} GB to work with\n")
    if cfg_ckpt := (fixed_gb > vram * 0.5):
        print("Over half of VRAM is optimiser state. On a card this size use")
        print("gradient_checkpointing=True and/or optim_8bit=True.\n")

    fe = SeamlessM4TFeatureExtractor.from_pretrained(DONDO)
    collate = Collator(fe, tok, {"Kusaal_kus": 0})

    results = {}
    print(f"{'batch_dur':>10s} {'clips':>6s} {'peak GB':>8s} {'aud s/s':>9s} {'status'}")
    print("-" * 52)

    for bd in args.batch_durations:
        # Fill a batch up to `bd` seconds of padded audio.
        picked, longest = [], 0.0
        for r in rows:
            d = float(r["duration"])
            if picked and max(longest, d) * (len(picked) + 1) > bd:
                break
            picked.append(r)
            longest = max(longest, d)
        if not picked:
            continue

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = {k: v.cuda() for k, v in collate(picked).items()}
            audio_s = sum(float(r["duration"]) for r in picked)

            for _ in range(2):  # warmup
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(**batch)["loss"]
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            n, t0 = 0, time.time()
            while time.time() - t0 < args.minutes * 60 / len(args.batch_durations):
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(**batch)["loss"]
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                n += 1
            torch.cuda.synchronize()
            el = time.time() - t0

            peak = torch.cuda.max_memory_allocated() / 1e9
            rate = audio_s * n / el
            results[str(bd)] = {"clips": len(picked), "peak_gb": round(peak, 2),
                                "steps": n, "audio_sec_per_sec": round(rate, 1),
                                "sec_per_step": round(el / n, 3)}
            print(f"{bd:>10.0f} {len(picked):>6d} {peak:>8.2f} {rate:>9.1f}  ok")
        except torch.cuda.OutOfMemoryError:
            results[str(bd)] = {"oom": True}
            print(f"{bd:>10.0f} {len(picked):>6d} {'-':>8s} {'-':>9s}  OOM")
            torch.cuda.empty_cache()
        except Exception as e:
            results[str(bd)] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{bd:>10.0f} {len(picked):>6d} {'-':>8s} {'-':>9s}  {type(e).__name__}")
            torch.cuda.empty_cache()

    ok = {k: v for k, v in results.items() if "audio_sec_per_sec" in v}
    if not ok:
        print("\nNothing ran. Fix the errors above before Thursday.")
        sys.exit(1)

    best = max(ok.values(), key=lambda v: v["audio_sec_per_sec"])
    rate = best["audio_sec_per_sec"]
    clips = best["clips"]
    here_h = args.target_hours * 3600 * args.epochs / rate / 3600

    _, h200_tflops = gpu_class("H200")
    scale = h200_tflops / tflops
    h200_h = here_h / scale

    print(f"\n{'='*62}")
    print(f"best throughput here : {rate:.1f} s audio / s wall  "
          f"(batch of {clips} clips)")
    print(f"{args.target_hours:.0f} h x {args.epochs:g} epochs here : {here_h:.0f} h")
    print(f"H200 estimate        : {h200_h:.1f} h   (x{scale:.0f} on peak TFLOPS)")

    # A tiny batch leaves the GPU mostly idle between kernel launches, so a
    # TFLOPS-only scaling understates a card that can hold a far larger batch.
    # Do not quantify the gain — just say which way the error points.
    if clips <= 8:
        print(f"\nThis is PESSIMISTIC. Only {clips} clips fit per batch on {vram:.0f} GB, so")
        print(f"the GPU is largely idle. The H200's 141 GB holds a far larger batch,")
        print(f"and throughput improves substantially with batch size until the")
        print(f"kernels become compute-bound. Treat {h200_h:.0f} h as an upper bound.")

    print(f"\n{'-'*62}")
    if h200_h > 12:
        # Solve for the settings that land inside a comfortable window.
        for ep in (3, 2):
            budget = 10.0 * scale * rate * 3600 / (ep * 3600)
            if budget >= 200:
                print(f"To target ~10 h: --budget-hours {budget:.0f} with {ep} epochs")
        print("\nSet these in data/mixture.py. Lower budget-hours mainly trims the")
        print("largest languages, which temperature sampling was already capping —")
        print("the tail languages keep their data either way.")
    else:
        print("Comfortably inside the window.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "gpu": name, "vram_gb": round(vram, 1), "bf16_native": native,
        "compute_capability": f"sm_{cc[0]}{cc[1]}",
        "amp": str(amp_dtype), "params_m": round(params), "vocab": len(vocab),
        "by_batch_duration": results,
        "estimate": {"target_hours": args.target_hours, "epochs": args.epochs,
                     "hours_here": round(here_h, 2), "hours_h200": round(h200_h, 2),
                     "scale_factor": round(scale, 1)},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
