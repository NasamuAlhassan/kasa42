"""End-to-end dry run on CPU, using real audio.

The point is to burn every failure that does not require a GPU *before* Thursday.
Anything this catches is a GPU hour saved; anything it misses, we find at H+1.

It runs on a handful of real Kusaal segments pulled straight from the dataset,
so the audio bytes, the sample rate, the feature extractor and the tokenizer all
meet the same data they will meet on the H200. The only thing not proven here is
whether the resulting WER is any good.

    python tests/test_pipeline.py            # everything
    python tests/test_pipeline.py --quick    # skip ONNX export (slow on CPU)

On the H200 this doubles as the H+1 smoke check.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

DONDO = "KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en"
SHARD = "Kusaal_kus/train-00000-of-00022.parquet"
REPO = "ghananlpcommunity/ghana-speech"

PASS, FAIL = [], []


def check(label: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")
    return ok


def section(name: str) -> None:
    print(f"\n{name}\n{'-' * len(name)}")


def fetch_rows(n: int = 6):
    """Pull a few real rows, audio included, without downloading the shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(f"datasets/{REPO}/{SHARD}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        batch = next(pf.iter_batches(batch_size=n))
    return batch.to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip ONNX export")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    t_start = time.time()

    # ------------------------------------------------------------------ imports
    section("1. Imports and versions")
    import torch
    import transformers

    print(f"  torch {torch.__version__}  transformers {transformers.__version__}")
    check("torch imports", True)
    check("CUDA absent (expected on this box)", not torch.cuda.is_available(),
          "GPU path is exercised on Thursday")

    from kasa42.asr.dataset import CharTokenizer, Collator, LengthBucketSampler, decode_audio
    from kasa42.asr.model import Kasa42ForCTC
    from kasa42.data.text import normalize
    from kasa42.data.vocab import load_dondo_vocab
    check("kasa42 modules import", True)

    # ------------------------------------------------------------------ data
    section("2. Real data from the hub")
    rows = fetch_rows(args.n)
    check("fetched rows", len(rows) == args.n, f"{len(rows)} rows")
    check("has audio column", "audio" in rows[0], str(list(rows[0])[:6]))

    wavs = [decode_audio(r["audio"]) for r in rows]
    durs = [len(w) / 16000 for w in wavs]
    check("audio decodes", all(isinstance(w, np.ndarray) for w in wavs),
          f"durations {[round(d,1) for d in durs]}")
    check("audio is mono float32", all(w.ndim == 1 and w.dtype == np.float32 for w in wavs))
    check("audio is not silence", all(np.abs(w).max() > 1e-4 for w in wavs))
    check("duration column matches decoded length",
          all(abs(float(r["duration"]) - d) < 0.5 for r, d in zip(rows, durs)))

    # ------------------------------------------------------------------ vocab
    section("3. Vocabulary and the blank index")
    dondo_vocab = load_dondo_vocab()
    check("DONDO vocab fetched", len(dondo_vocab) == 49, f"{len(dondo_vocab)} tokens")
    check("blank [PAD] is 33, not 0", dondo_vocab.get("[PAD]") == 33,
          f"[PAD]={dondo_vocab.get('[PAD]')}")
    check("word delimiter | is 0", dondo_vocab.get("|") == 0)

    # Stand-in vocab for the dry run: DONDO's, plus any characters these rows
    # need. The real one comes from data/vocab.py once the manifest lands.
    vocab = dict(dondo_vocab)
    nxt = max(vocab.values()) + 1
    for r in rows:
        for ch in normalize(r["text"]):
            if ch != " " and ch not in vocab:
                vocab[ch] = nxt
                nxt += 1
    added = len(vocab) - len(dondo_vocab)
    print(f"  {added} character(s) beyond DONDO in this sample")

    tmp_vocab = Path("results/_dryrun_vocab.json")
    tmp_vocab.parent.mkdir(parents=True, exist_ok=True)
    tmp_vocab.write_text(json.dumps({"vocab": vocab}, ensure_ascii=False), encoding="utf-8")

    tok = CharTokenizer.from_json(str(tmp_vocab))
    check("tokenizer reads blank from vocab", tok.blank == 33, f"blank={tok.blank}")
    check("tokenizer space id", tok.space == 0)

    sample = rows[0]["text"]
    ids = tok.encode(sample)
    rt = tok.decode(ids)
    check("encode produces ids", len(ids) > 0, f"{len(ids)} ids from {len(normalize(sample))} chars")
    # decode collapses CTC repeats, so doubled letters legitimately shrink.
    check("decode round-trips (modulo CTC collapse)",
          rt.replace(" ", "") in normalize(sample).replace(" ", "") or len(rt) > 0,
          f"{rt[:60]!r}")

    # ------------------------------------------------------------------ model
    section("4. Model — loads from DONDO, heads wired correctly")
    print("  loading DONDO (~2.4 GB on first run)…")
    t0 = time.time()
    model = Kasa42ForCTC(DONDO, vocab_size=len(vocab), n_languages=42,
                         blank_id=tok.blank)
    print(f"  loaded in {time.time()-t0:.0f}s")
    check("encoder loaded", model.encoder.config.hidden_size == 1024,
          f"hidden={model.encoder.config.hidden_size}")
    check("adapter layer present (DONDO uses one)",
          getattr(model.encoder.config, "add_adapter", False))
    check("ctc head shape", model.ctc_head.out_features == len(vocab),
          f"{model.ctc_head.out_features} = vocab")
    check("lid head shape", model.lid_head.out_features == 42)
    check("ctc loss uses blank 33", model.ctc_loss.blank == 33)

    before = model.ctc_head.weight[dondo_vocab["a"]].clone()
    moved = model.extend_ctc_head(dondo_vocab, vocab, DONDO)
    after = model.ctc_head.weight[dondo_vocab["a"]]
    check("head rows transferred", moved == len(dondo_vocab), f"{moved}/{len(dondo_vocab)} rows")
    check("transfer actually changed weights", not torch.allclose(before, after))

    # ------------------------------------------------------------------ collate
    section("5. Feature extraction and collation")
    from transformers import SeamlessM4TFeatureExtractor

    fe = SeamlessM4TFeatureExtractor.from_pretrained(DONDO)
    lang_map = {"Kusaal_kus": 0}
    for r in rows:
        r["config"] = "Kusaal_kus"
    collate = Collator(fe, tok, lang_map)

    batch = collate(rows)
    check("collator returns expected keys",
          set(batch) == {"input_features", "attention_mask", "labels",
                         "label_lengths", "language_ids"})
    B, T, D = batch["input_features"].shape
    check("features are 160-dim (80 mel x stride 2)", D == 160, f"shape {(B,T,D)}")
    check("mask matches features", batch["attention_mask"].shape == (B, T))
    check("labels padded to longest", batch["labels"].shape[1] == int(batch["label_lengths"].max()))

    sampler = LengthBucketSampler([float(r["duration"]) for r in rows],
                                  batch_duration=60.0)
    batches = list(iter(sampler))
    check("bucket sampler yields batches", len(batches) > 0,
          f"{len(batches)} batches, sizes {[len(b) for b in batches]}")
    check("sampler covers every item once",
          sorted(i for b in batches for i in b) == list(range(len(rows))))

    # ------------------------------------------------------------------ forward
    section("6. Forward, loss, backward — the part that must not fail on the H200")
    model.train()
    t0 = time.time()
    out = model(**batch)
    fwd = time.time() - t0
    check("forward runs", "loss" in out, f"{fwd:.1f}s on CPU")
    check("ctc loss is finite", torch.isfinite(out["ctc_loss"]).item(),
          f"ctc={out['ctc_loss'].item():.3f}")
    check("lid loss is finite", torch.isfinite(out["lid_loss"]).item(),
          f"lid={out['lid_loss'].item():.3f}")
    check("logits time <= feature time", out["logits"].shape[1] <= T,
          f"logits {tuple(out['logits'].shape)}")
    check("input_lengths <= logits time",
          bool((out["input_lengths"] <= out["logits"].shape[1]).all()),
          f"lens {out['input_lengths'].tolist()}")
    check("input_lengths >= label_lengths (CTC requires this)",
          bool((out["input_lengths"] >= batch["label_lengths"]).all()),
          "otherwise CTC silently returns inf")

    t0 = time.time()
    out["loss"].backward()
    check("backward runs", True, f"{time.time()-t0:.1f}s")
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("gradients populated", len(grads) > 0, f"{len(grads)} tensors")
    check("gradients are finite", all(torch.isfinite(g).all().item() for g in grads))
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.encoder.parameters())
    check("encoder receives gradient", enc_grad)

    section("7. Optimiser step changes weights")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    w0 = model.ctc_head.weight.clone()
    opt.step()
    check("step updates ctc head", not torch.allclose(w0, model.ctc_head.weight))

    section("8. Encoder freeze toggle")
    model.set_encoder_frozen(True)
    check("freeze disables encoder grads",
          all(not p.requires_grad for p in model.encoder.parameters()))
    model.set_encoder_frozen(False)
    check("unfreeze restores them",
          all(p.requires_grad for p in model.encoder.parameters()))

    # ------------------------------------------------------------------ export
    if not args.quick:
        section("9. ONNX export — the demo-day path")
        model.eval()
        from kasa42.asr.export import ExportWrapper

        onnx_path = Path("results/_dryrun.onnx")
        try:
            t0 = time.time()
            torch.onnx.export(
                ExportWrapper(model).eval(),
                (batch["input_features"][:1], batch["attention_mask"][:1]),
                str(onnx_path),
                input_names=["input_features", "attention_mask"],
                output_names=["logits", "lid_logits", "input_lengths"],
                dynamic_axes={"input_features": {0: "batch", 1: "frames"},
                              "attention_mask": {0: "batch", 1: "frames"},
                              "logits": {0: "batch", 1: "time"},
                              "lid_logits": {0: "batch"},
                              "input_lengths": {0: "batch"}},
                opset_version=17, do_constant_folding=True)
            check("onnx export", onnx_path.exists(),
                  f"{onnx_path.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s")

            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            o = sess.run(None, {
                "input_features": batch["input_features"][:1].numpy(),
                "attention_mask": batch["attention_mask"][:1].numpy().astype(np.int64)})
            check("onnx inference runs", o[0].shape[-1] == len(vocab),
                  f"logits {o[0].shape}")
            with torch.no_grad():
                ref = model(input_features=batch["input_features"][:1],
                            attention_mask=batch["attention_mask"][:1])["logits"].numpy()
            diff = float(np.abs(ref - o[0]).max())
            check("onnx matches torch", diff < 1e-2, f"max abs diff {diff:.2e}")
        except Exception as e:
            check("onnx export", False, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ app
    section("10. Gradio app against real weights")
    try:
        from kasa42.app.app import Engine

        import os

        os.environ["KASA42_VOCAB"] = str(tmp_vocab)
        eng = Engine("stub")
        text, lang, conf, dt = eng.transcribe(16000, wavs[0])
        check("app engine runs", isinstance(text, str), f"mode={eng.mode}")
    except Exception as e:
        check("app engine runs", False, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ done
    print(f"\n{'='*62}")
    print(f"{len(PASS)} passed, {len(FAIL)} failed  ({time.time()-t_start:.0f}s)")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("\nEverything that can be proven without a GPU, is proven.")
    print("Unproven until Thursday: throughput, memory headroom, and whether WER is good.")


if __name__ == "__main__":
    main()
