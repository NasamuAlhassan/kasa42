"""Fine-tune w2v-BERT 2.0 + CTC + LID across the 42-language mixture.

Written to be driven from the Jupyter notebook on the H200:

    from kasa42.asr.train import train, TrainConfig
    train(TrainConfig(max_steps=200, smoke=True))   # H+1 smoke run
    train(TrainConfig())                            # the real thing

A custom loop rather than HF Trainer, because the dual CTC + LID objective, the
length-bucketed sampler and the encoder-freeze warmup are all easier to get
right explicitly than to coax out of Trainer callbacks — and on a 48 h clock,
predictable beats clever.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kasa42.asr.dataset import CharTokenizer, Collator, LengthBucketSampler
from kasa42.asr.model import Kasa42ForCTC


DONDO = "KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en"


@dataclass
class TrainConfig:
    # Start from DONDO, not raw w2v-BERT 2.0: it has already been fine-tuned on
    # 11 Northern Ghanaian languages including Kusaal, in this same read-speech
    # domain. Apache-2.0.
    encoder: str = DONDO
    transfer_head: bool = True
    manifest: str = "results/manifest.parquet"
    mixture: str = "results/mixture.json"
    splits: str = "results/splits.json"
    vocab: str = "results/vocab.json"
    data_dir: str = "data/parquet"          # local copy of the HF shards
    out_dir: str = "checkpoints/kasa42-asr"

    lr: float = 5e-5
    warmup_steps: int = 500
    freeze_encoder_steps: int = 300         # let the fresh heads settle first
    max_steps: int = 12000
    batch_duration: float = 320.0           # seconds of audio per batch
    grad_accum: int = 1
    max_grad_norm: float = 5.0
    lid_weight: float = 0.1
    weight_decay: float = 0.01
    # Trades ~30% speed for a large activation-memory saving. Unnecessary on the
    # H200's 141 GB; required on a 16 GB card, where AdamW alone costs ~9.7 GB
    # (606M params x 16 bytes for weights + grads + two moments).
    gradient_checkpointing: bool = False
    # 8-bit Adam cuts optimiser state from ~4.8 GB to ~1.2 GB. Needs bitsandbytes.
    optim_8bit: bool = False

    log_every: int = 25
    eval_every: int = 1000
    save_every: int = 1000
    seed: int = 20260730
    smoke: bool = False
    languages: list[str] = field(default_factory=list)


def has_native_bf16() -> bool:
    """True only where bf16 runs in hardware, i.e. Ampere (sm_80) and later.

    `torch.cuda.is_bf16_supported()` is not this check. It defaults to
    `including_emulation=True`, so a Turing T4 (sm_75) answers True and then
    runs bf16 in software — which measured 4.0 audio-seconds/second here, an
    order of magnitude off what fp16 gives on the same card. The H200 is
    Hopper (sm_90) and takes the real path.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def build_lang_map(mixture: dict) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(mixture["segment_ids"]))}


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return step / max(cfg.warmup_steps, 1)
    p = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def load_split(cfg: TrainConfig, split: str, lang_map: dict[str, int]):
    """Load segments for a split from local parquet, honouring the mixture.

    Selection is done on *indices*, never with `Dataset.filter`. Filtering runs
    the predicate over full rows — audio bytes included — and with `num_proc>1`
    each worker process materialises its own copy. On a 606M-row-of-audio
    dataset that exhausts RAM and restarts the kernel, which is precisely what
    it did on both Kaggle and Colab.

    Reading the two small columns we actually need, computing indices in plain
    Python, and calling `select()` keeps everything memory-mapped: no audio is
    touched until the DataLoader asks for a batch.
    """
    import datasets

    mixture = json.loads(Path(cfg.mixture).read_text(encoding="utf-8"))
    splits = json.loads(Path(cfg.splits).read_text(encoding="utf-8"))["configs"]
    configs = cfg.languages or sorted(lang_map)

    parts = []
    for config in configs:
        files = sorted(Path(cfg.data_dir, config).glob("*.parquet"))
        if not files:
            continue
        ds = datasets.load_dataset(
            "parquet", data_files=[str(p) for p in files], split="train",
        )
        book_split = splits.get(config, {}).get("book_to_split", {})

        if split == "train":
            keep = set(mixture["segment_ids"].get(config, []))
            # Column access on a memory-mapped dataset reads that column only.
            idx = [i for i, sid in enumerate(ds["id"]) if sid in keep]
        else:
            idx = [i for i, src in enumerate(ds["source_file"])
                   if book_split.get(str(src).split(".")[0], "train") == split]

        if not idx:
            continue
        ds = ds.select(idx)  # lazy: an index remapping, not a copy
        ds = ds.add_column("config", [config] * len(ds))
        parts.append(ds)
        print(f"  {config:24s} {len(ds):>7,} segments ({split})")

    if not parts:
        raise RuntimeError(
            f"no data for split={split}. Checked {cfg.data_dir} for "
            f"{len(configs)} configs — has the audio been downloaded?")
    return datasets.concatenate_datasets(parts)


def preflight(cfg: TrainConfig) -> None:
    """Fail early, and say exactly which command is missing.

    Training depends on four artefacts built by the data pipeline. A bare
    FileNotFoundError three frames deep is a poor way to discover that at hour
    four of a 48-hour window, so check them up front and name the fix.
    """
    needed = [
        (cfg.manifest, "python -m kasa42.data.build_manifest --workers 6"),
        (cfg.splits, "python -m kasa42.data.splits"),
        (cfg.vocab, "python -m kasa42.data.vocab"),
        (cfg.mixture, "python -m kasa42.data.mixture --alpha 0.5 --cap-hours 40"),
    ]
    missing = [(p, cmd) for p, cmd in needed if not Path(p).exists()]
    if not missing:
        return

    lines = ["", "Cannot train — the data pipeline has not been run.", ""]
    for p, cmd in missing:
        lines.append(f"  missing {p}")
        lines.append(f"     fix: {cmd}")
    lines += [
        "",
        "They must run in that order; each consumes the previous one's output.",
        "The manifest is the slow step (network-bound); the rest take seconds.",
        "",
        "If the manifest died partway, just rerun it — completed configs are",
        "cached in results/manifest_parts/ and skipped.",
        "",
    ]
    raise SystemExit("\n".join(lines))


def train(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    preflight(cfg)
    torch.manual_seed(cfg.seed)

    from transformers import SeamlessM4TFeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixture = json.loads(Path(cfg.mixture).read_text(encoding="utf-8"))
    lang_map = build_lang_map(mixture)
    tokenizer = CharTokenizer.from_json(cfg.vocab)

    if cfg.smoke:
        cfg.max_steps = min(cfg.max_steps, 30)
        cfg.warmup_steps, cfg.freeze_encoder_steps = 5, 5
        cfg.log_every, cfg.eval_every, cfg.save_every = 5, 10**9, 10**9

    print(f"device={device}  vocab={len(tokenizer)}  languages={len(lang_map)}")

    ds = load_split(cfg, "train", lang_map)
    print(f"train segments: {len(ds):,}")

    fe = SeamlessM4TFeatureExtractor.from_pretrained(cfg.encoder)
    collate = Collator(fe, tokenizer, lang_map)
    sampler = LengthBucketSampler(
        ds["duration"], batch_duration=cfg.batch_duration, seed=cfg.seed
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate, num_workers=4, pin_memory=True
    )

    vocab_meta = json.loads(Path(cfg.vocab).read_text(encoding="utf-8"))
    model = Kasa42ForCTC(
        cfg.encoder, vocab_size=len(tokenizer), n_languages=len(lang_map),
        lid_weight=cfg.lid_weight, blank_id=tokenizer.blank,
    )
    # Carry over DONDO's trained output rows for every character we share.
    if cfg.transfer_head:
        from kasa42.data.vocab import load_dondo_vocab

        moved = model.extend_ctc_head(load_dondo_vocab(), tokenizer.vocab, cfg.encoder)
        print(f"transferred {moved} CTC head rows from DONDO; "
              f"{len(tokenizer) - moved} rows freshly initialised")
    if cfg.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        print("gradient checkpointing on (~30% slower, much less activation memory)")
    model = model.to(device)

    if cfg.optim_8bit:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
        print("8-bit AdamW")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, cfg))

    amp_dtype = None
    if device == "cuda":
        amp_dtype = torch.bfloat16 if has_native_bf16() else torch.float16
        cc = torch.cuda.get_device_capability()
        print(f"gpu={torch.cuda.get_device_name(0)}  sm_{cc[0]}{cc[1]}  "
              f"amp={str(amp_dtype).split('.')[-1]}")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))

    def autocast_ctx():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "encoder": cfg.encoder, "vocab_size": len(tokenizer),
        "languages": sorted(lang_map), "lid_weight": cfg.lid_weight,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    step, t0, frozen = 0, time.time(), False
    model.train()
    while step < cfg.max_steps:
        sampler.set_epoch(step)
        for batch in loader:
            if step >= cfg.max_steps:
                break

            want_frozen = step < cfg.freeze_encoder_steps
            if want_frozen != frozen:
                model.set_encoder_frozen(want_frozen)
                frozen = want_frozen

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast_ctx():
                res = model(**batch)
            loss = res["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                el = time.time() - t0
                print(f"step {step:>6}/{cfg.max_steps}  loss {res['loss'].item():.3f}  "
                      f"ctc {res['ctc_loss'].item():.3f}  "
                      f"lid {res.get('lid_loss', torch.tensor(0.)).item():.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{step/max(el,1e-9):.2f} step/s  {el/60:.1f} min")

            if step and step % cfg.save_every == 0:
                torch.save(model.state_dict(), out / f"step{step}.pt")
            step += 1

    torch.save(model.state_dict(), out / "final.pt")
    print(f"\nsaved {out/'final.pt'} after {(time.time()-t0)/60:.1f} min")
    return out / "final.pt"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    for f in TrainConfig.__dataclass_fields__.values():
        if f.type is bool or f.name == "smoke":
            ap.add_argument(f"--{f.name.replace('_','-')}", action="store_true")
        elif f.name == "languages":
            ap.add_argument("--languages", nargs="*", default=[])
        else:
            ap.add_argument(f"--{f.name.replace('_','-')}", type=type(f.default),
                            default=f.default)
    train(TrainConfig(**vars(ap.parse_args())))
