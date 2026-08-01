"""Fine-tune w2v-BERT 2.0 + CTC on one language, from a parquet corpus.

`asr/train.py` drives the 42-language mixture and needs a manifest, splits,
vocab and mixture to do it. For a single language with its own train/val/test
already decided, that machinery is overhead: this reads the parquet directly,
uses the splits the corpus ships with, and trains.

Built for the Kusaal ASR corpus (`train-*.parquet` / `val-*.parquet`, columns
`audio` and `sentence`), which is why the column names default to those.

    python -u -m kasa42.asr.train_single \\
        --data-dir /workspace/kusaal_hub/data \\
        --vocab results/vocab.json --max-steps 5000

Initialising from DONDO rather than raw `facebook/w2v-bert-2.0` converges far
faster — it has already seen Kusaal — but DONDO's own training books are not
published, so a small unquantified overlap with any Kusaal test set is possible.
Say so when reporting. `--init facebook/w2v-bert-2.0` avoids the question
entirely at the cost of needing more steps.

The LID head from `Kasa42ForCTC` is present but unused here: one language means
nothing to identify, so `lid_weight=0` and the loss is pure CTC.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kasa42.asr.dataset import CharTokenizer, decode_audio
from kasa42.asr.model import Kasa42ForCTC, model_state
from kasa42.asr.train import has_native_bf16
from kasa42.data.text import normalize

DONDO = "KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en"


def load_split(data_dir: str, prefix: str, text_col: str):
    """Rows for one split, as (audio_bytes, text, duration) — audio left encoded."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{data_dir}/{prefix}-*.parquet"))
    if not files:
        raise SystemExit(f"no {prefix}-*.parquet under {data_dir}")

    rows = []
    for f in files:
        t = pq.read_table(f)
        audio = t.column("audio").to_pylist()
        text = t.column(text_col).to_pylist()
        dur = (t.column("duration_ms").to_pylist()
               if "duration_ms" in t.schema.names else [None] * len(text))
        for a, s, d in zip(audio, text, dur):
            raw = a["bytes"] if isinstance(a, dict) else a
            if not raw or not (s or "").strip():
                continue
            try:
                secs = float(d) / 1000.0
            except (TypeError, ValueError):
                secs = max(len(raw) - 44, 0) / 32000.0
            rows.append((raw, s.strip(), secs))
    return rows


class Clips(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw, text, _ = self.rows[i]
        return {"audio": raw, "text": text}


def collate(batch, fe, tok):
    wavs = [decode_audio(b["audio"]) for b in batch]
    feats = fe(wavs, sampling_rate=16000, return_tensors="pt",
               padding=True, return_attention_mask=True)
    ids = [tok.encode(b["text"]) for b in batch]
    lens = torch.tensor([len(x) for x in ids], dtype=torch.long)
    padded = torch.zeros(len(batch), int(lens.max().clamp(min=1)), dtype=torch.long)
    for i, x in enumerate(ids):
        if x:
            padded[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return {"input_features": feats["input_features"],
            "attention_mask": feats["attention_mask"],
            "labels": padded, "label_lengths": lens}


def batches_by_duration(rows, budget: float, cap: int):
    """Group indices so each batch costs about `budget` seconds of padded audio."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][2])
    out, cur, longest = [], [], 0.0
    for i in order:
        d = rows[i][2]
        if cur and (max(longest, d) * (len(cur) + 1) > budget or len(cur) >= cap):
            out.append(cur)
            cur, longest = [i], d
        else:
            cur.append(i)
            longest = max(longest, d)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def evaluate(model, rows, fe, tok, device, amp, budget, cap):
    """WER and CER over a split, greedy CTC."""
    from kasa42.asr.evaluate import error_rate

    model.eval()
    refs, hyps = [], []
    for idx in batches_by_duration(rows, budget, cap):
        batch = collate([{"audio": rows[i][0], "text": rows[i][1]} for i in idx], fe, tok)
        ctx = (torch.autocast("cuda", dtype=amp) if amp is not None
               else torch.autocast("cpu", enabled=False))
        with ctx:
            res = model(input_features=batch["input_features"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
        pred = res["logits"].argmax(-1).cpu()
        lens = res["input_lengths"].cpu()
        for j, i in enumerate(idx):
            hyps.append(tok.decode(pred[j][: int(lens[j])]))
            refs.append(rows[i][1])
    model.train()
    wer, _, _ = error_rate(refs, hyps, unit="word")
    cer, _, _ = error_rate(refs, hyps, unit="char")
    return wer, cer, list(zip(refs[:3], hyps[:3]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", default="results/vocab.json")
    ap.add_argument("--out-dir", default="checkpoints/kusaal-w2vbert")
    ap.add_argument("--init", default=DONDO,
                    help="Encoder to start from, or a kasa42 .pt checkpoint.")
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--train-prefix", default="train")
    ap.add_argument("--eval-prefix", default="val")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--batch-duration", type=float, default=160.0)
    ap.add_argument("--batch-cap", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--freeze-encoder-steps", type=int, default=200)
    ap.add_argument("--eval-clips", type=int, default=400)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = CharTokenizer.from_json(args.vocab)
    train_rows = load_split(args.data_dir, args.train_prefix, args.text_col)
    eval_rows = load_split(args.data_dir, args.eval_prefix, args.text_col)[: args.eval_clips]
    hrs = sum(r[2] for r in train_rows) / 3600
    print(f"train {len(train_rows):,} clips ({hrs:.1f} h)  "
          f"eval {len(eval_rows):,}  vocab {len(tok)}  blank={tok.blank}")

    # Characters the vocabulary cannot represent become [UNK], and the model is
    # then charged for text it was never able to emit. Report it rather than
    # letting it sit inside the WER.
    chars = {c for _, t, _ in train_rows[:2000] for c in normalize(t)}
    missing = sorted(c for c in chars if c != " " and c not in tok.vocab)
    if missing:
        print(f"WARNING {len(missing)} char(s) not in the vocab, will be [UNK]: "
              f"{''.join(missing)}")

    from transformers import SeamlessM4TFeatureExtractor

    encoder = DONDO if args.init.endswith(".pt") else args.init
    fe = SeamlessM4TFeatureExtractor.from_pretrained(encoder)
    model = Kasa42ForCTC(encoder, vocab_size=len(tok), n_languages=1,
                         lid_weight=0.0, blank_id=tok.blank)
    if args.init.endswith(".pt"):
        model.load_state_dict(model_state(args.init))
        print(f"initialised from checkpoint {args.init}")
    else:
        from kasa42.data.vocab import load_dondo_vocab

        moved = model.extend_ctc_head(load_dondo_vocab(), tok.vocab, encoder)
        print(f"initialised from {encoder}; {moved} CTC head rows transferred")
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(s):
        if s < args.warmup:
            return s / max(args.warmup, 1)
        p = (s - args.warmup) / max(args.max_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = torch.bfloat16 if (device == "cuda" and has_native_bf16()) else None
    print(f"device={device} amp={'bf16' if amp else 'fp32'} "
          f"budget={args.batch_duration}s")

    ds = Clips(train_rows)
    step, t0, frozen, skipped = 0, time.time(), None, 0
    while step < args.max_steps:
        loader = DataLoader(
            ds, batch_sampler=batches_by_duration(train_rows, args.batch_duration,
                                                  args.batch_cap),
            collate_fn=lambda b: collate(b, fe, tok), num_workers=4, pin_memory=True)
        for batch in loader:
            if step >= args.max_steps:
                break
            want = step < args.freeze_encoder_steps
            if want != frozen:
                model.set_encoder_frozen(want)
                frozen = want
            try:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                ctx = (torch.autocast("cuda", dtype=amp) if amp is not None
                       else torch.autocast("cpu", enabled=False))
                with ctx:
                    res = model(**batch)
                res["loss"].backward()
            except torch.OutOfMemoryError:
                # Shared card; drop the batch rather than the run.
                skipped += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if step % 25 == 0:
                el = time.time() - t0
                print(f"step {step:>5}/{args.max_steps}  ctc {res['ctc_loss'].item():.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{step/max(el,1e-9):.2f} step/s  {el/60:.1f} min"
                      + (f"  oom-skipped {skipped}" if skipped else ""))
            if step and step % 1000 == 0:
                wer, cer, _ = evaluate(model, eval_rows, fe, tok, device, amp,
                                       args.batch_duration, args.batch_cap)
                print(f"  [eval @ {step}] WER {wer:.1%}  CER {cer:.1%}")
                torch.save(model.state_dict(), out / f"step{step}.pt")
            step += 1

    torch.save(model.state_dict(), out / "final.pt")
    (out / "config.json").write_text(json.dumps({
        "encoder": encoder, "vocab_size": len(tok), "languages": ["Kusaal_kus"],
        "lid_weight": 0.0, "init": args.init, "max_steps": args.max_steps,
        "train_clips": len(train_rows), "train_hours": round(hrs, 2),
    }, indent=2), encoding="utf-8")

    wer, cer, samples = evaluate(model, eval_rows, fe, tok, device, amp,
                                 args.batch_duration, args.batch_cap)
    print(f"\nFINAL on {args.eval_prefix}: WER {wer:.1%}  CER {cer:.1%}  "
          f"({len(eval_rows):,} clips)")
    for ref, hyp in samples:
        print(f"  REF: {ref[:80]}\n  HYP: {hyp[:80]}\n")
    (out / "eval.json").write_text(json.dumps(
        {"split": args.eval_prefix, "n": len(eval_rows),
         "wer": round(wer, 4), "cer": round(cer, 4)}, indent=2), encoding="utf-8")
    print(f"saved {out/'final.pt'} after {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
