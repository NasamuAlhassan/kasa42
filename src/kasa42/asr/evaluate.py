"""Per-language WER/CER, and the leaked-vs-honest comparison.

The headline artifact is a 42-row table. The second artifact is the one that
earns trust: the *same weights* evaluated on a book-disjoint test set and on a
random-segment test set drawn from the same pool. The gap between them is the
size of the leak every random-split submission is unknowingly reporting.

Every hypothesis and reference passes through data/text.normalize before
scoring, including baselines' outputs. Normalizing one side only is the classic
way to publish a number that flatters the wrong system.
"""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from pathlib import Path

from kasa42.data.text import normalize


def _levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_rate(refs: list[str], hyps: list[str], *, unit: str) -> tuple[float, int, int]:
    """WER (unit='word') or CER (unit='char'). Returns (rate, errors, total)."""
    errs = total = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r), normalize(h)
        rt = r.split() if unit == "word" else list(r.replace(" ", ""))
        ht = h.split() if unit == "word" else list(h.replace(" ", ""))
        errs += _levenshtein(rt, ht)
        total += len(rt)
    return (errs / total if total else 0.0), errs, total


def score_by_language(records: list[dict]) -> dict[str, dict]:
    """records: [{config, reference, hypothesis, language_pred?}, ...]"""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["config"]].append(r)

    out: dict[str, dict] = {}
    for config, rows in sorted(by_lang.items()):
        refs = [r["reference"] for r in rows]
        hyps = [r["hypothesis"] for r in rows]
        wer, w_err, w_tot = error_rate(refs, hyps, unit="word")
        cer, c_err, c_tot = error_rate(refs, hyps, unit="char")
        entry = {"n": len(rows), "wer": round(wer, 4), "cer": round(cer, 4),
                 "word_errors": w_err, "words": w_tot,
                 "char_errors": c_err, "chars": c_tot}
        preds = [r.get("language_pred") for r in rows if r.get("language_pred")]
        if preds:
            entry["lid_acc"] = round(sum(p == config for p in preds) / len(preds), 4)
        out[config] = entry

    # Micro-average: pooled errors over pooled tokens, so big languages do not
    # get to hide a bad tail behind a favourable per-language mean.
    tw_e = sum(v["word_errors"] for v in out.values())
    tw_t = sum(v["words"] for v in out.values())
    tc_e = sum(v["char_errors"] for v in out.values())
    tc_t = sum(v["chars"] for v in out.values())
    out["__micro__"] = {
        "n": sum(v["n"] for v in out.values()),
        "wer": round(tw_e / tw_t, 4) if tw_t else 0.0,
        "cer": round(tc_e / tc_t, 4) if tc_t else 0.0,
    }
    langs = [v for k, v in out.items() if k != "__micro__"]
    out["__macro__"] = {
        "wer": round(sum(v["wer"] for v in langs) / max(len(langs), 1), 4),
        "cer": round(sum(v["cer"] for v in langs) / max(len(langs), 1), 4),
    }
    return out


def report(scores: dict[str, dict], title: str = "") -> str:
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"{'language':24s} {'n':>7s} {'WER':>8s} {'CER':>8s} {'LID':>7s}")
    lines.append("-" * 58)
    for k, v in scores.items():
        if k.startswith("__"):
            continue
        lid = f"{v['lid_acc']:>6.1%}" if "lid_acc" in v else "     -"
        lines.append(f"{k:24s} {v['n']:>7,} {v['wer']:>7.1%} {v['cer']:>7.1%} {lid}")
    lines.append("-" * 58)
    for k in ("__micro__", "__macro__"):
        if k in scores:
            v = scores[k]
            lines.append(f"{k.strip('_'):24s} {v.get('n',''):>7} "
                         f"{v['wer']:>7.1%} {v['cer']:>7.1%}")
    return "\n".join(lines)


def leak_report(honest: dict, leaked: dict) -> str:
    """The money table: identical weights, two evaluation protocols."""
    lines = ["", "Leaked (random split) vs. honest (book-disjoint) — same weights",
             "=" * 64,
             f"{'language':24s} {'leaked':>9s} {'honest':>9s} {'inflation':>11s}",
             "-" * 64]
    for k in sorted(honest):
        if k.startswith("__") or k not in leaked:
            continue
        h, l = honest[k]["wer"], leaked[k]["wer"]
        # WERs are fractions; the gap is quoted in percentage points, so it has
        # to be scaled. Formatting it with .1f straight off printed a 33-point
        # gap as "+0.3pp" — in the one table the whole project rests on.
        lines.append(f"{k:24s} {l:>8.1%} {h:>8.1%} {(h - l) * 100:>+9.1f}pp")
    lines.append("-" * 64)
    hm, lm = honest["__micro__"]["wer"], leaked["__micro__"]["wer"]
    lines.append(f"{'micro-average':24s} {lm:>8.1%} {hm:>8.1%} {(hm - lm) * 100:>+9.1f}pp")
    if lm > 0:
        lines.append(f"\nA random split understates WER by {(hm/lm - 1):.0%} on these weights.")
    return "\n".join(lines)


def save(scores: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- inference

def transcribe(test_set: str, checkpoint: str, vocab: str, batch_size: int = 16,
               device: str | None = None) -> list[dict]:
    """Run the trained model over a test-set parquet and return scoreable records.

    Torch is imported here rather than at module scope so that the scoring
    functions above stay importable on a machine with no torch — the demo and
    the notebook both do that.
    """
    import pyarrow.parquet as pq
    import torch
    from transformers import SeamlessM4TFeatureExtractor

    from kasa42.asr.dataset import CharTokenizer, decode_audio
    from kasa42.asr.model import Kasa42ForCTC
    from kasa42.asr.train import has_native_bf16

    ckpt = Path(checkpoint)
    meta = json.loads((ckpt.parent / "config.json").read_text(encoding="utf-8"))
    languages = meta["languages"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = CharTokenizer.from_json(vocab)
    model = Kasa42ForCTC(meta["encoder"], vocab_size=meta["vocab_size"],
                         n_languages=len(languages), blank_id=tok.blank)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(device)

    fe = SeamlessM4TFeatureExtractor.from_pretrained(meta["encoder"])
    tbl = pq.read_table(test_set)
    durations = tbl.column("duration").to_pylist()
    configs = tbl.column("config").to_pylist()
    texts = tbl.column("text").to_pylist()

    # Sort by duration so each batch pads to something close to its own longest
    # member, exactly as LengthBucketSampler does for training.
    order = sorted(range(tbl.num_rows), key=lambda i: durations[i])
    amp = torch.bfloat16 if (device == "cuda" and has_native_bf16()) else None

    out: list[dict] = []
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        sub = tbl.take(idx)
        wavs = [decode_audio(b) for b in sub.column("audio").to_pylist()]
        feats = fe(wavs, sampling_rate=16000, return_tensors="pt",
                   padding=True, return_attention_mask=True)

        ctx = (torch.autocast("cuda", dtype=amp) if amp is not None
               else contextlib.nullcontext())
        with torch.no_grad(), ctx:
            res = model(input_features=feats["input_features"].to(device),
                        attention_mask=feats["attention_mask"].to(device))

        pred = res["logits"].argmax(-1).cpu()
        lens = res["input_lengths"].cpu()
        lid = res["lid_logits"].argmax(-1).cpu()
        for j, i in enumerate(idx):
            out.append({
                "config": configs[i],
                "reference": texts[i],
                # Trim to real frames: everything past input_lengths is padding,
                # and decoding it invents characters at the end of every line.
                "hypothesis": tok.decode(pred[j][: int(lens[j])]),
                "language_pred": languages[int(lid[j])],
            })
        if start % (batch_size * 25) == 0:
            print(f"  {start:>6,}/{len(order):,}")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/kasa42-asr/final.pt")
    ap.add_argument("--vocab", default="results/vocab.json")
    ap.add_argument("--honest", default="results/testset_honest.parquet")
    ap.add_argument("--leaked", default="results/testset_leaked.parquet")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("--- honest (book-disjoint) ---")
    honest_recs = transcribe(args.honest, args.checkpoint, args.vocab, args.batch_size)
    honest = score_by_language(honest_recs)
    print("\n" + report(honest, "KASA-42 — book-disjoint test") + "\n")
    save(honest, f"{args.out_dir}/honest.json")

    if not Path(args.leaked).exists():
        print(f"no {args.leaked} — skipping the leak comparison. "
              f"Build it with `python -m kasa42.asr.testset`.")
        return

    print("--- leaked (random split over seen books) ---")
    leaked_recs = transcribe(args.leaked, args.checkpoint, args.vocab, args.batch_size)
    leaked = score_by_language(leaked_recs)
    save(leaked, f"{args.out_dir}/leaked.json")

    # Score the honest side again over only the configs the leaked set covers.
    # Some configs have too small a leaked pool to report, and comparing two
    # micro-averages taken over different language sets would not be a
    # like-for-like number.
    shared = {k for k in leaked if not k.startswith("__")}
    honest_matched = score_by_language([r for r in honest_recs if r["config"] in shared])
    save(honest_matched, f"{args.out_dir}/honest_matched.json")

    text = leak_report(honest_matched, leaked)
    print("\n" + text)
    dropped = len({k for k in honest if not k.startswith("__")} - shared)
    if dropped:
        print(f"\n{dropped} language(s) omitted: leaked pool below the floor. "
              f"Both columns above cover the same {len(shared)} languages.")
    Path(f"{args.out_dir}/leak_report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_dir}/")


if __name__ == "__main__":
    main()
