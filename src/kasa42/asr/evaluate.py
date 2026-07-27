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
    ratios = []
    for k in sorted(honest):
        if k.startswith("__") or k not in leaked:
            continue
        h, l = honest[k]["wer"], leaked[k]["wer"]
        if h > 0:
            ratios.append(h / max(l, 1e-9))
        lines.append(f"{k:24s} {l:>8.1%} {h:>8.1%} {h - l:>+10.1f}pp")
    lines.append("-" * 64)
    hm, lm = honest["__micro__"]["wer"], leaked["__micro__"]["wer"]
    lines.append(f"{'micro-average':24s} {lm:>8.1%} {hm:>8.1%} {hm - lm:>+10.1f}pp")
    if lm > 0:
        lines.append(f"\nA random split understates WER by {(hm/lm - 1):.0%} on these weights.")
    return "\n".join(lines)


def save(scores: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
