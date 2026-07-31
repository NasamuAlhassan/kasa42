"""Compare against baselines on the languages they actually cover.

The raw micro-averages are not comparable. MMS scores 34 of the 42 languages
because it has no adapter for the rest; DONDO scores 8, being trained on 11 of
which 8 appear here; ours scores all 42. Reading "25.7% vs 30.2%" off those
three rows compares an average over easy-and-covered against an average over
everything, and flatters whichever system declined the hardest languages.

So for each baseline this recomputes *both* sides over the intersection, pooling
errors and reference tokens rather than averaging per-language rates — a macro
average would let a system hide a bad tail behind languages it happened to skip.

Coverage is reported alongside, because refusing to transcribe a language is a
result about that system, not a neutral omission.

    python -m kasa42.asr.compare
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def micro(scores: dict, langs: set[str]) -> tuple[float, float, int]:
    """Pooled WER, CER and utterance count over `langs`."""
    we = ww = ce = cw = n = 0
    for k, v in scores.items():
        if k.startswith("__") or k not in langs:
            continue
        we += v["word_errors"]
        ww += v["words"]
        ce += v["char_errors"]
        cw += v["chars"]
        n += v["n"]
    return (we / ww if ww else 0.0), (ce / cw if cw else 0.0), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", default="results/eval_matched/honest.json")
    ap.add_argument("--baselines", default="results/baselines")
    ap.add_argument("--out", default="results/comparison.json")
    args = ap.parse_args()

    ours = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    our_langs = {k for k in ours if not k.startswith("__")}
    print(f"KASA-42: {len(our_langs)} languages, "
          f"micro WER {ours['__micro__']['wer']:.1%}  "
          f"CER {ours['__micro__']['cer']:.1%}\n")

    rows = []
    for f in sorted(glob.glob(f"{args.baselines}/*.json")):
        name = Path(f).stem
        base = json.loads(Path(f).read_text(encoding="utf-8"))
        base_langs = {k for k in base if not k.startswith("__")}
        shared = base_langs & our_langs
        if not shared:
            print(f"{name}: no languages in common — skipped")
            continue

        b_wer, b_cer, n = micro(base, shared)
        o_wer, o_cer, _ = micro(ours, shared)
        rows.append({
            "system": name, "languages": len(shared), "utterances": n,
            "not_covered": sorted(our_langs - base_langs),
            "baseline_wer": round(b_wer, 4), "baseline_cer": round(b_cer, 4),
            "kasa42_wer": round(o_wer, 4), "kasa42_cer": round(o_cer, 4),
            "wer_delta_pp": round((b_wer - o_wer) * 100, 2),
            "cer_delta_pp": round((b_cer - o_cer) * 100, 2),
        })

    print("On the languages each system covers — both sides recomputed there:\n")
    print(f"{'system':10s} {'langs':>5s} {'their WER':>10s} {'ours':>8s} "
          f"{'their CER':>10s} {'ours':>8s} {'WER gap':>9s}")
    print("-" * 66)
    for r in sorted(rows, key=lambda r: -r["languages"]):
        print(f"{r['system']:10s} {r['languages']:>5d} "
              f"{r['baseline_wer']:>9.1%} {r['kasa42_wer']:>7.1%} "
              f"{r['baseline_cer']:>9.1%} {r['kasa42_cer']:>7.1%} "
              f"{r['wer_delta_pp']:>+8.1f}pp")
    print("-" * 66)
    print("WER gap is baseline minus ours: positive means we are ahead.\n")

    for r in sorted(rows, key=lambda r: -r["languages"]):
        miss = len(r["not_covered"])
        print(f"{r['system']}: covers {r['languages']}/{len(our_langs)}"
              + (f", no output for {miss} ({', '.join(r['not_covered'][:6])}"
                 + ("…" if miss > 6 else "") + ")" if miss else ""))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"kasa42_all_languages": {"n_languages": len(our_langs),
                                  "wer": ours["__micro__"]["wer"],
                                  "cer": ours["__micro__"]["cer"]},
         "matched": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
