"""Temperature-sampled training mixture across the 42 languages.

Left alone, the corpus is dominated by a handful of languages — Asante Twi has
200 h while Mampruli has ~5 h. Training on the raw distribution produces a model
that is excellent at Twi and close to useless on the tail, which defeats the
entire point of covering 42 languages.

The standard fix (XLSR, MMS, NLLB) is temperature sampling: draw language i with
probability proportional to (h_i / H)^alpha. At alpha=1.0 you get the raw
distribution; at alpha=0 you get uniform. alpha=0.5 is the usual compromise and
what we use — it roughly square-roots the imbalance, lifting the tail sharply
while still respecting that more data deserves more weight.

A hard per-language cap runs on top. Capping Asante Twi is what turns 2,247 h
into ~700 h of training data, which is the single reason the run fits in hours
rather than days.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from kasa42.data.text import is_trainable

SEED = 20260730


def temperature_weights(hours: dict[str, float], alpha: float) -> dict[str, float]:
    total = sum(hours.values()) or 1.0
    raw = {k: (v / total) ** alpha for k, v in hours.items()}
    z = sum(raw.values()) or 1.0
    return {k: v / z for k, v in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/manifest.parquet")
    ap.add_argument("--splits", default="results/splits.json")
    ap.add_argument("--out", default="results/mixture.json")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--budget-hours", type=float, default=700.0,
                    help="Total training hours to draw. ~700 h keeps the ASR run at 4-7 h.")
    ap.add_argument("--cap-hours", type=float, default=40.0,
                    help="Hard per-language ceiling. This is what tames Asante Twi.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))["configs"]
    t = pq.read_table(args.manifest, columns=["config", "id", "text", "duration", "book"])

    # Available trainable hours per language, restricted to the train books.
    pool: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for config, sid, text, dur, book in zip(
        t.column("config").to_pylist(), t.column("id").to_pylist(),
        t.column("text").to_pylist(), t.column("duration").to_pylist(),
        t.column("book").to_pylist(),
    ):
        if splits.get(config, {}).get("book_to_split", {}).get(book) != "train":
            continue
        if not is_trainable(text):
            continue
        pool[config].append((sid, float(dur or 0.0)))

    avail = {c: sum(d for _, d in v) / 3600 for c, v in pool.items()}
    weights = temperature_weights(avail, args.alpha)

    # Allocate the budget by weight, then clamp to cap and to what exists.
    # Whatever the clamps free up is redistributed over the unclamped languages.
    target = {c: min(weights[c] * args.budget_hours, args.cap_hours, avail[c]) for c in avail}
    for _ in range(20):
        slack = args.budget_hours - sum(target.values())
        if slack <= 0.5:
            break
        open_ = {c for c in avail if target[c] < min(args.cap_hours, avail[c]) - 1e-6}
        if not open_:
            break
        z = sum(weights[c] for c in open_) or 1.0
        for c in open_:
            target[c] = min(target[c] + slack * weights[c] / z, args.cap_hours, avail[c])

    rng = random.Random(args.seed)
    chosen: dict[str, list[str]] = {}
    for config, items in pool.items():
        items = sorted(items)
        rng.shuffle(items)
        want, acc, ids = target[config] * 3600, 0.0, []
        for sid, d in items:
            if acc >= want:
                break
            ids.append(sid)
            acc += d
        chosen[config] = ids

    print(f"alpha={args.alpha}  budget={args.budget_hours} h  cap={args.cap_hours} h\n")
    print(f"{'config':24s} {'avail h':>8s} {'weight':>7s} {'used h':>7s} {'segs':>8s} {'share':>6s}")
    print("-" * 66)
    used_total = sum(target.values())
    for config in sorted(avail, key=lambda c: -avail[c]):
        used = target[config]
        print(f"{config:24s} {avail[config]:>8.1f} {weights[config]:>6.2%} "
              f"{used:>7.1f} {len(chosen[config]):>8,} {used/max(used_total,1e-9):>5.1%}")

    raw_share = max(avail.values()) / max(sum(avail.values()), 1e-9)
    mix_share = max(target.values()) / max(used_total, 1e-9)
    print(f"\navailable {sum(avail.values()):,.0f} h  ->  training {used_total:,.0f} h")
    print(f"largest language share: {raw_share:.1%} raw  ->  {mix_share:.1%} after mixing")
    print(f"segments selected: {sum(len(v) for v in chosen.values()):,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "alpha": args.alpha, "budget_hours": args.budget_hours,
        "cap_hours": args.cap_hours, "seed": args.seed,
        "available_hours": {k: round(v, 2) for k, v in avail.items()},
        "target_hours": {k: round(v, 2) for k, v in target.items()},
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "segment_ids": chosen,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
