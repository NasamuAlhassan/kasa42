"""Book-disjoint train/dev/test splits, plus the leaked split we compare against.

The dataset is Bible audio: `source_file` is `BOOK.CHAPTER.VERSION`. A random
segment split puts adjacent verses of the same chapter, read by the same narrator
in the same session, on both sides of the boundary. Chapter-level splitting is
better but still leaks lexically — genealogies and narrative arcs repeat proper
nouns across a whole book.

So we split by **book**. No book appears in more than one split.

We deliberately also emit a `random` split. Training one model and evaluating it
on both is what turns "we were careful" into a number: the gap between leaked and
honest WER on identical weights.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED = 20260730  # the morning the GPU opens


def assign_books(books: dict[str, float], dev_frac: float, test_frac: float,
                 seed: int) -> dict[str, str]:
    """Assign whole books to splits, targeting duration fractions.

    Books are shuffled (not sorted by size) so no split systematically inherits
    the longest or shortest books, then filled by accumulated duration.
    """
    total = sum(books.values())
    if total <= 0:
        return {b: "train" for b in books}

    order = sorted(books)  # deterministic base order before seeding
    random.Random(seed).shuffle(order)

    want_test, want_dev = total * test_frac, total * dev_frac
    got_test = got_dev = 0.0
    out: dict[str, str] = {}

    for b in order:
        d = books[b]
        # Fill test first, then dev; a book only lands in an eval split if it
        # does not overshoot the target by more than half its own duration.
        if got_test < want_test and got_test + d <= want_test + d / 2:
            out[b], got_test = "test", got_test + d
        elif got_dev < want_dev and got_dev + d <= want_dev + d / 2:
            out[b], got_dev = "dev", got_dev + d
        else:
            out[b] = "train"

    # A config with too few books to spare any must still yield an eval set;
    # fall back to the single smallest book rather than silently shipping none.
    if "test" not in out.values() and len(order) > 1:
        out[min(order, key=lambda b: books[b])] = "test"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/manifest.parquet")
    ap.add_argument("--out", default="results/splits.json")
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t = pq.read_table(args.manifest, columns=["config", "book", "duration", "id"])
    configs = sorted(set(t.column("config").to_pylist()))

    result: dict[str, dict] = {}
    print(f"{'config':24s} {'books':>6s} {'train h':>8s} {'dev h':>7s} {'test h':>7s} {'test %':>7s}")
    print("-" * 66)

    for config in configs:
        sub = t.filter(pc.equal(t.column("config"), config))
        books: dict[str, float] = defaultdict(float)
        for b, d in zip(sub.column("book").to_pylist(), sub.column("duration").to_pylist()):
            books[b] += float(d or 0.0)

        mapping = assign_books(dict(books), args.dev_frac, args.test_frac, args.seed)
        hours = defaultdict(float)
        for b, d in books.items():
            hours[mapping[b]] += d / 3600
        tot = sum(hours.values()) or 1.0

        result[config] = {
            "book_to_split": mapping,
            "hours": {k: round(v, 2) for k, v in hours.items()},
            "n_books": len(books),
        }
        print(f"{config:24s} {len(books):>6d} {hours['train']:>8.1f} "
              f"{hours['dev']:>7.1f} {hours['test']:>7.1f} {hours['test']/tot:>6.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "configs": result},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    tot_books = sum(v["n_books"] for v in result.values())
    print(f"\nWrote {out} — {len(result)} configs, {tot_books:,} books")
    print("Book-disjoint by construction: no book appears in two splits.")


if __name__ == "__main__":
    main()
