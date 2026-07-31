"""Materialise the two evaluation sets the leak comparison needs.

`baselines.py` and `evaluate.py` both consume records of
`{config, id, text, duration, source_file, book, audio}`. This builds them out of
the pre-staged parquet, driven by the book assignment in `splits.json`.

Two sets come out of one pass:

  * **honest** — segments from books assigned to `test`. No book here appears in
    training, so nothing about them has been seen.
  * **leaked** — a random sample of segments from books assigned to `train`,
    *excluding every id the mixture actually trained on*.

That exclusion is the whole experiment. Both sets contain only segments the
model has never seen; the single difference is whether the *book* was seen. That
isolates leakage. Sampling segments the model did train on would measure
memorisation instead — a different and much weaker claim, and the first thing a
reviewer would object to.

Output is parquet, not the JSONL the old `--test-set` default implied: audio is
raw bytes, and base64 inside JSON would inflate a 42-language test set by about
a third for no benefit.

Reads are targeted. The manifest records which shard each id came from, so only
shards that actually contain wanted rows are opened, and within a shard only the
row groups holding those rows are decompressed. The shard column is treated as a
hint rather than gospel — any config with ids still outstanding falls back to
scanning its remaining shards, and anything never found is reported rather than
silently dropped.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kasa42.data.build_manifest import parse_source
from kasa42.data.text import is_trainable

SEED = 20260730
READ_COLS = ["id", "text", "duration", "source_file", "audio"]

SCHEMA = pa.schema([
    ("config", pa.string()),
    ("id", pa.string()),
    ("text", pa.string()),
    ("duration", pa.float32()),
    ("source_file", pa.string()),
    ("book", pa.string()),
    ("audio", pa.binary()),
])


def _one_per_id(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Collapse repeated ids, keeping the lowest shard so the result is stable.

    `id` is not unique in every config. Where a shard holds two recording
    projects, the same verse id appears once per version — same text, different
    narrator — and the manifest lists each physical row. Sampling raw rows would
    draw one utterance twice while claiming the cap had been met, and then
    weight it double in the WER table.
    """
    seen: set[str] = set()
    out: list[tuple[str, int]] = []
    for sid, shard in pairs:
        if sid not in seen:
            seen.add(sid)
            out.append((sid, shard))
    return out


def choose(manifest: str, splits: dict, mixture: dict, max_per_config: int,
           seed: int, min_leaked: int, match_books: bool = False) -> dict[str, dict]:
    """Pick honest and leaked ids per config from metadata alone.

    `is_trainable` gates both sets identically. Segments carrying digits (verse
    numbers leaking into the transcript) are excluded from training, so scoring
    them here would charge the model for text it was never taught to emit.
    """
    t = pq.read_table(manifest, columns=["config", "id", "text", "book", "shard"])

    trained = {c: set(v) for c, v in mixture.get("segment_ids", {}).items()}
    honest: dict[str, list] = defaultdict(list)
    leaked: dict[str, list] = defaultdict(list)

    # Without this, the two sets differ in more than whether the book was seen.
    # The honest side draws from the 2-5 held-out books of a language; the
    # leaked side draws across all 20-60 train books. Measured on the real
    # split that is 4.2 books against 27.2 — 6.5x the content diversity, and
    # 4.9% shorter references, both of which make the leaked side harder for
    # reasons that have nothing to do with leakage. Restricting the leaked draw
    # to as many train books as the config has test books removes the dominant
    # confound, at the cost of a smaller pool.
    allowed: dict[str, set[str]] = {}
    if match_books:
        pick = random.Random(seed ^ 0x5EED)
        for config, sp in splits.items():
            b2s = sp.get("book_to_split", {})
            n_test = sum(1 for v in b2s.values() if v == "test")
            train_books = sorted(b for b, v in b2s.items() if v == "train")
            if train_books and n_test:
                allowed[config] = set(pick.sample(
                    train_books, min(n_test, len(train_books))))

    b2s_cache: dict[str, dict] = {}
    for config, sid, text, book, shard in zip(
        t.column("config").to_pylist(), t.column("id").to_pylist(),
        t.column("text").to_pylist(), t.column("book").to_pylist(),
        t.column("shard").to_pylist(),
    ):
        b2s = b2s_cache.get(config)
        if b2s is None:
            b2s = b2s_cache[config] = splits.get(config, {}).get("book_to_split", {})
        where = b2s.get(book)
        if where == "test":
            if is_trainable(text):
                honest[config].append((sid, int(shard or 0)))
        elif where == "train":
            # Held out from this model, but from a book it has seen.
            if match_books and book not in allowed.get(config, {book}):
                continue
            if sid not in trained.get(config, ()) and is_trainable(text):
                leaked[config].append((sid, int(shard or 0)))

    rng = random.Random(seed)
    out: dict[str, dict] = {}
    for config in sorted(set(t.column("config").to_pylist())):
        h = _one_per_id(sorted(honest.get(config, [])))
        l = _one_per_id(sorted(leaked.get(config, [])))
        rng.shuffle(h)
        rng.shuffle(l)
        enough = len(l) >= min_leaked
        out[config] = {
            "honest": h[:max_per_config],
            "leaked": l[:max_per_config] if enough else [],
            "honest_pool": len(h),
            "leaked_pool": len(l),
            # Below the floor, a per-language WER is noise rather than a number.
            # Drop the config from the leak table and say so.
            "leaked_excluded": not enough,
        }
    return out


def take_rows(path: str, wanted: set[str]) -> pa.Table | None:
    """Pull the rows whose `id` is in `wanted`, touching as little as possible.

    Pass one reads only the `id` column — cheap, and it decompresses no audio.
    Pass two reads only the row groups those positions fall in. On a shard laid
    out as a single row group this degrades to one full read, which is the floor
    anyway.
    """
    pf = pq.ParquetFile(path)
    ids = pf.read(columns=["id"]).column("id").to_pylist()
    pos = [i for i, s in enumerate(ids) if str(s) in wanted]
    if not pos:
        return None

    sizes = [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)]
    starts, acc = [], 0
    for n in sizes:
        starts.append(acc)
        acc += n

    group_of = {p: bisect.bisect_right(starts, p) - 1 for p in pos}
    groups = sorted(set(group_of.values()))

    present = [c for c in READ_COLS if c in pf.schema_arrow.names]
    tbl = pf.read_row_groups(groups, columns=present)

    # Positions are global to the shard; remap them into the concatenation of
    # just the groups we read.
    base, acc = {}, 0
    for g in groups:
        base[g] = acc
        acc += sizes[g]
    local = [base[group_of[p]] + (p - starts[group_of[p]]) for p in pos]
    return tbl.take(local)


def to_schema(tbl: pa.Table, config: str) -> pa.Table:
    """Normalise a taken table into SCHEMA, unwrapping the audio column.

    HF writes `audio` as struct<bytes, path>; a plain binary column is also
    accepted so this keeps working if the layout changes.
    """
    n = tbl.num_rows
    audio = tbl.column("audio")
    if pa.types.is_struct(audio.type):
        audio = pc.struct_field(audio, "bytes")
    src = [str(x) for x in tbl.column("source_file").to_pylist()]
    return pa.Table.from_pydict({
        "config": [config] * n,
        "id": [str(x) for x in tbl.column("id").to_pylist()],
        "text": [str(x) for x in tbl.column("text").to_pylist()],
        "duration": [float(x) if x is not None else 0.0
                     for x in tbl.column("duration").to_pylist()],
        "source_file": src,
        "book": [parse_source(s)[0] for s in src],
        "audio": audio.to_pylist(),
    }, schema=SCHEMA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/manifest.parquet")
    ap.add_argument("--splits", default="results/splits.json")
    ap.add_argument("--mixture", default="results/mixture.json")
    ap.add_argument("--data-dir", default="/data/ghana-speech",
                    help="Root holding <config>/*.parquet.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-per-config", type=int, default=200,
                    help="Cap per language per set. 200 x 42 x 2 is roughly "
                         "27 h of audio; raise it if you want tighter per-language "
                         "confidence and can afford the baseline inference time.")
    ap.add_argument("--min-leaked", type=int, default=30,
                    help="Configs with a smaller leaked pool are excluded from "
                         "the leak table rather than reported as noise.")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--match-books", action="store_true",
                    help="Draw the leaked set from as many train books as the "
                         "config has test books, so the two sets differ in "
                         "whether the book was seen and not in how many books "
                         "they span. Without it the comparison is confounded.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))["configs"]
    mixture = json.loads(Path(args.mixture).read_text(encoding="utf-8"))

    plan = choose(args.manifest, splits, mixture, args.max_per_config,
                  args.seed, args.min_leaked, match_books=args.match_books)
    if args.configs:
        plan = {k: v for k, v in plan.items() if k in args.configs}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = out_dir / "testset_honest.parquet"
    l_path = out_dir / "testset_leaked.parquet"

    hw = pq.ParquetWriter(h_path, SCHEMA, compression="zstd")
    lw = pq.ParquetWriter(l_path, SCHEMA, compression="zstd")

    summary: dict[str, dict] = {}
    missing_total = repeat_total = 0
    print(f"{'config':24s} {'honest':>7s} {'leaked':>7s} {'h pool':>8s} {'l pool':>8s} {'shards':>7s}")
    print("-" * 68)
    try:
        for config, sel in plan.items():
            h_set = {sid for sid, _ in sel["honest"]}
            l_set = {sid for sid, _ in sel["leaked"]}
            wanted = h_set | l_set
            if not wanted:
                summary[config] = {**{k: v for k, v in sel.items()
                                      if k not in ("honest", "leaked")},
                                   "honest_n": 0, "leaked_n": 0, "missing": 0}
                continue

            shard_files = sorted(Path(args.data_dir, config).glob("*.parquet"))
            if not shard_files:
                print(f"{config:24s} no parquet under {args.data_dir}/{config} — skipped")
                continue

            hint = {s for _, s in sel["honest"]} | {s for _, s in sel["leaked"]}
            order = [i for i in range(len(shard_files)) if i in hint]
            order += [i for i in range(len(shard_files)) if i not in hint]

            todo = set(wanted)
            h_n = l_n = touched = repeats = 0
            for i in order:
                if not todo:
                    break
                tbl = take_rows(str(shard_files[i]), todo)
                touched += 1
                if tbl is None:
                    continue
                got = to_schema(tbl, config)
                got_ids = got.column("id").to_pylist()
                # Take each id once. `take_rows` filters on `todo`, which
                # already excludes anything written by an earlier shard, but a
                # single shard can still hold the same id twice — so discard as
                # we go rather than matching the whole selection set.
                h_idx, l_idx = [], []
                for j, s in enumerate(got_ids):
                    if s not in todo:
                        repeats += 1
                        continue
                    todo.discard(s)
                    (h_idx if s in h_set else l_idx).append(j)
                if h_idx:
                    hw.write_table(got.take(h_idx))
                    h_n += len(h_idx)
                if l_idx:
                    lw.write_table(got.take(l_idx))
                    l_n += len(l_idx)

            missing_total += len(todo)
            repeat_total += repeats
            flag = f"  {len(todo)} ids not found" if todo else ""
            if repeats:
                flag += f"  ({repeats} repeated id rows skipped)"
            print(f"{config:24s} {h_n:>7,} {l_n:>7,} {sel['honest_pool']:>8,} "
                  f"{sel['leaked_pool']:>8,} {touched:>7}{flag}")
            summary[config] = {
                "honest_n": h_n, "leaked_n": l_n,
                "honest_pool": sel["honest_pool"], "leaked_pool": sel["leaked_pool"],
                "leaked_excluded": sel["leaked_excluded"],
                "missing": len(todo), "repeated_id_rows": repeats,
            }
    finally:
        hw.close()
        lw.close()

    excluded = sorted(c for c, v in summary.items() if v.get("leaked_excluded"))
    total_h = sum(v["honest_n"] for v in summary.values())
    total_l = sum(v["leaked_n"] for v in summary.values())

    print("-" * 68)
    print(f"honest  {total_h:>7,} utts -> {h_path} "
          f"({h_path.stat().st_size / 1e6:.0f} MB)")
    print(f"leaked  {total_l:>7,} utts -> {l_path} "
          f"({l_path.stat().st_size / 1e6:.0f} MB)")
    if excluded:
        print(f"\n{len(excluded)} config(s) excluded from the leak table for a "
              f"leaked pool under {args.min_leaked}:")
        print("  " + ", ".join(excluded))
        print("  They still appear in the honest table. evaluate.py compares the")
        print("  two sets on the intersection so the micro-average stays fair.")
    if missing_total:
        print(f"\n{missing_total} selected id(s) were never found in the shards. "
              f"That means the manifest and {args.data_dir} disagree — "
              f"rebuild the manifest with --local-root before trusting the split.")
    if repeat_total:
        print(f"\n{repeat_total} row(s) repeated an id already taken — kept once each.")
        print("  `id` is not a unique key in every config: where one shard holds two")
        print("  recording projects, the same verse appears once per version.")
        print("  Worth knowing beyond this file — asr/train.load_split selects by id")
        print("  too, so those configs contribute more rows than the mixture's")
        print("  target hours. Check with:")
        print("    python -m kasa42.data.build_manifest --check-ids")

    (out_dir / "testset_summary.json").write_text(json.dumps({
        "seed": args.seed, "max_per_config": args.max_per_config,
        "min_leaked": args.min_leaked, "data_dir": args.data_dir,
        "match_books": args.match_books,
        "honest_total": total_h, "leaked_total": total_l,
        "leak_excluded": excluded, "configs": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'testset_summary.json'}")


if __name__ == "__main__":
    main()
