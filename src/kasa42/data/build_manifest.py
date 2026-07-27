"""Build a compact metadata manifest for all 42 configs.

One pass over every parquet shard reading only the non-audio columns, producing
a single ~150 MB table for 1.4 M segments. Everything downstream — splits,
mixture weights, vocab — operates on this instead of touching audio.

Doing this locally before the GPU window means the splits are frozen and
reproducible on Thursday rather than being derived under time pressure. It is
HTTP-bound, so it runs threaded and is safe to leave in the background.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "ghananlpcommunity/ghana-speech"
COLS = ["id", "language", "text", "duration", "source_file"]
SOURCE_RE = re.compile(r"^([A-Z0-9]+)\.(\d+)\.(\d+)$")

SCHEMA = pa.schema([
    ("config", pa.string()),
    ("iso", pa.string()),
    ("shard", pa.int16()),
    ("id", pa.string()),
    ("text", pa.string()),
    ("duration", pa.float32()),
    ("source_file", pa.string()),
    ("book", pa.string()),
    ("chapter", pa.int32()),
    ("version", pa.string()),
])


def parse_source(s: str) -> tuple[str, int, str]:
    """`1CO.15.3752` -> ('1CO', 15, '3752'). Unparseable values fall back to the
    whole string as the book, so they still group into exactly one split."""
    m = SOURCE_RE.match(s)
    if not m:
        return s, -1, ""
    return m.group(1), int(m.group(2)), m.group(3)


def read_shard(fs: HfFileSystem, config: str, iso: str, path: str, shard: int) -> pa.Table:
    with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        present = [c for c in COLS if c in pf.schema_arrow.names]
        d = pf.read(columns=present).to_pydict()

    n = len(d.get("id") or d.get("text") or [])
    src = [str(x) for x in d.get("source_file", [""] * n)]
    parsed = [parse_source(s) for s in src]

    return pa.Table.from_pydict({
        "config": [config] * n,
        "iso": [iso] * n,
        "shard": [shard] * n,
        "id": [str(x) for x in d.get("id", [""] * n)],
        "text": [str(x) for x in d.get("text", [""] * n)],
        "duration": [float(x) if x is not None else 0.0 for x in d.get("duration", [0.0] * n)],
        "source_file": src,
        "book": [p[0] for p in parsed],
        "chapter": [p[1] for p in parsed],
        "version": [p[2] for p in parsed],
    }, schema=SCHEMA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/manifest.parquet")
    ap.add_argument("--workers", type=int, default=8, help="HTTP-bound; 8-16 is sensible.")
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()

    api, fs = HfApi(), HfFileSystem()
    jobs: list[tuple[str, str, str, int]] = []
    seen: dict[str, int] = {}
    for f in sorted(api.list_repo_files(REPO, repo_type="dataset")):
        if not f.endswith(".parquet") or "/" not in f:
            continue
        config = f.split("/")[0]
        if args.configs and config not in args.configs:
            continue
        m = re.match(r"^(.*)_([a-z]{3})$", config)
        iso = m.group(2) if m else ""
        idx = seen.get(config, 0)
        seen[config] = idx + 1
        jobs.append((config, iso, f, idx))

    print(f"Reading metadata from {len(jobs)} shards across {len(seen)} configs "
          f"({args.workers} workers)\n")

    tables: list[pa.Table] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(read_shard, fs, c, i, p, s): (c, p) for c, i, p, s in jobs}
        for fut in as_completed(futs):
            config, path = futs[fut]
            done += 1
            try:
                tables.append(fut.result())
            except Exception as e:
                print(f"  [{done}/{len(jobs)}] FAILED {path}: {type(e).__name__}: {e}")
                continue
            if done % 25 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print(f"  [{done:>3}/{len(jobs)}] {rate:.1f} shard/s  ETA {eta/60:.1f} min")

    table = pa.concat_tables(tables)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")

    size_mb = out.stat().st_size / 1e6
    hours = pa.compute.sum(table.column("duration")).as_py() / 3600
    print(f"\nWrote {out}  {table.num_rows:,} rows  {hours:,.1f} h  {size_mb:.1f} MB "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
