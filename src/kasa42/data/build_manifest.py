"""Build a compact metadata manifest for all 42 configs.

One pass over every parquet shard reading only the non-audio columns, producing
a single ~150 MB table for 1.4 M segments. Everything downstream — splits,
mixture weights, vocab — operates on this instead of touching audio.

**Resumable by design.** The first version of this script held everything in
memory and wrote once at the end; a DNS blip at shard 411 of 533 destroyed two
hours of work. Now each config is written to `results/manifest_parts/` as soon
as it completes, and a rerun skips whatever is already on disk. A network drop
costs one config, not the run.

Shards are also retried individually, because transient 5xx and DNS failures are
normal over a run this long.
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


def read_shard(config: str, iso: str, path: str, shard: int, retries: int = 4) -> pa.Table:
    """Read one shard's metadata, retrying transient network failures.

    A fresh HfFileSystem per attempt matters: once an httpx client has been
    closed by a DNS failure it stays closed, and every later call through it
    fails with 'Cannot send a request, as the client has been closed'.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            fs = HfFileSystem()
            with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                present = [c for c in COLS if c in pf.schema_arrow.names]
                d = pf.read(columns=present).to_pydict()
            break
        except Exception as e:  # noqa: BLE001 - any transport error is retryable here
            last = e
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    else:  # pragma: no cover
        raise last  # type: ignore[misc]

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


def build_config(config: str, iso: str, files: list[str], workers: int) -> pa.Table:
    tables: list[pa.Table] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(read_shard, config, iso, p, i): p
                for i, p in enumerate(files)}
        for fut in as_completed(futs):
            tables.append(fut.result())  # a failed shard aborts this config only
    return pa.concat_tables(tables)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/manifest.parquet")
    ap.add_argument("--parts-dir", default="results/manifest_parts")
    ap.add_argument("--workers", type=int, default=8, help="HTTP-bound; 8-16 is sensible.")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--force", action="store_true", help="Rebuild configs already on disk.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip fetching; just merge whatever parts exist.")
    args = ap.parse_args()

    parts = Path(args.parts_dir)
    parts.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    by_config: dict[str, list[str]] = {}
    for f in sorted(api.list_repo_files(REPO, repo_type="dataset")):
        if f.endswith(".parquet") and "/" in f:
            by_config.setdefault(f.split("/")[0], []).append(f)
    if args.configs:
        by_config = {k: v for k, v in by_config.items() if k in args.configs}

    if not args.merge_only:
        todo = [c for c in sorted(by_config)
                if args.force or not (parts / f"{c}.parquet").exists()]
        done_already = len(by_config) - len(todo)
        print(f"{len(by_config)} configs; {done_already} already on disk, {len(todo)} to fetch\n")

        t0 = time.time()
        failed: list[str] = []
        for i, config in enumerate(todo, 1):
            m = re.match(r"^(.*)_([a-z]{3})$", config)
            iso = m.group(2) if m else ""
            files = by_config[config]
            try:
                table = build_config(config, iso, files, args.workers)
                pq.write_table(table, parts / f"{config}.parquet", compression="zstd")
                hrs = pa.compute.sum(table.column("duration")).as_py() / 3600
                el = time.time() - t0
                eta = (len(todo) - i) * el / max(i, 1)
                print(f"[{i:>2}/{len(todo)}] {config:24s} {table.num_rows:>7,} rows "
                      f"{hrs:>7.1f} h  ({len(files)} shards)  ETA {eta/60:.0f} min")
            except Exception as e:
                failed.append(config)
                print(f"[{i:>2}/{len(todo)}] {config:24s} FAILED: {type(e).__name__}: {e}")
                print("            rerun this script to retry it; finished configs are kept")

        if failed:
            print(f"\n{len(failed)} config(s) failed: {', '.join(failed)}")
            print("Rerun the same command — completed configs are skipped.")

    # ------------------------------------------------------------------ merge
    have = sorted(parts.glob("*.parquet"))
    if not have:
        print("\nNo parts to merge.")
        return
    table = pa.concat_tables([pq.read_table(p) for p in have])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")

    hours = pa.compute.sum(table.column("duration")).as_py() / 3600
    n_cfg = len(set(table.column("config").to_pylist()))
    print(f"\nmerged {len(have)} parts -> {out}")
    print(f"  {table.num_rows:,} rows  {n_cfg} configs  {hours:,.1f} h  "
          f"{out.stat().st_size/1e6:.1f} MB")
    if n_cfg < 42:
        print(f"  NOTE: only {n_cfg}/42 configs present — rerun to fetch the rest")


if __name__ == "__main__":
    main()
