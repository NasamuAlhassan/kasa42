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
import gc
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


# fsspec buffers file blocks in RAM. Left at its default, N concurrent readers
# against 1.5 GB shards will exhaust a 16 GB box — this OOM'd and restarted a
# Kaggle kernel at 16 workers. Cap the block size so memory stays bounded by
# roughly (workers x BLOCK_SIZE) rather than by the size of the shards.
BLOCK_SIZE = 4 * 1024 * 1024


def _scan(pf: pq.ParquetFile) -> dict[str, list]:
    """Pull the metadata columns out of an open parquet file.

    Stream in batches rather than pf.read(), which materialises every row group
    at once. Peak memory becomes one batch of small metadata columns instead of
    a whole shard's worth — this is what was exhausting RAM and restarting
    kernels.
    """
    present = [c for c in COLS if c in pf.schema_arrow.names]
    d: dict[str, list] = {c: [] for c in present}
    for rb in pf.iter_batches(batch_size=2048, columns=present):
        chunk = rb.to_pydict()
        for c in present:
            d[c].extend(chunk[c])
        del rb, chunk
    return d


def read_shard(config: str, iso: str, path: str, shard: int, retries: int = 4,
               local: bool = False) -> pa.Table:
    """Read one shard's metadata.

    `local=True` reads `path` straight off the filesystem — the case when the
    host has the dataset pre-staged (the H200 box mounts it at
    /data/ghana-speech). No transport, so no retry loop: a failure there is a
    real error and should surface immediately rather than being slept over.

    Otherwise stream it out of the Hub. A fresh HfFileSystem per attempt
    matters: once an httpx client has been closed by a DNS failure it stays
    closed, and every later call through it fails with 'Cannot send a request,
    as the client has been closed'.
    """
    if local:
        d = _scan(pq.ParquetFile(path))
    else:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                fs = HfFileSystem()
                with fs.open(f"datasets/{REPO}/{path}", "rb",
                             block_size=BLOCK_SIZE, cache_type="readahead") as fh:
                    d = _scan(pq.ParquetFile(fh))
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


def build_config(config: str, iso: str, files: list[str], workers: int,
                 local: bool = False) -> pa.Table:
    tables: list[pa.Table] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(read_shard, config, iso, p, shard=i, local=local): p
                for i, p in enumerate(files)}
        for fut in as_completed(futs):
            tables.append(fut.result())  # a failed shard aborts this config only
    out = pa.concat_tables(tables)
    tables.clear()
    gc.collect()
    return out


def check_ids(manifest: str) -> None:
    """Report configs where `id` repeats, and what it costs downstream.

    Everything that selects segments — the mixture, `train.load_split`, the test
    sets — treats `id` as a key. Where a config holds two recording projects,
    the same verse can appear once per version under one id, so selecting an id
    pulls in more than one row and a language contributes more than its target
    hours. Silent, and invisible in the hours accounting, which counts rows.
    """
    from collections import Counter

    t = pq.read_table(manifest, columns=["config", "id", "duration"])
    rows: dict[str, Counter] = {}
    hours: dict[str, float] = {}
    for c, i, d in zip(t.column("config").to_pylist(), t.column("id").to_pylist(),
                       t.column("duration").to_pylist()):
        rows.setdefault(c, Counter())[i] += 1
        hours[c] = hours.get(c, 0.0) + float(d or 0.0) / 3600

    bad = {c: n for c, n in rows.items() if len(n) != sum(n.values())}
    if not bad:
        print(f"{manifest}: `id` is unique in all {len(rows)} configs.")
        return

    print(f"{'config':24s} {'rows':>9s} {'unique':>9s} {'factor':>7s} {'h if deduped':>13s}")
    print("-" * 66)
    for c in sorted(bad, key=lambda c: -(sum(rows[c].values()) / len(rows[c]))):
        n, u = sum(rows[c].values()), len(rows[c])
        print(f"{c:24s} {n:>9,} {u:>9,} {n/u:>6.2f}x {hours[c]*u/n:>12.1f}")
    print("-" * 66)
    print(f"{len(bad)} of {len(rows)} configs repeat ids.")
    print("Selecting by id therefore over-samples them relative to the mixture's")
    print("target hours. The last column is what each would contribute if one row")
    print("per id were taken.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/manifest.parquet")
    ap.add_argument("--parts-dir", default="results/manifest_parts")
    ap.add_argument("--workers", type=int, default=6,
                    help="HTTP-bound, but each worker buffers ~4 MB blocks. "
                         "6 is safe on a 16 GB Kaggle kernel; raise only with headroom.")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--local-root",
                    help="Read pre-staged shards from <root>/<config>/*.parquet "
                         "instead of the Hub (the H200 box has the dataset at "
                         "/data/ghana-speech). Disk-bound, so it takes minutes.")
    ap.add_argument("--force", action="store_true", help="Rebuild configs already on disk.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip fetching; just merge whatever parts exist.")
    ap.add_argument("--check-ids", action="store_true",
                    help="Report configs where `id` is not unique, and exit.")
    args = ap.parse_args()

    if args.check_ids:
        check_ids(args.out)
        return

    parts = Path(args.parts_dir)
    parts.mkdir(parents=True, exist_ok=True)

    by_config: dict[str, list[str]] = {}
    if args.local_root:
        root = Path(args.local_root)
        if not root.is_dir():
            raise SystemExit(f"--local-root {root} is not a directory")
        # One level of config directories, each holding the shards. Anything
        # without parquet in it is not a config — skip quietly rather than
        # emitting an empty part file for it.
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            shards = sorted(str(p) for p in d.glob("*.parquet"))
            if shards:
                by_config[d.name] = shards
        if not by_config:
            raise SystemExit(
                f"no <config>/*.parquet under {root} — check the layout with "
                f"`ls {root}` and point --local-root at the level above the "
                f"config directories")
        print(f"local root {root}: {len(by_config)} configs, "
              f"{sum(len(v) for v in by_config.values())} shards")
    else:
        api = HfApi()
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
                table = build_config(config, iso, files, args.workers,
                                     local=bool(args.local_root))
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

    # Stream part-by-part rather than concatenating all 42 in memory first.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    seconds = 0.0
    writer: pq.ParquetWriter | None = None
    try:
        for p in have:
            t = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(out, t.schema, compression="zstd")
            writer.write_table(t)
            rows += t.num_rows
            seconds += pa.compute.sum(t.column("duration")).as_py() or 0.0
            del t
    finally:
        if writer is not None:
            writer.close()

    print(f"\nmerged {len(have)} parts -> {out}")
    print(f"  {rows:,} rows  {len(have)} configs  {seconds/3600:,.1f} h  "
          f"{out.stat().st_size/1e6:.1f} MB")
    if len(have) < 42:
        print(f"  NOTE: only {len(have)}/42 configs present — rerun to fetch the rest")


if __name__ == "__main__":
    main()
