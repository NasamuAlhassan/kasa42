"""How many recording projects back each language?

`source_file` is `BOOK.CHAPTER.VERSION`. The VERSION field identifies the Bible
edition, and in this corpus family one edition means one recording project,
which usually means one narrator.

This matters twice:

  * **TTS.** VITS fine-tuning wants roughly single-speaker audio. A language with
    one version is a good candidate; a language blending several is not, unless
    we filter to the dominant one first.
  * **Splits.** If a language contains two versions of the same book, then the
    *same scripture text* exists twice, read by different narrators. Splitting on
    (version, book) would put Genesis-v1 in train and Genesis-v2 in test — an
    identical reference transcript on both sides, which is a worse leak than the
    chapter one because the model can simply memorise the text.

    The fix is to split on **book alone**, so every version of a book travels
    together into the same split. That is what data/splits.py does.

Metadata columns only — no audio is fetched.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "ghananlpcommunity/ghana-speech"
SOURCE_RE = re.compile(r"^([A-Z0-9]+)\.(\d+)\.(\d+)$")


def scan(fs: HfFileSystem, path: str) -> tuple[Counter, dict[str, set]]:
    with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
        d = pq.ParquetFile(fh).read(columns=["source_file"]).to_pydict()
    versions: Counter[str] = Counter()
    books: dict[str, set] = defaultdict(set)
    for s in d["source_file"]:
        m = SOURCE_RE.match(str(s))
        if m:
            versions[m.group(3)] += 1
            books[m.group(3)].add(m.group(1))
        else:
            versions[f"UNPARSED:{s}"] += 1
    return versions, books


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-per-config", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results/versions.json")
    args = ap.parse_args()

    api, fs = HfApi(), HfFileSystem()
    by_config: dict[str, list[str]] = defaultdict(list)
    for f in sorted(api.list_repo_files(REPO, repo_type="dataset")):
        if f.endswith(".parquet") and "/" in f:
            by_config[f.split("/")[0]].append(f)

    jobs = []
    for config, files in sorted(by_config.items()):
        n = len(files)
        # Spread the sample across the shard range; versions may be contiguous.
        idxs = sorted({int(i * (n - 1) / max(args.shards_per_config - 1, 1))
                       for i in range(args.shards_per_config)}) if n > 1 else [0]
        jobs += [(config, files[i]) for i in idxs]

    print(f"Scanning {len(jobs)} shards across {len(by_config)} configs\n")

    versions: dict[str, Counter] = defaultdict(Counter)
    books: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan, fs, p): c for c, p in jobs}
        for fut in as_completed(futs):
            config = futs[fut]
            try:
                v, b = fut.result()
            except Exception as e:
                print(f"  {config}: FAILED {type(e).__name__}: {e}")
                continue
            versions[config].update(v)
            for ver, bk in b.items():
                books[config][ver] |= bk

    print(f"{'config':24s} {'vers':>5s} {'dominant':>9s} {'books':>6s}  detail")
    print("-" * 78)
    multi, overlap = [], []
    for config in sorted(versions):
        c = versions[config]
        total = sum(c.values()) or 1
        top, top_n = c.most_common(1)[0]
        share = top_n / total
        nbooks = len(books[config].get(top, set()))
        detail = ""
        if len(c) > 1:
            multi.append(config)
            detail = " ".join(f"{k}:{v/total:.0%}" for k, v in c.most_common(4))
            # Same book under two versions => identical text recorded twice.
            # Book-level grouping keeps them together; a version-aware split
            # would split them apart and leak the transcript.
            sets = list(books[config].values())
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    if sets[i] & sets[j]:
                        if config not in overlap:
                            overlap.append(config)
        print(f"{config:24s} {len(c):>5d} {share:>8.1%} {nbooks:>6d}  {detail}")

    print(f"\nsingle-version configs : {len(versions) - len(multi)} / {len(versions)}")
    print(f"multi-version configs  : {len(multi)}")
    if multi:
        print(f"  {', '.join(multi)}")
    print(f"\nconfigs where one book appears under >1 version: {len(overlap)}")
    if overlap:
        print(f"  {', '.join(overlap)}")
        print("  -> the same scripture text exists twice in different voices.")
        print("  -> split by BOOK ALONE so all versions travel together;")
        print("     a (version, book) split would leak identical transcripts.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "versions": {k: dict(v) for k, v in sorted(versions.items())},
        "books_per_version": {k: {vv: sorted(bb) for vv, bb in v.items()}
                              for k, v in sorted(books.items())},
        "multi_version": multi,
        "book_version_overlap": overlap,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
