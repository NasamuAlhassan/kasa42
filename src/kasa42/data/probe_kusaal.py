"""One-off probe: confirm Bible provenance details that shape the plan.

Checks, in order of consequence:
  1. Is the version suffix constant? A single recording project implies a single
     narrator, which decides whether Kusaal VITS fine-tuning is viable.
  2. How many distinct books/chapters? Splitting by *book* is a stronger
     source-disjoint split than by chapter (adjacent chapters share vocabulary).
  3. Is the text all-caps? Casing wrecks CTC vocab if inconsistent.
"""

from __future__ import annotations

import re
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "ghananlpcommunity/ghana-speech"
SHARDS = [f"Kusaal_kus/train-{i:05d}-of-00022.parquet" for i in (0, 7, 15, 21)]

fs = HfFileSystem()
books: Counter[str] = Counter()
versions: Counter[str] = Counter()
chapters: set[str] = set()
case = Counter()
texts: list[str] = []

for path in SHARDS:
    with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
        d = pq.ParquetFile(fh).read(columns=["text", "source_file", "duration"]).to_pydict()
    for s in d["source_file"]:
        s = str(s)
        chapters.add(s)
        m = re.match(r"^([A-Z0-9]+)\.(\d+)\.(\d+)$", s)
        if m:
            books[m.group(1)] += 1
            versions[m.group(3)] += 1
        else:
            books[f"UNPARSED:{s}"] += 1
    for t in d["text"]:
        t = str(t)
        texts.append(t)
        letters = [c for c in t if c.isalpha()]
        if not letters:
            case["no-letters"] += 1
        elif all(c.isupper() for c in letters):
            case["ALL CAPS"] += 1
        elif all(c.islower() for c in letters):
            case["all lower"] += 1
        else:
            case["Mixed"] += 1

print(f"shards sampled : {len(SHARDS)} of 22")
print(f"segments       : {len(texts):,}")
print(f"distinct chaps : {len(chapters):,}")
print(f"distinct books : {len([b for b in books if not b.startswith('UNPARSED')])}")
print(f"\nversion suffix : {dict(versions)}")
print(f"  -> {'SINGLE project (one narrator likely)' if len(versions) == 1 else 'MULTIPLE projects — check narrators'}")

print(f"\ncasing:")
for k, v in case.most_common():
    print(f"  {k:12s} {v:>7,} ({v/len(texts):>6.1%})")

print(f"\ntop books: {[b for b, _ in books.most_common(12)]}")
print(f"\nsample texts:")
for t in texts[:8]:
    print(f"  {t[:100]}")

charset = sorted(set("".join(texts)))
print(f"\ncharset ({len(charset)}): {''.join(charset)}")
