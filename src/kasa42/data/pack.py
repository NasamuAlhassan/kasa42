"""Pack a CSV-plus-audio corpus into Hub-ready parquet shards.

Uploading an audio dataset as loose WAVs fails, and fails in ways that are not
obvious until it is public:

  * A Hub repo is git-backed and caps each directory at **10,000 files**. A
    24,597-clip train directory is rejected mid-push, leaving a partial upload.
  * Subdirectories get read as **class labels**, so the dataset lands as
    `soundfolder` — audio classification — with a null `label` column and the
    transcripts nowhere in sight.
  * 30,000 small files are slow to clone and impossible to stream.

Parquet shards solve all three, which is why `ghana-speech` ships that way and
why its metadata could be audited without downloading 222 GB of audio.

    python -m kasa42.data.pack \\
        --csv asr_dataset/metadata.csv --root asr_dataset \\
        --out hub_upload --shard-mb 400

The `huggingface` schema metadata is written by hand rather than through
`datasets`, so packing needs no torchcodec — `datasets` 5.x pulls that in merely
to *encode* an audio column, and it is not otherwise required here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def hf_features(text_col: str, extra: list[str]) -> bytes:
    """The metadata blob that makes the Hub render `audio` as playable audio."""
    feats: dict[str, dict] = {
        "audio": {"_type": "Audio"},
        text_col: {"dtype": "string", "_type": "Value"},
    }
    for c in extra:
        feats[c] = {"dtype": "string", "_type": "Value"}
    return json.dumps({"info": {"features": feats}}).encode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--path-col", default="file")
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--extra-cols", nargs="*",
                    default=["id", "book", "chapter", "verse", "duration_ms"])
    ap.add_argument("--shard-mb", type=float, default=400.0,
                    help="Target shard size. The Hub is happiest in the "
                         "200-500 MB range.")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.csv} is empty")

    extra = [c for c in args.extra_cols if c in rows[0]]
    schema = pa.schema(
        [("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
         (args.text_col, pa.string())]
        + [(c, pa.string()) for c in extra],
        metadata={b"huggingface": hf_features(args.text_col, extra)})

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r.get(args.split_col) or "train"].append(r)

    limit = args.shard_mb * 1e6
    written: dict[str, int] = {}
    missing = 0

    for split, items in sorted(by_split.items()):
        buf: list[dict] = []
        size = 0
        shards: list[list[dict]] = []
        for r in items:
            wav = root / r[args.path_col]
            try:
                raw = wav.read_bytes()
            except OSError:
                missing += 1
                continue
            buf.append({
                "audio": {"bytes": raw, "path": Path(r[args.path_col]).name},
                args.text_col: (r.get(args.text_col) or "").strip(),
                **{c: str(r.get(c, "")) for c in extra},
            })
            size += len(raw)
            if size >= limit:
                shards.append(buf)
                buf, size = [], 0
        if buf:
            shards.append(buf)

        for i, chunk in enumerate(shards):
            name = f"{split}-{i:05d}-of-{len(shards):05d}.parquet"
            pq.write_table(pa.Table.from_pylist(chunk, schema=schema),
                           out / "data" / name, compression="zstd")
            mb = (out / "data" / name).stat().st_size / 1e6
            print(f"  {name}  {len(chunk):>6,} rows  {mb:>7.1f} MB")
        written[split] = len(items)
        print(f"{split}: {len(items):,} rows in {len(shards)} shard(s)\n")

    if missing:
        print(f"WARNING: {missing} row(s) referenced files that do not exist\n")

    # A configs block, so the Hub reads these as splits rather than guessing.
    yaml = ["---", "configs:", "  - config_name: default", "    data_files:"]
    for split in sorted(written):
        yaml.append(f"      - split: {split}")
        yaml.append(f'        path: "data/{split}-*.parquet"')
    yaml += ["---", ""]
    readme = out / "README.md"
    if readme.exists():
        body = readme.read_text(encoding="utf-8")
        body = body.split("---", 2)[-1] if body.startswith("---") else body
    else:
        body = "\n# Dataset\n\nReplace this with your dataset card.\n"
    readme.write_text("\n".join(yaml) + body.lstrip("\n"), encoding="utf-8")

    total = sum(f.stat().st_size for f in (out / "data").glob("*.parquet"))
    print(f"{sum(written.values()):,} rows, {total/1e9:.2f} GB -> {out}")
    print(f"wrote {readme} with a configs block for the splits")
    print(f"\n  hf upload <user>/<dataset> {out} --repo-type dataset")


if __name__ == "__main__":
    main()
