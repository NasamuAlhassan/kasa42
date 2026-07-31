"""Package an outside corpus into the test-set schema, so it can be scored here.

Evaluating on somebody else's recordings is the only way to find out whether a
model has learned a language or learned a recording project. `ghana-speech`'s
Kusaal is one ministry's audio; an independent corpus of the same language, cut
by a different pipeline, is a different question entirely.

Takes a CSV manifest plus a directory of WAVs and writes the same parquet
`asr/testset.py` produces, so `asr/evaluate.py` consumes it unchanged.

    python -m kasa42.asr.external \\
        --csv  asr_dataset/metadata.csv --root asr_dataset \\
        --books 1CH 1TH 2CO 2SA DAN GAL JUD LAM MIC OBA SNG \\
        --config Kusaal_kus --out results/testset_kaggle.parquet

**`--books` is not a convenience.** Scoring an outside corpus on books the model
trained on measures nothing: scripture corpora share a translation, so the model
has already seen those sentences, and only the audio is new. Pass the books your
splits held out. The tool refuses to run without them for that reason.

Labelling rows with a `config` this model knows also gets language ID scored for
free — whether it still recognises the language on unfamiliar recordings is part
of the same question.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kasa42.asr.testset import SCHEMA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Manifest with audio paths and text.")
    ap.add_argument("--root", required=True, help="Directory the paths are relative to.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True,
                    help="Label rows with a config this model knows, e.g. "
                         "Kusaal_kus, so per-language scoring and LID line up.")
    ap.add_argument("--books", nargs="+", required=True,
                    help="Books to keep — must be ones the model did NOT train on.")
    ap.add_argument("--path-col", default="file")
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--book-col", default="book")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--duration-col", default="duration_ms",
                    help="Milliseconds. Omitted or absent falls back to file size.")
    ap.add_argument("--max-clips", type=int, default=4000)
    args = ap.parse_args()

    root = Path(args.root)
    keep = set(args.books)
    rows, seen_books, missing, skipped = [], Counter(), 0, 0

    with open(args.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            book = (r.get(args.book_col) or "").strip()
            if book not in keep:
                continue
            wav = root / r[args.path_col]
            if not wav.exists():
                missing += 1
                continue
            raw = wav.read_bytes()
            text = (r.get(args.text_col) or "").strip()
            if not text or len(raw) < 64:
                skipped += 1
                continue

            ms = r.get(args.duration_col)
            try:
                secs = float(ms) / 1000.0
            except (TypeError, ValueError):
                # 16 kHz mono 16-bit PCM, minus a 44-byte header.
                secs = max(len(raw) - 44, 0) / 32000.0

            rows.append({
                "config": args.config,
                "id": r.get(args.id_col) or wav.stem,
                "text": text,
                "duration": secs,
                "source_file": r.get(args.id_col) or wav.name,
                "book": book,
                "audio": raw,
            })
            seen_books[book] += 1
            if len(rows) >= args.max_clips:
                print(f"stopping at --max-clips {args.max_clips}")
                break

    if not rows:
        raise SystemExit(
            f"nothing matched books {sorted(keep)} in {args.csv} — check "
            f"--book-col (currently '{args.book_col}') and the book codes")

    absent = sorted(keep - set(seen_books))
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{len(rows):,} clips, {hours:.2f} h, {len(seen_books)} of "
          f"{len(keep)} requested books")
    for b, n in sorted(seen_books.items()):
        print(f"  {b:6s} {n:>5,}")
    if absent:
        print(f"  not in this corpus: {' '.join(absent)}")
    if missing:
        print(f"  {missing} row(s) pointed at files that do not exist")
    if skipped:
        print(f"  {skipped} row(s) had empty text or unusable audio")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out, compression="zstd")
    print(f"\nwrote {out} ({out.stat().st_size/1e6:.0f} MB)")
    print("\nScore it with:")
    print(f"  python -m kasa42.asr.evaluate --honest {out} \\")
    print(f"      --leaked {out} --checkpoint checkpoints/kasa42-asr/final.pt \\")
    print(f"      --out-dir results/eval_external")
    print("  (the leak comparison is meaningless here — read the honest table only)")


if __name__ == "__main__":
    main()
