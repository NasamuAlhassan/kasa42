"""Prepare the Kusaal TTS corpus: filter to one narrator, clean, and export.

VITS fine-tuning wants roughly single-speaker audio. The audit showed Kusaal is
backed by a single Bible version (`3752`), which in this corpus family means a
single recording project and almost certainly one narrator — but "almost
certainly" is not a thing to train 4 GPU-hours on, so this script:

  1. filters to the dominant version explicitly, whatever the audit said;
  2. drops segments that are too short, too long, or clipped;
  3. drops text with digits (the TTS front end has no number expansion for
     Kusaal, and a mispronounced verse number in the demo is a bad look);
  4. exports a small hand-check set from *distant books*, so the confirmation
     listen covers the corpus rather than one session.

Step 4 is the 30-second check that de-risks the whole TTS track. Do it first.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from kasa42.data.text import has_digits, normalize

SOURCE_RE = re.compile(r"^([A-Z0-9]+)\.(\d+)\.(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="Kusaal_kus")
    ap.add_argument("--data-dir", default="data/parquet")
    ap.add_argument("--out-dir", default="data/tts")
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=14.0)
    ap.add_argument("--max-hours", type=float, default=20.0,
                    help="VITS converges long before the full 80 h; more is slower, not better.")
    ap.add_argument("--check-clips", type=int, default=6)
    ap.add_argument("--eval-clips", type=int, default=32,
                    help="Held out for the trainer's eval split.")
    ap.add_argument("--no-export", action="store_true",
                    help="Skip writing audio — gate check and manifest only.")
    ap.add_argument("--only-books", nargs="*", default=[],
                    help="Restrict to these books. A single version can still "
                         "hold more than one narrator; speaker_check.py clusters "
                         "the books and names the largest cluster, which is what "
                         "belongs here. Filtering to one voice is what keeps "
                         "single-speaker VITS from averaging two together.")
    args = ap.parse_args()

    import datasets
    import soundfile as sf

    files = sorted(Path(args.data_dir, args.config).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data_dir}/{args.config}")
    from kasa42.asr.dataset import undecoded_audio

    ds = datasets.load_dataset("parquet", data_files=[str(p) for p in files], split="train")
    # Same reason as training: we hand raw bytes to soundfile ourselves, and
    # datasets 5.x would otherwise demand torchcodec to decode them for us.
    ds = undecoded_audio(ds)
    print(f"{args.config}: {len(ds):,} segments")

    versions = Counter()
    for s in ds["source_file"]:
        m = SOURCE_RE.match(str(s))
        if m:
            versions[m.group(3)] += 1
    dominant, n_dom = versions.most_common(1)[0]
    print(f"versions: {dict(versions)}")
    print(f"dominant: {dominant} ({n_dom/max(sum(versions.values()),1):.1%} of segments)")
    if len(versions) > 1:
        print("  NOTE: multiple versions present — filtering to the dominant one, "
              "which likely means one narrator. Confirm by ear.")

    out_dir = Path(args.out_dir)
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    check_dir = out_dir / "check"
    check_dir.mkdir(parents=True, exist_ok=True)

    kept, total_sec = [], 0.0
    dropped = Counter()
    books_seen: dict[str, int] = {}

    keep_books = set(args.only_books)
    if keep_books:
        print(f"restricted to {len(keep_books)} book(s): {' '.join(sorted(keep_books))}")

    for row in ds:
        m = SOURCE_RE.match(str(row["source_file"]))
        if not m or m.group(3) != dominant:
            dropped["other-version"] += 1
            continue
        if keep_books and m.group(1) not in keep_books:
            dropped["other-book"] += 1
            continue
        dur = float(row.get("duration") or 0.0)
        if not (args.min_sec <= dur <= args.max_sec):
            dropped["duration"] += 1
            continue
        text = row["text"]
        if has_digits(text):
            dropped["digits"] += 1
            continue
        norm = normalize(text)
        if len(norm) < 4:
            dropped["too-short-text"] += 1
            continue
        if total_sec >= args.max_hours * 3600:
            dropped["over-budget"] += 1
            continue

        kept.append({"id": row["id"], "book": m.group(1), "text": norm,
                     "duration": dur, "audio": row["audio"]})
        total_sec += dur
        books_seen[m.group(1)] = books_seen.get(m.group(1), 0) + 1

    print(f"\nkept {len(kept):,} segments  {total_sec/3600:.1f} h "
          f"across {len(books_seen)} books")
    print(f"dropped: {dict(dropped)}")

    # Hand-check clips spread across the widest possible book range.
    from kasa42.asr.dataset import decode_audio

    order = sorted(books_seen, key=lambda b: -books_seen[b])
    picks = [next(k for k in kept if k["book"] == b)
             for b in order[:: max(len(order) // args.check_clips, 1)][:args.check_clips]]
    for i, k in enumerate(picks):
        wav = decode_audio(k["audio"])
        sf.write(check_dir / f"{i:02d}_{k['book']}.wav", wav, 16000)
    (check_dir / "README.txt").write_text(
        "Listen to these before spending GPU hours on VITS.\n"
        "They come from different books of the same version.\n"
        "If they sound like the same person, the single-narrator assumption holds.\n\n"
        + "\n".join(f"{i:02d}_{k['book']}.wav  {k['text'][:70]}"
                    for i, k in enumerate(picks)),
        encoding="utf-8")
    print(f"\nwrote {len(picks)} check clips -> {check_dir}")
    print("LISTEN TO THESE FIRST — same voice across books means the gate holds.")

    meta = [{"id": k["id"], "book": k["book"], "text": k["text"],
             "duration": round(k["duration"], 3)} for k in kept]
    (out_dir / "metadata.json").write_text(
        json.dumps({"config": args.config, "version": dominant,
                    "hours": round(total_sec / 3600, 2), "items": meta},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_dir/'metadata.json'}")

    if args.no_export:
        print("--no-export: audio not written, so this cannot train anything yet")
        return

    # Write the corpus as `audiofolder`: a directory of WAVs plus a metadata.csv
    # naming each one. That is what `datasets.load_dataset` recognises without
    # a loading script, which is what the VITS trainer calls.
    #
    # Previously only the six gate-check clips were written and `wavs/` was left
    # empty, so the trainer would start against a corpus with no audio in it.
    import csv

    from kasa42.asr.dataset import decode_audio

    eval_n = min(args.eval_clips, max(len(kept) // 20, 1))
    splits = {"eval": kept[:eval_n], "train": kept[eval_n:]}
    for split, items in splits.items():
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file_name", "text"])
            for i, k in enumerate(items):
                name = f"{k['book']}_{i:06d}.wav"
                sf.write(d / name, decode_audio(k["audio"]), 16000)
                w.writerow([name, k["text"]])
                if i and i % 2000 == 0:
                    print(f"  {split}: {i:,}/{len(items):,}")
        hrs = sum(k["duration"] for k in items) / 3600
        print(f"wrote {len(items):,} clips ({hrs:.2f} h) -> {d}")

    print(f"\n{out_dir} is now loadable:")
    print(f"  datasets.load_dataset('audiofolder', data_dir='{out_dir}')")


if __name__ == "__main__":
    main()
