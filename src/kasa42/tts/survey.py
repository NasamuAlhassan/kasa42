"""Which of the 42 languages are usable for single-speaker TTS?

Kusaal is 100% one version code, and that version turned out to contain two
narrators — Old Testament and New Testament read by different people, split
cleanly by ECAPA-TDNN embeddings. Fine-tuning single-speaker VITS across that
boundary averages two voices into a muddy one, and nothing in the metadata warns
you: the version field says one project.

Nobody has checked the other 41. This does, in one pass, on CPU.

For each config it samples clips per book, embeds them, clusters against a
threshold calibrated from same-book pairs, and reports the hours behind the
largest cluster — which is the number that decides whether a voice can be
trained at all. VITS wants roughly 10 h of one speaker.

    python -m kasa42.tts.survey --data-dir /data/ghana-speech

Roughly half an hour for all 42, no GPU, safe to run beside anything else. The
output is a ranking of the corpus by TTS readiness, which is a dataset result
rather than a model one: it holds for anyone building on `ghana-speech`, not
just for this project.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kasa42.tts.speaker_check import (
    ecapa_embeddings,
    fallback_embeddings,
    load_clips,
    pick_clips,
)


def analyse(config: str, manifest: str, data_dir: str, hours: dict[str, float],
            per_book: int, max_books: int, min_sec: float, max_sec: float,
            seed: int) -> dict:
    """Cluster one config's books by speaker. Returns a summary dict."""
    picks = pick_clips(manifest, config, min_sec, max_sec, max_books, per_book, seed)
    if len({b for b, _, _ in picks}) < 2:
        return {"config": config, "error": "fewer than 2 usable books"}

    clips = load_clips(data_dir, config, picks)
    if len(clips) < 4:
        return {"config": config, "error": f"only {len(clips)} clips decoded"}

    books = [b for b, _, _ in clips]
    emb = ecapa_embeddings([w for _, _, w in clips])
    method = "ECAPA-TDNN"
    if emb is None:
        method = "F0+MFCC"
        emb = fallback_embeddings([w for _, _, w in clips])

    norms = np.linalg.norm(emb, axis=1)
    ok = norms > 1e-6
    emb, books = emb[ok], [b for b, k in zip(books, ok) if k]
    if len(set(books)) < 2:
        return {"config": config, "error": "no usable audio"}

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T
    bk = np.array(books)
    within = sim[(bk[:, None] == bk[None, :]) & ~np.eye(len(bk), dtype=bool)]
    across = sim[bk[:, None] != bk[None, :]]
    if not within.size:
        return {"config": config, "error": "no same-book pairs to calibrate on"}

    threshold = float(1.0 - np.percentile(within, 5))

    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average").fit_predict(emb)

    votes: dict[str, list[int]] = defaultdict(list)
    for lab, book in zip(labels, books):
        votes[book].append(int(lab))
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for book, labs in votes.items():
        by_cluster[max(set(labs), key=labs.count)].append(book)

    ranked = sorted(by_cluster.items(),
                    key=lambda kv: -sum(hours.get(b, 0.0) for b in kv[1]))
    top_books = ranked[0][1]
    top_h = sum(hours.get(b, 0.0) for b in top_books)
    total_h = sum(hours.get(b, 0.0) for b in votes)

    return {
        "config": config, "method": method,
        "sampled_books": len(votes), "clips": len(books),
        "n_speakers": len(by_cluster),
        "largest_cluster_books": sorted(top_books),
        "largest_cluster_hours": round(top_h, 1),
        "sampled_hours": round(total_h, 1),
        "single_speaker_fraction": round(top_h / total_h, 3) if total_h else 0.0,
        "same_book_similarity": round(float(within.mean()), 3),
        "cross_book_similarity": round(float(across.mean()), 3),
        "threshold": round(threshold, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/manifest.parquet")
    ap.add_argument("--data-dir", default="/data/ghana-speech")
    ap.add_argument("--out", default="results/tts_survey.json")
    ap.add_argument("--configs", nargs="*", help="Default: every config.")
    ap.add_argument("--per-book", type=int, default=2)
    ap.add_argument("--max-books", type=int, default=24)
    ap.add_argument("--min-sec", type=float, default=4.0)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--viable-hours", type=float, default=10.0,
                    help="Hours of one speaker VITS needs to be worth attempting.")
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()

    t = pq.read_table(args.manifest, columns=["config", "book", "duration"])
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for c, b, d in zip(t.column("config").to_pylist(), t.column("book").to_pylist(),
                       t.column("duration").to_pylist()):
        hours[c][b] += float(d or 0.0) / 3600

    configs = args.configs or sorted(hours)
    print(f"surveying {len(configs)} configs, {args.per_book} clips/book, "
          f"up to {args.max_books} books each\n")

    rows, t0 = [], time.time()
    for i, config in enumerate(configs, 1):
        try:
            r = analyse(config, args.manifest, args.data_dir, hours[config],
                        args.per_book, args.max_books, args.min_sec,
                        args.max_sec, args.seed)
        except Exception as e:  # noqa: BLE001 - one bad config must not end the sweep
            r = {"config": config, "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        el = time.time() - t0
        eta = (len(configs) - i) * el / max(i, 1)
        note = r.get("error") or (f"{r['n_speakers']} speaker(s), "
                                  f"{r['largest_cluster_hours']:.1f} h largest")
        print(f"[{i:>2}/{len(configs)}] {config:24s} {note}   ETA {eta/60:.0f} min")

    good = [r for r in rows if "error" not in r]
    good.sort(key=lambda r: -r["largest_cluster_hours"])

    print(f"\n{'config':24s} {'spk':>4s} {'largest h':>10s} {'of total':>9s} "
          f"{'TTS?':>6s}")
    print("-" * 60)
    for r in good:
        viable = r["largest_cluster_hours"] >= args.viable_hours
        print(f"{r['config']:24s} {r['n_speakers']:>4d} "
              f"{r['largest_cluster_hours']:>10.1f} "
              f"{r['single_speaker_fraction']:>8.0%} "
              f"{'yes' if viable else 'no':>6s}")
    print("-" * 60)

    viable = [r for r in good if r["largest_cluster_hours"] >= args.viable_hours]
    single = [r for r in good if r["n_speakers"] == 1]
    print(f"{len(viable)}/{len(good)} configs have {args.viable_hours:.0f} h+ behind "
          f"one speaker — enough to attempt single-speaker VITS.")
    print(f"{len(single)}/{len(good)} look like a single narrator throughout; "
          f"the rest need filtering to their largest cluster first.")
    if bad := [r for r in rows if "error" in r]:
        print(f"{len(bad)} could not be analysed: "
              + ", ".join(f"{r['config']} ({r['error']})" for r in bad[:4]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "viable_hours": args.viable_hours, "per_book": args.per_book,
        "n_viable": len(viable), "n_single_speaker": len(single),
        "configs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
