"""Decide whether a config is one narrator, and if not, who has the most audio.

`prepare.py` writes six clips and asks you to listen. That is the right first
gate, but ears on laptop speakers cannot reliably separate *a different person*
from *the same person recorded months apart* — and only the first ends the TTS
track. Session variation shows up as loudness, room tone, pace and energy;
speaker identity shows up in pitch and timbre. This measures it.

It also answers the more useful question. A config with three narrators is not
untrainable: it is trainable on whichever narrator has the most books. So this
samples one clip per book, embeds, clusters, and reports the hours behind each
cluster — turning a failed gate into a filter.

Speaker embeddings come from ECAPA-TDNN if speechbrain is installed, which is
what the number should rest on. Without it there is an F0-plus-MFCC fallback
that will separate obviously different voices and should not be trusted for
anything finer; it says so when it runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TARGET_SR = 16000


def pick_clips(manifest: str, config: str, min_sec: float, max_sec: float,
               max_books: int, per_book: int, seed: int) -> list[tuple[str, str, int]]:
    """Representative segments per book: (book, id, shard).

    Mid-length segments only. Very short clips carry too little voice to embed
    reliably, and very long ones are more likely to span a splice.

    Several per book, because one is not enough to two ends. A single noisy or
    unusual clip would otherwise misassign a whole book — and more importantly,
    clips from the same book give a *known same-speaker* baseline to calibrate
    the threshold against, instead of picking a number and hoping.
    """
    t = pq.read_table(manifest, columns=["config", "id", "book", "duration", "shard"])
    per: dict[str, list] = defaultdict(list)
    for c, sid, book, dur, shard in zip(
        t.column("config").to_pylist(), t.column("id").to_pylist(),
        t.column("book").to_pylist(), t.column("duration").to_pylist(),
        t.column("shard").to_pylist(),
    ):
        if c != config:
            continue
        d = float(dur or 0.0)
        if min_sec <= d <= max_sec:
            per[book].append((sid, int(shard or 0), d))

    rng = np.random.default_rng(seed)
    books = sorted(per)
    if len(books) > max_books:
        step = len(books) / max_books
        books = [books[int(i * step)] for i in range(max_books)]

    picks = []
    for book in books:
        rows = sorted(per[book])
        take = min(per_book, len(rows))
        for j in rng.choice(len(rows), size=take, replace=False):
            sid, shard, _ = rows[int(j)]
            picks.append((book, sid, shard))
    return picks


def load_clips(data_dir: str, config: str, picks: list[tuple[str, str, int]]):
    """Fetch and decode the chosen segments, reusing the test-set reader."""
    from kasa42.asr.dataset import decode_audio
    from kasa42.asr.testset import take_rows, to_schema

    want = {sid for _, sid, _ in picks}
    book_of = {sid: book for book, sid, _ in picks}
    files = sorted(Path(data_dir, config).glob("*.parquet"))
    hint = {sh for _, _, sh in picks}
    order = [i for i in range(len(files)) if i in hint]
    order += [i for i in range(len(files)) if i not in hint]

    todo = set(want)
    out = []
    for i in order:
        if not todo:
            break
        tbl = take_rows(str(files[i]), todo)
        if tbl is None:
            continue
        got = to_schema(tbl, config)
        for sid, blob in zip(got.column("id").to_pylist(),
                             got.column("audio").to_pylist()):
            if sid not in todo:
                continue
            todo.discard(sid)
            out.append((book_of[sid], sid, decode_audio(blob)))
    return out


def ecapa_embeddings(wavs: list[np.ndarray]):
    """ECAPA-TDNN speaker embeddings, or None if speechbrain is unavailable."""
    try:
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        return None

    # CPU deliberately: this is a 5 MB model over a few dozen clips, and the
    # GPU is busy with a training run that has already been OOM'd once by a
    # neighbour. Not worth the risk for a job that takes seconds either way.
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa", run_opts={"device": "cpu"})
    embs = []
    for w in wavs:
        sig = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            e = enc.encode_batch(sig).squeeze().cpu().numpy()
        embs.append(e)
    return np.stack(embs)


def fallback_embeddings(wavs: list[np.ndarray]) -> np.ndarray:
    """F0 statistics plus MFCC means. Coarse, and only honest about voices that
    differ a lot — a low-pitched man against a woman, say. It will not
    distinguish two similar speakers, and it is confounded by what was said."""
    import librosa

    feats = []
    for w in wavs:
        f0 = librosa.yin(w, fmin=60, fmax=400, sr=TARGET_SR)
        f0 = f0[np.isfinite(f0)]
        mfcc = librosa.feature.mfcc(y=w, sr=TARGET_SR, n_mfcc=13)
        feats.append(np.concatenate([
            [np.median(f0) if f0.size else 0.0, np.percentile(f0, 25) if f0.size else 0.0,
             np.percentile(f0, 75) if f0.size else 0.0],
            mfcc.mean(axis=1), mfcc.std(axis=1)]))
    x = np.stack(feats)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="Kusaal_kus")
    ap.add_argument("--manifest", default="results/manifest.parquet")
    ap.add_argument("--data-dir", default="/data/ghana-speech")
    ap.add_argument("--out", default="results/speaker_check.json")
    ap.add_argument("--max-books", type=int, default=40)
    ap.add_argument("--min-sec", type=float, default=4.0)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--per-book", type=int, default=3,
                    help="Clips per book. 2 or more is what makes the same-book "
                         "baseline — and the auto threshold — possible.")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Cosine distance at which clusters stop merging. "
                         "0 (default) derives it from the same-book pairs, which "
                         "is the only calibration this corpus can supply.")
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()

    picks = pick_clips(args.manifest, args.config, args.min_sec, args.max_sec,
                       args.max_books, args.per_book, args.seed)
    n_books = len({b for b, _, _ in picks})
    if n_books < 2:
        raise SystemExit(f"only {n_books} usable book(s) for {args.config}")
    print(f"{args.config}: {len(picks)} clips over {n_books} books "
          f"({args.per_book} per book, {args.min_sec}-{args.max_sec}s each)")

    clips = load_clips(args.data_dir, args.config, picks)
    print(f"decoded {len(clips)} clips")
    books = [b for b, _, _ in clips]
    wavs = [w for _, _, w in clips]

    emb = ecapa_embeddings(wavs)
    method = "ECAPA-TDNN"
    if emb is None:
        method = "F0+MFCC fallback"
        print("\nspeechbrain not installed — using the coarse fallback.")
        print("For a number worth quoting:  pip install speechbrain\n")
        emb = fallback_embeddings(wavs)

    # Drop degenerate embeddings before normalising. A silent or corrupt clip
    # produces a zero vector, which makes cosine distance undefined — sklearn
    # then fails with "Cosine affinity cannot be used when X contains zero
    # vectors", several frames from anything that names the clip responsible.
    norms = np.linalg.norm(emb, axis=1)
    ok = norms > 1e-6
    if not ok.all():
        dropped = [books[i] for i in np.flatnonzero(~ok)]
        print(f"\ndropped {len(dropped)} clip(s) with no usable signal "
              f"(silent or corrupt): {' '.join(sorted(set(dropped)))}")
        emb, books = emb[ok], [b for b, keep in zip(books, ok) if keep]
    if len(books) < 2 or len(set(books)) < 2:
        raise SystemExit(
            "not enough usable clips to compare — every clip was silent or "
            "degenerate. Check the audio decodes before trusting any of this.")

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T

    # Clips from the same book are the same session and so certainly the same
    # speaker. That distribution is the "same voice" reference this corpus
    # actually produces — channel noise, clip length and all — which beats
    # asserting a threshold from general ECAPA folklore.
    bk = np.array(books)
    same_book = (bk[:, None] == bk[None, :]) & ~np.eye(len(bk), dtype=bool)
    cross_book = (bk[:, None] != bk[None, :])
    within = sim[same_book]
    across = sim[cross_book]

    print(f"\nmethod: {method}")
    if within.size:
        print(f"same-book pairs   (known same speaker): mean {within.mean():.3f}  "
              f"p5 {np.percentile(within, 5):.3f}  n={within.size}")
    print(f"cross-book pairs                        : mean {across.mean():.3f}  "
          f"p5 {np.percentile(across, 5):.3f}  n={across.size}")

    threshold = args.threshold
    if args.threshold <= 0:
        if not within.size:
            raise SystemExit("--threshold auto needs --per-book 2 or more")
        # Split where same-speaker pairs bottom out: books whose clips sit below
        # what this corpus produces for one voice are a different voice.
        threshold = float(1.0 - np.percentile(within, 5))
        print(f"auto threshold: cosine distance {threshold:.3f} "
              f"(= similarity {1 - threshold:.3f}, the 5th percentile of same-book pairs)")

    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average").fit_predict(emb)

    # Hours behind each cluster, so "how many narrators" becomes "is the biggest
    # one enough to train on".
    t = pq.read_table(args.manifest, columns=["config", "book", "duration"])
    hours: dict[str, float] = defaultdict(float)
    for c, b, d in zip(t.column("config").to_pylist(), t.column("book").to_pylist(),
                       t.column("duration").to_pylist()):
        if c == args.config:
            hours[b] += float(d or 0.0) / 3600

    # A book is assigned by majority vote of its clips. A book whose clips
    # disagree is more interesting than one that lands cleanly — it means either
    # a genuinely mixed book or an unreliable clip — so those are named.
    votes: dict[str, list[int]] = defaultdict(list)
    for lab, book in zip(labels, books):
        votes[book].append(int(lab))

    by_cluster: dict[int, list[str]] = defaultdict(list)
    split_books = []
    for book, labs in votes.items():
        counts = defaultdict(int)
        for x in labs:
            counts[x] += 1
        winner = max(counts, key=lambda k: (counts[k], -k))
        by_cluster[winner].append(book)
        if len(counts) > 1:
            split_books.append((book, dict(counts)))

    print(f"\n{len(by_cluster)} cluster(s) over {len(votes)} books")
    print(f"{'cluster':>8s} {'books':>6s} {'hours':>8s}  {'books':<40s}")
    print("-" * 70)
    ranked = sorted(by_cluster.items(), key=lambda kv: -sum(hours[b] for b in kv[1]))
    for lab, bs in ranked:
        h = sum(hours[b] for b in bs)
        print(f"{lab:>8d} {len(bs):>6d} {h:>8.1f}  {' '.join(sorted(bs))[:40]:<40s}")
    if split_books:
        print(f"\n{len(split_books)} book(s) whose clips disagreed "
              f"(assigned by majority): "
              + ", ".join(f"{b} {c}" for b, c in split_books[:6]))

    top_books, top_h = ranked[0][1], sum(hours[b] for b in ranked[0][1])
    print("-" * 70)
    if len(by_cluster) == 1:
        print("One cluster: consistent with a single narrator. The TTS gate holds.")
    else:
        print(f"{len(by_cluster)} clusters — more than one voice in this config.")
        print(f"Largest covers {len(top_books)} sampled books, ~{top_h:.1f} h.")
        print("VITS wants roughly 10 h of one speaker, so "
              + ("that is enough — fine-tune on those books alone."
                 if top_h >= 10 else
                 "that is thin. Consider dropping the TTS track."))
        print(f"\n  python -m kasa42.tts.prepare --config {args.config} \\")
        print(f"      --data-dir {args.data_dir} --only-books {' '.join(sorted(top_books))}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": args.config, "method": method,
        "threshold": round(threshold, 4), "threshold_auto": args.threshold <= 0,
        "per_book": args.per_book, "clips": len(books),
        "n_clusters": len(by_cluster), "sampled_books": len(votes),
        "similarity": {
            "same_book_mean": float(within.mean()) if within.size else None,
            "same_book_p5": float(np.percentile(within, 5)) if within.size else None,
            "cross_book_mean": float(across.mean()),
            "cross_book_p5": float(np.percentile(across, 5)),
        },
        "books_with_disagreeing_clips": [b for b, _ in split_books],
        "clusters": {str(k): {"books": sorted(v),
                              "hours": round(sum(hours[b] for b in v), 2)}
                     for k, v in by_cluster.items()},
        "largest_cluster_books": sorted(top_books),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
