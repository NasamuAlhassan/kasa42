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


def pick_one_per_book(manifest: str, config: str, min_sec: float, max_sec: float,
                      max_books: int, seed: int) -> list[tuple[str, str, int]]:
    """One representative segment per book: (book, id, shard).

    Mid-length segments only. Very short clips carry too little voice to embed
    reliably, and very long ones are more likely to span a splice.
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
    picks = []
    for book in sorted(per):
        rows = sorted(per[book])
        sid, shard, _ = rows[int(rng.integers(len(rows)))]
        picks.append((book, sid, shard))
    if len(picks) > max_books:
        step = len(picks) / max_books
        picks = [picks[int(i * step)] for i in range(max_books)]
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

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa")
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
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="Cosine distance at which clusters stop merging. "
                         "0.75 is a workable ECAPA split; raise it to merge more.")
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()

    picks = pick_one_per_book(args.manifest, args.config, args.min_sec,
                              args.max_sec, args.max_books, args.seed)
    if len(picks) < 2:
        raise SystemExit(f"only {len(picks)} usable book(s) for {args.config}")
    print(f"{args.config}: sampling {len(picks)} books "
          f"({args.min_sec}-{args.max_sec}s each)")

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

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T
    off = sim[~np.eye(len(sim), dtype=bool)]

    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=args.threshold,
        metric="cosine", linkage="average").fit_predict(emb)

    # Hours behind each cluster, so "how many narrators" becomes "is the biggest
    # one enough to train on".
    t = pq.read_table(args.manifest, columns=["config", "book", "duration"])
    hours: dict[str, float] = defaultdict(float)
    for c, b, d in zip(t.column("config").to_pylist(), t.column("book").to_pylist(),
                       t.column("duration").to_pylist()):
        if c == args.config:
            hours[b] += float(d or 0.0) / 3600

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for lab, book in zip(labels, books):
        by_cluster[int(lab)].append(book)

    print(f"\nmethod: {method}")
    print(f"pairwise similarity: mean {off.mean():.3f}  min {off.min():.3f}  "
          f"max {off.max():.3f}")
    print(f"\n{len(by_cluster)} cluster(s) over {len(books)} sampled books")
    print(f"{'cluster':>8s} {'books':>6s} {'sampled h':>10s} {'books':<40s}")
    print("-" * 70)
    ranked = sorted(by_cluster.items(), key=lambda kv: -sum(hours[b] for b in kv[1]))
    for lab, bs in ranked:
        h = sum(hours[b] for b in bs)
        print(f"{lab:>8d} {len(bs):>6d} {h:>10.1f} {' '.join(sorted(bs))[:40]:<40s}")

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
        "config": args.config, "method": method, "threshold": args.threshold,
        "n_clusters": len(by_cluster), "sampled_books": len(books),
        "similarity": {"mean": float(off.mean()), "min": float(off.min()),
                       "max": float(off.max())},
        "clusters": {str(k): {"books": sorted(v),
                              "hours": round(sum(hours[b] for b in v), 2)}
                     for k, v in by_cluster.items()},
        "largest_cluster_books": sorted(top_books),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
