"""Build the shared CTC character vocabulary across all 42 languages.

One vocab, one model, no per-language output heads. CTC over characters handles
this comfortably and it is what makes "train on 42 for the price of 1" true.

**This extends DONDO's vocabulary rather than replacing it.** We fine-tune from
`KhayaAI/w2v-bert-...`, so every token that keeps its original index also keeps
its trained row in the CTC output head (see `asr/model.extend_ctc_head`).
Renumbering the vocabulary would throw that away for no reason — the rows would
still load, but they would be attached to the wrong characters.

Consequences worth knowing:

  * The blank is `[PAD]` at index **33**, DONDO's position, not 0. `Kasa42ForCTC`
    takes `blank_id` explicitly for this reason.
  * `|` (word delimiter) stays at index 0.
  * New characters — the ones the other 31 languages need — are appended after
    DONDO's 49.

A frequency floor still applies to new characters. Each rare character costs an
output unit that will never be predicted reliably, so those below the floor are
dropped and reported rather than silently absorbed.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from kasa42.data.text import is_trainable, normalize

DONDO_VOCAB_URL = (
    "https://huggingface.co/KhayaAI/"
    "w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en/raw/main/vocab.json"
)
UNK, SPACE = "[UNK]", "|"
BLANK = "[PAD]"


def load_dondo_vocab() -> dict[str, int]:
    with urllib.request.urlopen(DONDO_VOCAB_URL, timeout=60) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/manifest.parquet")
    ap.add_argument("--out", default="results/vocab.json")
    ap.add_argument("--min-count", type=int, default=50,
                    help="Global frequency floor for keeping a character.")
    args = ap.parse_args()

    t = pq.read_table(args.manifest, columns=["config", "text"])
    configs = t.column("config").to_pylist()
    texts = t.column("text").to_pylist()

    counts: Counter[str] = Counter()
    per_lang: dict[str, set[str]] = defaultdict(set)
    kept_segs = dropped_segs = 0

    for config, raw in zip(configs, texts):
        if not is_trainable(raw):
            dropped_segs += 1
            continue
        kept_segs += 1
        n = normalize(raw)
        for ch in n:
            if ch == " ":
                continue
            counts[ch] += 1
            per_lang[config].add(ch)

    dondo = load_dondo_vocab()
    # DONDO's indices are load-bearing — keep every one of them exactly as is.
    vocab = dict(dondo)
    next_idx = max(vocab.values()) + 1

    # A character is only "new" if DONDO does not already have it. Existing ones
    # bypass the frequency floor: they arrive with a trained head row, so the
    # floor's rationale (untrainable fresh rows) does not apply.
    new_chars = sorted(c for c in counts if c not in dondo)
    keep_new = [c for c in new_chars if counts[c] >= args.min_count]
    dropped_chars = [c for c in new_chars if counts[c] < args.min_count]

    for ch in keep_new:
        vocab[ch] = next_idx
        next_idx += 1

    total = kept_segs + dropped_segs
    print(f"segments        : {total:,}")
    print(f"  trainable     : {kept_segs:,} ({kept_segs/max(total,1):.1%})")
    print(f"  dropped       : {dropped_segs:,} (digits / empty / no letters)")
    print(f"\nDONDO vocab     : {len(dondo)} tokens (blank '{BLANK}'={dondo.get(BLANK)}, "
          f"space '{SPACE}'={dondo.get(SPACE)})")
    print(f"reused          : {len([c for c in counts if c in dondo])} chars keep "
          f"their trained CTC head rows")
    print(f"added           : {len(keep_new)} new chars -> {''.join(keep_new)}")
    print(f"vocab size      : {len(vocab)}")
    unused = [c for c in dondo if len(c) == 1 and c not in counts and c != SPACE]
    if unused:
        print(f"in DONDO, unseen here: {''.join(sorted(unused))}")
    if dropped_chars:
        rare = sum(counts[c] for c in dropped_chars)
        print(f"chars dropped   : {len(dropped_chars)} below {args.min_count} "
              f"({rare:,} occurrences, {rare/max(sum(counts.values()),1):.4%} of text)")
        print(f"  {''.join(dropped_chars)}")

    print(f"\nper-language charset sizes:")
    for config in sorted(per_lang):
        chars = per_lang[config]
        oov = chars - set(vocab)
        flag = f"  <- {len(oov)} rare: {''.join(sorted(oov))}" if oov else ""
        print(f"  {config:24s} {len(chars):>3d}{flag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "vocab": vocab,
        "blank_id": vocab[BLANK],
        "space_id": vocab[SPACE],
        "unk_id": vocab[UNK],
        "dondo_vocab_size": len(dondo),
        "added_chars": keep_new,
        "min_count": args.min_count,
        "counts": dict(counts.most_common()),
        "per_language_charset": {k: "".join(sorted(v)) for k, v in sorted(per_lang.items())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
