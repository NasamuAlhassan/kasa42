"""Provenance audit for ghananlpcommunity/ghana-speech.

The audio column is ~222 GB; every other column together is a rounding error.
Parquet is columnar, so we project only the metadata columns over HTTP range
requests and never touch a single audio byte. This lets the entire 42-language
audit run on a laptop in minutes, before the GPU window opens.

Two questions this answers, both of which gate the plan:

1. How many distinct `source_file` recordings back each language, and how are
   segments distributed across them? This decides whether a source-disjoint
   train/test split is feasible (it needs enough sources to split on) and
   confirms that a random split would leak.
2. Is Kusaal backed by a dominant narrator? VITS fine-tuning for TTS needs
   roughly single-speaker audio. `source_file` count is our proxy for narrator
   diversity; if Kusaal is spread thin across many recordings, TTS falls back
   to another language or is cut.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "ghananlpcommunity/ghana-speech"
META_COLS = ["id", "language", "text", "duration", "source_file"]

# 16 kHz, 16-bit, mono => 32000 bytes/sec. Verified against the dataset card:
# Asante_Twi 23.06 GB / 32000 = 200.2 h, matching its stated 200.02 h.
BYTES_PER_SEC = 32000


@dataclass
class LanguageAudit:
    config: str
    language: str
    iso: str
    n_files: int
    n_segments: int
    total_hours: float
    n_sources: int
    segments_per_source_median: float
    segments_per_source_max: int
    top_source_share: float
    top5_source_share: float
    duration_p50: float
    duration_p90: float
    duration_max: float
    charset_size: int
    sample_sources: list[str]
    sample_text: str


def list_configs(api: HfApi) -> dict[str, list[str]]:
    """Map each `Language_iso` directory to its parquet files."""
    configs: dict[str, list[str]] = {}
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        if f.endswith(".parquet") and "/" in f:
            configs.setdefault(f.split("/")[0], []).append(f)
    return {k: sorted(v) for k, v in sorted(configs.items())}


def read_meta(fs: HfFileSystem, path: str):
    """Read only the metadata columns from one parquet shard."""
    with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        present = [c for c in META_COLS if c in pf.schema_arrow.names]
        return pf.read(columns=present).to_pydict()


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(q * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def audit_config(fs: HfFileSystem, config: str, files: list[str], max_shards: int | None) -> LanguageAudit:
    shards = files if max_shards is None else files[:max_shards]

    sources: Counter[str] = Counter()
    durations: list[float] = []
    charset: set[str] = set()
    n_segments = 0
    sample_text = ""

    for path in shards:
        d = read_meta(fs, path)
        n = len(d.get("id", d.get("text", [])))
        n_segments += n
        for s in d.get("source_file", []):
            sources[str(s)] += 1
        durations.extend(float(x) for x in d.get("duration", []) if x is not None)
        for t in d.get("text", []):
            if t:
                charset.update(str(t))
                if not sample_text:
                    sample_text = str(t)[:120]

    durations.sort()
    total = sum(sources.values()) or 1
    ranked = [c for _, c in sources.most_common()]
    per_source = sorted(ranked)

    m = re.match(r"^(.*)_([a-z]{3})$", config)
    language, iso = (m.group(1), m.group(2)) if m else (config, "")

    # Scale sampled shards up to the full config so numbers stay comparable.
    scale = len(files) / len(shards) if shards else 1.0

    return LanguageAudit(
        config=config,
        language=language,
        iso=iso,
        n_files=len(files),
        n_segments=int(n_segments * scale),
        total_hours=round(sum(durations) * scale / 3600, 2),
        n_sources=len(sources),
        segments_per_source_median=percentile(per_source, 0.5),
        segments_per_source_max=max(ranked) if ranked else 0,
        top_source_share=round(ranked[0] / total, 4) if ranked else 0.0,
        top5_source_share=round(sum(ranked[:5]) / total, 4) if ranked else 0.0,
        duration_p50=round(percentile(durations, 0.50), 2),
        duration_p90=round(percentile(durations, 0.90), 2),
        duration_max=round(durations[-1], 2) if durations else 0.0,
        charset_size=len(charset),
        sample_sources=[s for s, _ in sources.most_common(3)],
        sample_text=sample_text,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", help="Config dirs, e.g. Kusaal_kus. Default: all.")
    ap.add_argument("--max-shards", type=int, default=2,
                    help="Shards per config. Low values sample; use 0 for all (slow).")
    ap.add_argument("--out", default="results/audit.json")
    args = ap.parse_args()

    max_shards = None if args.max_shards == 0 else args.max_shards
    api, fs = HfApi(), HfFileSystem()

    configs = list_configs(api)
    if args.configs:
        configs = {k: v for k, v in configs.items() if k in args.configs}

    print(f"Auditing {len(configs)} configs "
          f"({'all shards' if max_shards is None else f'{max_shards} shard(s) each'})\n")

    rows: list[LanguageAudit] = []
    for i, (config, files) in enumerate(configs.items(), 1):
        try:
            a = audit_config(fs, config, files, max_shards)
        except Exception as e:  # a bad shard must not kill a 42-config run
            print(f"[{i:2d}/{len(configs)}] {config:24s} FAILED: {type(e).__name__}: {e}")
            continue
        rows.append(a)
        print(f"[{i:2d}/{len(configs)}] {a.config:24s} "
              f"{a.n_segments:>7,} seg  {a.total_hours:>7.1f} h  "
              f"{a.n_sources:>5,} src  top={a.top_source_share:>6.1%}  "
              f"p50={a.duration_p50:>5.1f}s  chars={a.charset_size:>3d}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {out} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
