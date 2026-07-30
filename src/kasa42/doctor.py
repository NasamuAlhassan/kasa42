"""Check the environment before committing hours to it.

Every environment failure during the GPU window was cheap to detect and
expensive to hit late:

  * `HF_HOME` unset because the login shell was dash, which does not read
    `~/.bashrc` — so model downloads went to a `/` with 8.9 GB free instead of
    the 2 TB volume.
  * The data pre-staged at a path the pipeline did not know about.
  * Training started before `splits.json` and `mixture.json` existed.
  * A neighbouring tenant holding 100 GB of the shared card, discovered by an
    OOM at step 300 rather than by looking.

None of that needs to be found the hard way.

    python -m kasa42.doctor --data-dir /data/ghana-speech

Exits non-zero if anything is FAIL, so it can gate a long run:

    python -m kasa42.doctor && python -m kasa42.asr.train ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

OK, WARN, FAIL = "ok", "WARN", "FAIL"
_ROWS: list[tuple[str, str, str]] = []


def note(check: str, status: str, detail: str = "") -> None:
    _ROWS.append((check, status, detail))


def check_versions() -> None:
    note("python", OK, sys.version.split()[0])
    for mod in ("torch", "transformers", "datasets", "pyarrow", "numpy"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
        except ImportError:
            note(mod, WARN, "not installed")
            continue
        # datasets 5.x routes Audio through torchcodec; transformers 5.x moved
        # private helpers this codebase calls. Both were real breakages.
        extra = ""
        if mod == "datasets" and v.split(".")[0] >= "5":
            extra = " (5.x: Audio needs decode=False — handled)"
        if mod == "transformers" and v.split(".")[0] >= "5":
            extra = " (5.x: private _get_feat_extract_output_lengths in use)"
        note(mod, OK, v + extra)


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        note("gpu", WARN, "torch not installed")
        return
    if not torch.cuda.is_available():
        note("gpu", WARN, "no CUDA — CPU only")
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = torch.cuda.mem_get_info()[0] / 1e9
    note("gpu", OK, f"{name} sm_{major}{minor} {total:.0f} GB")

    # bf16 in hardware needs Ampere or later. torch.cuda.is_bf16_supported()
    # answers True on a Turing card and then runs it in software.
    note("bf16", OK if major >= 8 else WARN,
         "hardware" if major >= 8 else f"sm_{major}{minor} would emulate — expect fp16")

    used = total - free
    # The card is shared; a neighbour's footprint moved by 60 GB in one evening.
    status = OK if free > 30 else (WARN if free > 15 else FAIL)
    note("gpu free", status,
         f"{free:.0f} GB free, {used:.0f} GB in use by all tenants")

    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        note("PYTORCH_CUDA_ALLOC_CONF", WARN,
             "unset — expandable_segments:True reduces fragmentation")
    else:
        note("PYTORCH_CUDA_ALLOC_CONF", OK, os.environ["PYTORCH_CUDA_ALLOC_CONF"])


def check_env_and_disk(paths: list[Path]) -> None:
    hf = os.environ.get("HF_HOME")
    if not hf:
        note("HF_HOME", WARN, "unset — caches land in ~/.cache, often a small volume")
    elif not Path(hf).is_dir():
        note("HF_HOME", FAIL, f"{hf} does not exist")
    elif not os.access(hf, os.W_OK):
        note("HF_HOME", FAIL, f"{hf} not writable")
    else:
        note("HF_HOME", OK, hf)

    try:
        from huggingface_hub import get_token

        note("hf token", OK if get_token() else WARN,
             "present" if get_token() else "none — `hf auth login` before uploading")
    except Exception:  # noqa: BLE001 - an old hub version is not a failure here
        note("hf token", WARN, "could not check")

    seen = set()
    for p in paths:
        try:
            mount = shutil.disk_usage(p)
        except OSError:
            continue
        if mount.total in seen:
            continue
        seen.add(mount.total)
        gb = mount.free / 1e9
        # A 24k-step run writes six 2.4 GB checkpoints plus an ONNX export.
        status = OK if gb > 60 else (WARN if gb > 20 else FAIL)
        note(f"disk {p}", status, f"{gb:.0f} GB free of {mount.total/1e9:.0f} GB")


def check_data(data_dir: str) -> None:
    root = Path(data_dir)
    if not root.is_dir():
        note("data dir", FAIL, f"{root} missing — pass --data-dir")
        return
    configs = [d for d in sorted(root.iterdir())
               if d.is_dir() and any(d.glob("*.parquet"))]
    note("data dir", OK if configs else FAIL,
         f"{root}: {len(configs)} configs, "
         f"{sum(len(list(d.glob('*.parquet'))) for d in configs)} shards")
    if not configs:
        return

    import pyarrow.parquet as pq

    want = {"id", "language", "text", "duration", "source_file", "audio"}
    shard = next(configs[0].glob("*.parquet"))
    cols = set(pq.ParquetFile(shard).schema_arrow.names)
    missing = want - cols
    note("data columns", FAIL if missing else OK,
         f"missing {sorted(missing)}" if missing else "id/text/duration/source_file/audio")


def check_artifacts() -> None:
    """The four files training refuses to start without, plus the eval sets."""
    import pyarrow.parquet as pq

    checks = [
        ("manifest.parquet", "python -m kasa42.data.build_manifest --local-root <dir>"),
        ("splits.json", "python -m kasa42.data.splits"),
        ("vocab.json", "python -m kasa42.data.vocab"),
        ("mixture.json", "python -m kasa42.data.mixture --alpha 0.5 --cap-hours 40"),
    ]
    for fname, fix in checks:
        p = Path("results") / fname
        if not p.exists():
            note(fname, FAIL, f"missing — {fix}")
            continue
        detail = f"{p.stat().st_size/1e6:.1f} MB"
        try:
            if fname.endswith(".parquet"):
                t = pq.read_table(p, columns=["config"])
                detail += f", {t.num_rows:,} rows, {len(set(t.column('config').to_pylist()))} configs"
            else:
                d = json.loads(p.read_text(encoding="utf-8"))
                key = "configs" if "configs" in d else "segment_ids" if "segment_ids" in d else "vocab"
                detail += f", {len(d.get(key, d)):,} {key}"
        except Exception as e:  # noqa: BLE001 - an unreadable artefact is the finding
            note(fname, FAIL, f"unreadable: {type(e).__name__}: {e}")
            continue
        note(fname, OK, detail)

    for fname in ("testset_honest.parquet", "testset_leaked.parquet"):
        p = Path("results") / fname
        if not p.exists():
            note(fname, WARN, "missing — python -m kasa42.asr.testset")
            continue
        t = pq.read_table(p, columns=["config"])
        note(fname, OK, f"{t.num_rows:,} utts, "
                        f"{len(set(t.column('config').to_pylist()))} configs")


def check_checkpoints(out_dir: str) -> None:
    d = Path(out_dir)
    if not d.is_dir():
        note("checkpoints", WARN, f"{d} not created yet")
        return
    pts = sorted(d.glob("*.pt"))
    if not pts:
        note("checkpoints", WARN, f"{d} has no .pt yet")
        return
    newest = max(pts, key=lambda p: p.stat().st_mtime)
    note("checkpoints", OK,
         f"{len(pts)} file(s), newest {newest.name} ({newest.stat().st_size/1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.environ.get("GS", "/data/ghana-speech"))
    ap.add_argument("--out-dir", default="checkpoints/kasa42-asr")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    check_versions()
    if not args.skip_gpu:
        check_gpu()
    check_env_and_disk([Path.cwd(), Path("/"), Path(os.environ.get("HF_HOME", "."))])
    check_data(args.data_dir)
    check_artifacts()
    check_checkpoints(args.out_dir)

    width = max(len(c) for c, _, _ in _ROWS)
    print(f"\n{'check':<{width}s}  {'':<4s}  detail")
    print("-" * (width + 60))
    for check, status, detail in _ROWS:
        mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}[status]
        print(f"{check:<{width}s}  {mark}  {detail}")
    print("-" * (width + 60))

    fails = [c for c, s, _ in _ROWS if s == FAIL]
    warns = [c for c, s, _ in _ROWS if s == WARN]
    print(f"{len(_ROWS)} checks: {len(_ROWS)-len(fails)-len(warns)} ok, "
          f"{len(warns)} warn, {len(fails)} fail")
    if fails:
        print(f"\nblocking: {', '.join(fails)}")
        raise SystemExit(1)
    if warns:
        print(f"worth a look: {', '.join(warns)}")


if __name__ == "__main__":
    main()
