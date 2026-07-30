"""Upload checkpoints to the Hub as they appear, so the run survives the wipe.

The container is deleted at the end of the GPU window with no grace period, so
every artefact only counts once it is off the box. Doing that by hand means
remembering to come back every 50 minutes for five hours, at night, which is
exactly the sort of thing that does not happen.

This polls the checkpoint directory and pushes anything new. Run it detached
alongside training:

    nohup python -u -m kasa42.asr.watch_upload \\
        --repo PrinceAlhassanNasamu/kasa42-asr > upload.log 2>&1 &

A file is only uploaded once its size has stopped changing — `torch.save` on a
2.4 GB checkpoint takes seconds, and a half-written file uploads perfectly
happily and restores as a corrupt one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def stable_size(p: Path, settle: float) -> int | None:
    """Size of `p`, or None if it is still being written."""
    try:
        first = p.stat().st_size
    except FileNotFoundError:
        return None
    time.sleep(settle)
    try:
        second = p.stat().st_size
    except FileNotFoundError:
        return None
    return second if first == second and second > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="e.g. you/kasa42-asr")
    ap.add_argument("--checkpoint-dir", default="checkpoints/kasa42-asr")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=20.0,
                    help="Seconds a file's size must hold steady before upload.")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--stop-after", default="final.pt",
                    help="Exit once this has been uploaded. Empty means never.")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model",
                    private=not args.public, exist_ok=True)
    print(f"watching {args.checkpoint_dir} -> {args.repo}", flush=True)

    ckpt = Path(args.checkpoint_dir)
    done: set[str] = set()

    while True:
        found = sorted(ckpt.glob("*.pt")) + sorted(ckpt.glob("*.json"))
        for p in found:
            if p.name in done:
                continue
            size = stable_size(p, args.settle)
            if size is None:
                print(f"{p.name}: still being written, will retry", flush=True)
                continue

            print(f"uploading {p.name} ({size/1e9:.2f} GB) ...", flush=True)
            t0 = time.time()
            try:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                                repo_id=args.repo, repo_type="model")
            except Exception as e:  # noqa: BLE001 - keep watching; try again next pass
                print(f"  FAILED {p.name}: {type(e).__name__}: {e}", flush=True)
                continue
            done.add(p.name)
            print(f"  done {p.name} in {time.time()-t0:.0f}s "
                  f"({len(done)} uploaded)", flush=True)

            if args.stop_after and p.name == args.stop_after:
                print(f"\n{args.stop_after} is on the Hub. Nothing else to wait for.",
                      flush=True)
                return

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
