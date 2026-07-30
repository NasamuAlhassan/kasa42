"""Restart training if it dies while nobody is watching.

A run that stops at 02:00 costs every hour until someone wakes up. The GPU is
shared with a tenant whose footprint moved 60 GB in one evening, and while
`train.py` now skips an OOM batch rather than dying on it, a sustained squeeze
still ends the run by design — correctly, because retrying into a full card
wastes the window. What was missing is anything to try again later.

This polls, and relaunches from the newest checkpoint when all of these hold:

  * `final.pt` does not exist, so there is still work to do;
  * no training process is alive;
  * a `stepN.pt` exists to resume from, so a restart costs minutes not hours.

It does **not** adopt or disturb a run already in flight — it watches until that
run is gone. So it can be started right now, alongside a healthy run, and will
simply sit there unless something goes wrong.

    nohup python -u -m kasa42.watchdog --max-steps 24000 > watchdog.log 2>&1 &

Restarts are capped and spaced. A run that dies twice inside a few minutes is
failing for a reason a third attempt will not fix, and thrashing a shared GPU
is worse than stopping.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

STEP_FILE = re.compile(r"step(\d+)\.pt$")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def training_alive(pattern: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True,
                           text=True, timeout=10)
        return bool(r.stdout.split())
    except Exception:  # noqa: BLE001 - a failed probe must not trigger a restart
        log("WARN could not probe for the training process; assuming alive")
        return True


def newest_step(d: Path) -> int | None:
    steps = [int(m.group(1)) for p in d.glob("step*.pt")
             if (m := STEP_FILE.search(p.name))]
    return max(steps) if steps else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="checkpoints/kasa42-asr")
    ap.add_argument("--data-dir", default="/data/ghana-speech")
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--batch-duration", type=float, default=160.0)
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--max-restarts", type=int, default=4)
    ap.add_argument("--min-gap-min", type=float, default=10.0,
                    help="A run that dies again this fast is failing for a "
                         "reason another attempt will not fix.")
    ap.add_argument("--log", default="train.log")
    args = ap.parse_args()

    out = Path(args.out_dir)
    final = out / "final.pt"
    restarts, last = 0, 0.0
    log(f"watching {out} for max_steps={args.max_steps:,}; "
        f"up to {args.max_restarts} restart(s)")

    while True:
        if final.exists():
            log(f"{final} exists — training finished. Nothing to supervise.")
            return

        if training_alive("kasa42.asr.train"):
            time.sleep(args.interval)
            continue

        step = newest_step(out)
        if step is None:
            log("no training process and no checkpoint to resume from — "
                "not guessing. Start the run by hand.")
            return
        if step >= args.max_steps:
            log(f"newest checkpoint is step {step:,} of {args.max_steps:,}; "
                f"nothing left to run")
            return

        gap = (time.time() - last) / 60
        if restarts and gap < args.min_gap_min:
            log(f"died again after {gap:.1f} min — that is a real failure, not a "
                f"transient. Stopping so it can be diagnosed; see {args.log}.")
            return
        if restarts >= args.max_restarts:
            log(f"{restarts} restart(s) already used. Stopping.")
            return

        restarts += 1
        last = time.time()
        cmd = [sys.executable, "-u", "-m", "kasa42.asr.train",
               "--resume", "auto",
               "--max-steps", str(args.max_steps),
               "--batch-duration", str(args.batch_duration),
               "--gradient-checkpointing",
               "--save-every", str(args.save_every),
               "--data-dir", args.data_dir,
               "--out-dir", str(out)]
        log(f"training is gone at step {step:,}. Restart {restarts}/"
            f"{args.max_restarts}: {' '.join(cmd)}")

        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(f"\n--- watchdog restart {restarts} at "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')} from step {step} ---\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            code = proc.wait()

        log(f"restarted run exited with {code}")
        if code == 0 and final.exists():
            log("training completed. Done.")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
