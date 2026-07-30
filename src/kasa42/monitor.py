"""Watch a training run that is buffering its output.

`python -m kasa42.asr.train ... | tee train.log` gives Python a pipe, not a
terminal, so stdout is block-buffered at 8 KB. At ~87 bytes a line and one line
every 25 steps that is roughly 2,350 steps — the better part of an hour — before
anything appears. The run is fine; you just cannot see it.

Restarting with `python -u` would fix the buffering and cost every step so far,
which is the wrong trade once a run is hours in. So infer progress from what is
observable without touching the process:

  * **Checkpoint mtimes.** `step4000.pt` and `step8000.pt` are 4,000 steps apart
    and their timestamps say how long that took. That is a real measured rate,
    and unlike elapsed-time arithmetic it excludes dataset loading.
  * **Process elapsed time**, as a fallback while only one checkpoint exists.
  * **nvidia-smi**, to confirm the GPU is actually busy rather than blocked.
  * **The log**, for whatever has flushed.

    python -m kasa42.monitor              # refreshes until you Ctrl-C
    python -m kasa42.monitor --once       # one snapshot

Read-only throughout: it never signals, writes to, or otherwise disturbs the run.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

STEP_FILE = re.compile(r"step(\d+)\.pt$")
STEP_LINE = re.compile(r"step\s+(\d+)/(\d+)")


def checkpoints(d: Path) -> list[tuple[int, float, int]]:
    """[(step, mtime, bytes)] sorted by step."""
    out = []
    for p in d.glob("step*.pt"):
        m = STEP_FILE.search(p.name)
        if m:
            st = p.stat()
            out.append((int(m.group(1)), st.st_mtime, st.st_size))
    return sorted(out)


def training_pid(pattern: str = "kasa42.asr.train") -> tuple[int, float] | None:
    """(pid, elapsed_seconds) of the training parent, or None."""
    try:
        pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True,
                              text=True, timeout=10).stdout.split()
        if not pids:
            return None
        # Lowest pid is the parent; the rest are DataLoader workers, forked and
        # therefore sharing its command line.
        pid = min(int(p) for p in pids)
        secs = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        return pid, float(secs) if secs else 0.0
    except Exception:  # noqa: BLE001 - monitoring must never raise
        return None


def gpu_state() -> str:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        util, used, total = (x.strip() for x in q.split(","))
        return f"{util}% util   {int(used)/1024:.1f}/{int(total)/1024:.0f} GiB used (all tenants)"
    except Exception:  # noqa: BLE001
        return "unavailable"


def tail(path: Path, n: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def estimate(pts: list[tuple[int, float, int]], elapsed: float | None,
             now: float) -> tuple[float | None, float | None]:
    """(steps per second, current step). Prefers checkpoint deltas."""
    if len(pts) >= 2:
        (s0, t0, _), (s1, t1, _) = pts[0], pts[-1]
        if t1 > t0:
            rate = (s1 - s0) / (t1 - t0)
            return rate, s1 + rate * (now - t1)
    if pts and elapsed:
        s1, t1, _ = pts[-1]
        # Elapsed covers dataset loading too, so this reads slightly slow. It
        # sharpens the moment a second checkpoint lands and the branch above
        # takes over with a startup-free delta.
        rate = s1 / max(elapsed, 1e-9)
        return rate, s1 + rate * (now - t1)
    return None, None


def render(args) -> str:
    d = Path(args.checkpoint_dir)
    pts = checkpoints(d)
    proc = training_pid()
    now = time.time()
    rate, step = estimate(pts, proc[1] if proc else None, now)

    L = ["=" * 68, f"  kasa42 training monitor      {time.strftime('%Y-%m-%d %H:%M:%S')}",
         "=" * 68]

    if proc:
        h, m = divmod(int(proc[1]) // 60, 60)
        L.append(f"  process    pid {proc[0]}, running {h}h {m:02d}m")
    else:
        L.append("  process    NOT RUNNING — no kasa42.asr.train found")

    L.append(f"  gpu        {gpu_state()}")

    if step is not None and rate:
        pct = 100 * step / args.max_steps
        bar = "#" * int(pct / 2.5) + "." * (40 - int(pct / 2.5))
        left = max(args.max_steps - step, 0) / rate / 3600
        L += [f"  progress   [{bar}] {pct:5.1f}%",
              f"  step       ~{int(step):,} of {args.max_steps:,}  "
              f"at {rate:.2f} step/s",
              f"  eta        {left:.1f} h  "
              f"(~{time.strftime('%H:%M', time.localtime(now + left * 3600))})"]
        if len(pts) < 2:
            L.append("             (rate from elapsed time; sharpens at the next checkpoint)")
    else:
        L.append("  progress   no checkpoint yet — first lands at --save-every")

    if pts:
        L.append("  saved      " + "  ".join(
            f"{s//1000}k({(now - t)/60:.0f}m ago)" for s, t, _ in pts[-6:]))
        L.append(f"             {sum(b for _, _, b in pts)/1e9:.1f} GB across {len(pts)} file(s)")

    steps_seen = [STEP_LINE.search(x) for x in tail(Path(args.log), 400)]
    seen = [m for m in steps_seen if m]
    if seen:
        L.append(f"  log        flushed to step {int(seen[-1].group(1)):,}")
    for line in tail(Path(args.log), args.log_lines):
        if line.strip():
            L.append(f"    | {line[:96]}")

    up = tail(Path(args.upload_log), 3)
    if up:
        L.append("  uploads")
        for line in up:
            if line.strip():
                L.append(f"    | {line[:96]}")

    L.append("=" * 68)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", default="checkpoints/kasa42-asr")
    ap.add_argument("--log", default="train.log")
    ap.add_argument("--upload-log", default="upload.log")
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--log-lines", type=int, default=4)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    try:
        while True:
            print("\033[2J\033[H" + render(args), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped — training is untouched")


if __name__ == "__main__":
    main()
