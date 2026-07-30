"""Run everything that comes after training, unattended.

Training finishing at 01:00 is worth little if evaluation, export and upload
wait for someone to wake up and type them. This waits for the checkpoint, then
drives the rest of the pipeline to a finished submission:

    evaluate -> export -> verify -> upload

Design rules, all of them learned the hard way during the window:

  * **A failing stage must not take down the ones after it.** Export does not
    need evaluation to have succeeded. Each stage is attempted, its outcome
    recorded, and the run continues; the summary at the end says what happened.
  * **Re-running must be cheap.** Every stage declares the file it produces and
    is skipped when that file already exists, so an interrupted overnight run
    can be restarted without repeating hours of work. `--force` overrides.
  * **Everything is logged with timestamps, to a file.** A failure discovered
    at 07:00 has to be diagnosable from the log alone.
  * **The clock is a hard constraint.** The container is deleted at the end of
    the GPU window, so `--stop-by` refuses to begin a stage that cannot finish
    in time, and upload is ordered before the optional extras.

    python -u -m kasa42.pipeline --repo you/kasa42-asr --stop-by 2026-08-01T07:00

Nothing here needs the GPU except the stages that obviously do, and nothing
here touches a training run already in flight.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("kasa42.pipeline")


@dataclass
class Stage:
    name: str
    cmd: list[str]
    produces: Path | None = None
    optional: bool = False
    needs: list[Path] = field(default_factory=list)
    est_minutes: int = 30


def setup_logging(logfile: Path) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("kasa42.pipeline")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)):
        h.setFormatter(fmt)
        root.addHandler(h)


def wait_for(path: Path, timeout_min: float, poll: float = 60.0) -> bool:
    """Block until `path` exists and has stopped growing."""
    if path.exists():
        log.info(f"{path} already present")
        return True

    log.info(f"waiting for {path} (timeout {timeout_min:.0f} min)")
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        if path.exists():
            # torch.save on a 2.4 GB checkpoint takes seconds; a half-written
            # file loads as a corrupt one.
            a = path.stat().st_size
            time.sleep(20)
            if path.exists() and path.stat().st_size == a and a > 0:
                log.info(f"{path} appeared ({a/1e9:.2f} GB)")
                return True
        time.sleep(poll)
    log.error(f"timed out after {timeout_min:.0f} min waiting for {path}")
    return False


def run(stage: Stage) -> tuple[bool, float]:
    """Run one stage, streaming its output into the log."""
    log.info(f"--- {stage.name}: {' '.join(stage.cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(stage.cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as e:  # noqa: BLE001 - a missing interpreter is a stage failure, not a crash
        log.error(f"{stage.name}: could not start: {type(e).__name__}: {e}")
        return False, 0.0

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info(f"  | {line}")
    code = proc.wait()
    mins = (time.time() - t0) / 60

    if code != 0:
        log.error(f"{stage.name} FAILED (exit {code}) after {mins:.1f} min")
        return False, mins
    if stage.produces is not None and not stage.produces.exists():
        log.error(f"{stage.name} exited 0 but did not write {stage.produces}")
        return False, mins
    log.info(f"{stage.name} ok in {mins:.1f} min")
    return True, mins


def parse_deadline(s: str | None) -> datetime | None:
    if not s:
        return None
    txt = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(txt)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/kasa42-asr/final.pt")
    ap.add_argument("--model-config", default="checkpoints/kasa42-asr/config.json")
    ap.add_argument("--repo", help="HF model repo. Omitted skips the upload stage.")
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--log", default="results/pipeline.log")
    ap.add_argument("--wait-timeout-min", type=float, default=600.0)
    ap.add_argument("--stop-by", help="ISO time, e.g. 2026-08-01T07:00Z. Stages "
                                      "that cannot finish before it are skipped.")
    ap.add_argument("--with-baselines", action="store_true",
                    help="Also score DONDO/MMS/Whisper. Slow; runs last.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages whose outputs already exist.")
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    setup_logging(Path(args.log))
    deadline = parse_deadline(args.stop_by)
    py = [sys.executable, "-u", "-m"]
    ckpt, export_dir = Path(args.checkpoint), Path(args.export_dir)
    eval_dir = Path(args.eval_dir)

    log.info("=" * 64)
    log.info(f"kasa42 pipeline | checkpoint={ckpt} repo={args.repo or '(none)'}")
    if deadline:
        left = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
        log.info(f"deadline {deadline.isoformat()} — {left:.1f} h from now")

    if not args.skip_wait and not wait_for(ckpt, args.wait_timeout_min):
        log.error("no checkpoint; nothing to do")
        raise SystemExit(1)

    stages = [
        Stage("evaluate", py + ["kasa42.asr.evaluate", "--checkpoint", str(ckpt),
                                "--out-dir", str(eval_dir)],
              produces=eval_dir / "honest.json", est_minutes=60),
        Stage("export", py + ["kasa42.asr.export", "--checkpoint", str(ckpt),
                              "--model-config", args.model_config,
                              "--out-dir", str(export_dir)],
              produces=export_dir / "kasa42_asr.int8.onnx", est_minutes=20),
        Stage("verify-onnx", py + ["kasa42.asr.verify_onnx", "--out-dir", str(export_dir),
                                   "--checkpoint", str(ckpt),
                                   "--model-config", args.model_config],
              optional=True, est_minutes=15),
    ]
    if args.repo:
        stages.append(Stage(
            "upload", py + ["kasa42.asr.watch_upload", "--repo", args.repo,
                            "--checkpoint-dir", str(ckpt.parent),
                            "--interval", "5", "--settle", "5",
                            "--stop-after", ckpt.name],
            est_minutes=45))
    if args.with_baselines:
        stages.append(Stage(
            "baselines", py + ["kasa42.asr.baselines", "--out-dir", "results/baselines"],
            produces=Path("results/baselines/dondo.json"),
            optional=True, est_minutes=90))

    results: list[tuple[str, str, float]] = []
    for st in stages:
        if st.produces is not None and st.produces.exists() and not args.force:
            log.info(f"--- {st.name}: skipped, {st.produces} exists (use --force)")
            results.append((st.name, "skipped", 0.0))
            continue
        missing = [p for p in st.needs if not Path(p).exists()]
        if missing:
            log.warning(f"--- {st.name}: skipped, missing {missing}")
            results.append((st.name, "skipped (deps)", 0.0))
            continue
        if deadline:
            left = (deadline - datetime.now(timezone.utc)).total_seconds() / 60
            if left < st.est_minutes:
                log.warning(f"--- {st.name}: skipped, needs ~{st.est_minutes} min "
                            f"but only {left:.0f} min left before {args.stop_by}")
                results.append((st.name, "skipped (clock)", 0.0))
                continue

        ok, mins = run(st)
        results.append((st.name, "ok" if ok else "FAILED", mins))
        if not ok and not st.optional:
            log.warning(f"{st.name} failed but is not fatal — continuing, "
                        f"later stages do not depend on it")

    log.info("=" * 64)
    log.info(f"{'stage':<14s} {'result':<16s} {'min':>6s}")
    for name, status, mins in results:
        log.info(f"{name:<14s} {status:<16s} {mins:>6.1f}")

    failed = [n for n, s, _ in results if s == "FAILED"]
    if failed:
        log.error(f"{len(failed)} stage(s) failed: {', '.join(failed)}")
        log.error(f"full output in {args.log}")
        raise SystemExit(1)
    log.info("pipeline complete")


if __name__ == "__main__":
    main()
