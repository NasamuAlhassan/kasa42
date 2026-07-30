"""Check an exported ONNX model before trusting it as the submission.

`export.py` prints three things to verify and provides no way to do any of
them. This is the first: does the exported graph still behave like the torch
model it came from?

The specific worry is length generalisation. Exporting w2v-BERT through the
legacy tracer emits

    TracerWarning: Converting a tensor to a Python boolean ...
      if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0

because transformers branches on a *shape* while building the attention mask.
A traced branch is frozen at whatever the dummy input made it, so a model
exported at 300 frames can be silently wrong at 150 or 900 — which is every
real utterance. Warnings alone cannot tell you whether it mattered; running it
at several lengths can.

Two comparisons, because they fail for different reasons:

  * **fp32 ONNX vs torch** isolates tracing. Any real gap here is a broken
    export, not rounding.
  * **int8 ONNX vs torch** additionally carries quantisation error, so it is
    judged on argmax agreement rather than on absolute logit distance.

The GPU is hidden throughout: this model has to run on CPU after the lease
expires, so that is where it should be tested.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Lengths to probe. 300 is what export.py traces with, so it is the one that
# would pass even from a fully frozen graph — the others are the test.
LENGTHS = (150, 300, 500, 900)


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def torch_reference(checkpoint: str, model_config: str, lengths, n_mel: int, seed: int):
    """Run the torch model on CPU and return {length: (logits, lid_logits)}."""
    import torch

    from kasa42.asr.model import Kasa42ForCTC

    cfg = json.loads(Path(model_config).read_text(encoding="utf-8"))
    model = Kasa42ForCTC(cfg["encoder"], vocab_size=cfg["vocab_size"],
                         n_languages=len(cfg["languages"]),
                         lid_weight=cfg.get("lid_weight", 0.1))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    out = {}
    rng = np.random.default_rng(seed)
    for t in lengths:
        feats = rng.standard_normal((1, t, n_mel), dtype=np.float32)
        mask = np.ones((1, t), dtype=np.int64)
        with torch.no_grad():
            res = model(input_features=torch.from_numpy(feats),
                        attention_mask=torch.from_numpy(mask))
        out[t] = (res["logits"].numpy(), res["lid_logits"].numpy(),
                  feats, mask)
    return out


def run_onnx(path: Path, feats: np.ndarray, mask: np.ndarray):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    t0 = time.time()
    outs = sess.run(None, {"input_features": feats, "attention_mask": mask})
    return outs, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="export")
    ap.add_argument("--checkpoint", default="checkpoints/kasa42-asr/final.pt")
    ap.add_argument("--model-config", default="checkpoints/kasa42-asr/config.json")
    ap.add_argument("--n-mel", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--skip-torch", action="store_true",
                    help="Only check that the ONNX models run at every length.")
    args = ap.parse_args()

    # Hide the GPU: this model has to work after the lease expires.
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    out_dir = Path(args.out_dir)
    fp32 = out_dir / "kasa42_asr.onnx"
    int8 = out_dir / "kasa42_asr.int8.onnx"

    print(f"{args.out_dir}: {dir_bytes(out_dir) / 1e6:.1f} MB total")
    for p in (fp32, int8):
        print(f"  {p.name:24s} {p.stat().st_size / 1e6:>8.1f} MB"
              if p.exists() else f"  {p.name:24s}   MISSING")
    if not int8.exists() and not fp32.exists():
        raise SystemExit("nothing to verify — run kasa42.asr.export first")

    ref = {}
    if not args.skip_torch:
        print("\nrunning torch reference on CPU ...")
        ref = torch_reference(args.checkpoint, args.model_config,
                              LENGTHS, args.n_mel, args.seed)

    rng = np.random.default_rng(args.seed)
    failures: list[str] = []

    print(f"\n{'frames':>7s} {'model':>6s} {'logits':>18s} {'max|d|':>9s} "
          f"{'argmax':>8s} {'lid':>5s} {'ms':>7s}")
    print("-" * 64)

    for t in LENGTHS:
        if ref:
            r_logits, r_lid, feats, mask = ref[t]
        else:
            feats = rng.standard_normal((1, t, args.n_mel), dtype=np.float32)
            mask = np.ones((1, t), dtype=np.int64)
            r_logits = r_lid = None

        for name, path in (("fp32", fp32), ("int8", int8)):
            if not path.exists():
                continue
            try:
                outs, dt = run_onnx(path, feats, mask)
            except Exception as e:  # noqa: BLE001 - report, do not abort the sweep
                print(f"{t:>7d} {name:>6s}  FAILED: {type(e).__name__}: {e}")
                failures.append(f"{name} @ {t} frames: {type(e).__name__}")
                continue

            logits, lid = outs[0], outs[1]
            shape_ok = logits.ndim == 3 and logits.shape[0] == 1
            diff = agree = lid_same = None
            if r_logits is not None and logits.shape == r_logits.shape:
                diff = float(np.abs(logits - r_logits).max())
                agree = float((logits.argmax(-1) == r_logits.argmax(-1)).mean())
                lid_same = int(lid.argmax(-1)[0]) == int(r_lid.argmax(-1)[0])
            elif r_logits is not None:
                failures.append(
                    f"{name} @ {t} frames: shape {logits.shape} != torch {r_logits.shape}")

            print(f"{t:>7d} {name:>6s} {str(logits.shape):>18s} "
                  f"{'-' if diff is None else f'{diff:9.2e}'} "
                  f"{'-' if agree is None else f'{agree:7.1%}'} "
                  f"{'-' if lid_same is None else ('ok' if lid_same else 'DIFF'):>5s} "
                  f"{dt * 1000:>7.0f}")

            if not shape_ok:
                failures.append(f"{name} @ {t} frames: bad output rank {logits.shape}")
            # fp32 carries no quantisation error, so a real gap here is a
            # broken trace. int8 is judged on decisions, not distances.
            if name == "fp32" and diff is not None and diff > 1e-2:
                failures.append(f"fp32 @ {t} frames: max|d| {diff:.2e} vs torch")
            if name == "int8" and agree is not None and agree < 0.95:
                failures.append(f"int8 @ {t} frames: only {agree:.1%} argmax agreement")

    print("-" * 64)
    if failures:
        print(f"\n{len(failures)} PROBLEM(S):")
        for f in failures:
            print(f"  - {f}")
        print("\nIf only non-300 lengths fail, the traced graph froze a shape "
              "comparison and the export is not usable for variable-length audio.")
        raise SystemExit(1)

    print("\nAll lengths ran and matched. Timings above are CPU, GPU hidden — "
          "compare them against the audio duration to confirm real-time.")


if __name__ == "__main__":
    main()
