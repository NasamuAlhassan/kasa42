"""Export the trained model to ONNX int8 so the demo outlives the GPU lease.

The H200 window closes Sat 1 Aug 08:31 GMT. If judging happens after that, a
checkpoint that only runs on an H200 is not a submission. This step is scheduled
work, not a stretch goal.

CTC helps here: the model is a plain encoder plus two linear heads, with no
autoregressive decode loop, so ONNX export is straightforward and int8 dynamic
quantization costs little accuracy. Expect roughly 4x smaller and comfortably
real-time on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kasa42.asr.model import Kasa42ForCTC


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class ExportWrapper(torch.nn.Module):
    """Inference-only forward: no loss, no labels, fixed output signature."""

    def __init__(self, model: Kasa42ForCTC) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_features=input_features, attention_mask=attention_mask)
        return out["logits"], out["lid_logits"], out["input_lengths"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/kasa42-asr/final.pt")
    ap.add_argument("--model-config", default="checkpoints/kasa42-asr/config.json")
    ap.add_argument("--out-dir", default="export")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Kasa42ForCTC(
        cfg["encoder"], vocab_size=cfg["vocab_size"],
        n_languages=len(cfg["languages"]), lid_weight=cfg.get("lid_weight", 0.1),
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    wrapper = ExportWrapper(model).eval()

    # w2v-BERT 2.0 consumes 80-dim log-mel filterbanks stacked in pairs -> 160.
    n_mel = getattr(model.encoder.config, "feature_projection_input_dim", 160)
    dummy_feats = torch.randn(1, 300, n_mel)
    dummy_mask = torch.ones(1, 300, dtype=torch.long)

    fp32 = out_dir / "kasa42_asr.onnx"
    print(f"exporting -> {fp32}")
    torch.onnx.export(
        wrapper, (dummy_feats, dummy_mask), str(fp32),
        input_names=["input_features", "attention_mask"],
        output_names=["logits", "lid_logits", "input_lengths"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "frames"},
            "attention_mask": {0: "batch", 1: "frames"},
            "logits": {0: "batch", 1: "time"},
            "lid_logits": {0: "batch"},
            "input_lengths": {0: "batch"},
        },
        opset_version=args.opset, do_constant_folding=True,
    )
    # fp32 weights exceed protobuf's 2 GB ceiling, so ONNX spills the tensors
    # into sibling files and leaves only the graph in the .onnx. Reporting the
    # .onnx alone claims a 606M-parameter model is 2.8 MB, and makes the
    # quantisation ratio come out as "0.0x smaller". Measure the directory,
    # which at this point holds nothing but the fp32 export.
    fp32_total = dir_bytes(out_dir)
    external = fp32_total - fp32.stat().st_size
    print(f"  fp32 {fp32_total/1e6:.1f} MB "
          f"({fp32.stat().st_size/1e6:.1f} MB graph + {external/1e6:.1f} MB external weights)")
    if external > 0:
        print("  NOTE: the .onnx alone is not loadable — ship the whole directory.")

    if not args.no_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8 = out_dir / "kasa42_asr.int8.onnx"
        print(f"quantizing -> {int8}")
        # The quantizer resolves external data and writes one self-contained
        # file, which is why the int8 model can be shipped on its own.
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        print(f"  int8 {int8.stat().st_size/1e6:.1f} MB "
              f"({fp32_total/max(int8.stat().st_size, 1):.1f}x smaller, self-contained)")

    (out_dir / "languages.json").write_text(
        json.dumps(cfg["languages"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nVerify before trusting this:")
    print(f"  python -m kasa42.asr.verify_onnx --out-dir {out_dir} \\")
    print(f"      --checkpoint {args.checkpoint} --model-config {args.model_config}")
    print("  Runs both graphs on CPU at four sequence lengths and compares them")
    print("  against torch. The tracer bakes shape comparisons in as constants,")
    print("  so an export can be correct at the traced length and wrong at every")
    print("  other one — which is every real utterance.")


if __name__ == "__main__":
    main()
