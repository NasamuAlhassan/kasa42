"""Fine-tune a Kusaal VITS voice from `facebook/mms-tts-kus`.

Honest framing, because this is where it would be easy to overclaim: MMS already
ships a Kusaal TTS checkpoint. We are not creating the first Kusaal voice — we
are making a measurably better one, starting from theirs and fine-tuning on
~20 h of in-domain audio versus the ~32 h of mixed Bible readings MMS trained
each language on.

Transformers has `VitsModel` for inference but no training loop: VITS needs a
discriminator plus mel/KL/duration losses. Rather than reimplement that under a
48 h clock, this script drives ylacombe/finetune-hf-vits, which is the
maintained community trainer for exactly this checkpoint family.

Run `prepare.py` first and listen to `data/tts/check/*.wav`. If those clips are
not the same voice, stop — fine-tuning multi-speaker audio into a single-speaker
VITS produces an averaged, muddy voice, and the GPU hours are better spent
elsewhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TRAINER_REPO = "https://github.com/ylacombe/finetune-hf-vits"
BASE_CHECKPOINT = "facebook/mms-tts-kus"


def ensure_trainer(root: Path) -> Path:
    """Clone the VITS trainer and build its monotonic-alignment extension."""
    repo = root / "finetune-hf-vits"
    if not repo.exists():
        print(f"cloning {TRAINER_REPO}")
        subprocess.run(["git", "clone", "--depth", "1", TRAINER_REPO, str(repo)], check=True)

    # The alignment search is a Cython extension; training fails without it.
    mas = repo / "monotonic_align"
    if mas.exists() and not list(mas.glob("**/*.so")) and not list(mas.glob("**/*.pyd")):
        try:
            import Cython  # noqa: F401
        except ImportError:
            # Not in the training extras, and the failure surfaces as a bare
            # ModuleNotFoundError from a setup.py three directories away.
            print("installing Cython (needed to build monotonic_align)")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "Cython"], check=True)
        print("building monotonic_align")
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                       cwd=mas, check=True)
    built = list(mas.glob("**/*.so")) + list(mas.glob("**/*.pyd"))
    print(f"monotonic_align: {len(built)} extension(s) built")
    return repo


def write_config(out: Path, data_dir: Path, run_name: str, epochs: int,
                 batch_size: int, lr: float) -> Path:
    cfg = {
        "project_name": "kasa42-kusaal-tts",
        "push_to_hub": False,
        "hub_model_id": run_name,
        "overwrite_output_dir": True,
        "output_dir": f"checkpoints/{run_name}",

        "model_name_or_path": BASE_CHECKPOINT,
        "dataset_name": str(data_dir),
        "audio_column_name": "audio",
        "text_column_name": "text",
        "train_split_name": "train",
        "eval_split_name": "eval",

        "full_generation_sample_text": "Fʋn na yel ye ba tʋm tʋʋma la.",

        "do_train": True,
        "num_train_epochs": epochs,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "per_device_train_batch_size": batch_size,
        "learning_rate": lr,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_ratio": 0.01,
        "group_by_length": False,

        "do_eval": True,
        "eval_steps": 500,
        "per_device_eval_batch_size": batch_size,
        "max_eval_samples": 24,
        "save_steps": 500,
        "logging_steps": 50,

        # VITS loss weights: the defaults from the original paper. Changing these
        # without listening to samples is a good way to waste GPU hours.
        "weight_disc": 3.0,
        "weight_fmaps": 1.0,
        "weight_gen": 1.0,
        "weight_kl": 1.5,
        "weight_duration": 1.0,
        "weight_mel": 35.0,

        "fp16": True,
        "seed": 20260730,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/tts")
    ap.add_argument("--work-dir", default="third_party")
    ap.add_argument("--run-name", default="kasa42-kusaal-tts")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--config-only", action="store_true",
                    help="Write the config without cloning or training.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "metadata.json").exists():
        raise SystemExit(f"run tts/prepare.py first — no {data_dir/'metadata.json'}")

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    print(f"corpus: {meta['config']} version {meta['version']} — "
          f"{meta['hours']} h, {len(meta['items']):,} clips")

    check = data_dir / "check"
    if check.exists():
        print(f"\nBefore training, listen to {check}/*.wav.")
        print("Same voice across books => single-narrator assumption holds.\n")

    cfg_path = write_config(Path(args.work_dir) / f"{args.run_name}.json",
                            data_dir, args.run_name, args.epochs,
                            args.batch_size, args.lr)
    print(f"wrote {cfg_path}")
    if args.config_only:
        return

    repo = ensure_trainer(Path(args.work_dir))
    cmd = [sys.executable, str(repo / "run_vits_finetuning.py"), str(cfg_path)]
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    print("\nNext: tts/roundtrip.py to score this against the MMS baseline.")


if __name__ == "__main__":
    main()
