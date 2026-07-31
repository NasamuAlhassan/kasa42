"""Gradio demo: speak any of 42 Ghanaian languages, it identifies and transcribes.

Runs in three modes so it is never blocked on the GPU:

  stub   no model at all — layout and wiring, usable today
  onnx   int8 CPU inference, the demo-day path once the H200 is gone
  torch  fp32/bf16, for checking parity against ONNX

Built against the stub first on purpose. On demo day the only thing that should
be new is the weights.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

TARGET_SR = 16000
MODE = os.environ.get("KASA42_MODE", "stub")
EXPORT_DIR = Path(os.environ.get("KASA42_EXPORT", "export"))
VOCAB_PATH = Path(os.environ.get("KASA42_VOCAB", "results/vocab.json"))

# Kusaal is the hero language; it leads the list rather than sorting alphabetically.
HERO = "Kusaal_kus"


def pretty(config: str) -> str:
    name, _, iso = config.rpartition("_")
    return f"{name.replace('_', ' ')} ({iso})"


class Engine:
    """Loads whichever backend is available; degrades to stub without complaint."""

    def __init__(self, mode: str = MODE) -> None:
        self.mode = mode
        self.languages: list[str] = []
        self.tokenizer = None
        self.session = None
        self.fe = None

        langs = EXPORT_DIR / "languages.json"
        if langs.exists():
            self.languages = json.loads(langs.read_text(encoding="utf-8"))
        if not self.languages and VOCAB_PATH.exists():
            v = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
            self.languages = sorted(v.get("per_language_charset", {}))
        if not self.languages:
            self.languages = [HERO]

        if mode == "stub":
            return

        try:
            from kasa42.asr.dataset import CharTokenizer
            from transformers import SeamlessM4TFeatureExtractor

            self.tokenizer = CharTokenizer.from_json(str(VOCAB_PATH))
            # Prefer the config shipped with the export: it is the encoder's own,
            # which is what training and evaluation used. Falling straight to
            # facebook/w2v-bert-2.0 assumes the two agree, and a mismatch here
            # changes the features under the model without raising anything.
            fe_src = (str(EXPORT_DIR)
                      if (EXPORT_DIR / "preprocessor_config.json").exists()
                      else "facebook/w2v-bert-2.0")
            self.fe = SeamlessM4TFeatureExtractor.from_pretrained(fe_src)
            print(f"[engine] feature extractor from {fe_src}")

            if mode == "onnx":
                import onnxruntime as ort

                path = EXPORT_DIR / "kasa42_asr.int8.onnx"
                if not path.exists():
                    path = EXPORT_DIR / "kasa42_asr.onnx"
                self.session = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"])
            else:
                import torch

                from kasa42.asr.model import Kasa42ForCTC

                cfg = json.loads((Path("checkpoints/kasa42-asr/config.json"))
                                 .read_text(encoding="utf-8"))
                m = Kasa42ForCTC(cfg["encoder"], vocab_size=cfg["vocab_size"],
                                 n_languages=len(cfg["languages"]))
                m.load_state_dict(torch.load("checkpoints/kasa42-asr/final.pt",
                                             map_location="cpu"))
                self.session = m.eval()
        except Exception as e:  # a broken backend must not take the demo down
            print(f"[engine] {mode} unavailable ({type(e).__name__}: {e}); using stub")
            self.mode = "stub"

    def transcribe(self, sr: int, wav: np.ndarray) -> tuple[str, str, float, float]:
        t0 = time.time()
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if np.abs(wav).max() > 1.0:
            wav = wav / 32768.0
        if sr != TARGET_SR:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

        if self.mode == "stub":
            time.sleep(0.2)
            return ("[stub] no model loaded — wiring only", HERO, 0.0,
                    time.time() - t0)

        feats = self.fe([wav], sampling_rate=TARGET_SR, return_tensors="np",
                        padding=True, return_attention_mask=True)
        if self.mode == "onnx":
            logits, lid_logits, _ = self.session.run(None, {
                "input_features": feats["input_features"].astype(np.float32),
                "attention_mask": feats["attention_mask"].astype(np.int64),
            })
        else:
            import torch

            with torch.no_grad():
                out = self.session(
                    input_features=torch.from_numpy(feats["input_features"]),
                    attention_mask=torch.from_numpy(feats["attention_mask"]),
                )
            logits = out["logits"].numpy()
            lid_logits = out["lid_logits"].numpy()

        text = self.tokenizer.decode(logits[0].argmax(-1))
        e = np.exp(lid_logits[0] - lid_logits[0].max())
        probs = e / e.sum()
        idx = int(probs.argmax())
        lang = self.languages[idx] if idx < len(self.languages) else "?"
        return text, lang, float(probs[idx]), time.time() - t0


def build_ui():
    import gradio as gr

    engine = Engine()
    banner = "" if engine.mode != "stub" else (
        "> **Stub mode** — interface only, no weights loaded yet.\n\n")

    def run(audio):
        if audio is None:
            return "", "", ""
        wav = np.asarray(audio[1])
        try:
            sr = audio[0]
            if wav.size < sr // 4:
                return "[too short] record at least half a second", "", ""
            text, lang, conf, dt = engine.transcribe(sr, wav)
        except Exception as e:  # noqa: BLE001 - the browser shows "Error" and nothing else
            # Gradio renders a bare "Error" chip and keeps the traceback in the
            # server log, which is useless to anyone testing from a phone. Put
            # the message where the person clicking the button can read it, and
            # still print the traceback for whoever has the terminal.
            import traceback

            traceback.print_exc()
            return f"[{type(e).__name__}] {e}", "", ""
        rtf = dt / max(len(wav) / sr, 1e-9)
        return text, f"{pretty(lang)} — {conf:.0%} confident", \
            f"{dt*1000:.0f} ms  ({rtf:.2f}x real time, {engine.mode})"

    with gr.Blocks(title="KASA-42") as demo:
        gr.Markdown(
            f"# KASA-42\n"
            f"{banner}"
            f"One speech model for **{len(engine.languages)} Ghanaian languages**. "
            f"Speak, and it works out which language you used before transcribing.\n\n"
            f"Trained on `ghananlpcommunity/ghana-speech` with **book-disjoint splits** — "
            f"no recording appears in both training and test."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                 label="Speak or upload")
                btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                out_text = gr.Textbox(label="Transcript", lines=4)
                out_lang = gr.Textbox(label="Detected language")
                out_time = gr.Textbox(label="Latency")

        btn.click(run, [audio], [out_text, out_lang, out_time])
        audio.stop_recording(run, [audio], [out_text, out_lang, out_time])

        with gr.Accordion("Languages covered", open=False):
            ordered = ([HERO] if HERO in engine.languages else []) + \
                      [c for c in sorted(engine.languages) if c != HERO]
            gr.Markdown("  ·  ".join(pretty(c) for c in ordered))

    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", action="store_true",
                    help="Public gradio.live link, for testing from a phone while "
                         "the box is still up. Expires in 72 h — a Space is the "
                         "durable answer.")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(share=args.share, server_name="0.0.0.0",
                      server_port=args.port)
