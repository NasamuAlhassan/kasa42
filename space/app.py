"""KASA-42 demo — standalone, CPU-only, no dependency on the kasa42 package.

Deliberately self-contained. A Space that imports the training package inherits
torch, datasets and pyarrow for the sake of a ten-line CTC decode, and it breaks
whenever that package moves. Everything it needs sits in this directory:

    kasa42_asr.int8.onnx      the graph and its weights, one file
    vocab.json                the character vocabulary
    languages.json            LID index -> config name
    preprocessor_config.json  the encoder's own feature extractor config

The blank is `[PAD]` = 33, not 0 — DONDO's position, kept so its trained CTC
head rows stay attached to the right characters. Nothing here may assume 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from transformers import SeamlessM4TFeatureExtractor

HERE = Path(__file__).parent
TARGET_SR = 16000
HERO = "Kusaal_kus"

vocab = json.loads((HERE / "vocab.json").read_text(encoding="utf-8"))["vocab"]
INV = {v: k for k, v in vocab.items()}
BLANK = vocab.get("[PAD]", 0)
SPACE = vocab.get("|", 0)
LANGUAGES = json.loads((HERE / "languages.json").read_text(encoding="utf-8"))

fe = SeamlessM4TFeatureExtractor.from_pretrained(str(HERE))
session = ort.InferenceSession(str(HERE / "kasa42_asr.int8.onnx"),
                               providers=["CPUExecutionProvider"])


def ctc_decode(ids) -> str:
    """Collapse repeats, then drop blanks."""
    out, prev = [], None
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(" " if i == SPACE else INV.get(i, ""))
        prev = i
    return "".join(out).strip()


def pretty(config: str) -> str:
    name, _, iso = config.rpartition("_")
    return f"{name.replace('_', ' ')} ({iso})"


def transcribe(audio):
    if audio is None:
        return "", "", ""
    sr, wav = audio
    wav = np.asarray(wav)
    t0 = time.time()

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if np.abs(wav).max() > 1.0:          # int16 arriving as float
        wav = wav / 32768.0
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    if wav.size < TARGET_SR // 10:
        return "", "clip too short — record at least a second", ""

    feats = fe([wav], sampling_rate=TARGET_SR, return_tensors="np",
               padding=True, return_attention_mask=True)
    logits, lid_logits, lengths = session.run(None, {
        "input_features": feats["input_features"].astype(np.float32),
        "attention_mask": feats["attention_mask"].astype(np.int64),
    })

    # Trim to real frames before decoding; padding frames invent characters.
    n = int(lengths[0]) if lengths is not None else logits.shape[1]
    text = ctc_decode(logits[0][:n].argmax(-1))

    e = np.exp(lid_logits[0] - lid_logits[0].max())
    probs = e / e.sum()
    idx = int(probs.argmax())
    lang = LANGUAGES[idx] if idx < len(LANGUAGES) else "?"

    dt = time.time() - t0
    rtf = dt / max(len(wav) / TARGET_SR, 1e-9)
    return (text or "[no speech detected]",
            f"{pretty(lang)} — {probs[idx]:.0%} confident",
            f"{dt*1000:.0f} ms  ({rtf:.2f}x real time, int8 CPU)")


with gr.Blocks(title="KASA-42") as demo:
    gr.Markdown(
        f"# KASA-42\n"
        f"One speech model for **{len(LANGUAGES)} Ghanaian languages**. Speak, and "
        f"it works out which language you used before transcribing — no language "
        f"picker.\n\n"
        f"Trained on `ghananlpcommunity/ghana-speech` with **book-disjoint splits**: "
        f"no book appears in both training and test. "
        f"**30.2% WER / 10.5% CER**, language ID **96.8%** over 42 languages.\n\n"
        f"⚠️ Trained entirely on *read scripture* — formal, studio-recorded, one or "
        f"two narrators per language. Expect it to do markedly worse on "
        f"conversational speech."
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

    btn.click(transcribe, [audio], [out_text, out_lang, out_time])
    audio.stop_recording(transcribe, [audio], [out_text, out_lang, out_time])

    with gr.Accordion(f"All {len(LANGUAGES)} languages", open=False):
        ordered = ([HERO] if HERO in LANGUAGES else []) + \
                  [c for c in sorted(LANGUAGES) if c != HERO]
        gr.Markdown("  ·  ".join(pretty(c) for c in ordered))

if __name__ == "__main__":
    demo.launch()
