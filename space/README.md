---
title: KASA-42
emoji: 🗣️
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# KASA-42

Speech recognition and language identification for **42 Ghanaian languages** in
one model. Speak, and it identifies the language before transcribing.

- **30.2% WER / 10.5% CER** on a book-disjoint test set (8,400 utterances)
- **96.8%** mean language-ID accuracy over 42 languages
- int8 ONNX, runs on CPU

Fine-tuned from [DONDO](https://arxiv.org/abs/2607.21540) on
[`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech).
Model card and full per-language results:
[`PrinceAlhassanNasamu/kasa42-asr`](https://huggingface.co/PrinceAlhassanNasamu/kasa42-asr).

Trained on read scripture, so conversational speech will score considerably
worse than these numbers suggest. Non-commercial use only (CC BY-NC 4.0).

Compute by **AI Skills and Compute Africa (AISCA)**.
