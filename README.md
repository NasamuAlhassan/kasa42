# KASA-42

One speech model for **42 Ghanaian languages**, plus a Kusaal voice, built on
[`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech).

## What this is

An extension of **DONDO** ([arXiv:2607.21540](https://arxiv.org/abs/2607.21540),
`KhayaAI/w2v-bert-...`, Apache-2.0), which covers 11 Northern Ghanaian languages
including Kusaal at ~10.3% average WER.

Three things are added:

1. **11 → 42 languages.** Every language in `ghana-speech`, in one checkpoint.
2. **Automatic language identification.** DONDO conditions on a one-hot language
   prefix, so the caller must already know what is being spoken. A jointly
   trained LID head removes that requirement — speak, and it works out which of
   the 42 it is heard.
3. **A book-disjoint benchmark.** Per-language WER/CER on splits where no
   recording appears in both training and test.

We fine-tune *from* DONDO rather than from raw `facebook/w2v-bert-2.0`: it has
already seen this language family and this recording domain, which is worth a
great deal when the budget is 48 hours.

## What makes the numbers real

The corpus is Bible audio: `source_file` parses as `BOOK.CHAPTER.VERSION`.

1. **A random split leaks acoustics and session.** Adjacent verses from the same
   chapter, same narrator, same recording, land on both sides.
2. **Worse: six languages contain the same book under multiple versions.**
   Asante Twi has the same scripture read by *three* different narrators, so a
   random split puts a byte-identical reference transcript in train and test.
   The model can memorise it.
3. **The fix is to split by BOOK alone**, so all versions of a book travel
   together. Splitting on `(version, book)` looks more careful and is actively
   worse — it separates Genesis-v1 from Genesis-v2, identical text in different
   voices.

We report the honest book-disjoint number *and* the leaked random-split number
from the same weights. The gap is the point.

See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full audit.

## Design decisions

| Component | Choice | Why |
|---|---|---|
| Encoder | **DONDO** (w2v-BERT 2.0 + CTC) | Already fine-tuned on 11 Ghanaian languages in this domain. w2v-BERT is 10–30× faster to fine-tune than Whisper-v3 at comparable WER, better on low-resource, and variable-length — segments average 5.7 s, so Whisper would spend ~80% of compute padding to 30 s |
| Vocab | DONDO's 49 tokens **extended**, not replaced | Shared characters keep their trained CTC head rows. Blank is `[PAD]`=**33**, not 0 — `blank_id` is explicit throughout |
| Homoglyphs | `ԑ→ɛ`, `ↄ→ɔ`, `ǝ→ə`, Greek/Cyrillic lookalikes folded | DONDO's own vocab contains both U+0511 and U+025B — the same letter splitting probability mass across two of only 49 output units |
| Language ID | Mean-pool + linear head, trained jointly | Near-free, and it is the capability DONDO explicitly lacks |
| Mixture | Temperature sampling α=0.5 + per-language cap | Raw data is ~16× imbalanced; capping also cuts 2,247 h to ~700 h so the run fits in hours |
| Serving | ONNX int8 | The GPU lease expires Sat 1 Aug 08:31 GMT; the demo must outlive it |

## Layout

```
src/kasa42/
  data/   audit  versions  build_manifest  splits  mixture  vocab  text
  asr/    model  dataset  train  evaluate  baselines  export
  tts/    prepare  finetune  roundtrip
  app/    app.py                    # Gradio, runs in stub/onnx/torch mode
notebooks/kasa42_h200.ipynb         # the H200 driver
docs/FINDINGS.md                    # what the audit established
tests/test_text.py                  # normalization guardrails
```

## Phase 0 (before the GPU)

Everything here runs on a laptop. The audit reads **only** the parquet metadata
columns via column projection, so all 42 languages can be analysed without
downloading any of the 222 GB of audio.

```bash
uv venv --python 3.12 && uv pip install -e .
python -m kasa42.data.audit --max-shards 1        # per-language overview
python -m kasa42.data.versions                    # recording projects per language
python -m kasa42.data.build_manifest              # ~150 MB metadata table
python -m kasa42.data.splits                      # book-disjoint splits
python -m kasa42.data.vocab                       # shared CTC charset
python -m kasa42.data.mixture --alpha 0.5 --cap-hours 40
python tests/test_text.py
```

## Phase 1 (H200 window)

Follow `notebooks/kasa42_h200.ipynb`. It clones this repo and drives the modules;
nothing of substance lives in the notebook, so a dead kernel costs nothing.

Two things not to skip:

- **The smoke run at H+1.** 30 steps, 5 languages. Never start a multi-hour run
  on unvalidated code.
- **The listen check before TTS.** `data/tts/check/*.wav` are clips from distant
  books. If they are not the same voice, single-speaker VITS fine-tuning will
  produce a muddy averaged voice — spend the hours on ASR instead.

## Claims we do not make

`facebook/mms-tts-kus` already exists, MMS-1B-all covers Kusaal ASR, and DONDO
covers Kusaal at 13.3% WER. **This is not the first Kusaal speech model, and not
the first multilingual Ghanaian ASR.** The claim is narrower and checkable:
broader language coverage, language ID that DONDO does not have, and results on
a split that does not leak.

Baselines are reported on our own book-disjoint test set so that every number
sits on the same axis. DONDO is evaluated *with* the language prefix supplied —
its favourable setting, and the fair one.

Models are Apache-2.0. The `ghana-speech` data is CC BY-NC 4.0, so trained
weights are released for non-commercial research use.

## Credits

- **DONDO** — Paul Azunre, [arXiv:2607.21540](https://arxiv.org/abs/2607.21540).
  Base checkpoint and the model this work extends.
- **`ghananlpcommunity/ghana-speech`** — GhanaNLP community.
- Kusaal MT and prior Kusaal ASR: `PrinceAlhassanNasamu/kusaal-nllb-600M`,
  `PrinceAlhassanNasamu/kusaal-whisper-small-lora`.
