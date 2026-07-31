---
license: cc-by-nc-4.0
language:
  - acd
  - ada
  - akp
  - any
  - avn
  - bib
  - bim
  - biv
  - bov
  - bud
  - bwu
  - dag
  - dga
  - ewe
  - fat
  - ffm
  - gjn
  - gur
  - hau
  - kbp
  - kdh
  - kma
  - kus
  - lef
  - lip
  - maw
  - mzw
  - naw
  - ncu
  - nko
  - ntr
  - nzi
  - sfw
  - sig
  - sil
  - snw
  - tpm
  - twi
  - vag
  - xon
  - xsm
tags:
  - ghana-nlp
  - speech
  - ghana-speech
  - automatic-speech-recognition
  - language-identification
  - w2v-bert
---

# KASA-42 — one speech model for 42 Ghanaian languages

**Author:** Prince Nasamu Alhassan

## Overview

A single CTC checkpoint that transcribes **42 Ghanaian language subsets** and
identifies which of them is being spoken, without being told. Fine-tuned from
[DONDO](https://arxiv.org/abs/2607.21540) (`KhayaAI/w2v-bert-…`, Apache-2.0),
which covers 11 Northern Ghanaian languages, and extended to the full
`ghana-speech` corpus with a jointly trained language-ID head.

The 42 configs map to 41 ISO 639-3 codes: Akuapem Twi and Asante Twi share
`twi` but are separate recording projects and separate rows below.

## Results

Book-disjoint test set — no book appears in both training and test.
8,400 utterances, 200 per language.

| Metric | Micro | Macro |
|---|---|---|
| WER | **30.2%** | 32.5% |
| CER | **10.5%** | 11.8% |
| Language ID accuracy | **96.8%** mean | 91.5% min, 100% max |

Language ID is the capability DONDO does not have: it conditions on a one-hot
language prefix, so the caller must already know the language. Here it is
inferred, at 96.8% mean accuracy over 42 classes (chance is 2.4%).

Per-language WER ranges from 13.0% (Gonja) to 69.8% (Kabiye). **Read those to
the nearest few points, not to one decimal** — see the variance note below.

## Against other systems

Every system scored on the same book-disjoint test set, with the same text
normalisation applied to references and hypotheses alike. Crucially, each row
recomputes **both** sides over only the languages that system covers — comparing
an average over 34 covered languages against an average over all 42 would
flatter whichever system declined the hardest ones.

| System | Languages | Their WER | Ours | Their CER | Ours |
|---|---|---|---|---|---|
| Whisper large-v3 | 42 | 102.3% | **30.2%** | 57.3% | **10.5%** |
| **MMS-1B-all** | **34** | **25.7%** | 29.2% | 12.1% | **9.7%** |
| DONDO (unconditioned) | 8 | 75.9% | **27.6%** | 40.8% | **14.1%** |

**MMS-1B-all beats us on WER by 3.5pp.** We beat it on CER by 2.4pp, and cover
eight languages it has no adapter for (both Twis, Dagbani, Dangme, Fante,
Fulfulde and two others). It is also roughly 1.7× our parameter count and does
not do language identification.

That WER/CER split is the informative part. MMS's ratio is 2.1×, ours 3.0× — we
get **more characters right and more words wrong**, which is a word-segmentation
weakness rather than an acoustic one. The same signature appears in our
per-language table (Bissa 41.7% WER against 9.5% CER). Space prediction is the
obvious place to look next, and we have not looked yet.

Whisper exceeds 100% WER because WER counts insertions: it transcribes these
languages into something else entirely and emits more words than the reference
contains. It was never trained on them.

**The DONDO row is not a fair comparison and should not be cited as one.**
DONDO conditions on a one-hot language prefix; `asr/baselines.py` supplies none
and runs it as a plain CTC model, so 75.9% is a lower bound on its ability, not a
measurement of it. Its published figure is ~10.3% average WER on 11 languages.
We report the number for transparency about what we ran, not as evidence about
DONDO. Fixing this needs the conditioning implemented, which the window did not
allow.

## The leakage experiment, and its negative result

`ghana-speech` is Bible audio: `source_file` parses as `BOOK.CHAPTER.VERSION`.
A random segment split therefore puts adjacent verses — same narrator, same
session — on both sides of the boundary, and in six languages the *same book*
appears under multiple recording versions, so a random split can place a
byte-identical reference transcript in train and test.

We split by **book** for that reason, and then tested whether it mattered.
Identical weights were scored on two sets: segments from held-out books, and
segments from *seen* books that the model was not trained on. Only the second
carries book-level overlap.

| Construction | honest − leaked | 95% CI |
|---|---|---|
| Unmatched | −1.1pp | [−1.9, −0.3] |
| Book-diversity matched | **−0.7pp** | **[−1.4, +0.1]** |

**The interval spans zero.** On these weights, book-level overlap gives no
measurable advantage. We report this as a negative result rather than omitting
it.

Two caveats we state rather than leave to the reader:

* This says the model is **not memorising books at this scale** — one pass over
  700 h at ~30% WER — not that leakage cannot occur. A longer-trained or
  higher-capacity model may behave differently.
* We did **not** test chapter-adjacent leakage, which is the strongest form: a
  naive split puts consecutive verses from one recording session on both sides.
  Our leaked set samples across whole books, which dilutes that.

Book-disjoint splitting remains the conservative default. Our result says it
cost us nothing.

The first construction was confounded — the honest set spanned 4.2 books per
language against the leaked set's 27.2, with 4.9% shorter references — so the
comparison varied content diversity alongside book-seen-ness. The matched row
above removes that.

## Sampling variance

Three independent draws of the same test design gave micro WER of 30.3%, 30.0%
and 30.2%, while **individual languages moved by up to 7.3pp** (Ninkare
57.2% → 64.5%). At n=200 per language, the micro-average is stable to a few
tenths and per-language figures are not. Quote them accordingly.

## Training data

[`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech),
CC BY-NC 4.0. 1,411,467 segments, 2,334.9 h across 42 configs.

Training used a temperature-sampled mixture (α=0.5, 40 h per-language cap) over
train-split books only: **700 h from 393,160 segments**, which reduces the
largest language's share from 8.5% to 4.9%.

### Three things we found in the corpus

1. **`id` is not unique in 13 of 42 configs**, by factors up to 1.52×. Anything
   selecting segments by id — the obvious design — silently over-samples those
   languages, and the hours accounting reconciles either way, so nothing looks
   wrong.
2. **One version code does not imply one narrator.** Kusaal is 100% version
   `3752`, yet ECAPA-TDNN speaker embeddings split its books cleanly into Old
   Testament (28 books, 39.7 h) and New Testament readers.
3. **21 Hebrew characters appear in Ewe and Hausa** — Psalm 119's acrostic
   stanza headings, 67 occurrences total. They look like an encoding fault and
   are not.

## Intended use and limitations

**Non-commercial use only** (CC BY-NC 4.0, inherited from the training data).

This is trained entirely on **read scripture**: formal, archaic, proper-noun
dense, studio-recorded, one or two narrators per language. It will transcribe
spontaneous conversational speech considerably worse than these numbers
suggest. Every model trained on this corpus has that limitation; ours is not
exempt.

## How to use

```python
import torch, soundfile as sf
from transformers import SeamlessM4TFeatureExtractor
from kasa42.asr.model import Kasa42ForCTC, model_state
from kasa42.asr.dataset import CharTokenizer

tok = CharTokenizer.from_json("vocab.json")
model = Kasa42ForCTC("KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en",
                     vocab_size=len(tok), n_languages=42, blank_id=tok.blank)
model.load_state_dict(model_state("final.pt"))
model.eval()

fe = SeamlessM4TFeatureExtractor.from_pretrained(model.encoder.name_or_path)
wav, sr = sf.read("clip.wav")                       # 16 kHz mono
feats = fe([wav], sampling_rate=16000, return_tensors="pt",
           padding=True, return_attention_mask=True)
with torch.no_grad():
    out = model(**feats)
print(tok.decode(out["logits"].argmax(-1)[0][: out["input_lengths"][0]]))
print("language:", languages[out["lid_logits"].argmax(-1).item()])
```

An ONNX export is included and verified against the torch model at 150/300/500/900
frames, so it is safe for variable-length audio and runs on CPU.

## Training details

| | |
|---|---|
| Base | DONDO `KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en` (Apache-2.0) |
| Architecture | w2v-BERT 2.0 encoder + CTC head + mean-pool linear LID head |
| Vocabulary | 96 chars — DONDO's 49 keep their indices and trained head rows, 47 added |
| Blank | `[PAD]` = **33**, not 0 (DONDO's position) |
| Steps | 24,000, batch budget 160 s of padded audio, gradient checkpointing |
| Optimiser | AdamW, lr 5e-5, 500 warmup, cosine decay, encoder frozen 300 steps |
| Precision | bf16 (Hopper) |
| Hardware | 1 × NVIDIA H200 (shared), ~6 h |

## Claims we do not make

`facebook/mms-tts-kus` exists, MMS-1B-all covers Kusaal ASR, and DONDO covers
Kusaal at 13.3% WER. **This is not the first Kusaal speech model, and not the
first multilingual Ghanaian ASR.** The claims are narrower: broader language
coverage in one checkpoint, language ID that DONDO does not have, and results on
a split that does not leak, with the leakage question actually tested.

See the comparison below for how this sits against MMS, Whisper and DONDO. It is
not first on every metric, and the section says so.

## Acknowledgements

Compute provided by **AI Skills and Compute Africa (AISCA)**, trained on the
Ghana NLP H200. Corpus by the **GhanaNLP community**. Base checkpoint by
**Paul Azunre** ([DONDO](https://arxiv.org/abs/2607.21540)).

Please keep derivatives non-commercial and share improvements back with
`ghananlpcommunity`.
