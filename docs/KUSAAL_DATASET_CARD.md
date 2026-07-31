---
license: other
license_name: derived-scripture-audio
license_link: LICENSE
language:
  - kus
task_categories:
  - automatic-speech-recognition
task_ids:
  - keyword-spotting
pretty_name: Kusaal ASR Dataset
size_categories:
  - 10K<n<100K
tags:
  - kusaal
  - gur
  - ghana
  - low-resource
  - african-languages
  - speech
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train-*.parquet"
      - split: val
        path: "data/val-*.parquet"
      - split: test
        path: "data/test-*.parquet"
---

# Kusaal ASR Dataset

Verse-level speech recognition data for **Kusaal** (ISO 639-3: `kus`), a Gur
language spoken by roughly 350,000 people in the Upper East Region of Ghana and
adjacent Burkina Faso.

| | |
|---|---|
| Clips | 30,820 |
| Audio | 81.7 h, 16 kHz mono 16-bit PCM |
| Books | 63 |
| Mean clip | ~9.5 s · ~25.6 words |
| Splits | book-held-out train / val / test |

```python
from datasets import load_dataset

ds = load_dataset("PrinceAlhassanNasamu/kusaal-asr-dataset")
print(ds["train"][0]["sentence"])
# streaming works too — no need to pull 6.5 GB to look at it
ds = load_dataset("PrinceAlhassanNasamu/kusaal-asr-dataset", streaming=True)
```

## Splits

Books are held out whole, so no clip from a held-out book appears in any other
split. Adjacent verses share proper nouns, place names and phrasing heavily, so
a random clip-level split would leak lexically and overstate generalisation.

| Split | Clips | Hours | Books |
|---|---|---|---|
| train | 24,597 | 69.33 | 55 |
| val | 4,622 | 8.80 | LUK, PHP, PRO, PSA |
| test | 1,601 | 3.58 | JON, MAT, REV, RUT |

Old Testament is ~79% of hours, New Testament ~21%.

## Fields

| Field | Type | Notes |
|---|---|---|
| `audio` | Audio | 16 kHz mono |
| `sentence` | string | Kusaal transcript, lowercase, includes `ŋ ɔ ɛ ʋ` |
| `id` | string | USFM verse reference, e.g. `MAT.6.24` |
| `book` | string | USFM book code |
| `chapter`, `verse` | string | Position within the book |
| `duration_ms` | string | Clip length |

## Licensing — read this before using the audio

Three layers, and they are not the same:

| Layer | Holder | Terms |
|---|---|---|
| **Old Testament audio** (~79% of hours) | © Davar Partners International (2023) — [davaraudiobibles.org](https://davaraudiobibles.org) | Non-commercial |
| **New Testament audio** (~21%) | Kusaal NT Drama `KUSTBLN2DA`, © Wycliffe / Hosanna (1996, 2005), distributed by Faith Comes By Hearing — [faithcomesbyhearing.com](https://faithcomesbyhearing.com) | Non-commercial |
| **Additional audio** | Global Recordings Network, Kusal "Look, Listen & Live" and "Words of Life" — [globalrecordings.net](https://globalrecordings.net) | Non-commercial |
| **Scripture text** | GILLBT Kusaal Bible — [gillbt.org](https://gillbt.org) | Non-commercial |
| **Alignment, segmentation, packaging** | Prince Nasamu Alhassan | CC BY 4.0, research use |

Only the last row is the dataset author's to license. The recordings come from
scripture ministries who publish audio in minority languages as a free public
service; their work is what makes this dataset possible.

**Use this for research, education, and non-commercial language technology.**
Commercial use requires clearing rights with **Davar, FCBH, GRN and GILLBT
directly**. This dataset grants nothing on their behalf, and nothing here
implies endorsement by any of them.

Models trained on it inherit these constraints.

## Provenance

Chapter-level scripture recordings from Davar (OT), FCBH/Hosanna (NT Drama)
and GRN, force-aligned at verse
level by CTC segmentation against GILLBT Kusaal Bible text, sliced to individual
verses and resampled to 16 kHz mono. Built 2026-05-23. Verse text lineage:
[`PrinceAlhassanNasamu/kusaal-english-parallel-corpus`](https://huggingface.co/datasets/PrinceAlhassanNasamu/kusaal-english-parallel-corpus).

Build skipped 60 verse attempts (58 slice failures, 2 bad text). 1 John, 2 John
and 3 John are absent — too short for reliable alignment.

## Limitations

- **Single domain.** Read scripture: formal, archaic, proper-noun dense,
  studio-recorded. Expect substantially worse performance on conversational or
  broadcast speech. This is measured, not assumed — see below.
- **Limited speaker diversity.** FCBH `KUSTBLN2DA` is a *dramatised* New
  Testament, so the NT portion carries multiple voice actors while the GRN
  material does not. Anyone fine-tuning single-speaker TTS on this should
  cluster speakers first rather than trusting the source label.
- **Residual boundary errors** from forced alignment at verse edges.
- **Not the first Kusaal speech corpus.**
  [`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
  contains 72.5 h of Kusaal, and DONDO covers Kusaal ASR. This corpus is larger
  for the language and independently sourced, which is a different claim.

## Known result on this data

[KASA-42](https://huggingface.co/PrinceAlhassanNasamu/kasa42-asr), a 42-language
model trained on `ghana-speech` and never on this corpus, scores **37.9% WER /
21.9% CER** on the 11 books it held out of its own training, against **30.2% /
16.3%** on `ghana-speech`'s own Kusaal. Language ID *improves*, 95.5% → 99.8%.

That is a clean cross-corpus measurement: same language, same domain, different
recording ministry and alignment pipeline. It gives anyone using this dataset a
published reference point.

Note that the other splits are **not** usable for evaluating models trained on
`ghana-speech`: scripture corpora share a translation, so a model trained on
those books has already seen the sentences.

## Citation

```bibtex
@dataset{alhassan_prince_kusaal_asr_2026,
  author    = {Alhassan, Prince Nasamu},
  title     = {{Kusaal ASR Dataset}: Verse-Level Scripture Speech Recognition Corpus},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/PrinceAlhassanNasamu/kusaal-asr-dataset},
  note      = {30,820 clips, 81.7 hours, 16 kHz mono, book-held-out splits}
}
```

## Related

- [`kusaal-english-parallel-corpus`](https://huggingface.co/datasets/PrinceAlhassanNasamu/kusaal-english-parallel-corpus) — 34,568 parallel verse pairs
- [`kusaal-nllb-600M`](https://huggingface.co/PrinceAlhassanNasamu/kusaal-nllb-600M) — NLLB-200 fine-tune, BLEU 27.57 kus→eng
- [`kasa42-asr`](https://huggingface.co/PrinceAlhassanNasamu/kasa42-asr) — 42-language Ghanaian ASR with language ID

Affiliated with GhanaNLP.
