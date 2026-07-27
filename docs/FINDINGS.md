# Phase 0 findings — `ghananlpcommunity/ghana-speech`

Established Mon 27 Jul 2026, before the GPU window, by projecting only the
metadata columns out of the parquet shards. No audio was downloaded.

## Provenance: it is Bible audio

`source_file` values parse cleanly as `BOOK.CHAPTER.VERSION`:

```
1CO.15.3752   ->  1 Corinthians, ch. 15, version 3752
1CH.2.3752    ->  1 Chronicles,  ch. 2,  version 3752
```

Sampling 4 of 22 Kusaal shards: 8,834 segments, 233 distinct chapters,
19 distinct books, and **exactly one version suffix (`3752`)**.

This is the open.bible / MMS-style corpus pattern.

### Three consequences

**1. Kusaal TTS is viable — the gate passes.**
A single version suffix means a single recording project, which in Bible audio
means a single narrator. The 54 `source_file` values in a shard are *chapters*,
not speakers — my earlier reading of that number as narrator diversity was wrong.
VITS fine-tuning from `facebook/mms-tts-kus` has the roughly-single-speaker audio
it needs. Worth confirming by ear on two clips from distant books at H+0.

**2. Split by BOOK, not chapter, and certainly not at random.**
Adjacent chapters share names, places and vocabulary heavily — the genealogies
alone repeat proper nouns across whole books. A chapter-level split still leaks
lexically. Book-level disjointness is the honest version, and there is plenty to
split on (~66 books, ~1,189 chapters at full coverage).

**3. The domain caveat is now confirmed, not suspected.**
This is read scripture: formal, archaic, proper-noun dense. No model trained on
it alone will transcribe spontaneous Kusaal well. Every team's WER will look
flattering. Saying this plainly, with the leaked-vs-honest split numbers to back
it, is the credibility play.

## The same text, read by different people

Scanning 139 shards for the VERSION field: **35 of 42 languages have exactly one
recording project. Seven have more**, and in six of those the *same book* appears
under more than one version.

| Config | Versions | Split |
|---|---|---|
| Asante Twi | **3** (39% / 31% / 30%) | overlapping books |
| Ewe | 2 (51% / 49%) | overlapping books |
| Fante | 2 (50% / 50%) | overlapping books |
| Hausa | 2 (50% / 50%) | overlapping books |
| Ninkare | 2 (54% / 46%) | overlapping books |
| Bimoba | 2 (77% / 23%) | overlapping books |
| Kasem | 2 (70% / 30%) | disjoint books |

**Asante Twi — the language most teams will target — contains the same scripture
read by three different narrators.** A random split there does not just leak
acoustics and session; it puts a *byte-identical reference transcript* on both
sides of the boundary. The model can memorise the text outright. Reported WER
becomes close to meaningless.

### The fix, and a trap next to it

Split on **book alone**, so every version of a book travels together into the
same split.

The tempting refinement — splitting on `(version, book)` to be "more granular" —
is actively wrong here. It would place Genesis-v1 in train and Genesis-v2 in
test: different audio, identical transcript. That is a worse leak than the one
we set out to avoid. `data/splits.py` groups by book for exactly this reason.

Note also that book counts are low (9–52, typically 18–27). At a 5% test
fraction some languages get one or two books, so per-language test sets are small
and their WER carries real variance. Worth stating alongside the numbers rather
than reporting three decimal places.

## Text

| Property | Value |
|---|---|
| Casing | 90.7% mixed/sentence case, 5.9% all-lower, 2.6% ALL-CAPS (headings) |
| Charset (Kusaal) | 76 chars |
| Special glyphs | `Ŋ ŋ Ɔ ɔ Ɛ ɛ Ʋ ʋ` |
| Noise to normalize | smart quotes `‘ ’ “ ”`, digits, `!,-.:;?` |

Sample: `Jafet biribis da anɛ: Goma nɛ Magog nɛ Madai nɛ Javan nɛ Tubal…`

Normalization for CTC: lowercase, fold smart quotes to ASCII, strip punctuation,
decide digit policy (verse numbers appear in text — likely drop affected segments
rather than spell them out across 42 orthographies).

## Kusaal at a glance

| Metric | Value |
|---|---|
| Segments | 48,598 (dataset card: 48,590 — scaling from 1 shard is accurate) |
| Audio | 81.6 h |
| Duration p50 / p90 / max | 4.86 s / 11.9 s / 29.3 s |
| Chapters (4 of 22 shards) | 233 |

Durations sit comfortably inside CTC's practical range; no 30 s padding waste,
which is exactly why w2v-BERT 2.0 over Whisper is the right call here.

## Method note

`BYTES_PER_SEC = 32000` (16 kHz, 16-bit, mono) was verified against the dataset
card: Asante Twi 23.06 GB / 32000 = 200.2 h vs. the stated 200.02 h.
