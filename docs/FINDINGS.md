# Findings — `ghananlpcommunity/ghana-speech`

Established Mon 27 Jul 2026, before the GPU window, by projecting only the
metadata columns out of the parquet shards over the network. No audio was
downloaded.

Revised Thu 30 Jul 2026 against the complete 42-config manifest — 1,411,467
segments, 2,334.9 h, built in about a minute from the copy pre-staged on the
H200 at `/data/ghana-speech`. Several Phase 0 figures had been extrapolated from
single shards and are corrected below; corrections are marked. One finding is
new, and it is the kind that only appears once you have every row.

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

**Corrected 30 Jul.** Book counts were estimated at 9–52 from sampled shards.
The full manifest gives **26–66 per language, 1,801 books in total** — small
configs cluster at 26–28, large ones at 60–66. The worry that a 5% test fraction
would leave some languages with only one or two books does not materialise:
realised test fractions run 5.0%–8.2%, and the ones at the top of that range are
simply languages whose books are large enough that a single book overshoots the
target. Per-language test sets are still small in absolute terms — we sample 200
utterances each — so WER carries real variance and should not be quoted to three
decimals.

## `id` is not a unique key

New, 30 Jul, from the full manifest. **13 of 42 configs contain repeated ids.**

| Config | Rows | Unique ids | Factor |
|---|---|---|---|
| Sisaala Tumulung | 47,096 | 31,074 | **1.52×** |
| Bimoba | 46,911 | 38,240 | 1.23× |
| Chumburung | 9,332 | 7,687 | 1.21× |
| Gikyode | 9,377 | 7,911 | 1.19× |
| nine others | | | 1.06–1.13× |

The other 29 configs, Asante Twi and Ewe among them, have wholly unique ids.

This is a trap for anything that selects segments by id — which is the obvious
design, and ours. The mixture drew ids until it had the hours it wanted, and the
loader then matched *every* row carrying a selected id. Sisaala's 22.9 h
allocation would have trained on roughly 34.8 h, making a tail language joint
largest in the mixture above Ewe and Hausa: exactly the imbalance temperature
sampling exists to remove. A test set built the same way returned 380 utterances
for Sisaala against a cap of 200, and those utterances would have carried double
weight in the per-language WER.

What makes it hard to catch is that **the accounting reconciles either way**.
Hours are summed over rows, so the printed totals are correct under both
readings and nothing looks wrong. The cap on the test set was the only thing
that gave it away, and only because the cap was an exact number to check
against.

The fix is to key on id at both ends and pick the same copy at each. We take the
shortest, because manifest row order is thread completion order and so is not
stable across rebuilds. `python -m kasa42.data.build_manifest --check-ids`
reports the affected configs and what each would contribute deduplicated.

**Mechanism not established.** The natural guess is the multi-version pattern
above — the same verse under two recording projects, sharing an id. It cannot be
the whole story: Asante Twi carries three versions and repeats no ids at all,
and the factors are not integral, so any overlap is partial rather than
wholesale. Settle it before citing a cause:

```bash
python - <<'EOF'
import pyarrow.parquet as pq
from collections import Counter, defaultdict
t = pq.read_table('results/manifest.parquet',
                  columns=['config','id','source_file','duration','shard'])
rows = defaultdict(list)
for c,i,s,d,sh in zip(*[t.column(k).to_pylist()
                        for k in ('config','id','source_file','duration','shard')]):
    if c == 'Sisaala_Tumulung_sil':
        rows[i].append((s, round(float(d),2), sh))
for i, v in list((i,v) for i,v in rows.items() if len(v) > 1)[:5]:
    print(i, v)
EOF
```

If the copies differ only in the VERSION field, it is the multi-version pattern
and the note above stands. If they are identical in `source_file` *and*
duration, the shards contain genuine duplicate rows, which is a different claim
and a more serious one about the corpus.

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

**Resolved 30 Jul, across all 1.41 M segments.** Dropping digit-bearing segments
costs **2.7%** of the corpus (38,252 of 1,411,467), which is cheap enough to
settle the question. The shared vocabulary comes to **96 tokens**: DONDO's 49
keep their indices and their trained CTC head rows, and 47 are new.

25 characters fell below the frequency floor of 50, together accounting for 67
occurrences — one ten-thousandth of a percent of the text. Twenty-one of them
are Hebrew: `א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת`, appearing in Ewe and
Hausa. Those are Psalm 119's acrostic section headings, which is exactly the
count you would expect — the psalm has 22 stanzas, one per Hebrew letter. Worth
recording because it looks like an encoding fault and is not.

## Kusaal at a glance

| Metric | Value | |
|---|---|---|
| Segments | **48,590** | est. 48,598 — scaling from one shard was accurate |
| Audio | **72.5 h** | est. 81.6 h — the estimate ran **13% high**, see below |
| Books | **66** | 65.2 h train / 3.6 dev / 3.6 test |
| Duration p50 / p90 / max | 4.86 s / 11.9 s / 29.3 s | |
| Ids | all unique | Kusaal is not among the 13 affected configs |

Durations sit comfortably inside CTC's practical range; no 30 s padding waste,
which is exactly why w2v-BERT 2.0 over Whisper is the right call here.

## Method note — where the byte estimate went wrong

Phase 0 had no duration column in hand, so hours were inferred from audio bytes
at `BYTES_PER_SEC = 32000` (16 kHz, 16-bit, mono). That was checked against the
dataset card and looked sound: Asante Twi 23.06 GB / 32000 = 200.2 h against a
stated 200.02 h.

It generalised badly. Summing the actual `duration` column across all 42 configs
gives **2,334.9 h**, and per language the byte estimate runs high — Kusaal 81.6 h
estimated against 72.5 h real, 13% over. The arithmetic was never wrong; the
assumption was, because WAV container overhead and any non-audio bytes in the
column are counted as if they were samples, and that overhead is proportionally
larger for corpora made of many short files. Asante Twi validated cleanly
precisely because it is the largest config, where the overhead disappears into
the rounding.

The lesson generalises past this dataset: a byte-derived estimate validated on
your **biggest** partition is validated where the error is smallest. Check the
smallest one too.

Every figure in this document dated 30 Jul comes from the `duration` column.
