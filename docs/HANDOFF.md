# Handoff — state of KASA-42 as of Fri 31 Jul 2026, ~18:30 GMT

Written for whoever picks this up next. It says what is done, what is still
open, and — most importantly — **what disappears and when**.

## The deadline that matters

The H200 container is **deleted at Sat 1 Aug 2026, 08:32 GMT**. No backup, no
grace period. Roughly 14 hours from this document's timestamp.

Everything on `/workspace` goes with it, including every intermediate artefact
listed under "Only on the box" below. Anything not on the Hub or in git by then
is gone permanently.

## What is finished and safe

All of this is published and survives the wipe.

| Artefact | Where |
|---|---|
| Trained ASR model + card | `PrinceAlhassanNasamu/kasa42-asr` |
| ONNX export (int8, verified) | same repo, `export/` |
| Kusaal ASR dataset, 30,820 clips | `PrinceAlhassanNasamu/kusaal-asr-dataset` |
| Code, 49 commits | `github.com/NasamuAlhassan/kasa42` |

### Headline results

KASA-42: w2v-BERT 2.0 + CTC + a joint language-ID head, fine-tuned from DONDO
(`KhayaAI/w2v-bert-…`) on 42 `ghana-speech` languages, 700 h mixture, 24k steps.

- **30.2% WER / 10.5% CER** micro, book-disjoint, 8,400 utterances
- **96.8%** mean language ID over 42 languages (chance 2.4%)
- Cross-corpus on an independent Kusaal set: **37.9% WER / 21.9% CER**, and
  **LID rises to 99.8%**
- Baselines on the same test set, recomputed over each system's own coverage:
  Whisper 102.3%, **MMS-1B 25.7% (beats us by 3.5pp on WER; we win CER 9.7 vs
  12.1 and cover 8 more languages)**, DONDO 75.9% — *unconditioned, see below*
- Leakage experiment: **negative result**, −0.7pp [−1.4, +0.1], interval spans
  zero

## Only on the box — will be lost at 08:32

These are **not** in git (`.gitignore` excludes them) and not on the Hub:

```
/workspace/kasa42/results/manifest.parquet      44 MB, 1.41M rows
/workspace/kasa42/results/splits.json           book -> split, all 42 configs
/workspace/kasa42/results/vocab.json            96 tokens
/workspace/kasa42/results/mixture.json          10.7 MB, the 393,160 chosen ids
/workspace/kasa42/results/testset_*.parquet     the two evaluation sets
/workspace/kasa42/results/eval_matched/         the canonical scores + records
/workspace/kasa42/results/eval_external/        cross-corpus scores
/workspace/kasa42/results/baselines/            dondo/mms/whisper scores
/workspace/kasa42/checkpoints/kasa42-asr/       6 checkpoints, final.pt uploaded
/workspace/kusaal/                              the Kaggle corpus, 6.5 GB
```

**`vocab.json` matters most.** Without it the published weights cannot be
decoded — it maps CTC indices to characters, and blank is `[PAD]` = **33**, not
0. If one thing gets rescued, make it that.

Suggested rescue, cheap and fast:

```bash
cd /workspace/kasa42
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi(); r = 'PrinceAlhassanNasamu/kasa42-asr'
for f in ('results/vocab.json', 'results/splits.json',
          'results/eval_matched/leak_report.txt', 'results/comparison.json'):
    api.upload_file(path_or_fileobj=f, path_in_repo=f.split('/')[-1],
                    repo_id=r, repo_type='model')
print('rescued')
EOF
```

Everything else is reproducible from `/data/ghana-speech` by rerunning the
pipeline, but only while the box exists.

## Open items, in priority order

1. **Record a demo video.** `cd space && python app.py --share`, speak Kusaal,
   capture 30 seconds. The only artefact that cannot be recreated after the
   window closes — the model survives, a running demo does not.
2. **Fix `kusaal-whisper-small-lora`'s card.** It links
   `alhassanprince/kusaal-asr-dataset`, a Kaggle handle, so those links 404. The
   HF dataset now exists at `PrinceAlhassanNasamu/kusaal-asr-dataset`. It also
   claims "the first open ASR model for Kusaal", which DONDO contradicts —
   KASA-42's card already concedes this and the two should not disagree.
3. **Host the KASA-42 Space.** `space/` is staged and tested locally. Creating a
   new Gradio Space returned `402 Payment Required` (PRO needed), but
   `PrinceAlhassanNasamu/kusaal-asr` is a live Gradio Space on the same account,
   so duplicating or extending that is worth trying.
4. **TTS survey** — `python -m kasa42.tts.survey --data-dir $GS`, 30 min CPU,
   safe beside anything. Reports which of the 42 languages have enough
   single-speaker audio for a voice. A dataset finding nobody has.
5. **Kusaal TTS fine-tune** — ~5 h, and the riskiest thing left. See below.

## TTS: what is done and what is not

The gate **passed, with a caveat that took measurement to find**. Kusaal is 100%
version `3752`, but ECAPA-TDNN speaker embeddings split its books cleanly into
Old and New Testament narrators. Train on the OT cluster only:

```
1CH 1KI 2CH 2KI 2SA AMO DEU EST EZK GEN HAG HOS ISA JDG JER JOB JON JOS
LAM MAL MIC NAM NEH OBA PRO RUT SNG ZEC     # 28 books, 39.7 h
```

`prepare.py --only-books <those>` is the command. Untested blockers:

- **torchcodec is required** and has never been installed. `datasets` 5.x needs
  it to encode an audiofolder. ffmpeg (its system dependency) is already
  installed.
- **`run_vits_finetuning.py` has never been executed here.** Third-party trainer
  against transformers 5.x. Smoke-test with `--epochs 1` before committing 5 h.

If the smoke test fails, drop TTS. The submission is complete without it.

## Things that will bite you

Every one of these cost hours today.

- **`id` is not unique** in 13 of 42 configs, up to 1.52×. Anything selecting
  segments by id over-samples those languages, and the hours accounting
  reconciles either way so nothing looks wrong. `build_manifest --check-ids`
  reports it. Already fixed in `mixture.py`, `train.load_split` and
  `asr/testset.py`.
- **`datasets` 5.x decodes audio through torchcodec** on every row fetch. Use
  `kasa42.asr.dataset.undecoded_audio()`, which sets `decode=False`.
- **The GPU is shared.** The other tenant moved between 42 GB and 100 GB in one
  evening. `--gradient-checkpointing` keeps training near 15 GB; training skips
  an OOM batch rather than dying.
- **Library defaults sized for text break on audio.** A fixed batch *count* in
  evaluation put every 29 s clip in the final batches; PyArrow's default row
  group made each parquet shard one 309 MB group and the Hub viewer refused it.
  Both are fixed; expect more of the same shape.
- **Piped stdout is block-buffered.** `python -m … | tee` shows nothing for
  ~2,350 steps. Use `python -u`, or `kasa42.monitor`.
- **The login shell is dash**, which does not read `~/.bashrc`. Run `bash`
  first, or `$GS` and `$HF_HOME` are empty.

## Tools built today

Beyond the pipeline: `doctor` (preflight), `monitor` (progress from checkpoint
timestamps when logs are buffered), `watchdog` (restart on death), `pipeline`
(unattended eval→export→verify→upload), `verify_onnx`, `compare` (coverage-matched
baselines), `external` (score an outside corpus), `data.pack` (Hub-ready parquet),
`tts.speaker_check` and `tts.survey`.

All have `--help`. `python -m kasa42.doctor --data-dir $GS` is the fastest way to
see whether the environment is sane.

## Standing rules for this project

- Report what was measured, including where it loses. The card says MMS beats us
  on WER and that the DONDO number is unfair to DONDO.
- Never claim a comparison without matched conditions. Two separate results were
  nearly published from confounded setups: the leak experiment (book diversity
  4.2 vs 27.2) and the baselines (34 languages vs 42).
- Per-language WER at n=200 swings by up to 7.3pp between samples. Quote it to
  the nearest few points; only the micro-average and its CI are firm.
- Credit **AI Skills and Compute Africa (AISCA)** and share models to
  `ghananlpcommunity` — the terms the GPU was granted under.
