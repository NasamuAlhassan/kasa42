"""Baselines on the identical book-disjoint test split.

Run these at H+2, before the main training run finishes. If the long run
disappoints, the comparison story still exists; if it succeeds, the numbers are
already on the same axis.

Three baselines, chosen because they are the honest ones to beat:

  * `KhayaAI/w2v-bert-...` (**DONDO**) — the one that counts. Same architecture
    family we build on, 11 Northern Ghanaian languages including Kusaal, and the
    checkpoint we fine-tune from. Reporting it on our own book-disjoint split is
    the only way our number and its number sit on the same axis. Note it expects
    a one-hot language prefix, so it is evaluated *with* the language supplied —
    the favourable setting for it, and the fair one.

  * `facebook/mms-1b-all` — actually covers many of these languages via ISO
    639-3 adapters, including Kusaal (`kus`). This is the real incumbent, and
    the reason "first-ever Kusaal ASR" is a claim we do not make.
  * `openai/whisper-large-v3` — what most hackathon entries will fine-tune.
    Zero-shot it has effectively no Ghanaian-language coverage; showing that
    plainly is more useful than pretending it is a peer.

Both hypotheses go through the same data/text.normalize as our own before
scoring. Scoring one system's output under different rules than another's is
how benchmarks end up lying.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kasa42.asr.evaluate import report, save, score_by_language

# MMS uses ISO 639-3, which is exactly what the config suffix encodes.
def iso_of(config: str) -> str:
    return config.rpartition("_")[2]


def run_mms(records, device: str, batch_size: int = 8):
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    from kasa42.asr.dataset import decode_audio

    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()

    by_lang: dict[str, list] = {}
    for r in records:
        by_lang.setdefault(r["config"], []).append(r)

    out = []
    for config, rows in sorted(by_lang.items()):
        iso = iso_of(config)
        try:
            processor.tokenizer.set_target_lang(iso)
            model.load_adapter(iso)
        except Exception as e:
            print(f"  {config:24s} no MMS adapter for '{iso}' ({type(e).__name__}) — skipped")
            continue

        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            wavs = [decode_audio(r["audio"]) for r in chunk]
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            hyps = processor.batch_decode(logits.argmax(-1).cpu().numpy())
            for r, h in zip(chunk, hyps):
                out.append({"config": config, "reference": r["text"], "hypothesis": h})
        print(f"  {config:24s} {len(rows):>5,} utts via adapter '{iso}'")
    return out


DONDO = "KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en"
# ISO codes DONDO was trained on. Anything else is out of its scope, and saying
# so is more useful than reporting a meaningless number.
DONDO_LANGS = {"gjn", "maw", "gur", "dag", "dga", "kus", "lxn", "wlx", "xon", "xsm", "en"}


def run_dondo(records, device: str, batch_size: int = 8):
    from transformers import AutoModelForCTC, AutoProcessor

    from kasa42.asr.dataset import decode_audio

    processor = AutoProcessor.from_pretrained(DONDO)
    model = AutoModelForCTC.from_pretrained(DONDO).to(device).eval()

    by_lang: dict[str, list] = {}
    for r in records:
        by_lang.setdefault(r["config"], []).append(r)

    out, skipped = [], []
    for config, rows in sorted(by_lang.items()):
        if iso_of(config) not in DONDO_LANGS:
            skipped.append(config)
            continue
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            wavs = [decode_audio(r["audio"]) for r in chunk]
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            hyps = processor.batch_decode(logits.argmax(-1).cpu().numpy())
            hyps = hyps.text if hasattr(hyps, "text") else hyps
            for r, h in zip(chunk, hyps):
                out.append({"config": config, "reference": r["text"], "hypothesis": h})
        print(f"  {config:24s} {len(rows):>5,} utts")
    if skipped:
        print(f"  outside DONDO's 11 languages ({len(skipped)}): "
              f"{', '.join(iso_of(c) for c in skipped)}")
    return out


def run_whisper(records, device: str, batch_size: int = 8):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from kasa42.asr.dataset import decode_audio

    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype).to(device).eval()

    out = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        wavs = [decode_audio(r["audio"]) for r in chunk]
        inputs = processor(wavs, sampling_rate=16000, return_tensors="pt",
                           return_attention_mask=True)
        feats = inputs.input_features.to(device, dtype)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=180, language=None, task="transcribe")
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for r, h in zip(chunk, hyps):
            out.append({"config": r["config"], "reference": r["text"], "hypothesis": h})
        if i % (batch_size * 20) == 0:
            print(f"  whisper {i:>6,}/{len(records):,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-set", default="results/testset.jsonl",
                    help="Records with config/text/audio, produced from the book-disjoint split.")
    ap.add_argument("--which", nargs="*", default=["dondo", "mms", "whisper"])
    ap.add_argument("--out-dir", default="results/baselines")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = [json.loads(l) for l in Path(args.test_set).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(records):,} test utterances on {device}\n")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    runners = {"dondo": run_dondo, "mms": run_mms, "whisper": run_whisper}
    for name in args.which:
        print(f"--- {name} ---")
        preds = runners[name](records, device, args.batch_size)
        if not preds:
            print(f"  no predictions for {name}\n")
            continue
        scores = score_by_language(preds)
        print("\n" + report(scores, f"{name} — book-disjoint test") + "\n")
        save(scores, f"{args.out_dir}/{name}.json")


if __name__ == "__main__":
    main()
