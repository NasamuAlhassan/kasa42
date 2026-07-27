"""Score TTS without a listening panel: ASR round-trip CER.

Synthesize held-out text, transcribe the result with the KASA-42 ASR, and measure
CER against the input. A voice that is clearer and better matched to the language
is transcribed more accurately, so lower round-trip CER means better TTS.

It is a proxy, not a MOS score, and it has a real failure mode worth naming: a
TTS that produces bland, over-smoothed speech can transcribe *well* while sounding
worse to a person. So this runs alongside a native-speaker A/B, not instead of it
— which is the one evaluation this project can do properly, because the builder
speaks Kusaal.

Both systems are scored with the identical ASR and the identical normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kasa42.asr.evaluate import error_rate

BASELINE = "facebook/mms-tts-kus"


def synthesize(model, tokenizer, text: str, device: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs).waveform
    return out[0].float().cpu().numpy()


def transcribe_all(wavs: list[np.ndarray], sr: int, vocab: str, checkpoint: str,
                   device: str) -> list[str]:
    from transformers import SeamlessM4TFeatureExtractor

    from kasa42.asr.dataset import CharTokenizer
    from kasa42.asr.model import Kasa42ForCTC

    cfg = json.loads(Path(checkpoint).with_name("config.json").read_text(encoding="utf-8"))
    tok = CharTokenizer.from_json(vocab)
    model = Kasa42ForCTC(cfg["encoder"], vocab_size=cfg["vocab_size"],
                         n_languages=len(cfg["languages"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval()
    fe = SeamlessM4TFeatureExtractor.from_pretrained(cfg["encoder"])

    hyps = []
    for i in range(0, len(wavs), 8):
        chunk = wavs[i:i + 8]
        feats = fe(chunk, sampling_rate=sr, return_tensors="pt", padding=True,
                   return_attention_mask=True)
        with torch.no_grad():
            out = model(input_features=feats["input_features"].to(device),
                        attention_mask=feats["attention_mask"].to(device))
        for row in out["logits"].argmax(-1).cpu().numpy():
            hyps.append(tok.decode(row))
    return hyps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetuned", default="checkpoints/kasa42-kusaal-tts")
    ap.add_argument("--asr-checkpoint", default="checkpoints/kasa42-asr/final.pt")
    ap.add_argument("--vocab", default="results/vocab.json")
    ap.add_argument("--metadata", default="data/tts/metadata.json")
    ap.add_argument("--n", type=int, default=100, help="Held-out sentences to score.")
    ap.add_argument("--out", default="results/tts_roundtrip.json")
    ap.add_argument("--save-audio", default="results/tts_samples")
    args = ap.parse_args()

    from transformers import AutoTokenizer, VitsModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    # Take from the tail so these sentences are unlikely to be in the training slice.
    texts = [it["text"] for it in meta["items"][-args.n:]]
    print(f"{len(texts)} held-out sentences on {device}\n")

    results = {}
    audio_dir = Path(args.save_audio)
    audio_dir.mkdir(parents=True, exist_ok=True)

    for name, path in (("mms_baseline", BASELINE), ("kasa42", args.finetuned)):
        if not (path == BASELINE or Path(path).exists()):
            print(f"{name}: {path} missing — skipped")
            continue
        model = VitsModel.from_pretrained(path).to(device).eval()
        tok = AutoTokenizer.from_pretrained(path)
        sr = model.config.sampling_rate

        wavs = [synthesize(model, tok, t, device) for t in texts]
        hyps = transcribe_all(wavs, sr, args.vocab, args.asr_checkpoint, device)
        cer, errs, total = error_rate(texts, hyps, unit="char")
        wer, _, _ = error_rate(texts, hyps, unit="word")
        results[name] = {"cer": round(cer, 4), "wer": round(wer, 4),
                         "chars": total, "errors": errs, "n": len(texts)}
        print(f"{name:14s} round-trip CER {cer:.1%}  WER {wer:.1%}")

        import soundfile as sf
        for i in range(min(8, len(wavs))):
            sf.write(audio_dir / f"{name}_{i:02d}.wav", wavs[i], sr)

    if len(results) == 2:
        b, k = results["mms_baseline"]["cer"], results["kasa42"]["cer"]
        delta = b - k
        print(f"\nbaseline {b:.1%} -> ours {k:.1%}  ({delta:+.1%} absolute)")
        print("BETTER than MMS" if delta > 0 else "NOT better than MMS — do not claim it")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"A/B clips in {audio_dir} — listen before claiming anything.")


if __name__ == "__main__":
    main()
