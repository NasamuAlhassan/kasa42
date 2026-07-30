"""Dataset and collator for the 42-language CTC mixture.

Audio lives in the parquet `audio` column as encoded bytes. We decode with
soundfile, run the SeamlessM4T feature extractor (log-mel filterbanks — the
front end w2v-BERT 2.0 expects), and emit variable-length features.

Bucketing matters more than it looks. Segment durations run from under a second
to 29 s; batching randomly means padding every batch to its longest member and
burning compute on zeros. Sorting into length buckets and batching within them
keeps padding low, which is a large part of why the run fits in hours.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
import torch

from kasa42.data.text import normalize

TARGET_SR = 16000


def undecoded_audio(ds):
    """Stop `datasets` decoding the audio column for us.

    From datasets 5.x the `Audio` feature decodes through **torchcodec**, and it
    fires on every row fetch — so a DataLoader worker dies with "To support
    decoding audio data, please install 'torchcodec'" the moment training starts,
    even though nothing here wants a decoded array.

    We decode with soundfile in `decode_audio` instead: the corpus is 16 kHz mono
    PCM-16 WAV, which soundfile reads with no ffmpeg and no extra dependency, and
    letting datasets decode first would only mean doing the work twice. Setting
    `decode=False` leaves the column as `{bytes, path}`, which `decode_audio`
    already handles.

    Metadata-only, so it costs nothing and is safe to call on any split.
    """
    import datasets

    if "audio" not in ds.column_names:
        return ds
    return ds.cast_column("audio", datasets.Audio(decode=False))


def decode_audio(blob) -> np.ndarray:
    """Decode the parquet audio cell to mono float32 at 16 kHz."""
    import soundfile as sf

    if isinstance(blob, dict):
        if blob.get("array") is not None:
            return np.asarray(blob["array"], dtype=np.float32)
        blob = blob.get("bytes")
    if blob is None:
        raise ValueError("empty audio cell")

    wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


class CharTokenizer:
    """Character tokenizer over the shared 42-language vocab.

    Indices come from DONDO, so nothing here may assume blank==0: in that
    vocabulary `|` is 0 and `[PAD]` (the CTC blank) is 33.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.inv = {v: k for k, v in vocab.items()}
        self.blank = vocab.get("[PAD]", vocab.get("<pad>", 0))
        self.unk = vocab.get("[UNK]", vocab.get("<unk>", 1))
        self.space = vocab["|"]

    @classmethod
    def from_json(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["vocab"])

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.space if c == " " else self.vocab.get(c, self.unk)
                for c in normalize(text)]

    def decode(self, ids) -> str:
        """Greedy CTC decode: collapse repeats, then drop blanks."""
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i != prev and i != self.blank:
                out.append(" " if i == self.space else self.inv.get(i, ""))
            prev = i
        return "".join(out).strip()


@dataclass
class Collator:
    """Pads a batch of variable-length features and label sequences."""

    feature_extractor: object
    tokenizer: CharTokenizer
    lang_to_id: dict[str, int]

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        wavs = [decode_audio(b["audio"]) for b in batch]
        feats = self.feature_extractor(
            wavs, sampling_rate=TARGET_SR, return_tensors="pt", padding=True,
            return_attention_mask=True,
        )

        label_ids = [self.tokenizer.encode(b["text"]) for b in batch]
        lengths = torch.tensor([len(x) for x in label_ids], dtype=torch.long)
        padded = torch.zeros(len(batch), int(lengths.max().clamp(min=1)), dtype=torch.long)
        for i, ids in enumerate(label_ids):
            if ids:
                padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {
            "input_features": feats["input_features"],
            "attention_mask": feats["attention_mask"],
            "labels": padded,
            "label_lengths": lengths,
            "language_ids": torch.tensor(
                [self.lang_to_id[b["config"]] for b in batch], dtype=torch.long
            ),
        }


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batch similar-length examples together to minimise padding.

    Buckets are shuffled each epoch so the model still sees a random order of
    batches; only the composition of each batch is length-correlated.
    """

    def __init__(self, durations, batch_duration: float = 320.0,
                 shuffle: bool = True, seed: int = 0) -> None:
        self.durations = list(durations)
        self.batch_duration = batch_duration
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _build(self) -> list[list[int]]:
        order = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches, cur, longest = [], [], 0.0
        for i in order:
            longest = max(longest, self.durations[i])
            # Cost is padded cost: longest member * batch size.
            if cur and longest * (len(cur) + 1) > self.batch_duration:
                batches.append(cur)
                cur, longest = [i], self.durations[i]
            else:
                cur.append(i)
        if cur:
            batches.append(cur)
        if self.shuffle:
            import random

            random.Random(self.seed + self.epoch).shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return iter(self._build())

    def __len__(self) -> int:
        return len(self._build())
