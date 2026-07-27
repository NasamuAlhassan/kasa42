"""w2v-BERT 2.0 encoder with a CTC head and a joint language-ID head.

Why this encoder rather than Whisper:

  * ~10-30x faster to fine-tune at comparable WER, and it beats Whisper outright
    on genuinely low-resource languages — which is 30 of our 42.
  * It is a plain encoder over variable-length audio. Whisper pads everything to
    30 s; our segments have a median of 4.9 s, so roughly 80% of Whisper's
    compute would be spent on padding.

We start from **DONDO** (`KhayaAI/w2v-bert-...`, Apache-2.0), which is already
fine-tuned on 11 Northern Ghanaian languages including Kusaal, rather than from
the raw `facebook/w2v-bert-2.0`. Starting from a checkpoint that has already seen
this language family and this recording domain converges faster and helps the
tail languages — which matters when the whole budget is 48 hours.

Two things differ from DONDO by design:

  * **Language ID is learned, not supplied.** DONDO conditions on a one-hot
    language prefix, so the caller must already know which language is being
    spoken. Our LID head is one mean-pool and one linear layer over features the
    encoder already computes, and it lets the demo accept any of the 42
    languages with nothing selected.
  * **The vocabulary is extended, not replaced.** DONDO's 49 tokens cover 11
    languages; 42 need more. Shared characters keep their original CTC head rows
    so that learned behaviour survives; only genuinely new characters get fresh
    rows. See `extend_ctc_head`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel

DONDO = "KhayaAI/w2v-bert-gjn_maw_gur_dag_dga_kus_lxn_wlx_xon_xsm_en"


class Kasa42ForCTC(nn.Module):
    def __init__(
        self,
        encoder_name: str = DONDO,
        vocab_size: int = 128,
        n_languages: int = 42,
        lid_weight: float = 0.1,
        dropout: float = 0.1,
        blank_id: int = 0,
        freeze_encoder_steps: int = 0,
    ) -> None:
        super().__init__()
        # DONDO is a Wav2Vec2BertForCTC; loading it as the base model keeps the
        # encoder (including its adapter layer) and discards only the old head.
        self.encoder = Wav2Vec2BertModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(hidden, vocab_size)
        self.lid_head = nn.Linear(hidden, n_languages)

        self.lid_weight = lid_weight
        self.blank_id = blank_id
        self.freeze_encoder_steps = freeze_encoder_steps
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.lid_loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def extend_ctc_head(self, old_vocab: dict[str, int], new_vocab: dict[str, int],
                        source: str = DONDO) -> int:
        """Copy DONDO's CTC head rows for characters both vocabularies share.

        Returns the number of rows transferred. Characters new to KASA-42 keep
        their freshly initialised rows, which is what we want: the model has
        never seen them and should not pretend otherwise.
        """
        from transformers import Wav2Vec2BertForCTC

        donor = Wav2Vec2BertForCTC.from_pretrained(source)
        w, b = donor.lm_head.weight, donor.lm_head.bias

        moved = 0
        for token, old_idx in old_vocab.items():
            new_idx = new_vocab.get(token)
            if new_idx is None or old_idx >= w.size(0) or new_idx >= self.ctc_head.weight.size(0):
                continue
            self.ctc_head.weight[new_idx].copy_(w[old_idx])
            self.ctc_head.bias[new_idx].copy_(b[old_idx])
            moved += 1
        del donor
        return moved

    def set_encoder_frozen(self, frozen: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not frozen

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_lengths: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_features=input_features, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)          # (B, T, H)

        logits = self.ctc_head(h)                        # (B, T, V)

        if attention_mask is not None:
            # The encoder subsamples; recover per-example output lengths from the
            # mask it actually produced rather than assuming a fixed ratio.
            enc_lens = self.encoder._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(logits.device)
            mask = torch.arange(h.size(1), device=h.device)[None, :] < enc_lens[:, None]
        else:
            enc_lens = torch.full((h.size(0),), h.size(1), device=h.device, dtype=torch.long)
            mask = torch.ones(h.size(0), h.size(1), dtype=torch.bool, device=h.device)

        # Mean-pool over real frames only, so padding cannot skew language ID.
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        lid_logits = self.lid_head(pooled)               # (B, L)

        result = {"logits": logits, "lid_logits": lid_logits, "input_lengths": enc_lens}
        if labels is None:
            return result

        # CTC wants (T, B, V) log-probs, and float32 for numerical stability even
        # when the rest of the step runs in bf16.
        log_probs = logits.transpose(0, 1).log_softmax(-1).float()
        ctc = self.ctc_loss(log_probs, labels, enc_lens, label_lengths)

        loss = ctc
        result["ctc_loss"] = ctc.detach()
        if language_ids is not None:
            lid = self.lid_loss(lid_logits, language_ids)
            loss = loss + self.lid_weight * lid
            result["lid_loss"] = lid.detach()
        result["loss"] = loss
        return result
