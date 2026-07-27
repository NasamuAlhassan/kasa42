"""Text normalization shared by training, evaluation and the demo.

Every WER number depends on this file, so it has to be applied identically to
references, our hypotheses, and every baseline's hypotheses. Normalizing only
one side is the classic way to publish a wrong number.

Design constraints from the Phase 0 audit (see docs/FINDINGS.md):

  * Ghanaian orthographies use `ɛ ɔ ŋ ʋ ɣ` and friends. These are letters, not
    decoration — they must survive normalization, which rules out any ASCII fold.
  * Source text carries smart quotes `‘ ’ “ ”` that are typographic noise.
  * Digits appear (verse numbers leaking into transcripts). Spelling them out
    across 42 orthographies is not feasible in the time available, so segments
    containing digits are dropped from training rather than half-handled.
"""

from __future__ import annotations

import re
import unicodedata

# Typographic variants -> plain equivalents. Applied before punctuation stripping.
_FOLD = {
    "‘": "'", "’": "'", "‛": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-",
    " ": " ", "​": "", "‌": "", "‍": "",
    "ʼ": "'",
}

# Homoglyphs: characters that *look* like Ghanaian letters but carry different
# codepoints. They appear in real corpora because typists reach for whatever
# their keyboard offers.
#
# This is not cosmetic. DONDO's published vocabulary contains both U+0511 and
# U+025B, and both U+2184 and U+0254 — the same letter competing for probability
# mass across two output units in a CTC head that only has 49 of them. Folding
# costs nothing and removes the split.
_HOMOGLYPH = {
    "ԑ": "ɛ",   # U+0511 CYRILLIC ABKHASIAN CHE   -> U+025B OPEN E
    "ↄ": "ɔ",   # U+2184 REVERSED C               -> U+0254 OPEN O
    "ǝ": "ə",   # U+01DD TURNED E                 -> U+0259 SCHWA
    "ε": "ɛ",   # U+03B5 GREEK EPSILON
    "ϵ": "ɛ",   # U+03F5 GREEK LUNATE EPSILON
    "ο": "o",   # U+03BF GREEK OMICRON
    "а": "a",   # U+0430 CYRILLIC A
    "е": "e",   # U+0435 CYRILLIC IE
    "о": "o",   # U+043E CYRILLIC O
    "р": "p",   # U+0440 CYRILLIC ER
    "с": "c",   # U+0441 CYRILLIC ES
    "у": "y",   # U+0443 CYRILLIC U
    "х": "x",   # U+0445 CYRILLIC HA
}

# Stripped entirely. Apostrophe and hyphen are kept: in several Ghanaian
# orthographies they are word-internal and carry meaning.
_PUNCT = re.compile(r"[!\"(),.:;?\[\]{}«»…·•*/\\|_=+<>@#$%^&~`]")
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def has_digits(text: str) -> bool:
    return bool(_DIGIT.search(text or ""))


def normalize(text: str, *, keep_apostrophe: bool = True) -> str:
    """Canonical form for both references and hypotheses."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    for a, b in _FOLD.items():
        t = t.replace(a, b)
    # Lowercase first: homoglyph folding is defined on lowercase forms, and
    # str.lower() already maps Ɛ->ɛ, Ɔ->ɔ, Ԑ->ԑ, Ↄ->ↄ, Ǝ->ǝ.
    t = t.lower()
    for a, b in _HOMOGLYPH.items():
        t = t.replace(a, b)
    t = _PUNCT.sub(" ", t)
    if not keep_apostrophe:
        t = t.replace("'", " ")
    # Hyphens and apostrophes are only meaningful between letters; elsewhere
    # they are dashes or quote residue.
    t = re.sub(r"(?<!\w)[-']|[-'](?!\w)", " ", t)
    return _WS.sub(" ", t).strip()


def is_trainable(text: str, *, min_chars: int = 2) -> bool:
    """Whether a segment should enter the training mixture."""
    if has_digits(text):
        return False
    n = normalize(text)
    return len(n) >= min_chars and any(c.isalpha() for c in n)


def charset(texts) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(normalize(t))
    out.discard(" ")
    return out
