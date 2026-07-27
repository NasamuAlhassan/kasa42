"""Guardrails on normalization. If these break, every WER in the project is wrong."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kasa42.data.text import charset, has_digits, is_trainable, normalize  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    FAILS.append(label) if not ok else None
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}\n        got={got!r}\n        want={want!r}"
          if not ok else f"  ok    {label}")


print("Ghanaian glyphs must survive (they are letters, not decoration)")
for g in "ɛɔŋʋɣ":
    check(f"keeps {g!r}", normalize(f"a{g}b"), f"a{g}b")
check("uppercase Ɛ -> ɛ", normalize("Ɛ"), "ɛ")
check("uppercase Ɔ -> ɔ", normalize("Ɔ"), "ɔ")
check("uppercase Ŋ -> ŋ", normalize("Ŋ"), "ŋ")
check("uppercase Ʋ -> ʋ", normalize("Ʋ"), "ʋ")

print("\nReal corpus lines")
check("genealogy line",
      normalize("Jafet biribis da anɛ: Goma nɛ Magog nɛ Madai nɛ Javan."),
      "jafet biribis da anɛ goma nɛ magog nɛ madai nɛ javan")
check("all-caps heading", normalize("YIIGA YƐLKƲDA"), "yiiga yɛlkʋda")
check("mixed case", normalize("Adam yaas yʋda."), "adam yaas yʋda")

print("\nHomoglyphs must collapse onto the canonical Ghanaian letters")
check("U+0511 ԑ -> ɛ", normalize("ԑ"), "ɛ")
check("U+0511 uppercase Ԑ -> ɛ", normalize("Ԑ"), "ɛ")
check("U+2184 ↄ -> ɔ", normalize("ↄ"), "ɔ")
check("U+2184 uppercase Ↄ -> ɔ", normalize("Ↄ"), "ɔ")
check("U+01DD ǝ -> ə", normalize("ǝ"), "ə")
check("greek epsilon -> ɛ", normalize("ε"), "ɛ")
check("cyrillic а -> a", normalize("а"), "a")
check("cyrillic о -> o", normalize("о"), "o")
check("real ɛ untouched", normalize("ɛ"), "ɛ")
check("real ɔ untouched", normalize("ɔ"), "ɔ")
check("homoglyph word collapses",
      normalize("bԑↄ") == normalize("bɛɔ"), True)

print("\nTypographic noise")
check("smart quotes folded", normalize("“Kus” nɛ ‘Seba’"), "kus nɛ seba")
check("em dash", normalize("a — b"), "a b")
check("nbsp", normalize("a b"), "a b")
check("whitespace collapsed", normalize("  a   b  "), "a b")

print("\nApostrophe / hyphen kept word-internally, dropped at edges")
check("internal apostrophe", normalize("da'an"), "da'an")
check("internal hyphen", normalize("ba-yela"), "ba-yela")
check("leading quote", normalize("'kus'"), "kus")
check("trailing hyphen", normalize("kus -"), "kus")

print("\nDigit policy: segments with digits are excluded, not half-handled")
check("has_digits true", has_digits("YIIGA 1"), True)
check("is_trainable false w/ digits", is_trainable("YIIGA YƐLKƲDA 1."), False)
check("is_trainable true", is_trainable("Adam yaas yʋda."), True)
check("is_trainable false empty", is_trainable("   "), False)
check("is_trainable false punct-only", is_trainable("!!! ..."), False)

print("\nIdempotence — normalize(normalize(x)) == normalize(x)")
for s in ["Jafet biribis da anɛ: Goma.", "YIIGA YƐLKƲDA", "  a—b  ", "da'an"]:
    check(f"idempotent {s[:18]!r}", normalize(normalize(s)), normalize(s))

print("\ncharset excludes space")
check("no space in charset", " " in charset(["a b c"]), False)
check("charset content", charset(["anɛ"]), {"a", "n", "ɛ"})

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("all passed")
