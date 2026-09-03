"""Frozen Telugu text frontend — the ONE romanizer used at train AND serve.

Never reimplement this on one side. A one-token drift between training and
inference silently destroys quality. Run #1 uses scheme="iso".

  iso    ISO-15919, diacritics kept (halō, mīru viṁṭunnāru) — phonemically faithful.
  ascii  ISO-15919 diacritics stripped — ablation only (bakes in English accent).
  native passthrough (byte-soup baseline) — ablation only.

Only Telugu Unicode runs are transliterated; English/code-switch spans, digits
and punctuation pass through untouched.
"""
from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript

_TELUGU_RUN = re.compile(r"[ఀ-౿]+")


def _iso_run(m: "re.Match") -> str:
    return sanscript.transliterate(m.group(0), sanscript.TELUGU, sanscript.ISO)


def romanize(text: str, scheme: str = "iso") -> str:
    if scheme == "native":
        return text
    iso = _TELUGU_RUN.sub(_iso_run, text)
    if scheme == "iso":
        return iso
    if scheme == "ascii":
        d = unicodedata.normalize("NFKD", iso)
        return "".join(c for c in d if not unicodedata.combining(c))
    raise ValueError(f"unknown scheme: {scheme!r}")


if __name__ == "__main__":
    for s in ["హలో my dear Telugu people, మీరు వింటున్నారు.", "పవనితో కబుర్లు."]:
        print("NATIVE:", s)
        print("  ISO :", romanize(s, "iso"))
