"""
Learn a native-script -> Latin word dictionary from the training ground truth.

Source 2/3 records in India sometimes carry the business name written in an Indic
script (e.g. "लक्ष्मी एनर्जी प्राइवेट लिमिटेड" for "Lakshmi Energy Private Limited").
Generic transliteration (anyascii) drops inherent vowels ("lksmi enrji"), so we align
the native tokens of matched training pairs with the Latin tokens of their Source 1
record and keep the most frequent mapping per native word. Unknown words at inference
time fall back to anyascii (see normalize.translit_text).
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple

from anyascii import anyascii
from rapidfuzz import fuzz

try:
    from .normalize import has_non_latin
except ImportError:
    from normalize import has_non_latin

_VOWELS = re.compile(r"[aeiou]")


def _skeleton(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.lower())
    return word[:1] + _VOWELS.sub("", word[1:])


def _latin_tokens(name: str):
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", name.lower())


def _native_tokens(name: str):
    return [t.strip(".,;:()[]{}\"'") for t in name.split() if t.strip(".,;:()[]{}\"'")]


def learn_translit_dict(pairs: Iterable[Tuple[str, str]], min_count: int = 2,
                        min_share: float = 0.5) -> Dict[str, str]:
    """pairs: iterable of (source1_latin_name, native_script_name)."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for latin, native in pairs:
        lt = _latin_tokens(latin)
        nt = _native_tokens(native)
        if not lt or not nt:
            continue
        if len(lt) == len(nt):
            for a, b in zip(nt, lt):
                if has_non_latin(a):
                    counts[a][b] += 1
            continue
        sk = [_skeleton(t) for t in lt]
        for a in nt:
            if not has_non_latin(a):
                continue
            guess = _skeleton(anyascii(a))
            best, best_s = None, 0.0
            for t, s in zip(lt, sk):
                sc = fuzz.ratio(guess, s)
                if sc > best_s:
                    best, best_s = t, sc
            if best is not None and best_s >= 60:
                counts[a][best] += 1
    out = {}
    for native, c in counts.items():
        word, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            out[native] = word
    return out


def save_translit_dict(mapping: Dict[str, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
