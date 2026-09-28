"""
Record normalisation: undo the noise operations observed in the ground truth.

Every name and address is reduced to a handful of canonical string fields once,
before blocking, so that the pairwise stage only compares already-clean strings.

Name noise undone here (catalogued from matched training pairs):
  * non-Latin scripts (Devanagari, Tamil, Telugu, ...) -> learned word dictionary,
    anyascii fallback
  * junk prefixes  "--", ">>", "...", "***", "#", "@", "<<", "(", "["
  * alias phrases  "<random> dba|d/b/a|t/a|fka|f/k/a|aka|a/k/a|formerly (known as)|
                    doing business as|nee <true name>"
  * "(ID: 12345)" tags, [brackets], (parentheses), hyphen / double-space noise
  * OCR swaps inside words: 0->o 1->l 5->s 8->b, leading l->i ("lnc", "lndia")
  * legal suffix variants (Pvt/Private, Ltd/Limited, L.L.C., S.A.S., ...)
  * honorifics (Shri, Sri, Smt, Mr, Dr, M/s) and generator filler words
    (Center, Services, Partners, ...)
  * domain forms (www.foo-bar.com -> foobar)
  * repeated words ("Mountain Mountain Best")

Address noise undone here:
  * native-script and abbreviated states  (TN <-> Tamil Nadu <-> தமிழ்நாடு,
    NC <-> North Carolina, Gironde -> Nouvelle-Aquitaine)
  * street types / directionals / ordinals (St/Street, Rd/Road, R./Rue, Bd, ...)
  * house-number decorations (#, ##, H.No, Door No, leading zeros, trailing '.')
  * filler components (null, <NULL>, N/A, PO Box, unit / floor designators)
  * city aliases (Bombay/Mumbai, Poona/Pune, St-Nazaire/Saint Nazaire, ...)
"""
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from anyascii import anyascii

# ─────────────────────────────────────────────────────────────────────────────
# Character helpers
# ─────────────────────────────────────────────────────────────────────────────

_NON_LATIN_RE = re.compile(r"[^\x00-ɏḀ-ỿ -⁯₠-⃏°’‘“”«»]")


def has_non_latin(text: str) -> bool:
    return bool(_NON_LATIN_RE.search(text))


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


# ─────────────────────────────────────────────────────────────────────────────
# Transliteration (learned dictionary + anyascii fallback)
# ─────────────────────────────────────────────────────────────────────────────

_TRANSLIT: Dict[str, str] = {}


def set_translit_dict(mapping: Dict[str, str]) -> None:
    _TRANSLIT.clear()
    _TRANSLIT.update(mapping)


def load_translit_dict(path: str) -> None:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            set_translit_dict(json.load(f))


def _strip_token_punct(tok: str) -> str:
    return tok.strip(".,;:()[]{}\"'")


def translit_text(text: str) -> str:
    """Word-by-word: learned native->Latin dictionary, anyascii for unknown words."""
    out = []
    for tok in text.split():
        if not has_non_latin(tok):
            out.append(tok)
            continue
        key = _strip_token_punct(tok)
        mapped = _TRANSLIT.get(key)
        if mapped is None:
            mapped = anyascii(key).lower()
        out.append(mapped)
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Name normalisation
# ─────────────────────────────────────────────────────────────────────────────

_ALIAS_RE = re.compile(
    r"^(.*?\S)\s+(?:doing business as|formerly known as|formerly:?|trading as|"
    r"d/b/a|f/k/a|a/k/a|dba:?|t/a|fka|aka|n[ée]e)\s+(.+)$",
    re.IGNORECASE,
)
_HONORIFIC_PREFIX_RE = re.compile(r"^(?:mr|mrs|ms|dr|shri|sri|shree|smt|m/s)\.?\s+(?=\S+\.(?:com|c0m|in|net|org|co|fr)\b)")
_ID_TAG_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|-?\s*\[\d{6,}\]|#\d{4,}", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\d?\.)?([a-z0-9][a-z0-9\-\.]*?)\."
    r"(com|c0m|co\.in|net|org|in|co|fr|biz|info|io|us|online|site|shop)$"
)

# canonical legal form tokens
LEGAL_CANON = {
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd", "limited": "ltd", "ltd": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "ltda": "ltd", "gmbh": "gmbh", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "sca": "sca", "gie": "gie", "ei": "ei", "scop": "scop",
    "selarl": "selarl", "scm": "scm",
}
# dotted forms collapse before tokenisation
_DOTTED_LEGAL = [
    (re.compile(r"\bl\.\s?l\.\s?c\.?"), " llc "), (re.compile(r"\bl\.\s?l\.\s?p\.?"), " llp "),
    (re.compile(r"\bp\.\s?l\.\s?l\.\s?c\.?"), " pllc "), (re.compile(r"\bp\.\s?c\.(?=\s|$)"), " pc "),
    (re.compile(r"\bl\.\s?p\.(?=\s|$)"), " lp "), (re.compile(r"\bs\.\s?a\.\s?s\.\s?u\.?"), " sasu "),
    (re.compile(r"\bs\.\s?a\.\s?s\.?"), " sas "), (re.compile(r"\be\.\s?u\.\s?r\.\s?l\.?"), " eurl "),
    (re.compile(r"\bs\.\s?a\.\s?r\.\s?l\.?"), " sarl "), (re.compile(r"\bs\.\s?c\.\s?i\.?"), " sci "),
    (re.compile(r"\bs\.\s?a\.(?=\s|$)"), " sa "), (re.compile(r"\bm/s\b"), " "),
    (re.compile(r"\bpvt\.\s?ltd\.?"), " pvt ltd "),
]
HONORIFICS = {"shri", "sri", "shree", "smt", "mr", "mrs", "ms", "dr", "messrs", "mssrs", "sh", "kumari", "km"}
# words the noise generator inserts far more often than it removes
FILLER_WORDS = {
    "the", "and", "of", "center", "centre", "services", "service", "partners",
    "labs", "sys", "one", "india", "france",
}
STOP_NAME = {"a", "an", "et", "de", "du", "des", "la", "le", "les", "l", "d", "&"}
LEGAL_TOKENS = set(LEGAL_CANON.values()) | {"pvt", "ltd"}
_LEGAL_SUFFIX_STR = sorted(
    ["pvtltd", "privatelimited", "private", "limited", "ltd", "pvt", "llc", "llp", "inc",
     "incorporated", "corp", "corporation", "co", "company", "sarl", "sas", "sasu",
     "eurl", "sci", "sa", "pllc", "lp", "pc"],
    key=len, reverse=True,
)

_OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "3": "e", "4": "a", "@": "a", "$": "s"})
_ORDINAL_TOKEN = re.compile(r"^\d+(st|nd|rd|th|er|e|eme|ere)$")
_LEAD_L_CONSONANT = re.compile(r"^l[bcdfghjkmnpqrstvwxz]")


def _fix_ocr_token(tok: str) -> str:
    if tok.isdigit() or _ORDINAL_TOKEN.match(tok) or tok in LEGAL_CANON:
        return tok
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        tok = tok.translate(_OCR_MAP)
    if _LEAD_L_CONSONANT.match(tok):
        tok = "i" + tok[1:]
    return tok


def _dedupe_consecutive(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def _strip_legal_suffix(stem: str) -> str:
    changed = True
    while changed and stem:
        changed = False
        for suf in _LEGAL_SUFFIX_STR:
            if stem.endswith(suf) and len(stem) - len(suf) >= 3:
                stem = stem[: -len(suf)]
                changed = True
                break
    return stem


def _tokenise_name(text: str) -> List[str]:
    text = text.replace("&", " and ").replace("+", " and ")
    for rx, rep in _DOTTED_LEGAL:
        text = rx.sub(rep, text)
    text = re.sub(r"['’`]", "", text)          # orelee's -> orelees
    text = re.sub(r"[^a-z0-9@$]+", " ", text)  # everything else is a separator
    toks = [_fix_ocr_token(t) for t in text.split()]
    toks = [t for t in toks if t]
    return toks


def normalize_name(raw: Optional[str]) -> Tuple[str, str, str, str, str, int, int, int]:
    """
    Returns (full, core, alias, legal, compact, is_domain, is_native, has_alias)

    full    – all tokens, legal forms canonicalised, noise prefixes/tags removed
    core    – full minus legal forms, honorifics, filler and stop words
    alias   – the random alias preceding a dba/fka marker (core-normalised)
    legal   – canonical legal tokens, sorted, space separated
    compact – core with spaces removed (for domain / concatenation comparisons)
    """
    if not raw:
        return "", "", "", "", "", 0, 0, 0
    text = raw
    is_native = 0
    if has_non_latin(text):
        text = translit_text(text)
        is_native = 1
    text = strip_accents(text).lower().strip()
    text = _ID_TAG_RE.sub(" ", text)
    text = re.sub(r"^[^a-z0-9]+", "", text)  # junk prefixes
    text = _HONORIFIC_PREFIX_RE.sub("", text)

    alias = ""
    has_alias = 0
    m = _ALIAS_RE.match(text)
    if m:
        alias, text = m.group(1), m.group(2)
        has_alias = 1

    is_domain = 0
    stripped = text.strip()
    if " " not in stripped:
        dm = _DOMAIN_RE.match(stripped)
        if dm:
            is_domain = 1
            stem = _fix_ocr_token(re.sub(r"[^a-z0-9]", "", dm.group(1)))
            core_stem = _strip_legal_suffix(stem)
            return stem, core_stem, "", "", core_stem, 1, is_native, 0

    toks = _dedupe_consecutive(_tokenise_name(text))
    full_toks, core_toks, legal_toks = [], [], []
    for t in toks:
        canon = LEGAL_CANON.get(t)
        if canon is not None:
            full_toks.append(canon)
            legal_toks.extend(canon.split())
            continue
        full_toks.append(t)
        if t in HONORIFICS or t in FILLER_WORDS or t in STOP_NAME:
            continue
        core_toks.append(t)
    if not core_toks:  # never leave the core empty if there were words
        core_toks = [t for t in full_toks if t not in LEGAL_TOKENS] or full_toks[:]
    alias_core = ""
    if alias:
        alias_core = " ".join(
            t for t in _tokenise_name(alias)
            if t not in LEGAL_CANON and t not in HONORIFICS and t not in FILLER_WORDS
        )
    full = " ".join(full_toks)
    core = " ".join(core_toks)
    legal = " ".join(sorted(set(legal_toks)))
    return full, core, alias_core, legal, core.replace(" ", ""), is_domain, is_native, has_alias


# ─────────────────────────────────────────────────────────────────────────────
# Address normalisation
# ─────────────────────────────────────────────────────────────────────────────

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "pr": "puerto rico", "gu": "guam", "vi": "virgin islands",
}
IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka",
    "tn": "tamil nadu", "wb": "west bengal", "gj": "gujarat", "tg": "telangana",
    "ts": "telangana", "hr": "haryana", "rj": "rajasthan", "kl": "kerala", "br": "bihar",
    "mp": "madhya pradesh", "ap": "andhra pradesh", "pb": "punjab", "od": "odisha",
    "or": "odisha", "as": "assam", "jh": "jharkhand", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "uk": "uttarakhand", "ut": "uttarakhand",
    "hp": "himachal pradesh", "jk": "jammu and kashmir", "ch": "chandigarh",
    "py": "puducherry", "tr": "tripura", "mn": "manipur", "ml": "meghalaya",
    "nl": "nagaland", "mz": "mizoram", "ar": "arunachal pradesh", "sk": "sikkim",
}
IN_STATE_VARIANTS = {
    "orissa": "odisha", "keralam": "kerala", "tamilnadu": "tamil nadu",
    "new delhi": None, "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
    "chattisgarh": "chhattisgarh", "nct of delhi": "delhi",
}
NATIVE_STATES = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh",
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "ગુજરાત": "gujarat",
    "পশ্চিমবঙ্গ": "west bengal", "తెలంగాణ": "telangana", "हरियाणा": "haryana",
    "राजस्थान": "rajasthan", "കേരളം": "kerala", "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab",
    "ଓଡ଼ିଶା": "odisha", "অসম": "assam", "झारखंड": "jharkhand", "छत्तीसगढ़": "chhattisgarh",
    "गोवा": "goa", "उत्तराखंड": "uttarakhand", "हिमाचल प्रदेश": "himachal pradesh",
}
# France: department -> region (canonical = region)
FR_REGIONS = {
    "hauts de france": "hauts de france", "nouvelle aquitaine": "nouvelle aquitaine",
    "pays de la loire": "pays de la loire", "ile de france": "ile de france",
    "bretagne": "bretagne", "normandie": "normandie", "grand est": "grand est",
    "occitanie": "occitanie", "auvergne rhone alpes": "auvergne rhone alpes",
    "provence alpes cote d azur": "provence alpes cote d azur", "corse": "corse",
    "centre val de loire": "centre val de loire", "bourgogne franche comte": "bourgogne franche comte",
    "nord": "hauts de france", "pas de calais": "hauts de france", "somme": "hauts de france",
    "aisne": "hauts de france", "oise": "hauts de france",
    "gironde": "nouvelle aquitaine", "landes": "nouvelle aquitaine",
    "dordogne": "nouvelle aquitaine", "lot et garonne": "nouvelle aquitaine",
    "pyrenees atlantiques": "nouvelle aquitaine", "charente maritime": "nouvelle aquitaine",
    "loire atlantique": "pays de la loire", "vendee": "pays de la loire",
    "maine et loire": "pays de la loire", "sarthe": "pays de la loire", "mayenne": "pays de la loire",
}


def _build_state_maps():
    maps = {}
    us = {}
    for code, name in US_STATES.items():
        us[code] = code
        us[name] = code
    maps["us"] = us
    ind = {}
    for code, name in IN_STATES.items():
        ind[code] = name
        ind[name] = name
    for var, name in IN_STATE_VARIANTS.items():
        if name:
            ind[var] = name
    maps["india"] = ind
    maps["france"] = dict(FR_REGIONS)
    return maps


STATE_MAPS = _build_state_maps()

STREET_TYPES_COMMON = {
    "street": "st", "st": "st", "str": "st", "road": "rd", "rd": "rd", "avenue": "ave",
    "ave": "ave", "av": "ave", "aveue": "ave", "avn": "ave", "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "boul": "blvd", "court": "ct", "ct": "ct", "circle": "cir", "cir": "cir", "place": "pl",
    "pl": "pl", "terrace": "ter", "terr": "ter", "parkway": "pkwy", "pkwy": "pkwy",
    "pky": "pkwy", "highway": "hwy", "hwy": "hwy", "way": "way", "wy": "way", "trail": "trl",
    "trl": "trl", "square": "sq", "sq": "sq", "plaza": "plz", "plz": "plz", "crossing": "xing",
    "xing": "xing", "point": "pt", "pt": "pt", "pike": "pike", "expressway": "expy",
    "expy": "expy", "freeway": "fwy", "fwy": "fwy", "turnpike": "tpke", "tpke": "tpke",
    "alley": "aly", "aly": "aly", "heights": "hts", "hts": "hts", "mount": "mt", "mt": "mt",
    "fort": "ft", "ft": "ft", "saint": "st", "route": "rte", "rte": "rte", "rt": "rte",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "ist": "1st", "marg": "marg", "mg": "marg",
}
STREET_TYPES_FR = {
    "rue": "rue", "r": "rue", "avenue": "ave", "av": "ave", "ave": "ave", "aveue": "ave",
    "boulevard": "blvd", "bd": "blvd", "blvd": "blvd", "bld": "blvd", "allee": "allee",
    "allees": "allee", "all": "allee", "impasse": "imp", "imp": "imp", "route": "rte",
    "rte": "rte", "place": "pl", "pl": "pl", "chemin": "ch", "ch": "ch", "chem": "ch",
    "quai": "quai", "q": "quai", "cours": "crs", "crs": "crs", "passage": "pass",
    "pass": "pass", "psg": "pass", "square": "sq", "sq": "sq", "cite": "cite",
    "residence": "res", "res": "res", "chaussee": "chs", "lotissement": "lot",
    "saint": "st", "st": "st", "sainte": "ste", "ste": "ste", "promenade": "prom",
    "esplanade": "esp", "sentier": "sent", "cour": "cour",
}
ADDR_DROP = {
    # fillers / unit designators / numbering words
    "null", "none", "n/a", "na", "nil", "unit", "suite", "ste", "apt", "apartment", "fl", "floor",
    "flr", "room", "rm", "bldg", "building", "no", "nos", "number", "hno", "h", "door", "d",
    "plot", "shop", "flat", "s", "kh", "khasra", "survey", "dno", "house", "po", "box",
    "etage", "eme", "er", "appartement", "appt", "bis", "ter", "quater", "bat", "batiment",
    "of", "the", "and", "de", "du", "des", "la", "le", "les", "l", "au", "aux", "et", "en",
    "near", "nr", "opp", "opposite", "behind", "beside", "besides", "next", "to",
    "ground", "gf", "g", "first", "floor", "b", "a", "c", "e",
}
ADDR_DROP_FR_KEEP = {"e"}  # nothing special yet
CITY_ALIASES = {
    "bombay": "mumbai", "calcutta": "kolkata", "bengaluru": "bangalore", "gurugram": "gurgaon",
    "poona": "pune", "madras": "chennai", "trivandrum": "thiruvananthapuram",
    "vishakhapatnam": "visakhapatnam", "ahmadabad": "ahmedabad", "mysuru": "mysore",
    "prayagraj": "allahabad", "baroda": "vadodara", "cochin": "kochi", "benares": "varanasi",
    "ctiy": "city",
}
# component classification (no house number): US/India street types come last,
# French voie types come first
STREET_TAILS = {"st", "rd", "ave", "dr", "ln", "blvd", "ct", "cir", "pl", "ter", "pkwy", "hwy",
                "way", "trl", "sq", "plz", "aly", "marg", "rte", "xing", "pike"}
FR_STREET_HEADS = {"rue", "ave", "blvd", "allee", "imp", "rte", "pl", "ch", "quai", "crs", "pass",
                   "sq", "cite", "res", "chs", "lot", "prom", "esp", "sent", "cour"}
FILLER_COMPONENTS = {"null", "<null>", "n/a", "na", "none", "nil", "-", "", "unknown"}
_PO_BOX_RE = re.compile(r"\bp\.?\s?o\.?\s?box\s*\d+", re.IGNORECASE)
_DIGIT_RUN = re.compile(r"\d+")
_POSTCODE_CITY_RE = re.compile(r"^\s*\d{5}\s+[a-z][a-z .]+$")
_SPLIT_DIGIT_ALPHA = re.compile("(\\d)(?!(?:st|nd|rd|th|er|eme|ere|e)(?![a-z]))([a-z])")
_SPLIT_ALPHA_DIGIT = re.compile("([a-z])(\\d)")


def _split_pair(m) -> str:
    return m.group(1) + " " + m.group(2)


_ID_PREFIX_RE = re.compile(
    r"\b(?:h\.?\s?no|house\s?no|door\s?no|dor\s?no|d\.?\s?no|plot\s?no|plt\s?no|flat\s?no|"
    r"shop\s?no|s\.?\s?no|sf\.?\s?no|sy\.?\s?no|p\.?\s?no|p\.?\s?n|kh\.?\s?no|khasra\s?no|"
    r"gate\s?no|no)\b[\s.:-]*")
_LEAD_ZERO_RE = re.compile(r"\d+")


def _compound_ids(comp: str) -> List[str]:
    """Letter/number identifiers such as D-12, UG-23, 22/235, 4141/B -> d12, ug23, 22/235, 4141/b."""
    comp = _ID_PREFIX_RE.sub(" ", comp.replace("#", ""))
    out = []
    for tok in re.split(r"[\s,;()]+", comp):
        tok = tok.strip(".:-'")
        if not tok or not any(c.isdigit() for c in tok):
            continue
        if not (any(c.isalpha() for c in tok) or "/" in tok):
            continue
        if _ORDINAL_TOKEN.match(tok):
            continue
        tok = re.sub(r"[.\-]", "", tok)
        tok = _LEAD_ZERO_RE.sub(lambda m: m.group(0).lstrip("0") or "0", tok)
        if len(tok) >= 2:
            out.append(tok)
    return out


def _canon_state(comp: str, cmap: Optional[dict]) -> Optional[str]:
    if cmap is None:
        return None
    key = re.sub(r"[^a-z ]", " ", comp)
    key = re.sub(r"\s+", " ", key).strip()
    return cmap.get(key)


def _addr_tokens(comp: str, street_map: dict) -> Tuple[List[str], List[str]]:
    """Return (word tokens canonicalised, digit runs) for one address component."""
    comp = re.sub(r"['’`]", "", comp)
    comp = _SPLIT_ALPHA_DIGIT.sub(_split_pair, _SPLIT_DIGIT_ALPHA.sub(_split_pair, comp))
    nums = [n.lstrip("0") or "0" for n in _DIGIT_RUN.findall(comp)]
    text = re.sub(r"[^a-z0-9]+", " ", comp)
    words = []
    for t in text.split():
        if t.isdigit():
            continue
        if _ORDINAL_TOKEN.match(t):
            words.append((t.lstrip("0") or "0").replace("eme", "e").replace("ere", "er"))
            continue
        if any(c.isdigit() for c in t):  # 61a, b2, 3503a -> letters only part is noise
            continue
        t = street_map.get(t, t)
        t = CITY_ALIASES.get(t, t)
        if t in ADDR_DROP or len(t) == 1:
            continue
        words.append(t)
    return words, nums


def normalize_address(raw: Optional[str], country: str):
    """
    Returns (norm, street, loc, state, nums, house, empty)

    norm   – all canonical word + number tokens, space separated
    street – canonical words of components that carry a number / street type
    loc    – canonical words of the remaining (locality / city) components
    state  – canonical state / region, '' if absent
    nums   – digit runs (leading zeros stripped), in order, de-duplicated
    house  – primary house number ('' if none)
    """
    if not raw:
        return "", "", "", "", "", "", 1, ""
    ckey = (country or "").strip().lower()
    cmap = STATE_MAPS.get(ckey)
    is_fr = ckey == "france"
    street_map = STREET_TYPES_FR if is_fr else STREET_TYPES_COMMON
    text = raw
    for native, st in NATIVE_STATES.items():
        if native in text:
            text = text.replace(native, st)
    if has_non_latin(text):
        text = translit_text(text)
    text = strip_accents(text).lower()
    text = _PO_BOX_RE.sub(" ", text)
    comps = [c.strip() for c in text.split(",")]
    comps = [c for c in comps if c not in FILLER_COMPONENTS and c.strip(" .-<>")]
    state = ""
    state_idx = -1
    for i in range(len(comps) - 1, -1, -1):  # states normally come last
        st = _canon_state(comps[i], cmap)
        if st:
            state, state_idx = st, i
            break
    street_words, loc_words, all_words, nums, ids = [], [], [], [], []
    house = ""
    for i, comp in enumerate(comps):
        if i == state_idx:
            continue
        if state and _canon_state(comp, cmap) == state:
            continue
        for tok in _compound_ids(comp):
            if tok not in ids:
                ids.append(tok)
        comp_c = comp.replace("-", " ")
        words, cnums = _addr_tokens(comp_c, street_map)
        if is_fr and _POSTCODE_CITY_RE.match(comp_c):  # "44600 st nazaire" = postcode + city
            loc_words.extend(words)
            all_words.extend(words)
            for n in cnums:
                if n not in nums:
                    nums.append(n)
            continue
        if cnums:
            is_street = True
        elif is_fr:
            is_street = bool(words) and words[0] in FR_STREET_HEADS
        else:
            is_street = bool(words) and words[-1] in STREET_TAILS
        if is_street:
            if cnums and not house:
                house = cnums[0]
            street_words.extend(words)
        else:
            loc_words.extend(words)
        all_words.extend(words)
        for n in cnums:
            if n not in nums:
                nums.append(n)
    norm = " ".join(all_words + nums)
    return (norm, " ".join(street_words), " ".join(loc_words), state,
            " ".join(nums), house, 0, " ".join(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers (used from multiprocessing workers)
# ─────────────────────────────────────────────────────────────────────────────

NAME_FIELDS = ["n_full", "n_core", "n_alias", "n_legal", "n_cmp", "is_domain", "is_native", "has_alias"]
ADDR_FIELDS = ["a_norm", "a_street", "a_loc", "a_state", "a_nums", "a_house", "a_empty", "a_ids"]


def normalize_batch(args):
    names, addrs, countries, translit_path = args
    if translit_path and not _TRANSLIT:
        load_translit_dict(translit_path)
    out_n = [normalize_name(n) for n in names]
    out_a = [normalize_address(a, c) for a, c in zip(addrs, countries)]
    return out_n, out_a


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible helpers (used by older scripts / tests)
# ─────────────────────────────────────────────────────────────────────────────

def clean_name(name: str) -> str:
    return normalize_name(name)[1]


def clean_compact_name(name: str) -> str:
    return normalize_name(name)[4]


def extract_addr_numbers(addr: str) -> List[str]:
    return [n.lstrip("0") or "0" for n in _DIGIT_RUN.findall(addr or "")]
