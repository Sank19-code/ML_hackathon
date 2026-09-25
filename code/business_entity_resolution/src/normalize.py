import re
import unicodedata

# Comprehensive legal entity terms across US, India, and France
LEGAL_TERMS = set([
    # US / UK / International
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'corp', 'corporation', 'co', 'company', 'llp', 'pllc', 'pc',
    'group', 'holdings', 'enterprises', 'enterprise', 'services', 'solutions',
    'technologies', 'consulting', 'global', 'international', 'intl',
    'trading', 'traders', 'center', 'centre', 'management',
    # French
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'sca', 'gie',
    'association', 'asso', 'fils', 'freres', 'cie',
    # Indian
    'proprietorship', 'opc', 'ngo', 'trust'
])

LEGAL_REGEX = re.compile(
    r'\b(' + '|'.join(sorted(LEGAL_TERMS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

# Common domain extensions appearing as business names in web fragments
DOMAIN_REGEX = re.compile(
    r'(www\d?\.)|(\.(com|co\.in|in|org|net|fr|io|biz|info|co|us|online|site))\b',
    re.IGNORECASE
)

# Street and address noise terms
ADDR_STOPWORDS = set([
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'av', 'boulevard', 'blvd', 'bd',
    'drive', 'dr', 'lane', 'ln', 'way', 'circle', 'cir', 'court', 'ct',
    'terrace', 'ter', 'place', 'pl', 'parkway', 'pkwy', 'highway', 'hwy',
    'rue', 'chemin', 'chem', 'impasse', 'imp', 'allee', 'all',
    'floor', 'fl', 'unit', 'apt', 'apartment', 'suite', 'ste', 'room',
    'near', 'opp', 'opposite', 'behind', 'beside', 'block', 'blk', 'sector', 'sec',
    'plot', 'door', 'hno', 'null', 'none', 'north', 'south', 'east', 'west',
    'main', 'market', 'nagar', 'town', 'city', 'village', 'colony', 'layout'
])


def _strip_accents(text: str) -> str:
    """NFKD-normalize and strip combining diacritic characters."""
    normalized = unicodedata.normalize('NFKD', text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_name(name: str) -> str:
    """
    Normalize business name:
    - NFKD + accent stripping → lowercase
    - Remove domain prefixes/suffixes (www., .com, etc.)
    - Remove legal suffixes (Inc, LLC, SARL, Pvt, etc.)
    - Remove 'the/a/an' leading articles
    - Remove punctuation (keep alphanumeric)
    - Strip very short or generic tokens
    """
    if not name or not isinstance(name, str):
        return ""
    text = _strip_accents(name).lower()
    # Remove domain patterns
    text = DOMAIN_REGEX.sub(' ', text)
    # Remove legal suffixes
    text = LEGAL_REGEX.sub(' ', text)
    # Remove leading articles
    text = re.sub(r'^(the|a|an)\s+', '', text)
    # Replace punctuation with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse and filter very short tokens (avoid noise like 'e', 'et')
    tokens = [t for t in text.split() if t not in LEGAL_TERMS and len(t) >= 2]
    return " ".join(tokens)


def clean_compact_name(name: str) -> str:
    """Return compact clean name with all whitespace removed."""
    return clean_name(name).replace(' ', '')


def extract_addr_numbers(addr: str):
    """Extract all numeric tokens from address, stripping leading zeroes."""
    if not addr or not isinstance(addr, str):
        return []
    raw_nums = re.findall(r'\d+', addr)
    nums = []
    for n in raw_nums:
        stripped = n.lstrip('0') or '0'
        # Keep realistic building/postal/plot numbers (not huge years etc.)
        if 1 <= len(stripped) <= 8 and stripped not in nums:
            nums.append(stripped)
    return nums


def extract_addr_words(addr: str):
    """Extract significant alphabetic words (length >= 4) excluding stopwords."""
    if not addr or not isinstance(addr, str):
        return []
    text = _strip_accents(addr).lower()
    words = re.findall(r'[a-z]{4,}', text)
    return [w for w in words if w not in ADDR_STOPWORDS]
