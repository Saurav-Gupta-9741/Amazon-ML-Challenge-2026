"""
normalizer.py — Comprehensive text normalizer for Business Entity Resolution.

Handles multi-country business names and addresses:
1. Devanagari transliteration (via transliterate.py)
2. Unicode NFKD accent stripping (French accents: é, è, ê, etc.)
3. Punctuation noise cleaning (--, <<, **, ##)
4. Legal suffix normalization (US, India, France)
5. Address token expansion & postal code / number extraction
"""

import re
import unicodedata
from transliterate import transliterate_text, contains_indic

# Canonical suffix replacements
LEGAL_SUFFIX_MAP = [
    # Multi-word replacements first
    (re.compile(r'\b(?:private\s+limited|pvt\s*\.?\s*ltd\.?|p\s*\.?\s*ltd\.?)\b', re.IGNORECASE), 'pvt_ltd'),
    (re.compile(r'\b(?:limited\s+liability\s+company|l\s*\.?\s*l\s*\.?\s*c\.?)\b', re.IGNORECASE), 'llc'),
    (re.compile(r'\b(?:corporation|corp\.?)\b', re.IGNORECASE), 'corp'),
    (re.compile(r'\b(?:incorporated|inc\.?)\b', re.IGNORECASE), 'inc'),
    (re.compile(r'\b(?:limited|ltd\.?)\b', re.IGNORECASE), 'ltd'),
    (re.compile(r'\b(?:company|co\.?)\b', re.IGNORECASE), 'co'),
    # French entity types
    (re.compile(r'\b(?:societe\s+a\s+responsabilite\s+limitee|s\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\.?)\b', re.IGNORECASE), 'sarl'),
    (re.compile(r'\b(?:s\s*\.?\s*a\s*\.?\s*s\s*\.?\s*u\.?)\b', re.IGNORECASE), 'sasu'),
    (re.compile(r'\b(?:s\s*\.?\s*a\s*\.?\s*s\.?)\b', re.IGNORECASE), 'sas'),
    (re.compile(r'\b(?:e\s*\.?\s*u\s*\.?\s*r\s*\.?\s*l\.?)\b', re.IGNORECASE), 'eurl'),
]

# Address token expansions
ADDRESS_EXPANSIONS = [
    (re.compile(r'\b(?:st|str)\b', re.IGNORECASE), 'street'),
    (re.compile(r'\brd\b', re.IGNORECASE), 'road'),
    (re.compile(r'\bave\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\b(?:dr|drv)\b', re.IGNORECASE), 'drive'),
    (re.compile(r'\b(?:blvd|bd)\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bln\b', re.IGNORECASE), 'lane'),
    (re.compile(r'\bct\b', re.IGNORECASE), 'court'),
    (re.compile(r'\bpkwy\b', re.IGNORECASE), 'parkway'),
    (re.compile(r'\bhwy\b', re.IGNORECASE), 'highway'),
    (re.compile(r'\bopp\b', re.IGNORECASE), 'opposite'),
    (re.compile(r'\bnr\b', re.IGNORECASE), 'near'),
    (re.compile(r'\b(?:ste|suite)\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\b(?:apt|apartment)\b', re.IGNORECASE), 'apt'),
    (re.compile(r'\bflr?\b', re.IGNORECASE), 'floor'),
]

# US State abbreviations mapping to state names
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee',
    'tx': 'texas', 'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington',
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming'
}


def strip_accents(text: str) -> str:
    """Strips diacritics / accents using Unicode NFKD normalization (e.g., é -> e)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def clean_noise_prefixes(text: str) -> str:
    """Removes common noise prefixes like '--', '<<', '**', '##', '***'."""
    if not text:
        return ""
    # Strip leading noise symbols
    return re.sub(r'^[-\s<*#>~|]+', '', text)


def _strip_domain(name: str) -> str:
    """Strips domain extensions and splits concatenated domain names."""
    # Remove common domain extensions
    name = re.sub(r'\.(com|in|org|net|co|io|biz|info)\b', '', name, flags=re.IGNORECASE)
    # Split camelCase or fully concatenated words if result looks like a domain
    # e.g., "summithealth" stays as is (model will handle via fuzzy matching)
    return name


def normalize_business_name(name: str) -> str:
    """
    Normalizes a business name:
    - Strips domain extensions (.com, .in, .org, etc.)
    - Transliterates Indic scripts if present
    - Strips accents and noise prefixes
    - Standardizes legal entity suffixes
    - Removes punctuation and standardizes spacing
    """
    if not name or not isinstance(name, str):
        return ""

    # Strip domain extensions first
    name = _strip_domain(name)

    # Transliterate Indic scripts if present
    name = transliterate_text(name)
    name = strip_accents(name)
    name = clean_noise_prefixes(name)
    name = name.lower()

    # Standardize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_MAP:
        name = pattern.sub(' ' + replacement + ' ', name)

    # Clean punctuation: replace non-alphanumeric (except underscores for suffixes) with space
    name = re.sub(r'[^a-z0-9_]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def normalize_address(address: str) -> str:
    """
    Normalizes a business address:
    - Transliterates Devanagari states/cities
    - Strips accents and punctuation
    - Expands standard street abbreviations (st -> street, rd -> road, etc.)
    - Expands standalone US state abbreviations
    """
    if not address or not isinstance(address, str):
        return ""

    address = transliterate_text(address)
    address = strip_accents(address)
    address = clean_noise_prefixes(address)
    address = address.lower()

    # Expand address abbreviations
    for pattern, replacement in ADDRESS_EXPANSIONS:
        address = pattern.sub(' ' + replacement + ' ', address)

    # Replace punctuation with spaces
    address = re.sub(r'[^a-z0-9]+', ' ', address)

    # Expand US state tokens
    tokens = address.split()
    expanded_tokens = [US_STATES.get(t, t) for t in tokens]
    
    return ' '.join(expanded_tokens).strip()


def extract_postal_code(address: str) -> str:
    """
    Extracts a postal code from an address string:
    - India: 6-digit PIN code (e.g. 400001)
    - US: 5-digit ZIP code (e.g. 90210)
    - France: 5-digit code (e.g. 33000)
    """
    if not address or not isinstance(address, str):
        return ""
    # Look for 6-digit PIN first (India)
    m6 = re.search(r'\b([1-9][0-9]{5})\b', address)
    if m6:
        return m6.group(1)
    # Look for 5-digit ZIP (US / France)
    m5 = re.search(r'\b([0-9]{5})\b', address)
    if m5:
        return m5.group(1)
    return ""


def extract_street_numbers(address: str) -> set:
    """Extracts set of street / building numbers from address."""
    if not address or not isinstance(address, str):
        return set()
    numbers = re.findall(r'\b([0-9]+[a-z]?)\b', address.lower())
    # Exclude likely postal codes (>4 digits)
    return {n for n in numbers if len(n) <= 4}


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    test_names = [
        "-- Holloway Peak Inc Seafood",
        "<< Team Ecole",
        "Orelee's Bárbershop",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Tata Motors Ltd",
        "ZNB Club SARL"
    ]
    for n in test_names:
        print(f"RAW : {n} -> NORM: {normalize_business_name(n)}")
        
    print()
    test_addrs = [
        "1795 Westchester Drive, High Point, NC",
        "IA, Iowa City, 1064 Newton Rd, Unit 11",
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110001",
        "PLOT NO B-78/1, THANE, महाराष्ट्र 400601"
    ]
    for a in test_addrs:
        print(f"ADDR: {a}")
        print(f"NORM: {normalize_address(a)}")
        print(f"PIN : {extract_postal_code(a)} | NUMS: {extract_street_numbers(a)}")
        print()
