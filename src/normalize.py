"""Unicode-safe normalization for business names and addresses.

Design (driven by EDA on the training data):
  * NFC + lowercase; Latin accents stripped (noise injects "Límited", "Ánnuities").
  * The 9 Indic scripts share the ISCII code layout, so each is mapped onto
    Devanagari offsets and romanised with one small table (schwa-deletion at
    word end). No external data or transliteration service is used.
  * Punctuation -> space, apostrophes deleted, runs of single letters merged
    ("L.L.C." -> "llc", "R & N" -> "rn").
  * Names: bracket tags like "[#21418]" removed, domains ("foo.com") reduced to
    their stem, leetspeak digits inside words mapped back ("seaf0od").
  * Addresses: junk tokens (None, N/A) removed, ordinals stripped (173rd -> 173),
    street-type / state words canonicalised.
The original fields are never modified; normalized values go in new columns.
"""
import re
import unicodedata
from multiprocessing import Pool

# ---------------------------------------------------------------- char table
_INHERENT, _KILL = "\x01", "\x02"
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOW = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
        0x0C: "l", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
        0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "l"}
_MATRA = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
          0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
          0x4C: "au", 0x62: "l", 0x63: "l"}
_SIGN = {0x01: "n", 0x02: "n", 0x03: "h", 0x3C: "", 0x4D: _KILL, 0x3D: ""}
_INDIC_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]


def _build_table():
    table = {}
    for c in range(0x3000):
        ch = chr(c)
        cat = unicodedata.category(ch)
        if ch.isascii():
            if ch.isalnum():
                table[c] = ch.lower()
            elif ch == "'":
                table[c] = ""
            else:
                table[c] = " "
            continue
        if cat == "Mn" and 0x0300 <= c <= 0x036F:  # stray combining accents
            table[c] = ""
            continue
        base = next((b for b in _INDIC_BASES if b <= c < b + 0x80), None)
        if base is not None:
            off = c - base
            if off in _CONS:
                table[c] = _CONS[off] + _INHERENT
            elif off in _VOW:
                table[c] = _VOW[off]
            elif off in _MATRA:
                table[c] = _KILL + _MATRA[off]
            elif off in _SIGN:
                table[c] = _SIGN[off]
            elif 0x66 <= off <= 0x6F:
                table[c] = str(off - 0x66)
            else:
                table[c] = ""
            continue
        if cat.startswith("L"):
            dec = unicodedata.normalize("NFKD", ch)
            if dec and dec[0].isascii() and dec[0].isalpha():  # accented Latin
                table[c] = dec[0].lower()
            else:
                table[c] = ch.lower()
        elif cat.startswith("N"):
            d = unicodedata.digit(ch, None)
            table[c] = str(d) if d is not None else " "
        elif cat.startswith("M"):
            table[c] = ""
        else:  # punctuation, symbols, separators
            table[c] = " "
    table[ord("°")] = " "
    table[ord("’")] = ""
    table[ord("ß")] = "ss"
    return str.maketrans(table)


_TABLE = _build_table()
_RE_KILL = re.compile(_INHERENT + _KILL + "|" + _KILL)
_RE_SCHWA = re.compile(_INHERENT + r"(?=\s|$)")
_RE_WS = re.compile(r"\s+")
_RE_SINGLES = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")


def _merge_singles(m):
    return m.group(0).replace(" ", "")


def base_normalize(text):
    """Lowercase, strip accents, romanise Indic scripts, punctuation -> space."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", text).translate(_TABLE)
    if _INHERENT in s or _KILL in s:
        s = _RE_KILL.sub("", s)
        s = _RE_SCHWA.sub("", s).replace(_INHERENT, "a")
    s = _RE_WS.sub(" ", s).strip()
    return _RE_SINGLES.sub(_merge_singles, s)


# ---------------------------------------------------------------- names
_RE_TAG = re.compile(r"[\[\(]?#\s*\d+[\]\)]?")
_RE_DOMAIN = re.compile(
    r"^(?:https?://)?(?:www\.)?([^\s/]+?)\.(?:com|net|org|co|in|fr|io|biz|info|us|co\.in|org\.in|net\.in)/?$",
    re.I)
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"})
_RE_MIXED = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b")
NAME_CANON = {"pvt": "private", "pvtltd": "private limited", "ltd": "limited", "ltda": "limited",
              "co": "company", "corp": "corporation", "inc": "incorporated", "incorp": "incorporated",
              "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "services",
              "assoc": "associates", "bros": "brothers", "mgmt": "management", "tech": "technologies",
              "dept": "department", "univ": "university", "hosp": "hospital", "ent": "enterprises",
              "and": "", "the": "", "of": "", "m s": "", "ms": ""}


def _deleet(m):
    tok = m.group(0)
    letters = sum(ch.isalpha() for ch in tok)
    return tok.translate(_LEET) if letters >= 2 and letters >= len(tok) - 2 else tok


def normalize_name(text):
    """Returns (normalized_name, is_domain)."""
    if text is None:
        return "", False
    t = _RE_TAG.sub(" ", text.strip())
    m = _RE_DOMAIN.match(t.strip())
    is_domain = bool(m)
    if m:
        t = m.group(1)
    s = base_normalize(t)
    s = _RE_MIXED.sub(_deleet, s)
    toks = [NAME_CANON.get(w, w) for w in s.split()]
    return " ".join(w for w in toks if w), is_domain


# ---------------------------------------------------------------- addresses
_JUNK = {"none", "null", "nan", "n a", "na", "nil", "unknown"}
_RE_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
ADDR_CANON = {
    "avenue": "ave", "av": "ave", "avn": "ave", "street": "st", "str": "st", "road": "rd",
    "lane": "ln", "drive": "dr", "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl",
    "suite": "ste", "apartment": "apt", "apartments": "apts", "floor": "fl", "highway": "hwy",
    "parkway": "pkwy", "circle": "cir", "square": "sq", "terrace": "ter", "north": "n",
    "south": "s", "east": "e", "west": "w", "building": "bldg", "near": "nr", "opposite": "opp",
    "behind": "bh", "sector": "sec", "number": "", "no": "", "nos": "", "n": "", "house": "h",
    "rue": "r", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte", "saint": "st",
    "sainte": "ste", "mount": "mt", "fort": "ft", "p o": "po", "post": "po", "office": "off",
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
}
ADDR_PHRASES = [  # multi-word states (applied before token canon)
    ("new hampshire", "nh"), ("new jersey", "nj"), ("new mexico", "nm"), ("new york", "ny"),
    ("north carolina", "nc"), ("north dakota", "nd"), ("south carolina", "sc"),
    ("south dakota", "sd"), ("west virginia", "wv"), ("rhode island", "ri"),
    ("district of columbia", "dc"), ("uttar pradesh", "up"), ("madhya pradesh", "mp"),
    ("andhra pradesh", "ap"), ("himachal pradesh", "hp"), ("arunachal pradesh", "ar"),
    ("tamil nadu", "tn"), ("tamilnadu", "tn"), ("west bengal", "wb"), ("maharashtra", "mh"),
    ("gujarat", "gj"), ("karnataka", "ka"), ("kerala", "kl"), ("rajasthan", "rj"),
    ("telangana", "tg"), ("ts", "tg"), ("haryana", "hr"), ("punjab", "pb"), ("bihar", "br"),
    ("odisha", "od"), ("orissa", "od"), ("jharkhand", "jh"), ("chhattisgarh", "cg"),
    ("uttarakhand", "uk"), ("assam", "as"), ("goa", "ga"), ("delhi", "dl"), ("new delhi", "dl"),
    ("jammu and kashmir", "jk"), ("jammu kashmir", "jk"), ("puducherry", "py"), ("pondicherry", "py"),
]
_RE_PHRASE = re.compile(r"\b(" + "|".join(re.escape(p) for p, _ in
                                          sorted(ADDR_PHRASES, key=lambda x: -len(x[0]))) + r")\b")
_PHRASE_MAP = dict(ADDR_PHRASES)


def normalize_address(text):
    if text is None:
        return ""
    s = base_normalize(text)
    if s in _JUNK:
        return ""
    s = _RE_ORD.sub(r"\1", s)
    s = _RE_PHRASE.sub(lambda m: _PHRASE_MAP[m.group(1)], s)
    toks = [ADDR_CANON.get(w, w) for w in s.split() if w not in _JUNK]
    return " ".join(w for w in toks if w)


# ---------------------------------------------------------------- bulk
def _norm_chunk(args):
    names, addrs = args
    nn = [normalize_name(x) for x in names]
    return [a for a, _ in nn], [b for _, b in nn], [normalize_address(x) for x in addrs]


def normalize_many(names, addrs, workers=10, chunk=50_000):
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    out_n, out_d, out_a = [], [], []
    with Pool(workers) as pool:
        for n, d, a in pool.imap(_norm_chunk, jobs):
            out_n += n
            out_d += d
            out_a += a
    return out_n, out_d, out_a


if __name__ == "__main__":
    tests = ["SURYANTH PRÓDUCER", "EAR NOSE & THRÓAT PARTNERS LLC", "Abalone Tráding Private (Limited)",
             "Márketing >> Royal Ram Private-Limited", "PVT. GUNTAKAL SÚPER LTD.", "Lucky Seaf0od",
             "R & N ANNUITIE5 L.L.C.", "delgadoprecious.com", "Novo Limited Private Insurance [#21418]",
             "Blount (Rocket)", "नोवो इंश्योरेंस प्राइवेट लिमिटेड", "यमुना एडवाइजरी", "Comité-des Recherche",
             "O'Brien's Café"]
    for t in tests:
        print(f"{t!r:50} -> {normalize_name(t)}")
    for t in ["##902 173ND AVENUE, N/A, BLLEVUE, WA", "C-36 TRANSPORT NAGAR, MATHURA, उत्तर प्रदेश",
              "815 Winters Lane, Unit Apartment 206, Catonsville, Maryland", "None",
              "N° 28 Av. Du Bourgailh, Gironde, Pessac", "20 Ter R Bergère, Nantes",
              "Uttar Pradesh, PARAS SEASONS GH 01- D SECTOR 168, EXPRESSWAY NOIDA"]:
        print(f"{t!r:60} -> {normalize_address(t)!r}")
