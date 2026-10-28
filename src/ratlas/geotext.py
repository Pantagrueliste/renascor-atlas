"""Place-name cleaning and matching keys for name-based geocoding (stage ``reconcile geocode``).

Pure functions, no I/O. The original surface is never modified: the cleaner returns *keys* (normalised strings, the
same normalisation as ``merge._norm``) that a name is looked up under, with the rule that produced each one.

Tiers, tried in this order, the first one that finds a candidate in the dictionary wins:

* ``t0``     the name as written (parentheses flattened, 'of ' dropped by ``_norm``)
* ``t1``     cleaned: leading prepositions / articles / administrative words stripped ('von', 'à', 'in', 'borough of',
             'Kloster'), trailing administrative words ('tithing'), first part before a comma, 's and German genitive
* ``paren``  the text inside parentheses (alternate name or qualifier: 'Cracow (Kraków)', 'Halle (Saale)')
* ``morph``  Latin / German case endings turned into the nominative ('Romae' -> 'roma', 'Argentorati' -> 'argentoratum')
* ``skel``   a spelling skeleton (y=i, double letters, ck=k, trailing e, no spaces ...) for early-modern spellings

Editorial junk ('of this towne', 'UNKNOWN', 'denha004', 's.l.') yields no key.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from .merge import _norm
from .namegate import small_letters
from . import placeclass as _PC

# ---------------------------------------------------------------------------------------------------------------------
PREP = frozenset('''of at in to from near by on upon within
von vom zu zum zur bei beim im nach aus an auf zwischen
a au aux de du des d en dans pres vers sur chez
di da ad al alla alle presso
del em do dos das na no
te ten ter van uit naar bij tot
apud ex ab prope circa iuxta supra inter ante post cis trans'''.split())
ARTICLE = frozenset('the der die das den dem le la les l il lo el los las gli un una'.split())
# administrative / generic words that precede a name ('Kloster Einsiedeln', 'borough of Dorne', 'Diocese of Lincoln')
LEAD_GENERIC = frozenset('''borough parish town city county township hundred liberty manor lordship village bishopric diocese
archdiocese bistum erzbistum bisthum stadt dorf kloster abtei stift kanton canton herrschaft grafschaft landgrafschaft
fürstentum furstentum herzogtum markgrafschaft reichsstadt amt landvogtei vogtei gemeinde pfarrei
ville cite chateau abbaye eveche diocese comte duche principaute
citta cittad contea diocesi abbazia ducato
ciudad villa obispado condado provincia
stad dorp bisdom graafschap'''.split())
TRAIL_GENERIC = frozenset('''tithing township parish hundred liberty hamlet manor borough ward wapentake precinct lordship
town city village county shire stadt dorf amt gemeinde'''.split())
# name keys that denote no particular place
GENERIC = frozenset('''stadt kloster bistum fluss insel kirche schloss burg haus hospital fürstentum furstentum königreich
konigreich herzogtum grafschaft republik amt church castle river island abbey palace street bridge gate parish manor
palast pass duche duchy comte kirche rathaus markt marktplatz cathedral dom closter cloister convent
hospital spital
priory college school inn house market lane ward tower town city village towne citie toune countrey country land ort place
locus urbs oppidum civitas castrum monasterium ecclesia''' .split())
JUNK = frozenset('''unknown unbekannt anonymus anonymous ibid ibidem idem ebenda s l sl sn s n n n nn n a na n d nd nil
none keine ohne unclear illegible illegibel
here there hence thence ubi ibi hic inde unde
this towne our towne same town the town
domi foris'''.split())
_JUNK_PHRASE = re.compile(r'^(?:this|our|the|that|my|his|her|their|same|ye|yt|y)\s+(?:town|towne|toun|toune|citie|city|parish|place|'
                          r'village|house|hous|countr\w*|shire|street|streete)$')
_ID_LIKE = re.compile(r'^[a-z]{2,8}\s?_?\d{2,5}$|^\d+$|^[a-z]\d+$|^\w*\d\w*$')

_ENDINGS = (('ae', 'a'), ('am', 'a'), ('ae', 'e'), ('i', 'um'), ('i', 'us'), ('i', 'ium'), ('ii', 'ium'), ('o', 'um'), ('o', 'us'), ('is', 'a'),
            ('is', 'um'), ('e', 'a'), ('s', ''), ('es', ''), ('ibus', 'a'), ('en', ''), ('n', ''))


@dataclass
class Forms:
    t0: list = field(default_factory=list)       # [(key, rule)]
    t1: list = field(default_factory=list)
    paren: list = field(default_factory=list)
    qual: str = ''                               # first parenthetical / comma remainder, normalised
    junk: bool = False
    parts: list = field(default_factory=list)    # normalised comma-separated parts ('España, Valladolid' -> 2 parts)
    qcc: frozenset = frozenset()                 # country codes implied by a leading country part (set by the geocoder)
    admin: bool = False                          # a hundred, ward, division, 'honour of ...': an administrative area, never a point
    leaf_only: bool = False                      # v2.3: 'Province, Town': the last part is the place, the others only qualify it (no fallback to the container)
    container: str = ''                          # ... the first part, the container
    container_generous: bool = False             # ... the anchor of the leaf is read with the radii of a region (a province named like its capital)

    def keys(self, tiers=('t0', 't1')):
        out = []
        for t in tiers:
            for k, _ in getattr(self, t):
                if k not in out:
                    out.append(k)
        return out

    def rule_of(self, key):
        for t in ('t0', 't1', 'paren'):
            for k, r in getattr(self, t):
                if k == key:
                    return r
        return ''


# administrative units that are not a point: 'Macclesfield Hundred', 'Kettlewell ward', 'honour of Tutbury', 'duchy of Lancaster'
ADMIN_TRAIL = frozenset('hundred hundreds wapentake ward division constabulary'.split())
_ADMIN_LEAD = re.compile(r'^(?:honou?r|duchy|earldom|county palatine|lordship|barony|liberty|hundred|wapentake|rape|lathe) of\b')


def is_admin_key(k: str) -> bool:
    toks = k.split()
    return bool(k) and (_ADMIN_LEAD.match(k) is not None or (len(toks) > 1 and toks[-1] in ADMIN_TRAIL))


def is_junk_key(k: str) -> bool:
    if not k or len(k) < 3:
        return True
    if k in JUNK or k in GENERIC or k in _PC.DEICTIC or k in _PC.COMMON_NOUN:
        return True
    if _JUNK_PHRASE.match(k) or _ID_LIKE.match(k):
        return True
    toks = k.split()
    if all(t in GENERIC or t in PREP or t in ARTICLE or t in JUNK for t in toks):
        return True
    return False


def _strip_tokens(k: str) -> list:
    """Successive reductions of a key: leading prepositions, articles and administrative words, trailing ones."""
    out = []
    cur = k
    for _ in range(4):
        toks = cur.split()
        new = toks
        if len(new) > 1 and new[0] in PREP | ARTICLE | LEAD_GENERIC:
            new = new[1:]
        elif len(new) > 1 and new[-1] in TRAIL_GENERIC:
            new = new[:-1]
        nk = ' '.join(new)
        if nk == cur or len(nk) < 3:
            break
        out.append(nk)
        cur = nk
    return out


def _flat(s: str) -> str:
    return _norm(s)


def forms(raw) -> Forms:
    """Matching keys of one surface (name or variant), see the module docstring."""
    f = Forms()
    if not isinstance(raw, str) or not raw.strip():
        f.junk = True
        return f
    s = small_letters(unicodedata.normalize('NFC', raw)).replace('’', "'").replace('`', "'")
    s = re.sub(r"'s\b", '', s)                                  # English possessive
    s = re.sub(r'\[[^\]]*\?\s*\]', ' ', s)                      # [?], [Wien?]
    s = s.replace('[', ' ').replace(']', ' ')
    s = re.sub(r'[?*†‡¶§]+', ' ', s)
    parens = re.findall(r'\(([^)]*)\)', s)
    base = re.sub(r'\([^)]*\)', ' ', s)
    flat = s.replace('(', ' ').replace(')', ' ')
    seen: set = set()

    def add(lst, key, rule):
        if key and key not in seen and not is_junk_key(key):
            seen.add(key)
            lst.append((key, rule))

    kb = _flat(base)
    if is_admin_key(kb):
        f.admin = True
        f.junk = True
        return f
    if is_junk_key(kb) and not parens:
        # 'of this towne', 'denha004', 'UNKNOWN': nothing to look up
        f.junk = True
    add(f.t0, kb, 'base')
    if parens:
        add(f.t0, _flat(flat), 'flat')
    # t1
    for sub in _strip_tokens(kb):
        add(f.t1, sub, 'strip')
    if ',' in base:
        first = base.split(',')[0]
        kf = _flat(first)
        add(f.t1, kf, 'comma')
        for sub in _strip_tokens(kf):
            add(f.t1, sub, 'comma+strip')
        rest = _flat(base.split(',', 1)[1])
        f.qual = rest
        f.parts = [k for k in (_flat(x) for x in base.split(',')) if k]
    if '/' in base:
        for part in base.split('/'):
            kp = _flat(part)
            add(f.t1, kp, 'slash')
            for sub in _strip_tokens(kp):
                add(f.t1, sub, 'slash+strip')
    # possessive / genitive 's' of German names ('Zürichs'), only as t1
    for k, _ in list(f.t0):
        if len(k) >= 6 and k.endswith('s') and ' ' not in k:
            add(f.t1, k[:-1], 'genitive')
    for p in parens:
        kp = _flat(p)
        if not f.qual:
            f.qual = kp
        add(f.paren, kp, 'paren')
        for sub in _strip_tokens(kp):
            add(f.paren, sub, 'paren+strip')
    if not f.t0 and not f.t1 and not f.paren:
        f.junk = True
    elif f.t0 or f.t1 or f.paren:
        f.junk = False
    return f


def morph_keys(key: str) -> list:
    """Nominative guesses for the first and the last word of a key (Latin / German endings)."""
    toks = key.split()
    out = []
    idx = sorted({0, len(toks) - 1})
    for i in idx:
        t = toks[i]
        if len(t) < 5:
            continue
        for a, b in _ENDINGS:
            if t.endswith(a) and len(t) - len(a) + len(b) >= 4:
                nt = t[:len(t) - len(a)] + b
                new = toks[:i] + [nt] + toks[i + 1:]
                k = ' '.join(new)
                if k != key and k not in out:
                    out.append(k)
    return out


_SAINT = re.compile(r'\b(?:saint|sainct|sainte|sankt|sanct|sanctus|sancta|sint|sant|santa|santo|san|sao|ste|st|s)\b')


@lru_cache(maxsize=2_000_000)
def skeleton(key: str) -> str:
    """Spelling skeleton of a normalised key: early-modern variants of one name collapse to the same string."""
    s = _SAINT.sub('st', key)
    for a, b in (('ae', 'a'), ('oe', 'o'), ('ue', 'u'), ('ph', 'f'), ('th', 't'), ('ck', 'k'), ('tz', 'z'), ('rh', 'r'), ('jh', 'j'),
                 ('gk', 'g'), ('mpb', 'mb'), ('mb', 'nb'), ('y', 'i'),
                 ('w', 'v'), ('j', 'i'), ('ou', 'u'), ('eu', 'u'), ('ei', 'i'), ('ie', 'i'), ('ee', 'e')):
        s = s.replace(a, b)
    s = re.sub(r'c(?![hei])', 'k', s)
    toks = []
    for t in s.split():
        if len(t) > 4 and t.endswith('e'):
            t = t[:-1]
        if len(t) > 4 and t.endswith('d'):
            t = t[:-1] + 't'                       # Hollant / Holland
        toks.append(t)
    s = ''.join(toks)
    s = re.sub(r'(.)\1+', r'\1', s)
    return s


def all_keys(f: Forms) -> set:
    """Every exact key plus morph variants (what the GeoNames index must be able to answer)."""
    ks = set(f.keys(('t0', 't1', 'paren')))
    ks.update(k for k in f.parts if not is_junk_key(k))
    for k in list(ks):
        ks.update(morph_keys(k))
    return ks
