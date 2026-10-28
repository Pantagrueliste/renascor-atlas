"""Name, class, date and homonym gates for authority ids (pure functions, no I/O; v2 reconcile).

Why. v1 accepted any authority id and took the heaviest QID without comparing the entry's names with the authority's labels
(Spain -> Famagusta, Kirsten -> a biographical article, Kalkar -> an Eifel hamlet). The audit of 2026-10-16 wrote two matchers
(persons: ``namecmp.py``, places: ``compat.py``); they are ported here unchanged in behaviour and tested.

An id is **accepted** when its authority is of the right class AND (the authority's names (labels and aliases in ten languages
incl. Latin, GND / GeoNames names) are compatible with the entry's names OR a second, independent id agrees). Otherwise it is
**suggested** (kept in the tables, never used for coordinates, labels or merging) or **rejected** (wrong class, homonym).

Statuses returned by the gates (``decide``): ``accepted`` | ``suggested`` | ``rejected`` with a ``reason``:

======================  =====================================================================================================
reason                  meaning
======================  =====================================================================================================
name_mismatch           labels / aliases are unrelated to every name of the entry (suggested)
given_name_only         only a common given name / title token agrees (suggested)
unverified              no label source for the id (VIAF, HLS, TGN not in Wikidata ...) and no second id (suggested)
cert                    the editor's @cert of every tag carrying the id is low / medium (suggested)
chronology              the person was born after the last document that mentions them (suggested)
register_dates          register and Wikidata dates differ by more than 50 years (suggested)
no_class                the item has no P31, cannot be class-checked (suggested)
wikipedia_title         the Wikipedia title does not match the entry (suggested)
class_<what>            wrong class: disambiguation page, given name, list, position, organisation, place for a person,
                        person for a place ... (rejected)
duplicate_item          a Wikimedia duplicate: followed to its target, rejected if no target (rejected)
qid_missing             the item no longer exists (rejected)
homonym                 population-0 feature with the town's name, more than 30 km from every other placed place of the edition (rejected)
homonym_unverified      the same conditions but the edition has fewer than 5 placed places to tell, or a larger namesake of the same region (150 km)
                        is as close to the edition's places as the village (suggested; v2.1, v2.2)
tgn_name_mismatch       the names of the TGN subject (local Getty index, all languages) are unrelated to the entry's names (rejected; v2.1)
tgn_fuzzy               only a fuzzy name relation of 0.80-0.90 with the TGN subject and no independent evidence (suggested; v2.2)
tgn_head_far            'Head, Qualifier': the TGN names match the qualifier only and the head lies far from the TGN point (rejected; v2.2)
manual_review           a decision of the manual review, including the hard overrides of the 'wrong' verdicts (v2.2)
place_id_on_person      a GeoNames / TGN / Pleiades id on a person is the birth place, not the person (ignored)
======================  =====================================================================================================
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# text folding
# ---------------------------------------------------------------------------------------------------------------------
PART = set('''von van vom zu zur zum de der den di da del della dei degli dello delle de' d l' du des le la les el al lo y e et und and the of ibn bin ben abu aus op ten ter te dal dalla dallo dai da' do dos das af av von' t' 't la' l del' a ad in sub
ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx jr jun sr sen
sir mr mrs dr lord lady saint st sainte san santo santa sanctus sancta sanctae sancti beatus beata king queen duke duchess earl count countess prince princess emperor empress pope bishop archbishop cardinal captain colonel general
herr frau graf grafen grafin herzog herzogs herzogin furst fursten furstin churfurst churfursten kurfurst kurfursten kaiser kaiserin konig konigin bischof erzbischof papst pabst sankt hl heilige heiliger pfalzgraf markgraf landgraf freiherr freiin ritter
don dona fra frate maestro messer mons monsieur madame mademoiselle sieur sr sra roi reine duc duchesse comte comtesse princesse pape eveque seigneur sire
rex regina dux ducis comes comitis princeps imperator imperatrix papa episcopus archiepiscopus doctor dominus domina magister frater pater mater cardinalis
re regno regina duca duchessa conte contessa principe principessa imperatore papa vescovo cardinale sig signor signora
rey reina conde condesa principe emperador
koning koningin hertog graaf prins keizer paus
the elder younger junior senior genannt gen called alias dit sive seu vel aka also known as of
mister master mistress lady dame sieur madame
dem das ein eine eines'''.split())
PART |= {'s', 'st', 'ste', 'th', 'nd', 'rd', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
ROMAN = set('titus marcus gaius caius lucius publius quintus sextus gnaeus aulus decimus tiberius appius manius servius flavius '
            'julius iulius claudius cornelius valerius aurelius antonius marc marcos lucio gaio tito'.split())

_COMB_SMALL = {0x363: 'a', 0x364: 'e', 0x365: 'i', 0x366: 'o', 0x367: 'u', 0x368: 'c', 0x369: 'd', 0x36A: 'h', 0x36B: 'm',
               0x36C: 'r', 0x36D: 't', 0x36E: 'v', 0x36F: 'x'}


def small_letters(s: str) -> str:
    """Combining small letters of early-modern print (o + U+0364 = oe) become the letter that follows the base."""
    if not any(0x363 <= ord(c) <= 0x36F for c in s):
        return s
    return ''.join(_COMB_SMALL[ord(c)] if ord(c) in _COMB_SMALL else c for c in s)


def fold(s) -> str:
    s = unicodedata.normalize('NFKD', small_letters(str(s)).casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    for a, b in (('ß', 'ss'), ('æ', 'ae'), ('œ', 'oe'), ('ø', 'o'), ('đ', 'd'), ('ł', 'l'), ('ı', 'i'), ('þ', 'th'), ('ð', 'd'),
                 ('ſ', 's')):
        s = s.replace(a, b)
    return s


def toks(s, keep_all=False):
    s = fold(s)
    s = re.sub(r'\([^)]*\)', ' ', s) if not keep_all else s
    s = re.sub(r"[’'`]", ' ', s)
    return [t for t in re.split(r'[^a-z0-9]+', s) if t]


def sig(s):
    return [t for t in toks(s) if t not in PART and len(t) >= 2 and not t.isdigit()]


def phon(t):
    t = (t.replace('ae', 'e').replace('oe', 'e').replace('ue', 'u').replace('ph', 'f').replace('th', 't').replace('ck', 'k')
         .replace('ch', 'k').replace('sch', 's').replace('tz', 'z').replace('w', 'v').replace('y', 'i').replace('j', 'i')
         .replace('c', 'k').replace('q', 'k'))
    t = re.sub(r'(.)\1+', r'\1', t)
    return t.replace('h', '')


ENDS = ('orum', 'arum', 'ius', 'ium', 'us', 'um', 'is', 'os', 'es', 'ae', 'as', 'ii', 'i', 'o', 'a', 'e', 's')


def stem(t):
    p = phon(t)
    for e in ENDS:
        if len(p) > len(e) + 3 and p.endswith(phon(e)):
            return p[:-len(phon(e))]
    return p


def tmatch(a, b) -> bool:
    """Do two name tokens denote the same name (spelling, Latin ending, abbreviation)? Port of the audit's matcher."""
    if a == b:
        return True
    if len(a) < 3 or len(b) < 3:
        return False
    pa, pb = phon(a), phon(b)
    if pa == pb:
        return True
    sa, sb = stem(a), stem(b)
    if sa == sb and len(sa) >= 3:
        return True
    m = min(len(sa), len(sb))
    if m >= 4:
        k = 0
        while k < m and sa[k] == sb[k]:
            k += 1
        if k >= m - 1 and k >= 4:
            return True
        if k >= 5 and k >= 0.75 * m:
            return True
    ka, kb = re.sub('[aeiou]', '', sa), re.sub('[aeiou]', '', sb)
    if len(ka) >= 3 and ka == kb and sa[:1] == sb[:1]:
        return True
    if len(ka) >= 5 and len(kb) >= 5 and sa[:1] == sb[:1] and difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.8:
        return True
    if min(len(a), len(b)) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.84:
        return True
    if m >= 4 and difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.86:
        return True
    return False


def build_weak(label_lists: Iterable[Iterable[str]], thresh: int = 40):
    """Name tokens that occur in at least ``thresh`` different items' labels (given names, titles): weak evidence."""
    c: Counter = Counter()
    for ls in label_lists:
        seen: set = set()
        for l in ls:
            seen |= set(sig(l))
        c.update(seen)
    return {t for t, n in c.items() if n >= thresh} | ROMAN, c


_NUMERAL = re.compile(r'^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)$')


def ruler_keys(s) -> list:
    """[(name token, regnal number)] of 'Ferdinand II', 'Ferdinand II.', 'Karl V.': a name and its number is specific enough."""
    ts = [t for t in re.split(r'[^a-z0-9]+', fold(re.sub(r'\([^)]*\)', ' ', str(s)))) if t]
    return [(ts[i - 1], ts[i]) for i in range(1, len(ts)) if _NUMERAL.match(ts[i]) and len(ts[i - 1]) >= 3 and not _NUMERAL.match(ts[i - 1])]


def compare_person(entry_names: Iterable[str], labels: Iterable[str], weak: set):
    """-> (status, matched tokens). status: ok | ok_weak | given_only | mismatch | no_label | no_name."""
    ent = [sig(n) for n in entry_names if isinstance(n, str)]
    lab: set = set()
    n_lab = 0
    for l in labels:
        if isinstance(l, str) and l:
            n_lab += 1
            lab |= set(sig(l))
    if not n_lab or not lab:
        return 'no_label', []
    ent_all = [t for ts in ent for t in ts]
    if not ent_all:
        return 'no_name', []
    strong, weakm = [], []
    for t in set(ent_all):
        if any(tmatch(t, l) for l in lab):
            (weakm if t in weak else strong).append(t)
    if strong:
        return 'ok', strong + weakm
    ek = {k for n in entry_names if isinstance(n, str) for k in ruler_keys(n)}
    lk = {k for l in labels if isinstance(l, str) for k in ruler_keys(l)}
    for a, n1 in ek:
        if any(n1 == n2 and tmatch(a, b) for b, n2 in lk):
            return 'ok', [a + ' ' + n1]            # a ruler: the name AND the regnal number agree (Ferdinand II)
    if any(tmatch(a, b) and n1 != n2 for a, n1 in ek for b, n2 in lk):
        return 'mismatch', []                      # the same name with another number: Ferdinand III is not Ferdinand II
    if weakm:
        prim = next((e for e in ent if e), [])
        if (prim and all(any(tmatch(t, l) for l in lab) for t in prim)) or len(weakm) >= 2:
            return 'ok_weak', weakm                 # all of the primary name, or two weak tokens (Anne + Denmark) agree
        return 'given_only', weakm
    return 'mismatch', []


# ---------------------------------------------------------------------------------------------------------------------
# places (port of the audit's compat.py)
# ---------------------------------------------------------------------------------------------------------------------
STOP = {'stadt', 'kloster', 'bistum', 'fluss', 'insel', 'kirche', 'schloss', 'burg', 'haus', 'republik', 'furstentum',
        'konigreich', 'herzogtum', 'grafschaft', 'church', 'castle', 'river', 'island', 'abbey', 'palace', 'street',
        'bridge', 'gate', 'parish', 'hall', 'manor', 'hospital', 'house', 'inn', 'school', 'college', 'market'}
GENERIC = set('''saint sainte sankt santa santo santi sant stadt city town village kingdom republic county province district region island isola insel lake river fluss mount monte berg
castle kloster abbey church kirche diocese bistum duchy duche herzogtum grafschaft roman empire reich holy saints sancti sanctus sancta oppidum civitas urbs regnum comitatus ducatus episcopus
nova novum novi neu neue neuen new alt alte altstadt nuova nuovo nueva nuevo nouvelle nouveau grosse grosser gross klein kleine upper lower oberen unteren ober unter under super inferior superior
municipality commune comune gemeinde region provinz provincia province departement department kreis landkreis bezirk canton kanton
north south east west nord sud ost west'''.split())
SMALL = set('st saint sainte san santa santo sankt sant de del della delle di du des la le les el al the of von vom zu zum zur am an in bei im ob upon on sur sous ad apud et y e da do dos das ten ter van den der het aan op te'.split())


def nrm(s) -> str:
    s = small_letters(str(s)).lower().replace('æ', 'ae').replace('œ', 'oe').replace('ſ', 's').replace('ß', 'ss').replace('ø', 'o') \
        .replace('ł', 'l').replace('đ', 'd')
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9 ]+', ' ', s).split())


def name_keys(name, variants) -> set:
    out: set = set()
    for n in [name] + list(variants if variants is not None else []):
        if not isinstance(n, str) or not n:
            continue
        for c in [n, re.sub(r'\([^)]*\)', ' ', n)] + re.split(r'[,/()]', n):
            k = nrm(c)
            if len(k) > 1 and k not in STOP:
                out.add(k)
    return out


def auth_keys(labels) -> set:
    out: set = set()
    for n in labels:
        if not isinstance(n, str) or not n:
            continue
        for c in [n, re.sub(r'\([^)]*\)', ' ', n)] + re.split(r'[,/()]', n):
            k = nrm(c)
            if len(k) > 1:
                out.add(k)
    return out


def pfold(k: str) -> str:
    toks_ = [t for t in k.split() if t not in SMALL]
    k = ' '.join(toks_) if toks_ else k
    for a, b in (('sch', 's'), ('ph', 'f'), ('th', 't'), ('ck', 'k'), ('qu', 'k'), ('ae', 'e'), ('oe', 'e'), ('ij', 'i'),
                 ('ss', 's'), ('tz', 'z'), ('gk', 'k'), ('jh', 'j')):
        k = k.replace(a, b)
    k = re.sub(r'c(?=[aouklrt]|$)', 'k', k).replace('c', 's')
    k = k.translate(str.maketrans({'j': 'i', 'v': 'u', 'y': 'i', 'w': 'u', 'q': 'k', 'x': 's', 'z': 's', 'h': '', 'k': 'k'}))
    k = re.sub(r'(.)\1+', r'\1', k)
    return ' '.join(k.split())


LAT = ('orum', 'arum', 'iae', 'ium', 'ius', 'ia', 'um', 'us', 'is', 'ae', 'os', 'es', 'en', 'er', 'a', 'e', 'i', 'o', 's')


def pstem(k: str) -> str:
    out = []
    for t in k.split():
        for e in LAT:
            if t.endswith(e) and len(t) - len(e) >= 3:
                t = t[:-len(e)]
                break
        out.append(t)
    return ' '.join(out)


def _tri(s):
    s = f'  {s} '
    return {s[i:i + 3] for i in range(len(s) - 2)}


_TRI_CACHE: dict = {}


def _trig(s):
    v = _TRI_CACHE.get(s)
    if v is None:
        if len(_TRI_CACHE) > 400_000:
            _TRI_CACHE.clear()
        v = _TRI_CACHE[s] = _tri(s)
    return v


def place_relation(ek: set, ak: set):
    """Best name relation between the entry's keys and the authority's keys -> (level, entry key, authority key).
    levels: exact fold stem substring token fuzzyNN noneNN noname noauthname"""
    if not ek or not ak:
        return ('noname' if not ek else 'noauthname', None, None)
    inter = ek & ak
    if inter:
        k = sorted(inter)[0]
        return ('exact', k, k)
    ef = {pfold(k): k for k in ek}
    af = {pfold(k): k for k in ak}
    common = {c for c in set(ef) & set(af) if len(c) >= 2}
    if common:
        c = sorted(common)[0]
        return ('fold', ef[c], af[c])
    es = {pstem(f): k for f, k in ef.items() if len(pstem(f)) >= 4}
    as_ = {pstem(f): k for f, k in af.items() if len(pstem(f)) >= 4}
    c = set(es) & set(as_)
    if c:
        c = sorted(c)[0]
        return ('stem', es[c], as_[c])
    for e in sorted(ek):
        et = [t for t in e.split() if len(t) >= 4 and t not in GENERIC and t not in SMALL]
        for a in ak:
            if len(e) >= 4 and len(a) >= 4:
                if re.search(r'(?<![a-z0-9])' + re.escape(e) + r'(?![a-z0-9])', a) or \
                        re.search(r'(?<![a-z0-9])' + re.escape(a) + r'(?![a-z0-9])', e):
                    if e not in STOP and a not in STOP:
                        return ('substring', e, a)
            at = a.split()
            if et and any(t in at for t in et):
                return ('token', e, a)
    best = (0.0, None, None)
    for f, e in ef.items():
        if len(f) < 4:
            continue
        te = _trig(f)
        for fa, a in af.items():
            if len(fa) < 4 or abs(len(fa) - len(f)) > max(4, 0.5 * len(f)):
                continue
            ta = _trig(fa)
            if len(te & ta) * 2 / (len(te) + len(ta)) < 0.4:
                continue
            ra = difflib.SequenceMatcher(None, f, fa).ratio()
            if ra > best[0]:
                best = (ra, e, a)
    if best[0] >= 0.8:
        return ('fuzzy%.2f' % best[0], best[1], best[2])
    return ('none%.2f' % best[0], best[1], best[2])


STRONG_LEVELS = ('exact', 'fold', 'stem')


#: v2.2: the Getty TGN is matched on thousands of historical and foreign forms, and a fuzzy name relation of 0.80-0.90 accepted Kythnos for Sifnos (0.83),
#: Piave for Pieve di Cadore, Vitto for Alvito; for a TGN id a fuzzy relation counts as a name match from 0.90 only (below it the pair needs independent
#: evidence, else it is only suggested)
TGN_FUZZY_MIN = 0.90


def fuzzy_ratio(level) -> Optional[float]:
    """0.83 for 'fuzzy0.83', else None."""
    if isinstance(level, str) and level.startswith('fuzzy'):
        try:
            return float(level[5:])
        except ValueError:
            return None
    return None


def place_name_ok(level: str, min_fuzzy: float = 0.0) -> bool:
    if level.startswith('fuzzy'):
        r = fuzzy_ratio(level)
        return r is None or r >= min_fuzzy
    return level in STRONG_LEVELS or level in ('substring', 'token')


def head_key(name) -> str:
    """Normalised head of an entry name: before the first comma or parenthesis ('Bad Düben, Alaunwerk' -> 'bad duben')."""
    if not isinstance(name, str):
        return ''
    return nrm(re.split(r'[,(]', name, maxsplit=1)[0])


def place_name_strong(level: str) -> bool:
    return level in STRONG_LEVELS


# ---------------------------------------------------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------------------------------------------------
HUMAN = 'Q5'
# Wikidata classes that can never be a place (a QID of one of them found through a place id is a wrong match)
NON_PLACE_CLASSES = {'Q5', 'Q4167410', 'Q16521', 'Q783794', 'Q4830453', 'Q11424', 'Q7725634', 'Q571', 'Q13442814', 'Q101352',
                     'Q202444', 'Q12308941', 'Q11879590', 'Q3409032', 'Q13406463', 'Q4167836', 'Q4164871', 'Q12737077',
                     'Q28640', 'Q15632617', 'Q95074', 'Q11266439'}
DUP_CLASSES = ('Q17362920', 'Q21286738')       # Wikimedia duplicated page / permanent duplicate item
DISAMBIG = 'Q4167410'

_MYTH = re.compile(r'devil|buddha|demon|deities|pleiades|dryad|naiad|mythical|anemoi|hamadryad|chthonic|nereid|oread|harpy|siren|'
                   r'cyclop|erinyes|maenad|gorgon|sphinx|race\b|muses|mytholog|deity|god\b|goddess|titan|nymph|giant|angel|'
                   r'legendary|fictional|biblical|folklore|muse|oceanid|potamoi|demigod|personification|character|protoplast|'
                   r'hero|satyr|centaur|monster|spirit|fairy|prophet', re.I)
_DYN = re.compile(r'dynasty|noble family|family|house of|sibling|royal house|princely house', re.I)
_DISP = re.compile(r'existence is disputed', re.I)
_PEOPLE = re.compile(r'ethnic group|ethnicity|\btribe\b|\bpeoples?\b|historical ethnic|ethnolinguistic', re.I)
_PLACE = re.compile(r'country|state|duchy|countship|county|principality|region|province|city|town|municipality|historical|kingdom|'
                    r'republic|territory|island|river|bishopric|archbishopric|empire|district|commune|settlement|village|'
                    r'lordship|margraviate|landgraviate|electorate|abbey|church|castle', re.I)


def person_category(p31: list, labels: dict) -> str:
    """Wikidata P31 classes of an item -> human | human_myth | unknown | err_* | legit_*  (audit s4_more.cat)."""
    if not p31:
        return 'unknown'
    labs = [labels.get(c, c) for c in p31]
    if HUMAN in p31:
        return 'human' if not any(_MYTH.search(l) for l in labs if l != 'human') else 'human_myth'
    s = ' | '.join(labs)
    if DISAMBIG in p31 or re.search(r'disambiguation', s, re.I):
        return 'err_disambiguation'
    if re.search(r'list article|wikimedia list|Wikimedia category', s, re.I):
        return 'err_list'
    if re.search(r'given name|family name|name\b', s, re.I) and not _DYN.search(s):
        return 'err_name'
    if any(c in DUP_CLASSES for c in p31) or re.search(r'duplicate', s, re.I):
        return 'err_duplicate'
    if _DISP.search(s):
        return 'legit_disputed'
    if _PEOPLE.search(s):
        return 'legit_people'
    if _MYTH.search(s):
        return 'legit_myth'
    if re.search(r'organi[sz]ation|company|business|enterprise|university|group of humans|institution|order\b', s, re.I) \
            and not _PLACE.search(s):
        return 'err_org'
    if _DYN.search(s) and not _PLACE.search(s.replace('noble family', '').replace('family', '')):
        return 'legit_dynasty'
    if _PLACE.search(s):
        return 'err_place'
    if re.search(r'^position|historical profession|\bposition\b|\boffice\b|title of', s, re.I):
        return 'err_position'
    if re.search(r'organi[sz]ation|company|group of humans', s, re.I):
        return 'err_org'
    if _DYN.search(s):
        return 'legit_dynasty'
    return 'other_nonhuman'


# person_class (v2 mention / entity class) -> person categories an authority may have. Mythological, legendary and biblical figures
# are accepted for any person entity (the audit: 557 legitimate items "to be marked", not removed): the item's own category is
# published (``wd_category`` / ``wd_person_class``). A dynasty or a people is accepted only for an entity the register calls one.
ANY_PERSON = {'human', 'human_myth', 'legit_myth', 'legit_disputed'}
OK_CATEGORIES = {
    'human': ANY_PERSON,
    'mythological': ANY_PERSON, 'biblical': ANY_PERSON, 'character': ANY_PERSON,
    'dynasty': ANY_PERSON | {'legit_dynasty'},
    'people': ANY_PERSON | {'legit_dynasty', 'legit_people'},
}
WD_PERSON_CLASS = {'human': 'human', 'human_myth': 'human', 'legit_myth': 'mythological', 'legit_disputed': 'mythological',
                   'legit_dynasty': 'dynasty', 'legit_people': 'people'}


_FAMILY = re.compile(r'\b(?:family|familie|famille|famiglia|familia|geschlecht|adelsgeschlecht|dynast\w*|house of|haus|casa de)\b', re.I)


def infer_person_class(name, person_class: Optional[str]) -> Optional[str]:
    """A register entry called 'Bellasis family' / '(Familie) Kambli' / 'Haus Österreich' is a family, whatever its class says."""
    if person_class in (None, 'human') and isinstance(name, str) and _FAMILY.search(name):
        return 'dynasty'
    return person_class


def person_class_ok(person_class: Optional[str], category: str):
    """-> (ok, reason)"""
    if category == 'unknown':
        return None, 'no_class'
    allowed = OK_CATEGORIES.get(person_class or 'human', ANY_PERSON)
    if category in allowed:
        return True, ''
    return False, 'class_' + category.replace('err_', '').replace('legit_', 'nonhuman_').replace('other_nonhuman', 'other')


def place_class_ok(p31: list):
    """-> (ok, reason) for an item found through a place id."""
    if not p31:
        return True, ''
    if DISAMBIG in p31:
        return False, 'class_disambiguation'
    if any(c in DUP_CLASSES for c in p31):
        return False, 'duplicate_item'
    if HUMAN in p31:
        return False, 'class_person'
    bad = [c for c in p31 if c in NON_PLACE_CLASSES]
    if bad and len(bad) == len(p31):
        return False, 'class_non_place'
    return True, ''


_SETTLEMENT_WORDS = re.compile(r'\b(city|big city|town|village|municipality|commune|human settlement|hamlet|borough|ortsteil|stadtteil|gemeindeteil|'
                               r'locality|neighbou?rhood|quarter)\b', re.I)
_POLITY_WORDS = re.compile(r'\b(country|sovereign state|state|kingdom|empire|duchy|principality|republic|territory|region|province|county|'
                           r'landgraviate|electorate|bishopric|confederation|dominion)\b', re.I)


def settlement_only(class_labels) -> bool:
    """True when the English labels of an item's classes name a settlement (a town, a village, an Ortsteil) and no polity: the item cannot be a
    country or a region. v2 audit, check 5: the register's *Hessen* (type country) had taken the Ortsteil of that name."""
    s = ' | '.join(l for l in class_labels if l)
    return bool(s) and bool(_SETTLEMENT_WORDS.search(s)) and not _POLITY_WORDS.search(s)


# GND types: a person record on a place entry (tagebuecher: 114 place entries carry a person's GND), and the reverse
GND_PERSON = {'Person', 'DifferentiatedPerson', 'UndifferentiatedPerson', 'RoyalOrMemberOfARoyalHouse', 'Pseudonym',
              'Family', 'NameOfThePerson'}
GND_PLACE = {'PlaceOrGeographicName', 'TerritorialCorporateBodyOrAdministrativeUnit', 'BuildingOrMemorial',
             'NaturalGeographicUnit', 'Country', 'AdministrativeUnit'}


def gnd_class_ok(types: Iterable[str], kind: str):
    ts = set(types or [])
    if not ts:
        return True, ''
    is_p, is_pl = bool(ts & GND_PERSON), bool(ts & GND_PLACE)
    if kind == 'place' and is_p and not is_pl:
        return False, 'class_gnd_person'
    if kind == 'person' and is_pl and not is_p:
        return False, 'class_gnd_place'
    return True, ''


# ---------------------------------------------------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------------------------------------------------
def wd_year(s) -> Optional[int]:
    if not isinstance(s, str) or not s:
        return None
    m = re.match(r'^([+-]?)(\d{1,6})-', s)
    if not m:
        return None
    y = int(m.group(2))
    return -y if m.group(1) == '-' else y


def reg_year(s) -> Optional[int]:
    """Year of a register date string; None for centuries, ranges, doubtful or empty."""
    if not isinstance(s, str) or not s.strip():
        return None
    if re.search(r'Jh|jahrh|\bc\d|century|saec|sec\.|si[eè]cle|/|\bEnde\b|\bAnfang\b|\bMitte\b|vor \d|nach \d|\?', s, re.I):
        return None
    m = re.search(r'(\d{1,4})', s)
    if not m:
        return None
    y = int(m.group(1))
    if re.search(r'v\.\s*chr|b\.?c\.?\b|a\.?\s*c\.|avant|vor chr', s, re.I):
        y = -y
    return y


def chronology_flag(wd_birth: Optional[int], wd_birth_prec, last_doc_year, tol: int = 5) -> bool:
    """True if the person was (by Wikidata) born after the last document that mentions them."""
    if wd_birth is None or last_doc_year is None or last_doc_year != last_doc_year:
        return False
    prec = 9 if wd_birth_prec is None or wd_birth_prec != wd_birth_prec else int(wd_birth_prec)
    lo = wd_birth - (99 if prec <= 7 else 9 if prec == 8 else 0)
    return lo > int(last_doc_year) + tol


def register_dates_flag(reg_birth, reg_death, wd_birth, wd_birth_prec, wd_death, wd_death_prec, limit: int = 50) -> bool:
    """True if the register's birth or death year differs from Wikidata's by more than ``limit`` years (both fully dated,
    both after year 500: BC years without sign in registers are skipped)."""
    def cmp(r, w, p):
        if r is None or w is None or w < 500 or r < 500:
            return False
        if p is not None and p == p and int(p) <= 8:
            return False
        return abs(r - w) > limit
    return cmp(reg_year(reg_birth), wd_birth, wd_birth_prec) or cmp(reg_year(reg_death), wd_death, wd_death_prec)


# ---------------------------------------------------------------------------------------------------------------------
# homonym gate
# ---------------------------------------------------------------------------------------------------------------------
HOMONYM_MAX_POP = 100          # the authority feature is (almost) uninhabited
HOMONYM_TOWN_POP = 1000        # a same-name feature that is a town
HOMONYM_TOWN_FACTOR = 10
HOMONYM_KM = 50.0
# v2.1 (recalibrated on the 114 pairs of the manual review, 2026-10-17): the v2 gate rejected every pair that met the conditions above
# (a village of population 0 with a larger namesake far away): 20 of its 114 rejections were right, 79 wrong (15 undecidable). What tells
# the editor's village from a wrong namesake is whether the feature lies in the edition's own territory: a feature more than
# HOMONYM_ISOLATED_KM from every other placed place of the edition is a namesake. An edition with fewer than HOMONYM_MIN_CONTEXT placed places
# cannot tell: the pair is only ``suggested``. Fitted on 70 % of the labels (``scripts/calibrate_rules.py``), reported on the other 30 %.
HOMONYM_ISOLATED_KM = 30.0
HOMONYM_MIN_CONTEXT = 5
# v2.2 (sample check of v2.1): the v2.1 gate accepted a village whenever the edition had a place within 30 km of it, which a dense edition always has
# (Bullinger: Switzerland and South Germany; the Religionsfrieden treaties: the whole Empire). 13 of the 34 pairs it re-accepted were wrong
# (Kalkar in the Eifel for Kalkar near Kleve, Brüggen, Mülheim, Münden, Marignano ...). A village that has a larger namesake in the SAME REGION
# (within HOMONYM_REGION_KM) is now accepted only when the edition's own places favour the village over the town (the town is itself more than
# HOMONYM_ISOLATED_KM from every other placed place of the edition); otherwise the pair is only suggested (``homonym_unverified``).
HOMONYM_REGION_KM = 150.0


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class HomonymVerdict:
    reject: bool
    reason: str = ''
    town_gid: Optional[str] = None
    town_km: Optional[float] = None
    suspect: bool = False              # the conditions are met but the edition gives no context to decide (the pair is only suggested)
    context_km: Optional[float] = None   # distance of the feature to the nearest other placed place of the edition
    town_context_km: Optional[float] = None   # v2.2: the same for the larger namesake of the region


_QUALIFIED = re.compile(r'[(),]|\b(?:bei|heute|near|prope|apud|près|pres|vicino)\b', re.I)


def has_qualifier(name) -> bool:
    """'Altenburg (bei Nienburg/Saale)', 'Neukirchen, heute Bergisch Neukirchen': the editor already says which one."""
    return bool(isinstance(name, str) and _QUALIFIED.search(name))


def homonym_towns(auth: dict, same: list, name=None) -> list:
    """[(town, km)] of the larger namesakes of the feature ``auth`` that make the pair a homonym candidate (empty: not a candidate), most
    populous first. A GeoNames settlement with (almost) no population that shares the entry's name with a town of the same country more
    than 50 km away (Kalkar 2893838 in the Eifel for Kalkar near Kleve, Montdidier in the Moselle for the Somme).

    ``auth``  {'gid','lat','lon','pop','cc','fcode'} the feature the id points to
    ``same``  features of the same normalised name [{'gid','lat','lon','pop','cc','fcode'}] (the id's own feature may be among them)
    ``name``  the entry's name: one that carries a qualifier (a parenthesis, a comma, 'bei ...') is left alone.
    Conditions: pop(auth) < 100, auth and the town are settlements (PPL*), the town has >= 1000 inhabitants and >= 10x the population,
    lies in the same country, more than 50 km away."""
    pa_ = int(auth.get('pop') or 0)
    if pa_ >= HOMONYM_MAX_POP or has_qualifier(name) or not str(auth.get('fcode') or '').startswith('PPL'):
        return []
    out = []
    for t in same:
        if str(t['gid']) == str(auth.get('gid')) or not str(t.get('fcode') or '').startswith('PPL'):
            continue
        if t.get('cc') != auth.get('cc') or int(t.get('pop') or 0) < max(HOMONYM_TOWN_POP, HOMONYM_TOWN_FACTOR * max(pa_, 1)):
            continue
        d = float(hav(auth['lat'], auth['lon'], t['lat'], t['lon']))
        if d > HOMONYM_KM:
            out.append((t, d))
    return sorted(out, key=lambda x: (-int(x[0]['pop']), str(x[0]['gid'])))


def homonym_candidate(auth: dict, same: list, name=None):
    """-> (town, km) of the most populous larger namesake when the pair meets the conditions of the homonym gate, else None (see ``homonym_towns``)."""
    t = homonym_towns(auth, same, name)
    return t[0] if t else None


def homonym_check(auth: dict, same: list, name=None, ctx=None) -> HomonymVerdict:
    """The homonym gate (v2.2). ``ctx``: (lats, lons) of the other placed places of the edition (None = unknown).

    Not a candidate (see ``homonym_towns``): not rejected. A candidate is **rejected** when the feature is more than ``HOMONYM_ISOLATED_KM`` from
    every context point (the editor's other places are nowhere near it: a namesake), **suspect** (suggested, not rejected) when the edition has
    fewer than ``HOMONYM_MIN_CONTEXT`` context points, and, v2.2, also when a larger namesake lies in the same region (within
    ``HOMONYM_REGION_KM``) and is just as close to the edition's other places (within ``HOMONYM_ISOLATED_KM`` of one of them): the edition cannot tell
    the village from the town. Kept otherwise."""
    towns = homonym_towns(auth, same, name)
    if not towns:
        return HomonymVerdict(False)
    t, d = towns[0]
    pa_ = int(auth.get('pop') or 0)
    base = f"pop {pa_} feature {auth.get('gid')}; same-name town {t['gid']} (pop {int(t['pop'])}) {d:.0f} km away"
    if ctx is None or len(ctx[0]) < HOMONYM_MIN_CONTEXT:
        return HomonymVerdict(False, base + '; the edition has too few placed places to tell', str(t['gid']), d, suspect=True)
    la, lo = np.asarray(ctx[0]), np.asarray(ctx[1])
    iso = float(np.min(hav(auth['lat'], auth['lon'], la, lo)))
    if iso > HOMONYM_ISOLATED_KM:
        return HomonymVerdict(True, base + f'; nearest other place of the edition {iso:.0f} km away', str(t['gid']), d, context_km=iso)
    near = [(tt, dd) for tt, dd in towns if dd <= HOMONYM_REGION_KM]
    if near:
        tt, dd = min(near, key=lambda x: x[1])
        iso_t = float(np.min(hav(tt['lat'], tt['lon'], la, lo)))
        if iso_t <= HOMONYM_ISOLATED_KM:
            return HomonymVerdict(False, f"pop {pa_} feature {auth.get('gid')}; same-name town {tt['gid']} (pop {int(tt['pop'])}) {dd:.0f} km away in the same region "
                                  f"is as close to the edition's other places ({iso_t:.0f} km) as the village ({iso:.0f} km)", str(tt['gid']), dd,
                                  suspect=True, context_km=iso, town_context_km=iso_t)
        return HomonymVerdict(False, '', str(tt['gid']), dd, context_km=iso, town_context_km=iso_t)
    return HomonymVerdict(False, '', str(t['gid']), d, context_km=iso)


# ---------------------------------------------------------------------------------------------------------------------
# Wikipedia titles
# ---------------------------------------------------------------------------------------------------------------------
def wikipedia_title_ok(title: str, entry_names: Iterable[str], kind: str, weak: Optional[set] = None) -> bool:
    """A Wikipedia link in a note is an id only if its title matches the entry (no 'Peasants' Revolt' for a knight)."""
    t = re.sub(r'\s*\([^)]*\)\s*$', '', title or '').strip()
    if not t:
        return False
    if kind == 'person':
        st, _ = compare_person(entry_names, [t], weak or set())
        return st in ('ok', 'ok_weak')
    lvl, _, _ = place_relation(name_keys(None, [n for n in entry_names if isinstance(n, str)]), auth_keys([t]))
    return place_name_ok(lvl)
