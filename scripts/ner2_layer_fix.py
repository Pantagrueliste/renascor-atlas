"""NER layer v2: the corrected model layer, built from the v1 tables (re-runnable).

    nice -n 19 python scripts/ner2_layer_fix.py --work DIR [--out model_layer_v2.json] [--report corrections_v2.md]

See the module docstring sections below, and ``RQ/derived/atlas/ner/v2/layer/corrections_v2.md`` (what changed, before / after). Pure logic (context cues, name
capitalisation, grouping, merging, radius floor, tiers) is tested in ``tests/test_ner2_layer_fix.py``. Nothing is committed or published.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# context cues (used on Hypatia by ner2_layer_fix_ctx.py, where the texts are)
# ---------------------------------------------------------------------------------------------------------------------
CONTEXT_KEYS = {
    'council': {'nice', 'nyce', 'niceam', 'nizza', 'niza'},
    'idf': {'isle de france', 'île de france', 'ile de france', 'yle de france', 'île-de-france', 'ile-de-france'},
    'jordan': {'iordan', 'iordanis'},
}
_KEY2RULE = {k: r for r, ks in CONTEXT_KEYS.items() for k in ks}


def context_rule_of(key):
    return _KEY2RULE.get(key)


def _norm_ctx(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '').replace('ſ', 's').replace('\u00ad', '')
    return ''.join(c for c in s if not unicodedata.combining(c)).casefold().replace('\u0131', 'i')


# a council word before the mention (Councel, Counsell, Councill, concile, synod, sinode, canon ...), a heresy / council cue, or a number of bishops
_COUNCIL_BEFORE = re.compile(
    r'(?:\bcou?n[cs]e?i?l+s?\b|\bcou?n[cs]e?i?l+\w*|\bcou?s+[ae]i?l+\w*|\bcounsail\w*|\bcoucel\w*|\bconcile\w*|\bconcil\w*|\bsyn+od\w*|\bsinod\w*|\bcanons?\b|\bcanon\w*|\bnicen\w*|\bnycen\w*|'
    r'\bconuenticle\w*|\bconventicle\w*|\bdecree\w*|\bdecret\w*|\barr?ians?\b|\barrius\b|\bathanasi\w*|\bgenerall?\b|\bconc\.|'
    r'\b\d{2,3}\.?\s*bi?s+h?o?p+s?\b)[^.;:!?]{0,32}$')
_COUNCIL_AFTER = re.compile(r'^[\s,()\-]{0,3}(?:creed|credo|symbol\w*|cou?n[cs]e?i?l+\w*|concil\w*|synod\w*|canons?|canon\w*|decree\w*|decret\w*)\b')
_MAURITIUS = re.compile(r'maurice|mauritius|mauricius|mascare|madagascar|bourbon|indian ocean|ocean indien|océan indien')
_RIVER = re.compile(r'\b(?:river|riuer|rivier|fleuve|fleuue|fleuves|riviere|rivière|flume|fiume|fiumi|fluvius|fluuius|flumen|fluuio|fluvio|flumine|fluminis|fluuii|fluvii|amnis|rio|río|ríos|rios|'
                    r'flood|floud|water|waters|streame|stream|ford|passage|passed|passe|passer|cross\w*|baptis\w*|baptiz\w*|baptism\w*|baptem\w*|bapt\w*|batt[eè]\w*|aqua|aquae|aquas|acque|aguas|eaux|'
                    r'banks?|shore|ripa|riva|rive|ribera|beyond|outre|oltre|ultra|trans|jenseits|jenseit|wasser|fluss|strom|ufer|ioh\w*|john|jean|giovanni|juan)\b')


def apply_context_rule(rule: str, before: str, after: str) -> bool:
    """True when the cue of the rule is in the text just before or after the mention.

    council   a council word right before (``council of``, ``concile de``, ``synod at``) or a creed / council word right after (``Nice Creed``)
    idf       Mauritius cue within 120 characters either side (the rule says: Île-de-France unless Maurice / Mauritius)
    jordan    a river cue within 100 characters either side
    """
    b, a = _norm_ctx(before), _norm_ctx(after)
    if rule == 'council':
        tail = re.split(r'(?<!\d)[.;!?]\s', b[-48:])[-1]
        return bool(_COUNCIL_BEFORE.search(tail)) or bool(_COUNCIL_AFTER.search(a[:30]))
    if rule == 'idf':
        return bool(_MAURITIUS.search(b[-120:] + ' ' + a[:120]))
    if rule == 'jordan':
        return bool(_RIVER.search(b[-100:] + ' ' + a[:100]))
    raise ValueError(rule)


# ---------------------------------------------------------------------------------------------------------------------
# pure helpers (tested)
# ---------------------------------------------------------------------------------------------------------------------
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

MIN_AREA_R = 40          # km, minimum radius of a country / province area
TIER_MAIN = 11           # mentions: "main" from 11, "rare" below
KM_EQUIV_CITY = 3.0      # two cities that are known equivalents (Wikidata id, exonym table) closer than this are one point
KM_SAME_NAME = 10.0      # the v1 rule (same modern name or spelling skeleton) keeps its 10 km
QID_KM = 60.0            # two points with the same Wikidata id, closer than this, are one point
AREA_SHARE = 0.30        # two areas: closer than 30 % of the larger radius
PPER_MAX = 0.9           # mean person probability of a form from which no point is made (exonym table excepted)

ARTICLES = ('the ', 'a ', 'an ', 'la ', 'le ', 'les ', 'el ', 'los ', 'las ', 'il ', 'lo ', 'gli ', 'der ', 'die ', 'das ', 'den ', "l'", "l’")
PARTICLES = {'de', 'di', 'del', 'della', 'delle', 'dei', 'degli', 'da', 'du', 'des', 'of', 'the', 'la', 'le', 'les', 'el', 'y', 'e', 'et', 'and', 'von', 'van', 'zu', 'am', 'im', 'an',
             'der', 'den', 'dem', 'in', 'sur', 'sous', 'en', 'upon', 'on', 'under', 'by', 'al', 'ad', 'ab', 'a', 'à', 'au', 'aux'}
BAD_LABEL = re.compile(r'[,()\[\]/;:0-9]|^(republic|kingdom|state|province|region|county|district|municipality|commune|canton|department|departement) of\b', re.I)


def fold(s) -> str:
    s = unicodedata.normalize('NFKD', str(s or '').casefold().replace('ſ', 's'))
    return ''.join(c for c in s if c.isalnum())


def hav_km(lat1, lon1, lat2, lon2):
    import numpy as np
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def capitalise_label(s: str) -> str:
    """First letter of the label and of each word that is not a particle (de, of, di ...). 'the bath' -> 'Bath' once the article is cut."""
    s = ' '.join(str(s or '').split())
    low = s.casefold()
    for art in ARTICLES:
        if low.startswith(art) and len(s) > len(art) + 2:
            s = s[len(art):]
            break
    out = []
    for i, w in enumerate(s.split(' ')):
        if not w:
            continue
        if i > 0 and w.casefold() in PARTICLES and w == w.casefold():
            out.append(w)
        elif w[:1].islower():
            out.append(w[:1].upper() + w[1:])
        else:
            out.append(w)
    return ' '.join(out)


def label_ok(s: str) -> bool:
    return len(s.strip()) >= 3 and not BAD_LABEL.search(s)


def pick_label(forms) -> str:
    """The display label of a point: the most frequent written form (summed over languages, case ignored), capitalised, never a label with a comma, a parenthesis or 'Republic of'.

    ``forms`` = iterable of (surface, mentions, lang).
    """
    tot = Counter()
    best_case = {}
    for sf, n, _lg in forms:
        k = capitalise_label(sf).casefold()
        tot[k] += n
        if k not in best_case or sum(1 for c in sf if c.isupper()) > sum(1 for c in best_case[k] if c.isupper()):
            best_case[k] = capitalise_label(sf)
    for k, _n in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0])):
        lab = best_case[k]
        if label_ok(lab):
            return lab
    return capitalise_label(forms[0][0]) if forms else ''


def radius_floor(kind: str, r) -> int:
    """Areas (countries, provinces) get at least MIN_AREA_R km; other kinds keep no radius."""
    if kind != 'a':
        return 0
    return int(round(max(float(r or 0), MIN_AREA_R)))


def tier_of(n: int) -> str:
    return 'main' if n >= TIER_MAIN else 'rare'


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return ra


def pair_thresholds(r_a: float, r_b: float):
    """(km for 'same place under two known equivalent names', km for 'same modern name or spelling'): cities 3 / 10 km; with an area, 30 % of the larger radius."""
    rmax = max(float(r_a or 0), float(r_b or 0))
    if rmax > 0:
        return max(KM_EQUIV_CITY, AREA_SHARE * rmax), max(KM_SAME_NAME, AREA_SHARE * rmax) if rmax > 15 else KM_SAME_NAME
    return KM_EQUIV_CITY, KM_SAME_NAME


def split_move(doc_n: dict, doc_move: dict) -> dict:
    """Mentions moved out of a form, document by document, never more than the form has there."""
    return {d: min(int(k), int(doc_n.get(d, 0))) for d, k in doc_move.items() if min(int(k), int(doc_n.get(d, 0))) > 0}


def pkey_of(qid, cand, lat, lon) -> str:
    if isinstance(qid, str) and qid:
        return qid
    if isinstance(cand, str) and cand.startswith('gn:'):
        return cand
    return f'xy:{round(float(lat), 3)},{round(float(lon), 3)}'


# ---------------------------------------------------------------------------------------------------------------------
# the build (Mac): units = written forms (language, key); D = mentions of a unit per document; groups = points
# ---------------------------------------------------------------------------------------------------------------------
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

NICE = (43.70313, 7.26608)
# explicit entries ("table edits"). Each: forms (lang, key) -> target. A target is either ('ref', (lang, key)) = the point of that existing form, or ('fix', name) = a new point with
# the given coordinates. kinds: c city, a area, n river / natural.
TABLE_EDITS = [
    # (label, [(lang, key) ...], target, extra) ; section 5 items 2 and 7 of controle.md
    ('Bergogna = Burgundy', [('it', 'bergogna')], ('ref', ('fr', 'bourgogne')), {'set': {'name': 'Burgundy'}}),
    ('Nerva = Narva', [('it', 'nerva')], ('ref', ('en', 'narva')), {'set': {'name': 'Narva'}}),
    ('Dan = Damme', [('en', 'dan')], ('ref', ('en', 'damme')), {'set': {'name': 'Damme'}}),
    ('Modana = Modena', [('it', 'modana'), ('es', 'modana')], ('ref', ('it', 'modena')), {'set': {'name': 'Modena'}}),
    ('Tage / Tago = Tagus river', [('fr', 'tage'), ('it', 'tago'), ('en', 'tago'), ('es', 'tago'), ('la', 'tagus'), ('en', 'tagus')], ('ref', ('la', 'tagus')),
     {'set': {'name': 'Tagus', 'kind': 'n', 'r': None}}),
    ('Monferrato = Montferrat region', [('it', 'monferrato'), ('en', 'monferrato'), ('es', 'monferrato')], ('ref', ('en', 'montferrat')),
     {'set': {'name': 'Montferrat', 'kind': 'a', 'r': 40}}),
    ('Geldria / Geldriam = Guelders region', [('la', 'geldria'), ('la', 'geldriam')], ('ref', ('en', 'gelderland')), {'set': {'name': 'Guelders'}}),
    ('Humber = Humber estuary', [('en', 'humber')], ('fix', 'humber'), {'set': {'name': 'Humber', 'lat': 53.65, 'lon': -0.20, 'kind': 'n', 'r': None}}),
    ('Iordan = Jordan river', [('en', 'iordan'), ('es', 'iordan'), ('en', 'iordanis'), ('la', 'iordanis')], ('ref', ('en', 'jordan')),
     {'set': {'name': 'Jordan River', 'lat': 31.76, 'lon': 35.56, 'kind': 'n', 'r': None}}),
]
ANTIOCH_LA = re.compile(r'^(antiochi|antyochi)')
BLACKLIST = {'uti', 'duron'}
MECKLENBURG_BAD = {('en', 'megalopolis')}
IDF_KEYS = CONTEXT_KEYS['idf']
IDF = {'name': 'Île-de-France', 'lat': 49.0, 'lon': 2.5, 'kind': 'a', 'r': 60}
NICE_TARGET = ('en', 'nicea')           # İznik (40.43, 29.72): the exonym-table point of Nicaea
ANTAKYA = ('en', 'antiochia')           # Antakya (36.20, 36.16)
MAURITIUS = ('fr', 'isle de france')

# equivalent names that share no root: used as a second proof of "known equivalents" next to Wikidata ids and the exonym table of the geocoder
EQUIV_NAMES = [
    {'napoli', 'naples', 'neapel', 'napels', 'neapolis', 'napoles', 'napolis', 'napolitano'},
    {'firenze', 'florence', 'florenz', 'florentia', 'florencia'},
    {'fraunce', 'france', 'frankreich', 'francia', 'francie', 'frantzreich', 'franckreich', 'gallia', 'republic of france'},
    {'venezia', 'venice', 'venedig', 'venise', 'venetia', 'venecia'},
    {'konstantinopel', 'istanbul', 'constantinople', 'costantinopoli', 'constantinopla', 'constantinopolis', 'konstantinopolis'},
    {'italia', 'italy', 'italien', 'italie', 'italya'},
]


def equiv_class(name: str):
    f = fold(name)
    for i, grp in enumerate(EQUIV_NAMES):
        if f in {fold(x) for x in grp}:
            return i
    return None


def _t(m):
    print(f'[layer-fix {time.strftime("%H:%M:%S")}] {m}', file=sys.stderr, flush=True)


class State:
    """Units (one per written form of a language), their mentions per document, and the groups they belong to."""

    def __init__(self, U, D, docs):
        self.U = U                      # DataFrame indexed by uid
        self.D = D                      # DataFrame uid, doc_id, n
        self.docs = docs                # DataFrame indexed by doc_id: collection, year, lang
        self.G_set = {}                 # group -> explicit overrides (name, lat, lon, kind, r)
        self.log = []                   # per step: dict
        self.next_uid = int(U.index.max()) + 1
        self._n = None

    # ---- counts
    def unit_n(self):
        return self.D.groupby('uid')['n'].sum()

    def snapshot(self):
        n = self.unit_n().reindex(self.U.index).fillna(0).astype(int)
        pl = self.U['placed'] & (n > 0)
        grp = self.U.loc[pl, 'grp']
        gn = n[pl].groupby(grp).sum()
        main = int((gn >= TIER_MAIN).sum())
        return {'points': int(len(gn)), 'mentions': int(gn.sum()), 'main': main, 'rare': int(len(gn) - main), 'main_mentions': int(gn[gn >= TIER_MAIN].sum()),
                'rare_mentions': int(gn[gn < TIER_MAIN].sum()), 'forms': int(pl.sum())}

    def step(self, name, **info):
        s = self.snapshot()
        prev = self.log[-1]['after'] if self.log else None
        self.log.append({'step': name, 'before': prev, 'after': s, **info})
        _t(f'{name}: points {s["points"]}, mentions {s["mentions"]}, main/rare {s["main"]}/{s["rare"]} {info}')
        return s

    # ---- unit ops
    def uid_of(self, lang, key, run=None):
        u = self.U
        m = (u['lang'] == lang) & (u['key'] == key) & (u['variant'] == '')
        if run:
            m &= (u['run'] == run)
        ids = u.index[m]
        return int(ids[0]) if len(ids) else None

    def new_unit(self, like_uid, variant):
        row = self.U.loc[like_uid].copy()
        uid = max(self.next_uid, int(self.U.index.max()) + 1)          # units added after the state was made must not be overwritten
        self.next_uid += 1
        row['variant'] = variant
        self.U.loc[uid] = row
        return uid

    def doc_counts(self, uid):
        d = self.D[self.D['uid'] == uid]
        return dict(zip(d['doc_id'], d['n']))

    def move_mentions(self, uid, move: dict, variant):
        """Moves ``move`` (doc -> k) mentions of a unit to a new unit (a copy of the unit); returns (new uid, mentions moved)."""
        have = self.doc_counts(uid)
        mv = split_move(have, move)
        if not mv:
            return None, 0
        nu = self.new_unit(uid, variant)
        m = self.D['uid'] == uid
        D = self.D
        rows_new = []
        for d, k in mv.items():
            idx = D.index[m & (D['doc_id'] == d)]
            i = idx[0]
            if D.at[i, 'n'] == k:
                D.at[i, 'uid'] = nu
            else:
                D.at[i, 'n'] = D.at[i, 'n'] - k
                rows_new.append((nu, d, k))
        if rows_new:
            self.D = pd.concat([D, pd.DataFrame(rows_new, columns=['uid', 'doc_id', 'n'])], ignore_index=True)
        return nu, int(sum(mv.values()))

    def retarget(self, uid, ref_uid, tag, set_attrs=None):
        """Puts a unit on the point of another unit (copying its coordinates, kind, radius, candidate)."""
        U = self.U
        for c in ('grp', 'lat', 'lon', 'kind', 'rad', 'cand', 'qid', 'cname', 'pkey', 'mfold'):
            U.at[uid, c] = U.at[ref_uid, c]
        U.at[uid, 'src'] = 'name_exonym'
        U.at[uid, 'conf'] = 'high'
        U.at[uid, 'placed'] = True
        U.at[uid, 'fx'] = (U.at[uid, 'fx'] + '|' if U.at[uid, 'fx'] else '') + tag
        if set_attrs:
            g = U.at[ref_uid, 'grp']
            self.G_set.setdefault(g, {}).update(set_attrs)

    def unplace(self, uid, tag):
        self.U.at[uid, 'placed'] = False
        self.U.at[uid, 'fx'] = (self.U.at[uid, 'fx'] + '|' if self.U.at[uid, 'fx'] else '') + tag

    # ---- groups
    def group_table(self, floor=False):
        U = self.U
        n = self.unit_n().reindex(U.index).fillna(0).astype(int)
        pl = U[U['placed'] & (n > 0)].copy()
        pl['n'] = n[pl.index]
        pl['_new'] = (pl['run'] == 'v3').astype(int)             # v3: the units of the earlier layer lead their point (coordinates, kind, radius), the new sets only add mentions
        pl = pl.sort_values(['_new', 'n', 'conf', 'form'], ascending=[True, False, True, True])
        G = pl.groupby('grp').agg(n=('n', 'sum'), lead=('n', lambda s: s.index[0]), units=('n', 'size'))
        lead = pl.loc[G['lead']]
        for c in ('lat', 'lon', 'kind', 'rad', 'cand', 'qid', 'cname', 'mfold'):
            G[c] = lead[c].to_numpy()
        for g, st in self.G_set.items():
            if g in G.index:
                for k, v in st.items():
                    if k in ('r', 'r_merge'):
                        G.at[g, 'rad'] = 0 if v is None else v
                    elif k in ('lat', 'lon', 'kind'):
                        G.at[g, k] = v
        if floor:
            G['rad'] = [radius_floor(k, r) for k, r in zip(G['kind'], G['rad'])]
            for g, st in self.G_set.items():           # a radius fixed by the sweep (23 Oct) is exact: no 40 km floor
                if g in G.index and st.get('r_exact') and G.at[g, 'kind'] == 'a':
                    G.at[g, 'rad'] = int(round(st['r'])) if st.get('r') else radius_floor('a', 0)
        return G, pl


def make_units(F, run):
    """Units from a forms_placed table (columns of ``geo/points/forms_placed.parquet``)."""
    u = pd.DataFrame({
        'run': run, 'lang': F['lang'].to_numpy(), 'key': F['key'].to_numpy(), 'variant': '', 'form': F['form'].to_numpy(),
        'placed': (F['fate'] == 'placed').to_numpy(), 'grp': np.where(F['pid'] >= 0, run + ':' + F['pid'].astype(str), ''),
        'lat': F['gc_lat'].to_numpy(), 'lon': F['gc_lon'].to_numpy(), 'kind': F['kind'].fillna('').to_numpy(),
        'rad': F['gc_radius_km'].fillna(0).to_numpy(), 'src': F['gc_source'].to_numpy(), 'conf': F['gc_conf'].to_numpy(),
        'cand': F['gc_cand'].fillna('').to_numpy(), 'qid': F['gc_qid'].where(F['gc_qid'].notna(), None).to_numpy(), 'cname': F['gc_cand_name'].fillna(F['form']).to_numpy(),
        'fate': F['fate'].to_numpy(), 'fx': ''})
    u['pkey'] = [pkey_of(q, c, la, lo) for q, c, la, lo in zip(u['qid'], u['cand'], u['lat'], u['lon'])]
    u['mfold'] = u['cname'].map(fold)
    return u.reset_index(drop=True)


def load_state(W: Path):
    F1 = pd.read_parquet(W / 'forms_placed.parquet')
    FE = pd.read_parquet(W / 'es/forms_placed.parquet')
    U1 = make_units(F1, 'v1')
    UE = make_units(FE, 'es')
    UE.index = UE.index + len(U1)
    U = pd.concat([U1, UE])
    U['grp0'] = U['grp']
    D1 = pd.read_parquet(W / 'doc_forms.parquet')
    DE = pd.read_parquet(W / 'es/doc_forms.parquet')
    ix = {('v1', lg, k): i for i, (lg, k) in zip(U1.index, zip(U1['lang'], U1['key']))}
    ix.update({('es', lg, k): i for i, (lg, k) in zip(UE.index, zip(UE['lang'], UE['key']))})
    D1['uid'] = [ix[('v1', lg, k)] for lg, k in zip(D1['lang'], D1['key'])]
    DE['uid'] = [ix[('es', lg, k)] for lg, k in zip(DE['lang'], DE['key'])]
    D = pd.concat([D1[['uid', 'doc_id', 'n']], DE[['uid', 'doc_id', 'n']]], ignore_index=True)
    docs = pd.concat([pd.read_parquet(W / 'docs_v1.parquet', columns=['doc_id', 'collection', 'year', 'lang']),
                      pd.read_parquet(W / 'es/docs_es.parquet', columns=['doc_id', 'collection', 'year', 'lang'])]).drop_duplicates('doc_id').set_index('doc_id')
    return U, D, docs


def stage_baseline(S: State):
    """The v1 layer: every unit of the v1 run (Spanish included) in its v1 point. The es-run units are set aside (not placed) until the Spanish step."""
    S.U.loc[S.U['run'] == 'es', 'placed'] = False
    return S.step('baseline v1 (reconstruction)')


def stage_nakala(S: State):
    coll = S.docs['collection']
    nak = set(coll.index[coll == 'Nakala'])
    m = S.D['doc_id'].isin(nak)
    un = S.D.loc[m, 'uid']
    placed = S.U['placed']
    moved = int(S.D.loc[m & S.D['uid'].map(placed).fillna(False).astype(bool), 'n'].sum())
    by_lang = S.D[m & S.D['uid'].map(placed).fillna(False).astype(bool)].assign(lang=lambda d: d['uid'].map(S.U['lang'])).groupby('lang')['n'].sum().to_dict()
    S.D = S.D[~m].reset_index(drop=True)
    return S.step('11 Nakala dropped', mentions_removed=moved, by_lang=by_lang, documents=len(nak))


def stage_spanish(S: State):
    """Replaces the Spanish part: the v1 Spanish units go, the es-run units (mDeBERTa, agreement >= 2, confidence >= 0.5) come, attached to a v1 point when the same place."""
    U = S.U
    old = (U['run'] == 'v1') & (U['lang'] == 'es')
    n = S.unit_n().reindex(U.index).fillna(0)
    old_pl = int(n[old & U['placed']].sum())
    old_groups_before = S.snapshot()['points']
    U.loc[old, 'placed'] = False
    new = (U['run'] == 'es') & (U['fate'] == 'placed')
    U.loc[new, 'placed'] = True
    nn = int(n[new].sum())
    # attach each es point (group es:<pid>) to a v1 point: same Wikidata / GeoNames id / coordinates, or same modern name within 10 km of the v1 point
    G, pl = S.group_table()
    v1_units = pl[pl['run'] == 'v1']
    pk = {}
    for uid, g, k in zip(v1_units.index, v1_units['grp'], v1_units['pkey']):
        pk.setdefault(k, g)
    gv1 = G[G.index.str.startswith('v1:')]
    by_name = defaultdict(list)
    for g, mf, la, lo in zip(gv1.index, gv1['mfold'], gv1['lat'], gv1['lon']):
        by_name[mf].append((la, lo, g))
    es_units = pl[pl['run'] == 'es'].sort_values('n', ascending=False)
    es_target = {}
    for g, d in es_units.groupby('grp', sort=False):
        tgt = None
        for _, r in d.iterrows():
            if r['pkey'] in pk:
                tgt = pk[r['pkey']]
                break
        if tgt is None:
            for _, r in d.iterrows():
                for (la, lo, gg) in by_name.get(r['mfold'], []):
                    if float(hav_km(r['lat'], r['lon'], la, lo)) <= KM_SAME_NAME:
                        tgt = gg
                        break
                if tgt:
                    break
        es_target[g] = tgt
    attached = sum(1 for v in es_target.values() if v)
    U.loc[new, 'grp'] = [es_target.get(g) or g for g in U.loc[new, 'grp']]
    return S.step('Spanish replaced (mDeBERTa, agreement >= 2, conf >= 0.5)', old_spanish_mentions_placed=old_pl, new_spanish_mentions_placed=nn,
                  es_groups=len(es_target), es_groups_joined_to_v1_points=attached, points_before_step=old_groups_before)


def stage_nice(S: State, ctx):
    """Correction 3: Nice / Nyce (and the forms of the same point) in a council context -> İznik."""
    U = S.U
    tgt = S.uid_of(*NICE_TARGET, run='v1')
    S.G_set.setdefault(U.at[tgt, 'grp'], {}).update({'name': 'Nicaea', 'modern': 'İznik'})
    c = ctx[ctx['rule'] == 'council']
    tot = 0
    seen = 0
    for (run, lg, key), d in c.groupby(['run', 'lang', 'key']):
        uid = S.uid_of(lg, key, run=run)
        if uid is None or not U.at[uid, 'placed']:
            continue
        if float(hav_km(U.at[uid, 'lat'], U.at[uid, 'lon'], *NICE)) > 30:
            continue
        seen += int(d['n_total'].sum())
        mv = {r.doc_id: int(r.n_match) for r in d.itertuples() if r.n_match > 0}
        nu, k = S.move_mentions(uid, mv, 'ctx:council')
        if nu is not None:
            S.retarget(nu, tgt, 'nice->iznik')
            tot += k
    return S.step('3 Nice / Nyce in a council context -> Iznik', mentions_on_nice_forms=seen, mentions_moved=tot)


def stage_antioch(S: State):
    """Correction 4: Latin Antiochia* placed away from Antakya -> Antakya."""
    U = S.U
    tgt = S.uid_of(*ANTAKYA, run='v1')
    S.G_set.setdefault(U.at[tgt, 'grp'], {}).update({'name': 'Antioch', 'modern': 'Antakya'})
    n = S.unit_n().reindex(U.index).fillna(0)
    m = U['placed'] & (U['lang'] == 'la') & U['key'].str.match(ANTIOCH_LA) & (U['grp'] != U.at[tgt, 'grp'])
    k = 0
    forms = []
    for uid in U.index[m]:
        k += int(n[uid])
        forms.append(f'{U.at[uid, "form"]} ({int(n[uid])})')
        S.retarget(uid, tgt, 'antiochia->antakya')
    return S.step('4 Latin Antiochia* -> Antakya', mentions_moved=k, forms=forms)


def explicit_keys():
    ks = set()
    for _label, forms, _t_, _x in TABLE_EDITS:
        ks |= set(forms)
    return ks


def stage_pper(S: State, pper):
    """Correction 5: no point for a form whose mean person probability is >= 0.9, unless the form comes from the exonym table (or one of the table corrections above)."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0)
    pp = {(r, lg, k): p for r, lg, k, p in zip(pper['run'], pper['lang'], pper['key'], pper['p_per'])}
    ex = explicit_keys() | {(lg, k) for ks in CONTEXT_KEYS.values() for k in ks for lg in ('en', 'fr', 'it', 'la', 'de', 'es')}
    hit = []
    for uid in U.index[U['placed']]:
        if U.at[uid, 'variant'] or U.at[uid, 'fx']:
            continue
        p = pp.get((U.at[uid, 'run'], U.at[uid, 'lang'], U.at[uid, 'key']))
        if p is not None and p >= PPER_MAX and U.at[uid, 'src'] != 'name_exonym' and (U.at[uid, 'lang'], U.at[uid, 'key']) not in ex:
            hit.append(uid)
    mm = int(n[hit].sum())
    top = sorted(((int(n[u]), U.at[u, 'form'], U.at[u, 'lang']) for u in hit), reverse=True)[:15]
    by_run = {r: [sum(1 for u in hit if U.at[u, 'run'] == r), int(sum(n[u] for u in hit if U.at[u, 'run'] == r))] for r in ('v1', 'es')}
    for uid in hit:
        S.unplace(uid, 'person')
    return S.step('5 person-like forms (mean p_per >= 0.9) unplaced', forms_unplaced=len(hit), mentions_removed=mm, by_run_forms_mentions=by_run, top=top)


def stage_idf(S: State, ctx):
    """Correction 6: Isle / Ile de France -> Ile-de-France (49.0, 2.5, 60 km) unless a Maurice / Mauritius cue is in the text."""
    U = S.U
    c = ctx[ctx['rule'] == 'idf']
    maur = S.uid_of(*MAURITIUS, run='v1')
    n = S.unit_n().reindex(U.index).fillna(0)
    tot = 0
    kept_maur = 0
    for (run, lg, key), d in c.groupby(['run', 'lang', 'key']):
        uid = S.uid_of(lg, key, run=run)
        if uid is None or not U.at[uid, 'placed']:
            continue
        mv = {r.doc_id: int(r.n_match) for r in d.itertuples() if r.n_match > 0}
        if mv:
            nu, k = S.move_mentions(uid, mv, 'ctx:mauritius')
            if nu is not None:
                S.retarget(nu, maur, 'idf-mauritius')
                kept_maur += k
    # every other Isle / Ile de France form -> the new point
    grp = 'fix:idf'
    S.G_set[grp] = {'name': IDF['name'], 'lat': IDF['lat'], 'lon': IDF['lon'], 'kind': IDF['kind'], 'r': IDF['r']}
    n = S.unit_n().reindex(S.U.index).fillna(0)
    for uid in list(U.index[U['placed'] & U['key'].isin(IDF_KEYS) & (U['variant'] == '')]):
        tot += int(n[uid])
        U.at[uid, 'grp'] = grp
        for c_, v in (('lat', IDF['lat']), ('lon', IDF['lon']), ('kind', 'a'), ('rad', IDF['r']), ('src', 'name_exonym'), ('conf', 'high'), ('cand', 'exonym:Ile-de-France'),
                      ('qid', None), ('cname', IDF['name'])):
            U.at[uid, c_] = v
        U.at[uid, 'fx'] = 'idf'
    return S.step('6 Isle / Ile de France -> Ile-de-France', mentions_moved=tot, mentions_kept_on_mauritius=kept_maur)


def stage_blacklist(S: State):
    """Correction 8: Uti, Duron unplaced; Megalopolis (en) no longer Mecklenburg."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0)
    rows = []
    for uid in U.index[U['placed']]:
        k, lg = U.at[uid, 'key'], U.at[uid, 'lang']
        if k in BLACKLIST or (lg, k) in MECKLENBURG_BAD:
            rows.append((U.at[uid, 'form'], lg, int(n[uid])))
            S.unplace(uid, 'blacklist')
    return S.step('8 blacklist Uti, Duron; Megalopolis out of Mecklenburg', forms=rows, mentions_removed=sum(r[2] for r in rows))


def stage_table(S: State, ctx):
    """Section 5 items 2 and 7: table edits (spelling variants and regions / rivers taken for a village)."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0)
    out = {}
    for label, forms, target, extra in TABLE_EDITS:
        kind, ref = target
        if kind == 'ref':
            ruid = S.uid_of(*ref, run='v1')
            if ruid is None:
                raise RuntimeError(f'reference form {ref} not found for {label}')
        moved = placed_new = 0
        fl = []
        for lg, key in forms:
            for uid in U.index[(U['lang'] == lg) & (U['key'] == key) & (U['variant'] == '')]:
                if n[uid] == 0 or (U.at[uid, 'run'] == 'v1' and lg == 'es'):
                    continue                     # the old Spanish units are out
                was, old_name, old_grp = bool(U.at[uid, 'placed']), str(U.at[uid, 'cname']), U.at[uid, 'grp']
                if kind == 'ref':
                    S.retarget(uid, ruid, 'table:' + label.split(' ')[0].lower(), extra.get('set'))
                else:
                    g = f'fix:{ref}'
                    st = dict(extra['set'])
                    S.G_set[g] = st
                    U.at[uid, 'grp'] = g
                    for c_, v in (('lat', st['lat']), ('lon', st['lon']), ('kind', st['kind']), ('rad', 0), ('src', 'name_exonym'), ('conf', 'high'), ('placed', True),
                                  ('cand', f'exonym:{st["name"]}'), ('qid', None), ('cname', st['name'])):
                        U.at[uid, c_] = v
                    U.at[uid, 'fx'] = 'table:' + ref
                changed = (not was) or U.at[uid, 'grp'] != old_grp or (extra.get('set') or {}).get('kind') not in (None,)
                if not was:
                    placed_new += int(n[uid])
                if (not was) or U.at[uid, 'grp'] != old_grp:
                    moved += int(n[uid])
                fl.append({'form': U.at[uid, 'form'], 'lang': lg, 'n': int(n[uid]), 'was': (old_name if was else 'sans point'), 'moved': bool((not was) or U.at[uid, 'grp'] != old_grp)})
        out[label] = {'mentions_moved': moved, 'of_which_newly_placed': placed_new, 'forms': fl}
    return S.step('2/7 table edits (variants, regions, rivers)', edits=out, mentions_moved=sum(v['mentions_moved'] for v in out.values()),
                  newly_placed=sum(v['of_which_newly_placed'] for v in out.values()))


def same_name(mf_a, mf_b, sk_a, sk_b) -> bool:
    """The v1 second-pass test: same folded modern name, same spelling skeleton, or the same first 4 letters with a shared 5-letter start."""
    a_, b_ = str(mf_a), str(mf_b)
    if not a_ or not b_:
        return False
    return a_ == b_ or (sk_a and sk_a == sk_b) or (len(a_) >= 4 and len(b_) >= 4 and (a_.startswith(b_[:5]) or b_.startswith(a_[:5])) and a_[:4] == b_[:4])


def stage_floor(S: State):
    """Item 9: minimum radius of 40 km for country / province areas (kind a); other kinds carry no radius."""
    G0, _ = S.group_table(floor=False)
    G1, _ = S.group_table(floor=True)
    a = G0['kind'] == 'a'
    low = a & (G0['rad'] < MIN_AREA_R)
    return S.step('9a radius floor 40 km', areas=int(a.sum()), areas_raised=int(low.sum()), mentions_in_raised_areas=int(G0.loc[low, 'n'].sum()),
                  areas_with_radius_5_or_less=int((a & (G0['rad'] <= 5)).sum()))


def stage_merge(S: State, keys_of, step_name='9b merge of duplicate points'):
    """Item 9: merge the points that are the same place under known equivalent names (Wikidata id, exonym table, curated pairs) or the same modern name."""
    import numpy as np
    G, pl = S.group_table(floor=True)
    gi = list(G.index)
    N = len(gi)
    old_g = set(pl.loc[pl['run'] != 'v3', 'grp'])
    pl_keys = pl[(pl['run'] != 'v3') | ~pl['grp'].isin(old_g)]          # v3: the name keys of a point of the earlier layer come from its earlier forms only
    top = pl_keys.sort_values('n', ascending=False).groupby('grp').head(12)
    forms_of = defaultdict(list)
    for g, f in zip(top['grp'], top['form']):
        forms_of[g].append(f)
    cache = {}

    def ks(x):
        if x not in cache:
            try:
                k = keys_of(x)
                cache[x] = k | {y[3:] for y in k if y.startswith('ex:')}      # the exonym table's own English name is a key too (Lüttich -> ex:liege matches Liège)
            except Exception:
                cache[x] = set()
        return cache[x]

    GT = None
    try:
        from ratlas import geotext as GT  # noqa: N811
    except Exception:
        pass
    keys, eq, skel = [], [], []
    for g in gi:
        kk = set(ks(re.sub(r'\s*\(.*?\)', '', str(G.at[g, 'cname'])).strip() or str(G.at[g, 'cname'])))
        for f in forms_of[g]:
            kk |= ks(f)
        keys.append(kk)
        cls = {equiv_class(G.at[g, 'cname'])} | {equiv_class(f) for f in forms_of[g][:5]}
        eq.append({c for c in cls if c is not None})
        skel.append(GT.skeleton(str(G.at[g, 'mfold'])) if GT and G.at[g, 'mfold'] else '')
    lat, lon = G['lat'].to_numpy(float), G['lon'].to_numpy(float)
    rad = np.where(G['kind'].to_numpy() == 'a', G['rad'].to_numpy(float), 0.0)
    qid = G['qid'].to_numpy(object)
    kinds = G['kind'].to_numpy(object)
    mf = G['mfold'].to_numpy(object)
    nn = G['n'].to_numpy()
    uf = UnionFind()
    pairs = []
    cand_pairs = []
    for i in range(N - 1):
        d = hav_km(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
        rmax = np.maximum(rad[i], rad[i + 1:])
        thr_eq = np.where(rmax > 0, np.maximum(KM_EQUIV_CITY, AREA_SHARE * rmax), KM_EQUIV_CITY)
        thr_same = np.where(rmax > 15, np.maximum(KM_SAME_NAME, AREA_SHARE * rmax), KM_SAME_NAME)
        if isinstance(qid[i], str):                  # the same Wikidata id is the same place even when the two geocodes are some tens of km apart
            q_same = np.array([isinstance(x, str) and x == qid[i] for x in qid[i + 1:]])
            thr_cand = np.maximum(thr_same, np.where(q_same, QID_KM, 0.0))
        else:
            thr_cand = thr_same
        for q in np.flatnonzero(d <= thr_cand):
            j = i + 1 + int(q)
            why = None
            if isinstance(qid[i], str) and qid[i] == qid[j]:
                why = 'wikidata'
            elif d[q] <= thr_eq[q] and (eq[i] & eq[j]):
                why = 'curated'
            elif d[q] <= thr_eq[q] and (keys[i] & keys[j]):
                why = 'exonym/key'
            elif same_name(mf[i], mf[j], skel[i], skel[j]):
                why = 'same-name'
            if why and not (S.G_set.get(gi[i], {}).get('no_merge') or S.G_set.get(gi[j], {}).get('no_merge')):      # v29: a point set apart by forms_move stays apart
                cand_pairs.append((gi[i], gi[j], why, float(d[q])))
    # v3: a point of the new sets alone never bridges two points of the earlier layer. Old-old pairs and new-new pairs are united as before; each cluster of new points joins
    # at most ONE old point (the best proof, then the biggest, then the nearest).
    rank_why = {'wikidata': 0, 'curated': 1, 'exonym/key': 2, 'same-name': 3}
    cross = []
    for a_, b_, why, dist in cand_pairs:
        ao, bo = a_ in old_g, b_ in old_g
        if ao != bo:
            cross.append((a_, b_, why, dist, ao))
            continue
        uf.union(a_, b_)
        pairs.append((a_, b_, why, dist))
    best_cross = {}
    for a_, b_, why, dist, a_is_old in cross:
        og, ng = (a_, b_) if a_is_old else (b_, a_)
        root = uf.find(ng)
        key = (rank_why.get(why, 9), -float(G.at[og, 'n']), dist)
        if root not in best_cross or key < best_cross[root][0]:
            best_cross[root] = (key, og, ng, why, dist)
    for root, (key, og, ng, why, dist) in best_cross.items():
        uf.union(og, ng)
        pairs.append((og, ng, why, dist))
    clusters = defaultdict(list)
    for g in gi:
        clusters[uf.find(g)].append(g)
    rep = {}
    merged = []
    for root, mem in clusters.items():
        mem.sort(key=lambda g: (g not in old_g, -G.at[g, 'n']))
        r0 = mem[0]
        for g in mem:
            rep[g] = r0
        if len(mem) > 1:
            merged.append(mem)
            # explicit overrides: the biggest member that has one wins
            st = {}
            for g in reversed(mem):
                sg = S.G_set.get(g, {})
                if sg.get('sweep') and g != r0:          # a point set by the sweep that is absorbed by a bigger point does not change it
                    continue
                st.update(sg)
            if st:
                S.G_set[r0] = st
            # an area keeps the largest radius of its members
            if G.at[r0, 'kind'] == 'a' and 'r' not in S.G_set.get(r0, {}):
                rr = max(float(G.at[g, 'rad']) for g in mem if G.at[g, 'kind'] == 'a' and (g in old_g or not (set(mem) & old_g)))
                S.G_set.setdefault(r0, {})['r_merge'] = rr
    S.U['grp'] = S.U['grp'].map(lambda g: rep.get(g, g))
    S.merge_proof = pairs
    ex = []
    for mem in sorted(merged, key=lambda m: -sum(G.at[g, 'n'] for g in m))[:40]:
        ex.append({'names': [f'{G.at[g, "cname"]} ({int(G.at[g, "n"])})' for g in mem[:6]], 'mentions': int(sum(G.at[g, 'n'] for g in mem))})
    why_c = Counter(w for _a, _b, w, _d in pairs)
    return S.step(step_name, clusters_merged=len(merged), points_absorbed=int(sum(len(m) - 1 for m in merged)), by_proof=dict(why_c),
                  mentions_in_merged=int(sum(G.at[g, 'n'] for m in merged for g in m)), biggest=ex), merged


# ---------------------------------------------------------------------------------------------------------------------
# pass 2 (23 Oct): the verdicts of the sweep of the 351 points, as data (data/model_layer_overrides.json)
# ---------------------------------------------------------------------------------------------------------------------
OVERRIDES = Path(__file__).resolve().parent.parent / 'data' / 'model_layer_overrides.json'
SWEEP_SRC = 'name_exonym'          # the method tag of a point whose coordinates were set by the sweep (a curated name -> place table)


def load_overrides(path=OVERRIDES):
    """The entries of the overrides file. Each aims at a point of the 22 Oct output by the (run, lang, key, variant) of its most frequent form."""
    return json.loads(Path(path).read_text('utf-8'))['entries']


def _anchor_uid(S: State, anchor):
    run, lang, key, variant = anchor
    U = S.U
    ids = U.index[(U['run'] == run) & (U['lang'] == lang) & (U['key'] == key) & (U['variant'] == variant)]
    if not len(ids):
        raise RuntimeError(f'override anchor {anchor} not found')
    return int(ids[0])


def _units_of(S: State, g):
    U = S.U
    return [int(u) for u in U.index[(U['grp'] == g) & U['placed']]]


def _tag(S: State, uid, tag):
    S.U.at[uid, 'fx'] = (S.U.at[uid, 'fx'] + '|' if S.U.at[uid, 'fx'] else '') + tag


def regroup(S: State, uid, ref_uid, tag):
    """Puts a unit on the point of another unit (group, coordinates, kind, radius, candidate) and keeps its own method and confidence."""
    U = S.U
    for c in ('grp', 'lat', 'lon', 'kind', 'rad', 'cand', 'qid', 'cname', 'pkey', 'mfold'):
        U.at[uid, c] = U.at[ref_uid, c]
    U.at[uid, 'placed'] = True
    _tag(S, uid, tag)


def stage_sweep(S: State, entries):
    """Applies the sweep verdicts in this order: bad forms out, forms sent to another point (``forms_into``) or to a point of their own (``forms_move``), drop / unplace, splits by collection, merges, moves,
    relabels. ``split`` (audit of the homonyms, 23 Oct): the mentions of the documents of ``collections`` leave the point and stay unplaced."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0).astype(int)
    rec = {'clean': [], 'drop': [], 'unplace': [], 'split': [], 'merge': [], 'move': [], 'relabel': [], 'forms_into': [], 'forms_move': [], 'missing': []}

    def grp_of(anchor):
        return U.at[_anchor_uid(S, anchor), 'grp']

    def pt_name(e):
        return f"{e['name_v2']} ({e['id_v2']})"

    def drop_forms(e, forms, why):
        want = {fold(f) for f in forms}
        got = {}
        for uid in _units_of(S, grp_of(e['anchor'])):
            if fold(U.at[uid, 'form']) in want:
                got[U.at[uid, 'form']] = got.get(U.at[uid, 'form'], 0) + int(n[uid])
                S.unplace(uid, why)
        for f in forms:
            if fold(f) not in {fold(x) for x in got}:
                rec['missing'].append(f"{pt_name(e)}: forme {f!r} introuvable")
        return got

    for e in entries:                                   # 1. bad forms inside otherwise good points
        if e['action'] in ('clean', 'move') and e.get('remove_forms'):
            got = drop_forms(e, e['remove_forms'], 'sweep2:form')
            rec['clean'].append({'point': pt_name(e), 'forms': got, 'mentions': sum(got.values())})
    for e in entries:                                   # 2. forms that belong to another point
        fi = e.get('forms_into')
        if fi:
            ref = _anchor_uid(S, fi['into_anchor'])
            want = {fold(f) for f in fi['forms']}
            moved = {}
            for uid in _units_of(S, grp_of(e['anchor'])):
                if fold(U.at[uid, 'form']) in want:
                    moved[U.at[uid, 'form']] = int(n[uid])
                    regroup(S, uid, ref, 'sweep2:form_into')
            rec['forms_into'].append({'point': pt_name(e), 'into': fi['into_id_v2'], 'forms': moved, 'mentions': sum(moved.values())})
            for f in fi['forms']:
                if f not in moved:
                    rec['missing'].append(f"{pt_name(e)}: forme {f!r} introuvable")
    for e in entries:                                   # 2b. forms that become a point of their own (v29: Blackheath grouped into Blackwall, 5 km apart)
        if e['action'] != 'forms_move':
            continue
        g = grp_of(e['anchor'])
        want = {fold(f) for f in e['forms']}
        label, r = e['label'], (e.get('radius_km') if e['kind'] == 'a' else None)
        new_g = f"sweep2:forms_move:{label}"
        moved = {}
        for uid in _units_of(S, g):
            if fold(U.at[uid, 'form']) in want:
                moved[U.at[uid, 'form']] = int(n[uid])
                U.at[uid, 'grp'] = new_g
                U.at[uid, 'lat'], U.at[uid, 'lon'], U.at[uid, 'kind'] = float(e['lat']), float(e['lon']), e['kind']
                U.at[uid, 'rad'] = float(r or 0)
                U.at[uid, 'src'], U.at[uid, 'conf'] = SWEEP_SRC, 'high'
                U.at[uid, 'cand'], U.at[uid, 'qid'] = 'sweep2:' + label, None
                U.at[uid, 'cname'], U.at[uid, 'mfold'] = label, fold(label)
                U.at[uid, 'pkey'] = pkey_of(None, 'sweep2:' + label, float(e['lat']), float(e['lon']))
                _tag(S, uid, 'sweep2:forms_move')
        if moved:
            S.G_set[new_g] = {'lat': float(e['lat']), 'lon': float(e['lon']), 'kind': e['kind'], 'r': r, 'r_exact': True, 'sweep': True, 'name': label, 'no_merge': True}
        rec['forms_move'].append({'point': pt_name(e), 'forms': moved, 'mentions': sum(moved.values()), 'to': (e['lat'], e['lon'], e['kind'], r), 'label': label})
        for f in e['forms']:
            if f not in moved:
                rec['missing'].append(f"{pt_name(e)}: forme {f!r} introuvable")
    for e in entries:                                   # 3. not a place / too uncertain: no point
        if e['action'] in ('drop', 'unplace'):
            us = _units_of(S, grp_of(e['anchor']))
            tot = sum(int(n[u]) for u in us)
            for u in us:
                S.unplace(u, 'sweep2:' + e['action'])
            rec[e['action']].append({'point': pt_name(e), 'mentions': tot, 'forms': len(us)})
    for e in entries:                                   # 3b. a point that mixes two homonyms: the mentions of some collections lose the point
        if e['action'] != 'split':                      # (audit of 23 Oct: Newbury, Dedham ... of New England in Evans, the English towns elsewhere)
            continue
        cols = set(e['collections'])
        tot = 0
        for uid in _units_of(S, grp_of(e['anchor'])):
            have = S.doc_counts(uid)
            mv = {d: k for d, k in have.items() if d in S.docs.index and S.docs.at[d, 'collection'] in cols}
            nu, k = S.move_mentions(uid, mv, 'sweep3:split')
            if nu is not None:
                S.unplace(nu, 'sweep3:split')
                tot += k
        rec['split'].append({'point': pt_name(e), 'collections': sorted(cols), 'mentions': tot})
    for e in entries:                                   # 4. merges
        if e['action'] != 'merge':
            continue
        g, ref = grp_of(e['anchor']), _anchor_uid(S, e['into_anchor'])
        excl = {fold(f) for f in e.get('exclude_forms', [])}
        tot = out = 0
        for uid in _units_of(S, g):
            if fold(U.at[uid, 'form']) in excl:
                out += int(n[uid])
                S.unplace(uid, 'sweep2:merge_excluded')
            else:
                tot += int(n[uid])
                regroup(S, uid, ref, 'sweep2:merge')
        rec['merge'].append({'point': pt_name(e), 'into': f"{e['into_name_v2']} ({e['into_id_v2']})", 'mentions': tot, 'left_out': out})
    for e in entries:                                   # 5. moves
        if e['action'] != 'move':
            continue
        g = grp_of(e['anchor'])
        us = _units_of(S, g)
        label = e.get('label') or e['name_v2']
        r = e.get('radius_km') if e['kind'] == 'a' else None
        old = (round(float(U.at[us[0], 'lat']), 3), round(float(U.at[us[0], 'lon']), 3), U.at[us[0], 'kind']) if us else None
        for uid in us:
            U.at[uid, 'lat'], U.at[uid, 'lon'], U.at[uid, 'kind'] = float(e['lat']), float(e['lon']), e['kind']
            U.at[uid, 'rad'] = float(r or 0)
            U.at[uid, 'src'], U.at[uid, 'conf'] = SWEEP_SRC, 'high'
            U.at[uid, 'cand'], U.at[uid, 'qid'] = 'sweep2:' + label, None
            U.at[uid, 'cname'], U.at[uid, 'mfold'] = label, fold(label)
            U.at[uid, 'pkey'] = pkey_of(None, 'sweep2:' + label, float(e['lat']), float(e['lon']))
            _tag(S, uid, 'sweep2:move')
        st = S.G_set.setdefault(g, {})
        st.update({'lat': float(e['lat']), 'lon': float(e['lon']), 'kind': e['kind'], 'r': r, 'r_exact': True, 'sweep': True})
        if e.get('label'):
            st['name'] = e['label']
        if e.get('no_merge'):                           # v29: the point stays apart from the neighbours that the second merge would join it to
            st['no_merge'] = True
        rec['move'].append({'point': pt_name(e), 'mentions': sum(int(n[u]) for u in us), 'from': old, 'to': (e['lat'], e['lon'], e['kind'], r), 'label': e.get('label')})
    for e in entries:                                   # 6. display names (and kind)
        if e['action'] != 'relabel':
            continue
        g = grp_of(e['anchor'])
        st = S.G_set.setdefault(g, {})
        st['name'] = e['label']
        st['sweep'] = True
        if e.get('kind'):
            st.update({'kind': e['kind'], 'r': None, 'r_exact': True})
            for uid in _units_of(S, g):
                U.at[uid, 'kind'], U.at[uid, 'rad'] = e['kind'], 0.0
        for uid in _units_of(S, g):
            _tag(S, uid, 'sweep2:label')
        rec['relabel'].append({'point': pt_name(e), 'label': e['label']})
    S.sweep_rec = rec
    gone = sum(x['mentions'] for k in ('drop', 'unplace', 'split') for x in rec[k]) + sum(x['mentions'] for x in rec['clean']) + sum(x['left_out'] for x in rec['merge'])
    return S.step('P2 sweep verdicts (drop, unplace, clean, merge, move, relabel)', counts={k: len(v) for k, v in rec.items()}, mentions_removed=gone,
                  mentions_moved=sum(x['mentions'] for x in rec['move']), mentions_merged=sum(x['mentions'] for x in rec['merge']))


def match_editors(points, G, keys_of, km=30.0):
    """Same matching as ner2_layer_geo_match.py: Wikidata id or < 30 km and a shared name key."""
    import numpy as np
    glat = np.array([g['lat'] for g in G])
    glon = np.array([g['lon'] for g in G])
    gkeys = [keys_of(g['name']) for g in G]
    gr = np.array([float(g.get('r') or 0) for g in G])
    by_qid = defaultdict(list)
    for i, g in enumerate(G):
        if g.get('qid'):
            by_qid[g['qid']].append(i)
    hit = set()
    how = Counter()
    for p in points:
        d = hav_km(p['lat'], p['lon'], glat, glon)
        r_l = float(p.get('r') or 0)
        thr = np.where((r_l > 0) & (gr > 0), np.maximum(km, np.minimum(r_l, gr)), km)
        near = np.flatnonzero(d <= thr)
        pk = keys_of(p['gname'])
        for f in p['forms']:
            pk |= keys_of(f[0])
        cands = []
        if p.get('qid') and p['qid'] in by_qid:
            cands += [(int(i), 'wikidata') for i in by_qid[p['qid']]]
        cands += [(int(i), 'name') for i in near if gkeys[i] & pk]
        p.pop('ed', None)
        p.pop('ed_n', None)
        if cands:
            i, h = max(cands, key=lambda c: (c[1] == 'wikidata', G[c[0]]['n']))
            g = G[i]
            p['ed_n'] = len(cands)
            p['ed'] = {'id': g['id'], 'name': g['name'], 'km': round(float(d[i]), 1), 'how': h, 'n': g['n']}
            if p['k'] == 'a' and g.get('r') and g['r'] > r_l and not p.get('_r_fixed'):          # an area drawn with a small radius: the editors' radius for the same area
                p['r0'] = int(r_l)
                p['r'] = int(g['r'])
            hit.add(i)
            how[h] += 1
    return {'points': sum(1 for p in points if p.get('ed')), 'mentions': sum(p['n'] for p in points if p.get('ed')), 'by': dict(how), 'editor_points_hit': len(hit)}


EEBO_ID = re.compile(r'^[AB]\d{5}$')


def doc_link(doc_id: str, collection: str):
    """A link built from the document's own identifier (not checked online), as in ner2_layer_geo_points.doc_link."""
    from urllib.parse import quote
    stem = doc_id.split('/', 1)[-1]
    if EEBO_ID.match(stem) and (collection.startswith('EEBO') or collection.startswith('LatinEEBO')):
        return f'https://quod.lib.umich.edu/e/eebogroup/{stem}.0001.001'
    if collection.startswith('EvansTCP') and re.match(r'^N\d{5}$', stem):
        return f'https://quod.lib.umich.edu/e/evans/{stem}.0001.001'
    m = re.match(r'^WS_([a-z]{2})_(.+)$', stem)
    if m and 'Wikisource' in collection:
        return f'https://{m.group(1)}.wikisource.org/wiki/{quote(m.group(2).replace(" ", "_"))}'
    return None


EN_SHARE = 0.25          # English label when English texts give at least this share of the mentions of the point (or 5 mentions)
LABEL_FIX = {'Istanbul': 'Constantinople'}       # the table's modern name, when the texts write the historical one


def exonym_label_of(forms, lookup):
    """The curated English label of the exonym table for the most frequent forms of a point (None when the table knows none of them)."""
    c = Counter()
    for f, k, _l in forms[:8]:
        try:
            h = lookup(f)
        except Exception:
            h = None
        if h:
            c[h] += k
    return c.most_common(1)[0][0] if c else None


def label_for(forms, exonym_label=None, ed_name=None) -> str:
    """The display label of a point (the globe is in English).

    1. the English label of the exonym table when the texts write it (Spain, Genoa, Jerusalem) ;
    2. the editor layer's name when it is also an English written form of the point ;
    3. the most frequent English form when English texts carry a quarter of the point ;
    4. the most frequent written form, capitalised.
    Never a label with a comma, a parenthesis or 'Republic of'. ``forms`` = (surface, mentions, lang).
    """
    forms = list(forms)
    n = sum(k for _f, k, _l in forms)
    written = Counter()
    for f, k, _l in forms:
        written[fold(f)] += k
    en = [(f, k, l) for f, k, l in forms if l == 'en']
    n_en = sum(k for _f, k, _l in en)

    def fix(lab):
        alt = LABEL_FIX.get(lab)
        return alt if alt and written.get(fold(alt), 0) >= 3 else lab

    if exonym_label and label_ok(exonym_label) and written.get(fold(exonym_label), 0) >= 3:
        return fix(capitalise_label(exonym_label))
    if ed_name and label_ok(ed_name) and n_en and sum(k for f, k, _l in en if fold(f) == fold(ed_name)) >= max(3, 0.10 * n_en):
        return fix(capitalise_label(ed_name))
    if en and n_en >= max(5, EN_SHARE * n):
        lab = pick_label(en)
        if label_ok(lab):
            return fix(lab)
    return fix(pick_label(forms))


def assemble(S: State, titles: dict, top_docs=5, top_forms=12):
    """Final point rows (without editor matching) and the per-point document lists."""
    import numpy as np
    U, docs = S.U, S.docs
    G, pl = S.group_table(floor=True)
    D = S.D.merge(pl[['grp']], left_on='uid', right_index=True, how='inner')
    dm = docs
    D['year'] = D['doc_id'].map(dm['year'])
    D['coll'] = D['doc_id'].map(dm['collection']).fillna(D['doc_id'].str.split('/').str[0])
    D['qi'] = np.floor((D['year'] - 1450) / 25)
    D.loc[~((D['qi'] >= 0)), 'qi'] = -1
    D['qi'] = D['qi'].clip(upper=9).fillna(-1).astype(int)
    Q = D[D['qi'] >= 0].groupby(['grp', 'qi'])['n'].sum().unstack(fill_value=0).reindex(columns=range(10), fill_value=0)
    GD = D.groupby(['grp', 'doc_id'])['n'].sum().reset_index().sort_values(['grp', 'n', 'doc_id'], ascending=[True, False, True])
    ND = GD.groupby('grp')['doc_id'].nunique()
    GC = D.groupby(['grp', 'coll'])['n'].sum().reset_index().sort_values(['grp', 'n', 'coll'], ascending=[True, False, True])
    NC = GC.groupby('grp')['coll'].nunique()
    topc = GC.groupby('grp').head(3).groupby('grp')['coll'].apply(list)
    docs_by = {g: list(zip(d['doc_id'], d['n'])) for g, d in GD.groupby('grp')}
    pl = pl.sort_values(['grp', 'n', 'form'], ascending=[True, False, True])
    pts = []
    for g, d in pl.groupby('grp', sort=False):
        lead = G.loc[g]
        n = int(d['n'].sum())
        forms_all = [[f, int(k), lg] for f, k, lg in zip(d['form'], d['n'], d['lang'])]
        langs = Counter()
        srcs = Counter()
        for lg, k, sr in zip(d['lang'], d['n'], d['src']):
            langs[lg] += int(k)
            srcs[sr] += int(k)
        fx = sorted({t for x in d['fx'] for t in x.split('|') if t})
        prev = sorted({x for x in d['grp0'] if x})
        st = S.G_set.get(g, {})
        kind = lead['kind'] if lead['kind'] else 'u'
        pt = {'id': g, 'name': st.get('name'), '_forms_all': forms_all, 'gname': str(lead['cname']), 'lat': round(float(lead['lat']), 5), 'lon': round(float(lead['lon']), 5), 'k': kind,
              'n': n, 'nd': int(ND.get(g, 0)), 'q': [int(Q.at[g, i]) if g in Q.index else 0 for i in range(10)], 's': 'm', 'gs': srcs.most_common(1)[0][0],
              'conf': 'high' if (d['conf'] == 'high').all() else 'medium', 'qid': (lead['qid'] if isinstance(lead['qid'], str) else None),
              'gid': (lead['cand'][3:] if str(lead['cand']).startswith('gn:') else None), 'cand': (lead['cand'] or None), 'forms': forms_all[:top_forms], 'nforms': len(forms_all),
              'langs': dict(langs), 'c': int(NC.get(g, 0)), 'cols': topc.get(g, []), 'fx': fx, 'prev': prev[:8]}
        if kind == 'a' and lead['rad'] > 0:
            pt['r'] = int(round(lead['rad']))
        if st.get('modern'):
            pt['modern'] = st['modern']
        if 'r' in st:
            pt['_r_fixed'] = True
        pt['_g'] = g
        pts.append(pt)
    return pts, docs_by


def exonym_lookup():
    """form -> label of the geocoder's exonym table (English label) or None."""
    from ratlas import exonyms as EX
    from ratlas.merge import _norm
    from ratlas.namegate import small_letters

    def look(form):
        h = EX.lookup(_norm(small_letters(form).replace('_', ' ').replace('-', ' ')))
        return h[3] if h else None
    return look


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def check_baseline(S: State, old_places):
    """The v1 layer rebuilt from the tables must be the v1 layer: same points, mentions, and the same count per point."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0)
    pl = U['placed'] & (n > 0)
    mine = n[pl].groupby(U.loc[pl, 'grp']).sum()
    bad = 0
    for p in old_places:
        g = 'v1:' + p['id'].split(':')[1]
        if int(mine.get(g, -1)) != p['n']:
            bad += 1
    return {'old_points': len(old_places), 'old_mentions': int(sum(p['n'] for p in old_places)), 'rebuilt_points': int(len(mine)), 'rebuilt_mentions': int(mine.sum()), 'points_with_different_n': bad}


def run(args):
    W = Path(args.work)
    t0 = time.time()
    U, D, docs = load_state(W)
    S = State(U, D, docs)
    old = json.loads(Path(args.old_layer).read_text('utf-8'))
    ctx = pd.read_parquet(W / 'fix/ctx.parquet')
    pper = pd.read_parquet(W / 'fix/form_pper.parquet')
    titles = json.loads((W / 'fix/titles.json').read_text('utf-8'))
    GM = load_module(Path(__file__).resolve().parent / 'ner2_layer_geo_match.py', 'ner2_layer_geo_match_mod')
    stage_baseline(S)
    base_check = check_baseline(S, old['places'])
    _t(f'baseline check {base_check}')
    stage_nakala(S)
    stage_spanish(S)
    stage_nice(S, ctx)
    stage_antioch(S)
    stage_pper(S, pper)
    stage_idf(S, ctx)
    stage_blacklist(S)
    stage_table(S, ctx)
    stage_floor(S)
    _, merged = stage_merge(S, GM.keys_of)
    if getattr(args, 'overrides', None) != 'none':
        stage_sweep(S, load_overrides(args.overrides or OVERRIDES))
        _, merged2 = stage_merge(S, GM.keys_of, step_name='P2 merge again after the sweep')
        merged = merged + merged2
    pts, docs_by = assemble(S, titles)
    pts.sort(key=lambda p: (-p['n'], p['id']))
    G = json.loads(Path(args.globe).read_text('utf-8'))['places']
    for p in pts:
        p['_old_id'] = p['id']
    for i, p in enumerate(pts):
        p['id'] = f'ml2:{i}'
    em = match_editors(pts, G, GM.keys_of)
    S.step('editor matching', **em)
    lookup = exonym_lookup()
    for p in pts:
        if not p['name']:
            p['name'] = label_for(p['_forms_all'], exonym_label_of(p['_forms_all'], lookup), (p.get('ed') or {}).get('name'))
        p['tier'] = tier_of(p['n'])
        p['prev'] = [('ml:' + x[3:] if x.startswith('v1:') else 'mles:' + x[3:]) if x[:3] in ('v1:', 'es:') else x for x in p['prev']]
    for p in pts:
        p.pop('_old_id', None)
        p.pop('_r_fixed', None)
        p.pop('_forms_all', None)
    ctx_live = ctx[~((ctx['run'] == 'v1') & (ctx['lang'] == 'es'))]          # the v1 Spanish units are out
    X = {'ctx': {r: [int(d['n_total'].sum()), int(d['n_match'].sum())] for r, d in ctx_live.groupby('rule')}, 'old': old, 'args': vars(args)}
    return S, pts, docs_by, titles, base_check, em, merged, X


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True, help='folder with the v1 and es tables copied from Hypatia')
    ap.add_argument('--old-layer', required=True)
    ap.add_argument('--globe', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--docs-out', default=None)
    ap.add_argument('--stats-out', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--overrides', default=None, help='sweep verdicts (default data/model_layer_overrides.json); "none" = the 22 Oct layer')
    ap.add_argument('--prev', default=None, help='the previous output (model_layer_v2_prev.json), for the before / after of pass 2')
    ap.add_argument('--report-pass2', default=None, help='corrections_v2.md: the section "Passe 2 (23 oct.)" is added or replaced')
    args = ap.parse_args()
    S, pts, docs_by, titles, base_check, em, merged, X = run(args)
    write_outputs(args, S, pts, docs_by, titles, base_check, em, merged, X)
    if args.report_pass2:
        prev = json.loads(Path(args.prev).read_text('utf-8')) if args.prev else None
        patch_report_pass2(Path(args.report_pass2), make_report_pass2(S, pts, prev, em))


def write_outputs(args, S, pts, docs_by, titles, base_check, em, merged, X):
    old = X['old']
    docs = S.docs
    dinfo = {}

    def info(did):
        if did not in dinfo:
            coll = docs['collection'].get(did) or did.split('/')[0]
            y = docs['year'].get(did)
            dinfo[did] = {'id': did, 'title': titles.get(did) or None, 'year': int(y) if y == y and y is not None else None, 'collection': coll, 'url': doc_link(did, str(coll))}
        return dinfo[did]

    full = {}
    for p in pts:
        lst = docs_by.get(p['_g'], [])
        full[p['id']] = [[d, int(n)] for d, n in lst]
        for d, _n in lst:
            info(d)
        p['docs'] = [{**info(d), 'n': int(n)} for d, n in lst[:5]]
        del p['_g']
    steps = [{'step': x['step'], **{k: v for k, v in x.items() if k not in ('step',)}} for x in S.log]
    hdr = {'title': 'Renascor Atlas, places found by the NER model on untagged texts (layer m), version 2: corrected', 'generated': time.strftime('%Y-%m-%d'), 'period': [1450, 1700],
           'quarters': list(range(1450, 1700, 25)),
           'fields': {'n': 'mentions', 'nd': 'documents', 'q': 'mentions per quarter-century by document year', 'k': 'kind c / a / n / b / u as in the globe (n = river, sea, mountain)',
                      'r': 'for areas: radius in km (at least 40)', 's': '"m" = found by the model (never an editor-tagged place)',
                      'gs': 'method of the coordinates', 'conf': 'high / medium (geocoder confidence)', 'forms': '[written form, mentions, language], 12 at most',
                      'tier': '"main" from 11 mentions, "rare" below (about 3 in 4 right)', 'name': 'label: most frequent written form, capitalised (English editor name when it is also a written form in English texts)',
                      'gname': 'the geocoder label (not shown)', 'fx': 'corrections that touched the point', 'prev': 'ids of the v1 points it comes from (ml:<n>, mles:<n> for the Spanish rerun)',
                      'docs': 'top 5 documents; the full list is in the documents file', 'ed': 'the editor-tagged point that coincides (same Wikidata id or < 30 km and a shared name key)'},
           'places': len(pts), 'mentions': int(sum(p['n'] for p in pts)), 'main': sum(p['tier'] == 'main' for p in pts), 'rare': sum(p['tier'] == 'rare' for p in pts),
           'coincide_with_editor_points': em, 'note': 'predictions of the model, no human review; a separate layer, distinct from the editors places', 'steps': steps}
    Path(args.out).write_text(json.dumps({'header': hdr, 'places': pts}, ensure_ascii=False, separators=(',', ':')), 'utf-8')
    if args.docs_out:
        Path(args.docs_out).write_text(json.dumps({'header': {'note': 'documents of each point of model_layer_v2.json: [doc id, mentions], most mentions first; docs = title, year, collection, url'},
                                                   'docs': dinfo, 'points': full}, ensure_ascii=False, separators=(',', ':')), 'utf-8')
    if args.report:
        Path(args.report).write_text(make_report(S, pts, base_check, em, merged, X), 'utf-8')
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps({'baseline_check': base_check, 'steps': steps, 'editor': em, 'merged_clusters': [[S.U.loc[S.U['grp'] == m[0], 'form'].head(1).tolist()] for m in merged[:0]]},
                                                   ensure_ascii=False, indent=1, default=str), 'utf-8')


def fr(n) -> str:
    return f'{int(n):,}'.replace(',', '\u202f')


def pct(a, b) -> str:
    return f'{100 * a / b:.1f}'.replace('.', ',') + ' %' if b else '-'


STEP_FR = {
    '11 Nakala dropped': 'point 11 : collection Nakala retirée',
    'Spanish replaced (mDeBERTa, agreement >= 2, conf >= 0.5)': 'espagnol remplacé (mDeBERTa, accord >= 2, confiance >= 0,5)',
    '3 Nice / Nyce in a council context -> Iznik': 'correction 3 : Nice du concile -> İznik',
    '4 Latin Antiochia* -> Antakya': 'correction 4 : Antiochia latin -> Antakya',
    '5 person-like forms (mean p_per >= 0.9) unplaced': 'correction 5 : noms de personnes (p_per >= 0,9) sans point',
    '6 Isle / Ile de France -> Ile-de-France': 'correction 6 : Isle de France -> Île-de-France',
    '8 blacklist Uti, Duron; Megalopolis out of Mecklenburg': 'correction 8 : Uti, Duron, Megalopolis',
    '2/7 table edits (variants, regions, rivers)': 'points 2 et 7 : variantes, régions, fleuves (9 entrées)',
    '9a radius floor 40 km': 'point 9 : rayon minimal 40 km',
    '9b merge of duplicate points': 'point 9 : fusion des doublons',
    'editor matching': 'coïncidence avec les points des éditeurs (recalculée)',
}


def make_report(S: State, pts, base_check, em, merged, X) -> str:
    """The French report: what each correction changed, before / after."""
    old = X['old']['places']
    log = {x['step']: x for x in S.log}
    steps = S.log
    b0 = steps[0]['after']
    last = steps[-1]['after']

    def row(x):
        a, b = x['before'], x['after']
        d = lambda k: f'{fr(a[k])} -> {fr(b[k])} ({b[k] - a[k]:+,})'.replace(',', '\u202f').replace('\u202f)', ')')  # noqa: E731
        return f"| {STEP_FR.get(x['step'], x['step'])} | {d('points')} | {d('mentions')} | {fr(b['main'])} / {fr(b['rare'])} |"

    o_main = [p for p in old if p['n'] >= TIER_MAIN]
    n_main = [p for p in pts if p['tier'] == 'main']
    n_rare = [p for p in pts if p['tier'] == 'rare']
    bad_re = re.compile(r'[,()]|republic|kingdom of|^the ', re.I)
    old_bad = [p for p in old if bad_re.search(p['name'])]
    new_bad = [p for p in pts if bad_re.search(p['name'])]
    small_old = [p for p in old if p['k'] == 'a' and p.get('r', 0) <= 10 and p['n'] > 50]
    small_new = [p for p in pts if p['k'] == 'a' and p.get('r', 0) <= 10 and p['n'] > 50]
    by_form = defaultdict(set)
    for p in pts:
        for f in p['forms']:
            by_form[fold(f[0])].add(p['id'])
    spec = [('Napoli / Naples / Neapel', ['napoli', 'naples', 'neapel']), ('Firenze / Florence', ['firenze', 'florence']), ('Fraunce / Republic of France / France / Frankreich', ['fraunce', 'france', 'frankreich']),
            ('Venezia / Venice', ['venezia', 'venice']), ('Konstantinopel / Istanbul', ['konstantinopel', 'istanbul', 'constantinople']), ('Italia / Italy', ['italia', 'italy'])]
    name_of = {p['id']: p['name'] for p in pts}
    spec_lines = []
    for lab, names in spec:
        ids = [by_form.get(fold(nm), set()) for nm in names]
        common = set.intersection(*[i for i in ids if i]) if any(ids) else set()
        spec_lines.append(f"| {lab} | {'oui, un seul point : ' + ', '.join(sorted(common)[:2]) + ' (' + name_of[sorted(common)[0]] + ')' if common else 'non'} |")
    t5 = log['5 person-like forms (mean p_per >= 0.9) unplaced']
    t3 = log['3 Nice / Nyce in a council context -> Iznik']
    t4 = log['4 Latin Antiochia* -> Antakya']
    t6 = log['6 Isle / Ile de France -> Ile-de-France']
    t8 = log['8 blacklist Uti, Duron; Megalopolis out of Mecklenburg']
    t2 = log['2/7 table edits (variants, regions, rivers)']
    t9a = log['9a radius floor 40 km']
    t9b = log['9b merge of duplicate points']
    tn = log['11 Nakala dropped']
    te = log['Spanish replaced (mDeBERTa, agreement >= 2, conf >= 0.5)']
    ctx = X['ctx']
    sp = {p['name']: p for p in pts}

    L = []
    A = L.append
    A('# Couche NER v2 : la couche corrigée (22 oct. 2026)')
    A('')
    A('Rien n\'est commité, publié ni envoyé. Aucune relecture humaine : ce sont des prédictions du modèle, corrigées par des règles. Script : `scripts/ner2_layer_fix.py`. Il peut être relancé tel quel.')
    A('')
    A('## En bref')
    A('')
    A('| | couche v1 | couche v2 |')
    A('|---|---|---|')
    A(f"| points | {fr(len(old))} | **{fr(len(pts))}** |")
    A(f"| mentions placées | {fr(sum(p['n'] for p in old))} | **{fr(sum(p['n'] for p in pts))}** |")
    A(f"| points « principaux » (11 mentions ou plus) | {fr(len(o_main))} ({fr(sum(p['n'] for p in o_main))} mentions) | **{fr(len(n_main))}** ({fr(sum(p['n'] for p in n_main))} mentions) |")
    A(f"| points « rares » (10 mentions ou moins) | {fr(len(old) - len(o_main))} ({fr(sum(p['n'] for p in old if p['n'] < TIER_MAIN))} mentions) | **{fr(len(n_rare))}** ({fr(sum(p['n'] for p in n_rare))} mentions) |")
    A(f"| points qui coïncident avec un point des éditeurs | 4 165 (321 631 mentions) | {fr(em['points'])} ({fr(em['mentions'])} mentions) |")
    A(f"| libellés avec virgule, parenthèse, « Republic of » ou article | {fr(len(old_bad))} points ({fr(sum(p['n'] for p in old_bad))} mentions) | {fr(len(new_bad))} |")
    A(f"| aires de rayon 10 km ou moins et plus de 50 mentions | {len(small_old)} ({fr(sum(p['n'] for p in small_old))} mentions) | {len(small_new)} |")
    A('')
    A(f"La reconstruction de v1 à partir des tables redonne la couche v1 : {fr(base_check['rebuilt_points'])} points, {fr(base_check['rebuilt_mentions'])} mentions, {base_check['points_with_different_n']} point avec un autre nombre de mentions. Le point de départ est donc exact.")
    A('')
    A('Les corrections sont appliquées l\'une après l\'autre, dans l\'ordre du tableau. Le « avant » d\'une ligne est le « après » de la ligne précédente.')
    A('')
    A('## Avant et après, correction par correction')
    A('')
    A('| correction | points | mentions placées | principaux / rares après |')
    A('|---|---|---|---|')
    A(f"| départ : couche v1 reconstruite | {fr(b0['points'])} | {fr(b0['mentions'])} | {fr(b0['main'])} / {fr(b0['rare'])} |")
    for x in steps[1:]:
        A(row(x))
    A(f"| **arrivée : couche v2** | **{fr(last['points'])}** | **{fr(last['mentions'])}** | **{fr(last['main'])} / {fr(last['rare'])}** |")
    A('')
    A('La ligne « coïncidence avec les points des éditeurs » ne change pas les nombres : elle recalcule seulement quels points coïncident avec un point des éditeurs.')
    A('')
    A('## Détail')
    A('')
    A('### Collection Nakala retirée (point 11)')
    A('')
    A(f"* {tn['documents']} documents Nakala ôtés. Mentions placées retirées : **{fr(tn['mentions_removed'])}** (fr {fr(tn['by_lang'].get('fr', 0))}, la {fr(tn['by_lang'].get('la', 0))}).")
    A('* Le contrôle indépendant annonçait 1 452 mentions. Ici, je compte les mentions des formes qui avaient un point : 1 268.')
    A(f"* Points : {fr(tn['before']['points'])} -> {fr(tn['after']['points'])}.")
    A('')
    A('### Espagnol remplacé')
    A('')
    A(f"* Les formes espagnoles de v1 sortent ({fr(te['old_spanish_mentions_placed'])} mentions placées). Les formes de la nouvelle exécution (mDeBERTa principal, accord d\'au moins 2 encodeurs sur 4, confiance 0,5 ou plus) entrent ({fr(te['new_spanish_mentions_placed'])} mentions placées, 1 493 formes).")
    A(f"* Ce n\'est plus une couche à part. Les {fr(te['es_groups'])} points espagnols sont recollés aux points de v1 quand c\'est le même lieu (même identifiant, ou même nom à moins de 10 km) : {fr(te['es_groups_joined_to_v1_points'])} points recollés, {fr(te['es_groups'] - te['es_groups_joined_to_v1_points'])} nouveaux points (avant les fusions du point 9).")
    A(f"* Points : {fr(te['before']['points'])} -> {fr(te['after']['points'])}. Mentions : {fr(te['before']['mentions'])} -> {fr(te['after']['mentions'])}.")
    A('* Variante stricte (confiance 0,98) : non retenue, elle reste disponible dans `es_v2/strict/`.')
    A('')
    A('### Correction 3 : « Nice » du concile de Nicée -> İznik (40,43 ; 29,72)')
    A('')
    A(f"* Formes concernées : Nice, Nyce, Nizza, Niza, Niceam (langues de v1 et espagnol). {fr(t3['mentions_on_nice_forms'])} mentions lues avec leur contexte.")
    A(f"* Le contexte est un mot de concile avant la forme (council, councell, concile, synod, canon, Nicene, decree, Arrians...) ou un mot comme Creed après elle. **{fr(t3['mentions_moved'])} mentions** passent à İznik, les autres restent à Nice.")
    A(f"* Point İznik (libellé « Nicaea », nom moderne `modern` = İznik) : {fr(sp['Nicaea']['n'])} mentions. Point Nice : {fr(sp['Nice']['n'])} mentions.")
    A('* Le contrôle indépendant annonçait 150 à 200 mentions. La règle est prudente : « évêque de Nice » ou « assemblé à Nice » sans mot de concile reste à Nice. Lecture de 25 mentions déplacées tirées au hasard (un seul lecteur) : 25 sur 25 sont le concile (intervalle de Wilson à 95 % : 87 à 100 %).')
    A('')
    A('### Correction 4 : Antiochia* latin -> Antakya (36,20 ; 36,16)')
    A('')
    A(f"* Formes : {', '.join(t4['forms'])}. **{fr(t4['mentions_moved'])} mentions** quittent Ptolémaïs (Akko) pour Antakya.")
    A('')
    A('### Correction 5 : pas de point pour les noms de personnes')
    A('')
    A(f"* Règle : probabilité moyenne de personne (`p_per`) de 0,9 ou plus, et forme absente de la table d\'exonymes. Les corrections de table de cette liste en sont exemptées.")
    A(f"* **{fr(t5['forms_unplaced'])} formes, {fr(t5['mentions_removed'])} mentions** perdent leur point (v1 : {t5['by_run_forms_mentions']['v1'][0]} formes, {t5['by_run_forms_mentions']['v1'][1]} mentions ; espagnol refait : {t5['by_run_forms_mentions']['es'][0]} formes, {t5['by_run_forms_mentions']['es'][1]} mentions).")
    A('* Les plus grosses : ' + ', '.join(f'{f} {lg} ({k})' for k, f, lg in t5['top'][:10]) + '.')
    A('* Perte connue : quelques vrais lieux (Istrana, 10 mentions).')
    A('')
    A('### Correction 6 : Isle / Île de France -> Île-de-France (49,0 ; 2,5 ; rayon 60 km)')
    A('')
    A(f"* **{fr(t6['mentions_moved'])} mentions** quittent l\'île Maurice. Aucune n\'a de mot Maurice, Mauritius, Madagascar ou Bourbon dans son contexte ({fr(ctx['idf'][0])} mentions lues), donc aucune ne reste à Maurice.")
    A(f"* Point Île-de-France : {fr(sp['Île-de-France']['n'])} mentions, rayon {sp['Île-de-France']['r']} km (le rayon des éditeurs, 150 km, n\'est pas repris).")
    A('')
    A('### Correction 8 : lieux fictifs')
    A('')
    A('* Liste noire : ' + ', '.join(f"{f} {lg} ({k})" for f, lg, k in t8['forms']) + f". **{fr(t8['mentions_removed'])} mentions** perdent leur point (Megalopolis n\'est plus rattachée au Mecklembourg et reste sans point).")
    A('')
    A('### Points 2 et 7 du contrôle : variantes d\'orthographe, régions, fleuves')
    A('')
    A('| entrée | forme, langue (mentions) | avant | après |')
    A('|---|---|---|---|')
    after_names = {'Bergogna = Burgundy': 'Burgundy (région, 47,0 ; 4,8)', 'Nerva = Narva': 'Narva (59,38 ; 28,20)', 'Dan = Damme': 'Damme (51,25 ; 3,28)', 'Modana = Modena': 'Modena (44,65 ; 10,93)',
                   'Tage / Tago = Tagus river': 'Tage (fleuve, 38,85 ; -9,01)', 'Monferrato = Montferrat region': 'Montferrat (aire, 45,13 ; 8,45 ; rayon 40 km)',
                   'Geldria / Geldriam = Guelders region': 'Guelders (aire, 52,08 ; 5,92)', 'Humber = Humber estuary': 'Humber (estuaire, 53,65 ; -0,20)', 'Iordan = Jordan river': 'Jordan (fleuve, 31,76 ; 35,56)'}
    after_names['Tage / Tago = Tagus river'] = 'Tagus (fleuve, 38,85 ; -9,01)'
    for lab, v in t2['edits'].items():
        for f in v['forms']:
            A(f"| {lab} | {f['form']} {f['lang']} ({f['n']}) | {f['was']} | {after_names[lab] if f['moved'] else 'déjà au bon endroit (genre et nom corrigés)'} |")
    A('')
    A(f"* Mentions déplacées ou placées : **{fr(t2['mentions_moved'])}**, dont **{fr(t2['newly_placed'])}** mentions de formes qui n\'avaient pas de point (Tago, Tagus en).")
    A(f"* Fleuves et estuaire : genre `n`, sans rayon. Aires : genre `a`. Je n\'ai pas inventé de coordonnées : Burgundy, Narva, Damme, Modena, Tagus, Montferrat et Guelders reprennent un point déjà dans l\'atlas ; Humber (53,65 ; -0,20) est le milieu de l\'estuaire, à quelques km près.")
    A(f"* Jordan : le contexte « fleuve » est présent dans {fr(ctx['jordan'][1])} des {fr(ctx['jordan'][0])} mentions de Iordan / Iordanis lues. J\'ai relu les {fr(ctx['jordan'][0] - ctx['jordan'][1])} mentions sans mot de fleuve : toutes sont le fleuve (« beyond Iordan », « went to Iordan », « edge of Iordan »). Les {fr(t2['edits']['Iordan = Jordan river']['mentions_moved'])} mentions vont donc au fleuve, sans condition de contexte.")
    A('* Non fait : « Tajo » (es, 174 mentions, sans point, « confiance basse ») n\'est pas dans la liste demandée. À ajouter si tu es d\'accord.')
    A('')
    A('### Point 9 : rayon minimal et doublons')
    A('')
    A(f"* Rayon minimal de 40 km pour les aires : {fr(t9a['areas_raised'])} aires sur {fr(t9a['areas'])} ont été relevées ({fr(t9a['mentions_in_raised_areas'])} mentions) ; {t9a['areas_with_radius_5_or_less']} avaient un rayon de 5 km ou moins.")
    A(f"* Fusions : **{fr(t9b['clusters_merged'])} groupes**, {fr(t9b['points_absorbed'])} points absorbés, {fr(t9b['mentions_in_merged'])} mentions dans des points fusionnés. Points : {fr(t9b['before']['points'])} -> {fr(t9b['after']['points'])}.")
    A(f"* Preuve de l\'équivalence des noms : {', '.join(f'{k} : {v}' for k, v in t9b['by_proof'].items())}. « exonym/key » = même clé de nom que la table d\'exonymes et le géocodeur (casse, accents, orthographe ancienne) ; « curated » = paires sans racine commune (Napoli / Naples) ; « wikidata » = même identifiant ; « same-name » = règle v1.")
    A('* Distance : villes à moins de 3 km ; aires à moins de 30 % du plus grand rayon ; même nom ou même orthographe à moins de 10 km (règle de v1).')
    A('')
    A('Les paires demandées :')
    A('')
    A('| paire | fusionnée ? |')
    A('|---|---|')
    L.extend(spec_lines)
    A('')
    A('Les plus gros groupes fusionnés :')
    A('')
    for m in t9b['biggest'][:15]:
        A(f"* {' + '.join(m['names'][:4])} : {fr(m['mentions'])} mentions")
    A('')
    A('Restent séparés : des paires à plus de 3 km sans preuve de nom (Hierosolymam, à 20 km de Jérusalem, est fusionnée par le contrôle du 23 oct.), ou de genres trop différents (province et ville de même nom). Ils sont faibles en mentions.')
    A('')
    A('### Point 10 : libellés')
    A('')
    A(f"* Règle : forme écrite la plus fréquente, avec majuscule, sans virgule, parenthèse ni « Republic of » ; « the bath » devient Bath. Libellés à corriger avant : {fr(len(old_bad))} points ({fr(sum(p['n'] for p in old_bad))} mentions). Après : {len(new_bad)}.")
    A('* Le globe est en anglais. Ordre de choix : (1) le nom anglais de la table d\'exonymes quand les textes l\'écrivent (Spain, Genoa, Jerusalem) ; (2) le nom du calque des éditeurs quand c\'est aussi une forme anglaise des textes (au moins 10 % des mentions anglaises) ; (3) la forme anglaise la plus fréquente quand les textes anglais portent le quart du point ; (4) la forme la plus fréquente toutes langues. « Istanbul » devient « Constantinople » quand les textes l\'écrivent.')
    A('* Le libellé du géocodeur est gardé dans `gname` (pas affiché). Le nom moderne d\'İznik et d\'Antakya est dans `modern`.')
    A('')
    A('### Niveau (tier)')
    A('')
    A(f"* `tier` = « main » à partir de 11 mentions, « rare » sinon, calculé après toutes les fusions : {fr(len(n_main))} points principaux ({fr(sum(p['n'] for p in n_main))} mentions, {pct(sum(p['n'] for p in n_main), sum(p['n'] for p in pts))}) et {fr(len(n_rare))} points rares ({fr(sum(p['n'] for p in n_rare))} mentions).")
    A('* Le contrôle indépendant a mesuré les points rares de v1 à environ 3 sur 4 justes. Cette mesure n\'a pas été refaite sur v2. Les nouveaux points rares espagnols ont été lus sur 30 mentions : 77 % justes.')
    A('')
    A('## Ce qui reste')
    A('')
    A('* Non fait ici (calcul lourd, prévu par le contrôle) : choisir le candidat GeoNames avec la liste complète et l\'indice des lieux du même document (point 1 du contrôle : homonymes de bourgs), table des aires de régions (Mosella, Garonne placés à la source).')
    A('* Point 12 (langue par passage) : non fait.')
    A('* La référence gold v2 de l\'espagnol est provisoire. Les corrections de contexte (Nice) reposent sur des listes de mots, lues sur un échantillon par un seul lecteur.')
    A('* Les sept documents WikisourceES_Renaissance présents en double sont encore lus deux fois (moins de 100 mentions).')
    A('')
    A('## Fichiers et reproduction')
    A('')
    A('* `scratchpad/atlas/globe/v24/model_layer_v2.json` : la couche (`header`, `places`). Champs ajoutés : `tier`, `gname`, `fx` (corrections qui ont touché le point), `prev` (ids de v1 : `ml:<n>`, `mles:<n>` pour la reprise espagnole), `modern`.')
    A('* `scratchpad/atlas/globe/v24/model_layer_v2_docs.json` : tous les documents de chaque point (`points` : id -> [[document, mentions]]), et `docs` : titre, année, collection, lien.')
    A('* Ids : `ml2:<n>`, classés par nombre de mentions. Ils ne sont pas ceux de v1.')
    A('* Étape Hypatia (Slurm, partition debug, 36 s) : `slurm/ner2_layer_fix_ctx.sbatch` lance `scripts/ner2_layer_fix_ctx.py` : `p_per` moyen par forme, contextes des règles (Nice, Île de France, Jordan), titres des 24 858 documents. Sortie : `layer_v1/fix/`. Aucun texte ne quitte Hypatia, sauf les fichiers `peek_*.jsonl` lus pour régler les listes de mots.')
    A('* Étape Mac : `nice -n 19 python scripts/ner2_layer_fix.py --work DIR --old-layer model_layer.json --globe globe_places.json --out model_layer_v2.json --docs-out model_layer_v2_docs.json --report corrections_v2.md`. Durée : 10 secondes. DIR contient `forms_placed.parquet`, `doc_forms.parquet`, `docs_v1.parquet` (de `layer_v1/geo` et `layer_v1`), `es/forms_placed.parquet`, `es/doc_forms.parquet`, `es/docs_es.parquet` (de `es_mdeberta`), et `fix/` (ci-dessus).')
    A('* Tests : `tests/test_ner2_layer_fix.py`.')
    A('')
    return '\n'.join(L) + '\n'


PASS2_TITLE = '## Passe 2 (23 oct.)'
BAD_NAME = re.compile(r'[,()]|republic|kingdom of|^the ', re.I)


def patch_report_pass2(path: Path, section: str):
    """Adds the pass 2 section at the end of corrections_v2.md, or replaces it when it is already there."""
    txt = path.read_text('utf-8') if path.exists() else ''
    i = txt.find('\n' + PASS2_TITLE)
    if i >= 0:
        txt = txt[:i + 1]
    elif txt and not txt.endswith('\n'):
        txt += '\n'
    path.write_text(txt.rstrip('\n') + '\n\n' + section, 'utf-8')


def make_report_pass2(S: State, pts, prev, em) -> str:
    """The French section of pass 2: what the sweep of 351 points changed, before / after."""
    rec = S.sweep_rec
    st = {x['step']: x for x in S.log}
    sw = st['P2 sweep verdicts (drop, unplace, clean, merge, move, relabel)']
    m2 = st['P2 merge again after the sweep']
    m1 = st['9b merge of duplicate points']
    pv = prev['places'] if prev else []
    pe = prev['header'].get('coincide_with_editor_points', {}) if prev else {}
    n_main = [p for p in pts if p['tier'] == 'main']
    n_rare = [p for p in pts if p['tier'] == 'rare']
    o_main = [p for p in pv if p['tier'] == 'main']
    o_rare = [p for p in pv if p['tier'] == 'rare']
    sm = lambda L: sum(p['n'] for p in L)  # noqa: E731
    bad_new = [p for p in pts if BAD_NAME.search(p['name'])]
    bad_old = [p for p in pv if BAD_NAME.search(p['name'])]
    L = []
    A = L.append
    A(PASS2_TITLE)
    A('')
    A('Rien n\'est commité, publié ni envoyé. Toujours aucune relecture humaine : les verdicts viennent d\'un balayage des 351 points les plus lus, fait par un modèle, avec les contextes des passages.')
    A('')
    A('### Avant et après')
    A('')
    A('| | avant (22 oct.) | après (23 oct.) |')
    A('|---|---|---|')
    A(f"| points | {fr(len(pv))} | **{fr(len(pts))}** ({len(pts) - len(pv):+,}) |".replace(',', ' ') if False else f"| points | {fr(len(pv))} | **{fr(len(pts))}** ({len(pts) - len(pv):+d}) |")
    A(f"| mentions placées | {fr(sm(pv))} | **{fr(sm(pts))}** ({sm(pts) - sm(pv):+d}) |")
    A(f"| points « principaux » (11 mentions ou plus) | {fr(len(o_main))} ({fr(sm(o_main))} mentions) | **{fr(len(n_main))}** ({fr(sm(n_main))} mentions) |")
    A(f"| points « rares » (10 mentions ou moins) | {fr(len(o_rare))} ({fr(sm(o_rare))} mentions) | **{fr(len(n_rare))}** ({fr(sm(n_rare))} mentions) |")
    A(f"| points qui coïncident avec un point des éditeurs | {fr(pe.get('points', 0))} ({fr(pe.get('mentions', 0))} mentions) | {fr(em['points'])} ({fr(em['mentions'])} mentions) |")
    A(f"| libellés avec virgule, parenthèse, « Republic of » ou article | {len(bad_old)} | {len(bad_new)} |")
    A('')
    A(f"Les niveaux (`tier`) sont recalculés après toutes les fusions. Les points principaux baissent surtout parce que {sum(1 for k in ('drop', 'unplace') for x in rec[k] if x['mentions'] >= 11)} points de 11 mentions ou plus perdent leur point (pas un lieu, ou lieu incertain) et que des doublons sont fusionnés ({len(rec['merge'])} demandés, {m2['points_absorbed']} trouvés ensuite).")
    A('')
    A('### Étapes')
    A('')
    A('| étape | points | mentions placées | principaux / rares après |')
    A('|---|---|---|---|')
    for x in (m1, sw, m2):
        a, b = x['before'], x['after']
        nm = {'9b merge of duplicate points': 'fusion des doublons (règle élargie, voir plus bas)', 'P2 merge again after the sweep': 'fusion des doublons relancée après le balayage'}.get(x['step'], 'verdicts du balayage')
        A(f"| {nm} | {fr(a['points'])} -> {fr(b['points'])} ({b['points'] - a['points']:+d}) | {fr(a['mentions'])} -> {fr(b['mentions'])} ({b['mentions'] - a['mentions']:+d}) | {fr(b['main'])} / {fr(b['rare'])} |")
    A('')
    A('Avant la passe 2, cette étape donnait 9 701 points : la règle élargie en fusionne un de plus (Saxonia, Saxony et Dresden en un seul point).')
    A('')
    A('### Ce que les verdicts ont fait')
    A('')
    A('| action | points | mentions |')
    A('|---|---|---|')
    A(f"| pas un lieu : sans point | {len(rec['drop'])} | {fr(sum(x['mentions'] for x in rec['drop']))} |")
    A(f"| incertain : sans point | {len(rec['unplace'])} | {fr(sum(x['mentions'] for x in rec['unplace']))} |")
    A(f"| déplacés (coordonnées, genre, rayon) | {len(rec['move'])} | {fr(sum(x['mentions'] for x in rec['move']))} |")
    A(f"| fusionnés dans un autre point | {len(rec['merge'])} | {fr(sum(x['mentions'] for x in rec['merge']))} |")
    A(f"| formes retirées d'un bon point | {sum(len(x['forms']) for x in rec['clean'])} formes dans {len(rec['clean'])} points | {fr(sum(x['mentions'] for x in rec['clean']))} |")
    A(f"| formes envoyées vers un autre point | {sum(len(x['forms']) for x in rec['forms_into'])} formes | {fr(sum(x['mentions'] for x in rec['forms_into']))} |")
    A(f"| libellé changé | {len(rec['relabel'])} | |")
    A('')
    A('Les 257 autres points sont jugés justes et ne bougent pas.')
    A('')
    A('**Sans point (pas un lieu).** ' + ', '.join(f"{x['point'].split(' (')[0]} ({x['mentions']})" for x in sorted(rec['drop'], key=lambda x: -x['mentions'])) + '.')
    A('')
    A('**Sans point (incertain).** ' + ', '.join(f"{x['point'].split(' (')[0]} ({x['mentions']})" for x in sorted(rec['unplace'], key=lambda x: -x['mentions'])) + '.')
    A('')
    A('**Déplacés, les plus gros.**')
    A('')
    A('| point | mentions | avant | après |')
    A('|---|---|---|---|')
    for x in sorted(rec['move'], key=lambda x: -x['mentions'])[:20]:
        f0, t0 = x['from'], x['to']
        A(f"| {x['label'] or x['point'].split(' (')[0]} | {x['mentions']} | {f0[0]} ; {f0[1]} ({f0[2]}) | {t0[0]} ; {t0[1]} ({t0[2]}{', ' + str(int(t0[3])) + ' km' if t0[3] else ''}) |".replace('.', ','))
    A('')
    A(f"Les {len(rec['move'])} déplacements sont dans `data/model_layer_overrides.json`. Les coordonnées sont celles du balayage. Je n'en ai pas ajouté. Un rayon donné pour un fleuve n'est pas gardé : seules les aires ont un rayon.")
    A('')
    A('**Fusions demandées.** ' + '; '.join(f"{x['point'].split(' (')[0]} -> {x['into'].split(' (')[0]} ({x['mentions']})" for x in rec['merge']) + '.')
    A('')
    ex = [x for x in rec['merge'] if x['left_out']]
    if ex:
        A('Une exception : ' + '; '.join(f"{x['point'].split(' (')[0]} garde sa forme « Yndias » à part ({x['left_out']} mentions) : c\'est souvent les Indes occidentales, donc elle est sans point" for x in ex) + '.')
        A('')
    A('**Formes retirées.** ' + '; '.join(f"{x['point'].split(' (')[0]} : {', '.join(x['forms'])}" for x in rec['clean']) + '.')
    A('')
    for x in rec['forms_into']:
        A(f"**Formes envoyées ailleurs.** {', '.join(x['forms'])} ({x['mentions']} mentions) vont de {x['point'].split(' (')[0]} à Hertfordshire.")
        A('')
    A('**Libellés.** ' + '; '.join(f"{x['point'].split(' (')[0]} -> {x['label']}" for x in rec['relabel']) + '.')
    A('')
    if bad_new:
        A('Libellés gardés avec une parenthèse parce que le balayage les demande ainsi : ' + ', '.join(p['name'] for p in bad_new) + ' (« Lusitania (Portugal) » a disparu : ce point est absorbé par Portugal).')
    A('')
    A('### La règle de fusion élargie')
    A('')
    A('* Les clés de nom comprennent maintenant le nom anglais de la table d\'exonymes lui-même. Avant, Lüttich donnait la clé `ex:liege`, mais Liège ne donnait que `liege` : les deux ne se rejoignaient jamais. Même chose pour Leodium, Dornic, Rhodum, Lucernam, Perpiñán, Malines, Swede, Argel.')
    A(f"* Deux points qui ont le même identifiant Wikidata fusionnent jusqu'à {QID_KM:.0f} km d'écart (avant : 3 km pour une ville, 30 % du rayon pour une aire).")
    A(f"* La fusion est relancée après le balayage : elle rassemble les points déplacés avec ceux qui sont maintenant au même endroit ({m2['clusters_merged']} groupes, {m2['points_absorbed']} points absorbés, {fr(m2['mentions_in_merged'])} mentions) : " + ', '.join(f"{' + '.join(m['names'][:2])}" for m in m2['biggest'][:12]) + '.')
    A('* Un point fixé par le balayage qui est absorbé par un plus gros ne change pas les coordonnées, le rayon ni le libellé du plus gros (Lusitaniam, Candie, Rossia, Melano, Mestrina, Glaston : leur point reprend celui de Portugal, Crete, Russia, Milan, Mestre, Glastonbury).')
    A('')
    A('### Gardé tel quel (à lire plus tard)')
    A('')
    A('* Les verdicts « juste » qui citent une forme douteuse sans la dire fausse restent en place : Nice (formes Nice, Niceam, Niceen, qui vont parfois à Nicée), Rodes (une occurrence est Rodez), Dan à Damme (tribu biblique possible), Tarso à Tarse (« reyno de Tarso » probablement romanesque), Tenedo, Malton, Wrentham, Lerma, Humber, St Andrews, North Muskham. Ce sont des formes majoritaires du point : les retirer viderait le point.')
    A('* Retirées parce que le verdict dit que la forme n\'est pas ce lieu : Manica, Lints, Saint Peters, Lym, Forlimpopoli, Salcraig.')
    A('* « Indo » (es, it) reste avec l\'Inde : le verdict ne dit rien contre cette forme.')
    A('')
    A('### Fichiers et reproduction')
    A('')
    ac = Counter(x['action'] for x in load_overrides())
    A(f"* Verdicts : `data/model_layer_overrides.json` ({sum(ac.values())} entrées : {ac['drop']} pas un lieu, {ac['unplace']} incertains, {ac['move']} déplacés, {ac['merge']} fusions, {ac['relabel']} libellés, {ac['clean']} nettoyages de formes). Chaque entrée vise un point par la forme la plus fréquente de son point (`anchor`), car les ids `ml2` changent à chaque exécution. Les 7 dernières entrées viennent du contrôle indépendant du 23 oct. (Hierosolymam vers Jérusalem, Rodes devient Rhodes, Litchfield devient Lichfield, Chilton sans point, doublons Japan, Epirus, Cor Sica).")
    A('* Script : `scripts/ner2_layer_fix.py` lit ce fichier tout seul (`--overrides none` redonne la couche du 22 oct., à la règle de fusion élargie près). Tests : `tests/test_ner2_layer_fix.py`.')
    A('* Ancienne sortie gardée : `model_layer_v2_prev.json` et `model_layer_v2_docs_prev.json`, à côté de `model_layer_v2.json`.')
    A('')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
