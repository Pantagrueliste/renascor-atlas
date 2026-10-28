"""Latin place names for the name-based geocoder (v2.3, path A; pure functions plus two loaders).

Why. 110,867 mentions were unplaced after v2.2. About 4,100 of them (and more in the long tail) are Latin case forms of places that the atlas knows under
their nominative or a vernacular name: Leidam (Leiden, 676), Hagam (The Hague, 461), Rhenum / Rheni / Rheno (the Rhine), Budam, Mosellam, Colmariam,
Transilvaniam, Scotiam, Masoviae. ``latin_names`` generates nominative candidates; this module is where they meet the lexicons and the gates.

Sources for a candidate (every one is local; nothing is fetched):

1. **Wikidata Latin labels and aliases** of the place items the atlas already holds (``wd_entities.parquet`` of the reconcile directory, 10 languages with ``la``)
   with the coordinates of the atlas's own placed locals of that item (``WikidataLatin``).
2. **Orbis Latinus** (Graesse-Benedict, 2nd ed. 1909, Internet Archive item ``bub_gb_oJt-AAAAMAAJ``, public domain; ``data/orbis_latinus.tsv``, tiers A and B): Latin form ->
   German-spelled modern name, type and country (``OrbisLatinus``). The modern name is then looked up in GeoNames by the geocoder (class from the type, country from the
   region text); the book gives no coordinates.
3. **The exonym table** (``exonyms.py``) and the reviewed polity table (``polities.py``) on the nominative candidates.
4. **Atlas names**: names of placed places of the Latin editions (``Geocoder.R``) on the candidates.
5. Pleiades: not available locally, not used.

Gates (the same family as the other name-based paths): the class of the name (celestial, ethnonym, common noun ... are never looked up, ``placeclass``), the
Latin context of the place (the mention's ``lang``, else the edition: ``latin_names.latin_context_level``), the strength of the de-inflection rule (a weak rule
is only a suggestion), the homonym rule (a candidate that leads to several different places is not placed), the regional prior of the edition and the London /
file priors (``Geocoder.row_keep``), and the agreement between the sources. Confidence: ``high`` only when two independent sources agree and the rule is strong;
``medium`` for one source with a strong rule in a Latin context; ``low`` (a suggestion, not on the globe) for the rest.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from . import latin_names as LN

#: v2.4: Latin adjectives and poetic words that the lexicons read as places ('Palatinus' = the court cook of the Palatine, not the Palatine Hill; 'Tiberina' = of the Tiber, not
#: Regensburg; 'Romuleae' = Roman, not Sevilla; Castalia is a spring). Folded keys (``latin_names.fold_key``); the Latin path never looks them up.
NOT_PLACES = frozenset(LN.fold_key(x) for x in '''palatinus palatina palatinum palatini palatinae palatinorum palatinis
tiberinus tiberina tiberinum tiberini tiberinae tiberinis romuleus romulea romuleum romuleae romulei castalia castaliae castalius castalium
hispanus hispana hispanum gallicus gallica gallicum germanicus germanica germanicum'''.split())

SOURCE_CODE = 'l'                 # globe source code of a Latin-name placement ('located by name')
METHOD = 'name_latin'

# --- Orbis Latinus: region text -> country codes ------------------------------------------------------------------------------
# The book's abbreviations and OCR variants of the first word of the region text. A set of codes (the historical country is not the modern one: 'Preußen' is
# Germany, Poland, Russia, Lithuania ...), None = no restriction. The match is on the folded first word(s); unknown text gives None.
_CC = {
    'frankr': 'FR', 'frankreich': 'FR', 'niederl': 'NL', 'niederlande': 'NL', 'belgien': 'BE', 'beigien': 'BE', 'schweiz': 'CH', 'ital': 'IT', 'italien': 'IT',
    'spanien': 'ES', 'portugal': 'PT', 'england': 'GB', 'schottland': 'GB', 'irland': 'IE', 'danemark': 'DK', 'schweden': 'SE', 'norwegen': 'NO',
    'bohmen': 'CZ', 'mahren': 'CZ', 'luxemburg': 'LU', 'sizilien': 'IT', 'bayern': 'DE', 'wurttemberg': 'DE', 'baden': 'DE', 'hessen': 'DE', 'sachsen': 'DE',
    'hannover': 'DE', 'braunschweig': 'DE', 'oldenburg': 'DE', 'anhalt': 'DE', 'waldeck': 'DE', 'mecklenburg': 'DE', 'holstein': 'DE', 'schleswig': 'DE',
    'lippe': 'DE', 'thuringen': 'DE', 'reuß': 'DE', 'reuss': 'DE', 'lubeck': 'DE', 'hamburg': 'DE', 'bremen': 'DE', 'griechenland': 'GR', 'bulgarien': 'BG',
    'serbien': 'RS', 'rumanien': 'RO', 'turkei': 'TR',
}
_CC_SET = {'preußen': {'DE', 'PL', 'RU', 'LT'}, 'preussen': {'DE', 'PL', 'RU', 'LT'}, 'preuß': {'DE', 'PL', 'RU', 'LT'}, 'preuss': {'DE', 'PL', 'RU', 'LT'},
           'elsaß': {'FR', 'DE'}, 'elsass': {'FR', 'DE'}, 'lothringen': {'FR', 'DE'}, 'lothr': {'FR', 'DE'},
           'österr': {'AT', 'IT', 'SI', 'HR', 'CZ', 'HU', 'DE'}, 'osterr': {'AT', 'IT', 'SI', 'HR', 'CZ', 'HU', 'DE'}, 'österreich': {'AT', 'IT', 'SI', 'HR', 'CZ', 'HU', 'DE'},
           'osterreich': {'AT', 'IT', 'SI', 'HR', 'CZ', 'HU', 'DE'}, 'tirol': {'AT', 'IT'}, 'steiermark': {'AT', 'SI'}, 'kärnten': {'AT', 'SI', 'IT'}, 'karnten': {'AT', 'SI', 'IT'},
           'ungarn': {'HU', 'SK', 'RO', 'RS', 'HR', 'AT', 'UA'}}
_CC_PREFIX = (('niederö', {'AT', 'CZ', 'HU', 'SK'}), ('niedero', {'AT', 'CZ', 'HU', 'SK'}), ('oberö', {'AT', 'DE'}), ('obero', {'AT', 'DE'}))

#: type of the book -> GeoNames feature classes (P populated place, A administrative, H water, T relief, L area). None = not used
_PT_CLASS = {'city': 'P', 'village': 'P', 'market_town': 'P', 'settlement': 'P', 'city_district': 'P', 'spa': 'P',
             'river': 'H', 'canal': 'H', 'lake': 'H', 'gulf': 'H', 'bay': 'H', 'strait': 'H',
             'mountain': 'T', 'mountains': 'T', 'island': 'T', 'islands': 'T', 'cape': 'T', 'peninsula': 'T', 'pass': 'T', 'valley': 'T', 'plateau': 'T',
             'region': 'AL', 'country_or_region': 'AL', 'province': 'A', 'county': 'A', 'duchy': 'A', 'kingdom': 'A', 'canton': 'A', 'principality': 'A', 'country': 'A',
             'district': 'A'}


def _fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize('NFKD', str(s or '').lower().strip(' .,;()'))
    return ''.join(c for c in s if not unicodedata.combining(c))


def region_countries(text) -> Optional[frozenset]:
    """Country codes the region text of an Orbis entry allows ('Frankr. (Nievre)' -> {FR}); None when the text says nothing usable."""
    t = str(text or '').strip()
    if not t:
        return None
    first = re.split(r'[\s(,]', t, 1)[0]
    f = first.lower().strip('. ')
    if f in _CC_SET:
        return frozenset(_CC_SET[f])
    fa = _fold(f)
    if fa in _CC:
        return frozenset({_CC[fa]})
    if f in _CC:
        return frozenset({_CC[f]})
    for pre, cc in _CC_PREFIX:
        if f.startswith(pre):
            return frozenset(cc)
    return None


@dataclass(frozen=True)
class OrbisEntry:
    latin: str
    modern: str
    ptype: str
    classes: str                   # GeoNames feature classes ('P', 'H' ...)
    ccs: Optional[frozenset]
    tier: str


class OrbisLatinus:
    """Latin form (``latin_names.fold_key``) -> entries of the book. Only tiers A and B of ``data/orbis_latinus.tsv``; a key that leads to more than one modern
    name is ambiguous and gives no entry (``key_n_modern`` > 1)."""

    def __init__(self, rows):
        self.idx: dict = defaultdict(list)
        for latin, modern, ptype, region, tier, nmod in rows:
            cls = _PT_CLASS.get(ptype)
            if not cls or str(tier) not in ('A', 'B'):
                continue
            try:
                if int(nmod or 1) > 1:
                    continue
            except ValueError:
                pass
            self.idx[LN.fold_key(latin)].append(OrbisEntry(latin, modern, ptype, cls, region_countries(region), str(tier)))

    @classmethod
    def load(cls, path: Path) -> 'OrbisLatinus':
        import csv
        rows = []
        with open(path, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                rows.append((r['latin_name'], r['modern_name'], r['place_type'], r['region_text'], r['tier'], r['key_n_modern']))
        return cls(rows)

    def get(self, lemma: str) -> list:
        """The entries of a lemma when they all agree on one modern name (case-folded), else []."""
        es = self.idx.get(LN.fold_key(lemma), [])
        if not es:
            return []
        mods = {_fold(e.modern) for e in es}
        return es if len(mods) == 1 else []

    def modern_names(self) -> set:
        return {e.modern for es in self.idx.values() for e in es}

    def __len__(self) -> int:
        return len(self.idx)


# --- Wikidata Latin labels --------------------------------------------------------------------------------------------------
class WikidataLatin:
    """Fold-key of a Latin label or alias -> {qid}. ``add(qid, label_json, alias_json)`` is fed from ``wd_entities.parquet``; ``points`` maps a qid to the atlas's
    own point for it (lat, lon, radius_km or None, is_area, n_mentions): built by the geocoder from the PLACED locals that carry the QID, so a hidden place of the
    hold-out cannot vouch for itself."""

    def __init__(self):
        self.idx: dict = defaultdict(set)
        self.label: dict = {}

    def add(self, qid: str, labels, aliases):
        for src in (labels, aliases):
            if isinstance(src, str):
                try:
                    src = json.loads(src)
                except ValueError:
                    continue
            if not isinstance(src, dict):
                continue
            v = src.get('la')
            vals = [v] if isinstance(v, str) else list(v or [])
            for x in vals:
                k = LN.fold_key(x)
                if len(k) >= 3:
                    self.idx[k].add(qid)
                    self.label.setdefault((qid, k), x)

    @classmethod
    def load(cls, path: Path, only: Optional[set] = None) -> 'WikidataLatin':
        import pandas as pd
        d = pd.read_parquet(path, columns=['qid', 'labels', 'aliases', 'missing'])
        w = cls()
        for q, lab, ali, miss in zip(d['qid'], d['labels'], d['aliases'], d['missing']):
            if miss or (only is not None and q not in only):
                continue
            w.add(q, lab, ali)
        return w

    def get(self, lemma: str) -> set:
        return self.idx.get(LN.fold_key(lemma), set())

    def __len__(self) -> int:
        return len(self.idx)


# --- confidence ---------------------------------------------------------------------------------------------------------------
STRONG_RULE = 0.8
WEAK_RULE = 0.6                  # below this a rule (a Greek accusative plural read as Latin: Delfos -> Delfi) gives a suggestion whatever the sources say
FAMILY = {'wikidata': 'atlas', 'atlas': 'atlas', 'exonym': 'table', 'polity': 'table', 'orbis': 'orbis'}
TIEBREAK_CTX = 0.5               # context margin that decides between two places of the same Latin name (Rhenus = the Rhine or the Reno)
TIEBREAK_RATIO = 20.0            # ... or a prominence (mentions in the atlas) ratio with no contrary context


class Hit(NamedTuple):
    src: str                     # wikidata | atlas | exonym | polity | orbis
    score: float                 # strength of the de-inflection rule
    lat: float
    lon: float
    radius: Optional[float]
    qid: Optional[str] = None
    lemma: str = ''
    row: Optional[int] = None    # the placed local the point comes from
    label: Optional[str] = None
    weight: float = 0.0          # mentions of that local in the atlas (prominence)


def _clusters(hits: list) -> list:
    """Hits of one place: the same Wikidata item (a river has several points), or points within the larger radius (at least 50 km)."""
    from .reconcile_geocode import hav          # lazy: reconcile_geocode imports this module
    n = len(hits)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a in range(n):
        for b in range(a + 1, n):
            ha, hb = hits[a], hits[b]
            same = bool(ha.qid) and ha.qid == hb.qid
            if not same:
                tol = max(50.0, min(max(ha.radius or 0.0, hb.radius or 0.0), 400.0))
                same = float(hav(ha.lat, ha.lon, hb.lat, hb.lon)) <= tol
            if same and find(a) != find(b):
                parent[find(a)] = find(b)
    out: dict = {}
    for k in range(n):
        out.setdefault(find(k), []).append(hits[k])
    return list(out.values())


def combine(hits: list, context: str, prefer=None, boost: bool = False) -> tuple:
    """hits: [Hit] -- the candidate points the sources lead to for ONE place. -> (confidence or None, reason, chosen Hit).

    * the hits must lie in one place (the same Wikidata item, or within the larger radius, at least 50 km): several different places are a homonym ->
      (None, 'latin_ambiguous'), unless ``prefer(clusters)`` names one (the regional prior of the edition or a large difference in prominence: Rhenus is the
      Rhine and, far less often, the Reno); the answer is then at most medium;
    * ``high``: two independent families agree, the best rule is strong (>= 0.8), Latin context;
    * ``medium``: one family with a strong rule in a Latin context, or two families in a mixed context or with a middling rule (>= 0.6);
    * ``boost`` (a strongly inflected form of a place the atlas mentions often: Rhenum, Mosellam in a mixed edition -- no French or Dutch word ends so): one family with a
      strong rule is medium in a mixed context too;
    * ``low``: the rest -- a weak rule (< 0.6) is a suggestion whatever the sources say.
    """
    if not hits:
        return None, 'latin_no_match', None
    cl = _clusters(hits)
    tie = False
    if len(cl) > 1:
        k = prefer(cl) if prefer is not None else None
        if k is None:
            return None, 'latin_ambiguous', None
        hits, tie = cl[k], True
    srcs = {FAMILY.get(h.src, h.src) for h in hits}          # a Wikidata label and an atlas name rest on the same placed locals: one family
    best = max(h.score for h in hits)
    strong = best >= STRONG_RULE
    latin = context == 'latin'
    if best < WEAK_RULE:
        conf = 'low'
    elif len(srcs) >= 2 and strong and latin:
        conf = 'high'
    elif len(srcs) >= 2:
        conf = 'medium'
    elif strong and (latin or boost):
        conf = 'medium'
    else:
        conf = 'low'
    if tie and conf == 'high':
        conf = 'medium'
    # the point: the Wikidata item's, else the best-scored hit
    chosen = max(hits, key=lambda h: (h.src == 'wikidata', h.score, h.weight))
    return conf, ('ok_tiebreak' if tie else 'ok'), chosen
