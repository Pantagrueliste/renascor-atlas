"""One hierarchical vocabulary of place types for the whole atlas, and the rules that map every source's own terms onto it (v2.4 preparation; pure functions).

Why. The atlas has so far a five-way *coarse kind* (``placekind``: c settlement, a area, n natural feature, b built feature, u other). Each source says more: Wikidata names
the class ('city of Switzerland', 'free imperial city', 'prince-bishopric'), GeoNames the feature code (PPLA3, ADM1H, CSTL), Getty TGN the place type ('inhabited place',
'former nation/state/empire'), the editions' registers their own type ('herrschaft, kastlanei', 'place_f_church', 'flurname'), Orbis Latinus an abbreviation ('St.', 'Mfl.',
'Gfsch.'), and the name itself a qualifier word ('Böhmen, Königreich', 'Kloster Zinna'). This module is the single place where these terms meet:

* ``TYPES``: the vocabulary, a tree of five groups (``area`` ``settlement`` ``natural`` ``built`` ``other``) and their leaves. Each type has a code, an English label, a parent,
  a coarse kind (``c a n b u``: the coarse kind of a type is a property of the type, so the five-way kind is *derived* from the fine type and stays what it is today) and the
  Wikidata class that stands for it when there is a verified one. A group is also a valid, *generic* type: 'we know it is a settlement, not which sort'.
* ``wd_type`` (Wikidata P31 class: QID and/or English label), ``gn_type`` (GeoNames feature code, optional population), ``tgn_type`` (TGN place type), ``orbis_type``
  (Orbis Latinus abbreviation or normalised type), ``register_type`` (an edition's own type), ``qualifier_type`` ('Head, Qualifier'), ``name_type`` (words of the name): each
  returns a type code or None (no information). The tables behind them are in this file; ``data/placetypes.tsv`` (``load_reviewed``) overrides them row by row: that file is
  the **reviewed** mapping (every class or code with 20 or more placed mentions in v2.3 has a row read by a human; ``status`` says which).
* ``classify(Evidence)``: the fine type of a place from all its sources (weighted vote; a specific type beats the generic one of its group; ``Evidence.kind`` restricts the
  vote to the types of the coarse kind the atlas has already decided, so the fine type never contradicts today's kind).
* ``python -m ratlas.placetypes inventory ...``: the analysis that produced ``data/placetypes.tsv`` and the coverage table (read-only, pandas).

Precision first: an unknown term gives None, never a guess; a place with no information at all is ``other``.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

# ======================================================================================================================
# 1. the vocabulary
# ======================================================================================================================
KINDS = ('c', 'a', 'n', 'b', 'u')
GROUPS = ('area', 'settlement', 'natural', 'built', 'other')


@dataclass(frozen=True)
class PlaceType:
    code: str
    label: str            # English label (the legend of the page)
    parent: Optional[str]
    kind: str             # coarse display kind c / a / n / b / u
    qid: str = ''         # Wikidata class that stands for the type ('' = none verified in the local tables or on wikidata.org on 2026-10-20)
    gloss: str = ''       # what is put under the type (the terms of the sources)


_T = [
    # code, label, parent, kind, qid, gloss
    ('area', 'area (kind of area unspecified)', None, 'a', 'Q56061', 'administrative or political territory of unknown sort'),
    ('continent', 'continent', 'area', 'a', 'Q5107', 'continent, subcontinent'),
    ('country', 'country', 'area', 'a', 'Q6256', 'sovereign or historical state, nation, constituent country'),
    ('kingdom', 'kingdom', 'area', 'a', 'Q417175', 'kingdom, sultanate, khanate (Konigreich, royaume, regno)'),
    ('empire', 'empire', 'area', 'a', 'Q48349', 'empire (Kaiserreich, Reich, imperium)'),
    ('republic', 'republic', 'area', 'a', 'Q7270', 'republic, commonwealth, confederation (Eidgenossenschaft), free state'),
    ('duchy', 'duchy', 'area', 'a', 'Q154547', 'duchy, grand duchy, archduchy (Herzogtum, duche, ducato)'),
    ('county', 'county (countship)', 'area', 'a', 'Q353344', 'countship, county palatine (Grafschaft, comte, contea); the English shire is its own type'),
    ('landgraviate', 'landgraviate', 'area', 'a', 'Q20856058', 'Landgrafschaft'),
    ('margraviate', 'margraviate', 'area', 'a', '', 'Markgrafschaft, marquisat, march of the Empire'),
    ('principality', 'principality', 'area', 'a', 'Q208500', 'principality (Furstentum, principaute, principato), hetmanate'),
    ('electorate', 'electorate', 'area', 'a', 'Q20521456', 'Kurfurstentum, electoral principate'),
    ('bishopric', 'bishopric / ecclesiastical territory', 'area', 'a', 'Q463742', 'Hochstift, Erzstift, Fürstbistum, Bistum, diocese, imperial abbey as territory (Reichsabtei)'),
    ('canton', 'canton', 'area', 'a', 'Q23058', 'Swiss canton, Eidgenossischer Ort, Land (Glarus, Appenzell)'),
    ('lordship', 'lordship', 'area', 'a', 'Q196068', 'Herrschaft, Freiherrschaft, seigneurie, signoria, fiefdom, honour, manor as a jurisdiction'),
    ('bailiwick', 'bailiwick (Amt, Vogtei)', 'area', 'a', 'Q4996207', 'Amt, Landvogtei, Vogtei, bailliage, gemeine Herrschaft, Gericht'),
    ('district', 'district', 'area', 'a', 'Q149621', 'district below the province (contado, Kreis, Bezirk, arrondissement, hundred, wapentake, liberty)'),
    ('shire', 'shire (English or Scottish county)', 'area', 'a', 'Q662296', 'shire, ceremonial or historic county of the British Isles'),
    ('province', 'province', 'area', 'a', 'Q34876', 'province, department, voivodeship, federated state (Bundesland), crown land, governorate, Kreis of the Empire'),
    ('region', 'region', 'area', 'a', 'Q82794', 'historical, geographic or cultural region, landscape (Landschaft)'),
    ('settlement', 'settlement (size unspecified)', None, 'c', 'Q486972', 'inhabited place of unknown size: municipality, commune, locality'),
    ('city', 'city', 'settlement', 'c', 'Q515', 'city, capital, free imperial city, city-state, metropolis'),
    ('town', 'town', 'settlement', 'c', 'Q3957', 'town (Stadt in the sense of a small town)'),
    ('market_town', 'market town', 'settlement', 'c', 'Q18511725', 'market town, Marktgemeinde, Flecken'),
    ('village', 'village', 'settlement', 'c', 'Q532', 'village, Dorf, vill'),
    ('hamlet', 'hamlet', 'settlement', 'c', 'Q5084', 'hamlet, Weiler, farm settlement'),
    ('city_quarter', 'city quarter', 'settlement', 'c', 'Q2983893', 'quarter, neighbourhood, suburb, ward, Ortsteil of a city'),
    ('parish', 'parish', 'settlement', 'c', 'Q102496', 'civil or ecclesiastical parish named after its village or church (Kirchspiel, paroisse)'),
    ('natural', 'natural feature (sort unspecified)', None, 'n', '', 'natural feature of unknown sort'),
    ('river', 'river', 'natural', 'n', 'Q4022', 'river, stream, canal, waterway'),
    ('lake', 'lake', 'natural', 'n', 'Q23397', 'lake, pond, reservoir'),
    ('sea', 'sea', 'natural', 'n', 'Q165', 'sea, ocean, strait'),
    ('bay', 'bay', 'natural', 'n', 'Q39594', 'bay, gulf, fjord, lagoon, inlet, harbour as a water body'),
    ('island', 'island', 'natural', 'n', 'Q23442', 'island, archipelago'),
    ('mountain', 'mountain', 'natural', 'n', 'Q8502', 'mountain, hill, range, volcano'),
    ('pass', 'pass', 'natural', 'n', 'Q133056', 'mountain pass'),
    ('valley', 'valley', 'natural', 'n', 'Q39816', 'valley, gorge, canyon, plain'),
    ('forest', 'forest', 'natural', 'n', 'Q4421', 'forest, wood, heath'),
    ('cape', 'cape', 'natural', 'n', 'Q185113', 'cape, headland, peninsula'),
    ('marsh', 'marsh', 'natural', 'n', 'Q30198', 'marsh, swamp, bog, fen, moor'),
    ('built', 'built feature (sort unspecified)', None, 'b', 'Q811979', 'building, structure or site of unknown sort'),
    ('church', 'church', 'built', 'b', 'Q16970', 'church, chapel, cathedral, temple, mosque, synagogue'),
    ('monastery', 'monastery', 'built', 'b', 'Q44613', 'monastery, abbey, convent, priory, collegiate foundation (Stift), commandery'),
    ('castle', 'castle', 'built', 'b', 'Q23413', 'castle, Burg, Schloss, manor house'),
    ('palace', 'palace', 'built', 'b', 'Q16560', 'palace, residence (Palast, palazzo, palais)'),
    ('inn', 'inn', 'built', 'b', 'Q256020', 'inn, tavern, alehouse, hotel'),
    ('street', 'street', 'built', 'b', 'Q79007', 'street, road, lane, alley'),
    ('square', 'square', 'built', 'b', 'Q174782', 'square, marketplace, piazza'),
    ('gate', 'gate', 'built', 'b', 'Q82117', 'city gate, gate, tower'),
    ('bridge', 'bridge', 'built', 'b', 'Q12280', 'bridge'),
    ('fort', 'fort', 'built', 'b', 'Q1785071', 'fort, fortress, citadel, fortification, wall'),
    ('building', 'building', 'built', 'b', 'Q41176', 'house, hall, hospital, school, theatre, mill, prison, farm ...'),
    ('site', 'site', 'built', 'b', 'Q839954', 'archaeological or historic site, ruin, cemetery, park, field name, battlefield'),
    ('other', 'other (no source says what it is)', None, 'u', '', 'nothing known'),
]
TYPES: dict = {c: PlaceType(c, lab, par, k, q, g) for c, lab, par, k, q, g in _T}
LEAVES = tuple(c for c, t in TYPES.items() if t.parent is not None)
GENERIC = frozenset(GROUPS)                                    # a group used as a type: 'a settlement', sort unknown
#: which coarse kind each group stands for
GROUP_OF_KIND = {'a': 'area', 'c': 'settlement', 'n': 'natural', 'b': 'built', 'u': 'other'}
#: tie-break inside one kind (earlier wins): the specific political form before the plain 'country', a city before a town ...
_ORDER = ('canton', 'lordship', 'bailiwick', 'shire', 'duchy', 'landgraviate', 'margraviate', 'principality', 'electorate', 'bishopric', 'county', 'kingdom', 'empire',
          'republic', 'country', 'continent', 'province', 'district', 'region', 'area',
          'city', 'market_town', 'town', 'village', 'hamlet', 'city_quarter', 'parish', 'settlement',
          'island', 'river', 'lake', 'bay', 'sea', 'pass', 'mountain', 'valley', 'forest', 'marsh', 'cape', 'natural',
          'church', 'monastery', 'castle', 'palace', 'fort', 'inn', 'gate', 'bridge', 'street', 'square', 'building', 'site', 'built', 'other')
_RANK = {c: i for i, c in enumerate(_ORDER)}
assert set(_ORDER) == set(TYPES), set(TYPES) ^ set(_ORDER)


def kind_of(code: Optional[str]) -> str:
    """Coarse kind of a type code (``u`` for None or an unknown code)."""
    t = TYPES.get(code or '')
    return t.kind if t else 'u'


def group_of(code: Optional[str]) -> str:
    """The group a type belongs to ('city' -> 'settlement'; a group is its own group)."""
    t = TYPES.get(code or '')
    if t is None:
        return 'other'
    return t.parent or t.code


def is_generic(code: Optional[str]) -> bool:
    return (code or 'other') in GENERIC


def types_json() -> dict:
    """``extra.json["types"]``: {code: [label_en, parent_code, coarse_kind]} (parent '' for a group)."""
    return {c: [t.label, t.parent or '', t.kind] for c, t in TYPES.items()}


# ======================================================================================================================
# 2. folding
# ======================================================================================================================
def fold(s) -> str:
    """lower case, no diacritics, 'ß' -> 'ss', every run of non-letters/digits one space ('Kurfürstentum' -> 'kurfurstentum', "St. Paul's" -> 'st paul s')."""
    s = unicodedata.normalize('NFKD', str(s or '').lower().replace('ß', 'ss'))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s).split())


# ======================================================================================================================
# 3. Wikidata P31 classes: English label rules (the QID table is data/placetypes.tsv)
# ======================================================================================================================
#: neutral heads: the class says nothing about the sort of place
_NEUTRAL_HEADS = frozenset('''location place object entity destination center centre cluster toponym feature thing phenomenon concept project hub site-of-interest
attraction landmark'''.split())
# head noun -> type (the last word of the label before ' of / in / with ...', singular)
_HEAD = {}


def _h(typ, words):
    for w in words.split():
        _HEAD[w] = typ


_h('continent', 'continent subcontinent')
_h('country', 'country nation state-nation dominion colony protectorate territory-state realm monarchy federation confederation')
_h('kingdom', 'kingdom sultanate khanate caliphate emirate')
_h('empire', 'empire')
_h('republic', 'republic commonwealth')
_h('duchy', 'duchy archduchy')
_h('county', 'county countship comitatus earldom')
_h('shire', 'shire')
_h('landgraviate', 'landgraviate')
_h('margraviate', 'margraviate')
_h('principality', 'principality hetmanate')
_h('electorate', 'electorate')
_h('bishopric', 'bishopric archbishopric diocese archdiocese hochstift erzstift stift see prince-bishopric prince-archbishopric')
_h('canton', 'canton')
_h('lordship', 'lordship seigneury seignory fiefdom barony manor-lordship honour freiherrschaft herrschaft')
_h('bailiwick', 'bailiwick amt vogtei landvogtei bailliage')
_h('district', 'district hundred wapentake riding division arrondissement lathe rape liberty prefecture contado kreis bezirk powiat gmina')
_h('province', 'province department voivodeship oblast governorate viceroyalty krai crown-land bundesland')
_h('region', 'region landscape area-region')
_h('margraviate', 'march marquisate')
_h('area', 'territory area')
_h('settlement', 'settlement municipality commune comune gemeinde township locality borough conurbation populated inhabited community frazione ortschaft area-settlement')
_h('city', 'city metropolis megalopolis megacity capital city-state polis regiopolis')
_h('town', 'town kurort resort')
_h('market_town', 'marktgemeinde flecken mestys')
_h('village', 'village dorf')
_h('hamlet', 'hamlet weiler')
_h('city_quarter', 'quarter neighbourhood neighborhood suburb ortsteil stadtteil ward')
_h('parish', 'parish')
_h('river', 'river stream brook watercourse tributary creek rivulet canal waterway streams')
_h('lake', 'lake lakes pond reservoir mere tarn loch lough')
_h('sea', 'sea ocean strait channel')
_h('bay', 'bay gulf fjord inlet lagoon cove firth estuary')
_h('island', 'island islands isle archipelago islet')
_h('mountain', 'mountain mountains mount hill hills volcano stratovolcano range massif ridge peak summit alp alps')
_h('pass', 'pass')
_h('valley', 'valley gorge canyon ravine plain plateau')
_h('forest', 'forest wood woods woodland heath')
_h('cape', 'cape headland promontory peninsula')
_h('marsh', 'marsh swamp bog fen moor wetland')
_h('church', 'church cathedral chapel basilica temple mosque synagogue minster')
_h('monastery', 'monastery abbey convent priory nunnery cloister charterhouse kloster')
_h('castle', 'castle schloss')
_h('palace', 'palace palazzo')
_h('inn', 'inn tavern hotel alehouse pub')
_h('street', 'street road lane alley avenue boulevard way')
_h('square', 'square plaza piazza marketplace')
_h('gate', 'gate')
_h('bridge', 'bridge')
_h('fort', 'fort fortress citadel fortification stronghold bastion rampart')
_h('building', 'building dwelling house hall hospital stadium school university college library theatre theater museum prison gaol mill farm estate villa brewery arsenal barracks '
               'townhall guildhall mansion tower')
_h('site', 'ruin ruins cemetery graveyard battlefield park garden monument memorial fountain well cave')
_h('natural', 'desert glacier spring waterfall falls beach coast cliff dune rock reef sandbank delta isthmus steppe oasis meadow lowland highland')
_HEAD.update({'state': 'country', 'principate': 'electorate'})
# whole phrases (checked first, longest first) -> type
_PHRASE = {
    'sovereign state': 'country', 'historical country': 'country', 'former country': 'country', 'constituent country': 'country', 'federal state': 'province',
    'federated state': 'province', 'free state': 'republic', 'free imperial city': 'city', 'imperial city': 'city', 'free city': 'city', 'city state': 'city',
    'city-state': 'city', 'market town': 'market_town', 'market municipality': 'market_town', 'market square': 'square', 'town hall': 'building',
    'city gate': 'gate', 'city district': 'city_quarter', 'urban district': 'city_quarter', 'mountain pass': 'pass', 'mountain range': 'mountain', 'mountain system': 'mountain',
    'island group': 'island', 'historical region': 'region', 'cultural region': 'region', 'cultural area': 'region', 'geographic region': 'region', 'natural region': 'region',
    'general region': 'region', 'autonomous region': 'province', 'autonomous community': 'province', 'human settlement': 'settlement', 'inhabited place': 'settlement',
    'populated place': 'settlement', 'country house': 'building', 'prince bishopric': 'bishopric', 'prince archbishopric': 'bishopric', 'imperial abbey': 'bishopric',
    'electoral principate': 'electorate', 'holy see': 'bishopric', 'administrative division': 'district', 'administrative territorial entity': 'area',
    'political territorial entity': 'area', 'territorial entity': 'area', 'archaeological site': 'site', 'historic site': 'site', 'historical site': 'site',
    'deserted settlement': 'settlement', 'lost settlement': 'settlement', 'ghost town': 'settlement', 'old town': 'city_quarter', 'city centre': 'city_quarter',
    'city center': 'city_quarter', 'civil parish': 'parish', 'parish village': 'village', 'college town': 'town', 'port city': 'city', 'big city': 'city', 'border city': 'city',
    'hanseatic city': 'city', 'county town': 'town', 'district town': 'town', 'free imperial': 'city', 'crown land': 'province', 'former municipality': 'settlement',
    'abolished municipality': 'settlement', 'rural municipality': 'settlement', 'urban municipality': 'settlement', 'statutory city': 'city', 'federal city': 'city',
    'independent city': 'city', 'city with powiat rights': 'city', 'unitary municipality': 'settlement', 'cadastral populated place': 'settlement',
    'church building': 'church', 'ceremonial county': 'shire', 'non metropolitan county': 'shire', 'administrative county': 'shire', 'county palatine': 'county',
    'territorial collectivity': 'province', 'federal subject': 'province', 'imperial circle': 'province',
    'urban area': 'settlement', 'recreation area': 'site', 'unparished area': 'settlement', 'unitary authority area': 'district',
    'drainage basin': 'natural', 'dwelling place': 'building',
    'former administrative territorial entity': 'area', 'former administrative division': 'area', 'second level subdivision': 'district', 'third level subdivision': 'district',
    'first level subdivision': 'province', 'dependent state': 'country', 'former nation state empire': 'country',
}
_SPLIT_OF = re.compile(r'\s+(?:of|in|within|on|at|from|for|by|near|to)(?:\s+.*)?$', re.I)
_WITH = re.compile(r'\s+(?:with|without|that|which|where|and|or|under|bordering|considered|sometimes|divided|located|situated)(?:\s+.*)?$', re.I)
_PAREN = re.compile(r'\([^)]*\)')
_URBAN_DISTRICT = re.compile(r'^urban district (?:of|in) ')


def _singular(w: str) -> str:
    for suf, rep in (('ies', 'y'), ('ses', 's'), ('es', ''), ('s', '')):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            c = w[:-len(suf)] + rep
            if c in _HEAD or c in _NEUTRAL_HEADS:
                return c
    return w


@lru_cache(maxsize=None)
def wd_label_type(label) -> Optional[str]:
    """Fine type named by the English label of a Wikidata class ('free imperial city' -> city, 'municipality of Switzerland' -> settlement (generic), 'river' -> river,
    'federated state of Germany' -> province); None when the label names no sort of place ('tourist destination', 'financial center'). Head noun first: 'island country'
    is a country, 'river island' an island."""
    if not isinstance(label, str) or not label.strip():
        return None
    s = fold(_PAREN.sub(' ', label))
    if not s:
        return None
    if _URBAN_DISTRICT.match(s):                         # 'urban district of Bavaria' = a kreisfreie Stadt (independent city), not a quarter
        return 'city'
    s = _SPLIT_OF.sub('', s)
    s = _WITH.sub('', s)
    toks = [t for t in s.replace('-', ' ').split() if t]
    if not toks:
        return None
    for n in (4, 3, 2):
        ph = ' '.join(toks[-n:])
        if len(toks) >= n and ph in _PHRASE:
            return _PHRASE[ph]
    ph = ' '.join(toks)
    if ph in _PHRASE:
        return _PHRASE[ph]
    last = _singular(toks[-1])
    if last in _HEAD:
        return _HEAD[last]
    if last in _NEUTRAL_HEADS:
        return None
    # 'Lutherstadt', 'Residenzstadt', 'Hansestadt': German compounds in -stadt are cities, in -dorf villages
    if last.endswith('stadt') and len(last) > 7:
        return 'city'
    return None


# ======================================================================================================================
# 4. GeoNames feature codes
# ======================================================================================================================
_GN: dict = {}


def _g(typ, codes):
    for c in codes.split():
        if _GN.get(c, typ) != typ:
            raise AssertionError(f'GeoNames code {c} mapped to {_GN[c]} and {typ}')
        _GN[c] = typ


_g('continent', 'CONT')
_g('country', 'PCL PCLD PCLF PCLH PCLI PCLIX PCLS')
_g('province', 'ADM1 ADM1H')
_g('district', 'ADM2 ADM2H ADMD ADMDH ADM5H ADM3H ADM4H')
_g('settlement', 'ADM3 ADM4 ADM5 PPLH PPLS PPLR STLMT PPLCH')
_g('parish', 'PRSH')
_g('region', 'RGN RGNE RGNH RGNL AREA TERR LTER')
_g('area', 'ZN ZNB')
_g('city', 'PPLC PPLG PPLA PPLA2')
_g('city_quarter', 'PPLX')
_g('hamlet', 'PPLL')
_g('village', 'PPLF')
_g('site', 'PPLQ PPLW LCTY CMP CMPMN CMPSC CMPTX CMPQ CMPRF CMPLA CMPFO CMPBS CMPHR CMPSK CMPSQ')
_g('river', 'STM STMA STMB STMC STMD STMH STMI STMIX STMM STMQ STMS STMSB STMX CNL CNLA CNLB CNLD CNLI CNLN CNLQ CNLSB CNLX DTCH DTCHD DTCHI DTCHM CRKT RPDS FLLS FLLSX')
_g('lake', 'LK LKC LKI LKN LKNI LKO LKOI LKS LKSB LKSC LKSI LKSN LKSNI LKX PND PNDI PNDN PNDNI PNDS PNDSF PNDSI PNDSN PNDQ POOL POOLI RSV RSVI RSVT')
_g('sea', 'SEA OCN OCNS OCNM STRT STRTM SD CHN CHNM CHNN CHNL')
_g('bay', 'RDST ESTY ESTYQ BAY BAYS BGHT COVE FJD FJDS GULF INLT INLTQ LGN LGNS LGNX LGNN RDST HBR HBRX ANCH ANCHD BNKU HBRP')
_g('island', 'ISL ISLS ISLET ISLF ISLM ISLT ISLX ATOL ATOLL')
_g('mountain', 'MT MTS MTSU MTU MTO PK PKS PKSU PKU HLL HLLS HLLU KNLL KNL RDGE RDGB RDGU RDGEU VLC MND MNDS CRTR SLP SLPS DPR CLF CLFS')
_g('pass', 'PASS')
_g('valley', 'VAL VALS VALG VALX VALU CNYN CNYU GRGE RAV TRGD GULD PLN PLNS PLAT PLATS DSRT DSRTX')
_g('forest', 'FRST FRSTF WOOD HTH HTHSE MDW GRSLD SCRB')
_g('cape', 'CAPE PT PEN PENX SPIT PRMT NSE')
_g('marsh', 'MRSH MRSHN MRSHP SWMP BOG BOGS FEN PEAT MOOR WTLD')
_g('natural', 'BCH BCHS CAV CAVE DUNE SPNG SPNS SPNT WLL WLLS WTRC WTRH GLCR ICEF SHOL SBED REEF RF RFX BNK BNKX BAR ISTH OAS RK RKS ROCK TUND FLD FLDI '
              'MGV LAVA SAND SNDR DEPU UPLD SLID SPRG ')
_g('church', 'CH CHRY TMPL MSQE MSSN MSSNQ SHRN SYG PGDA')
_g('monastery', 'MSTY CNVT ABBY ABYS')
_g('castle', 'CSTL')
_g('palace', 'PAL')
_g('fort', 'FT FTHSE WALL BSTN BTYD CTDL')
_g('gate', 'GATE')
_g('bridge', 'BDG BDGQ')
_g('inn', 'HTL INN RSRT SPA')
_g('street', 'RD RDA RDB RDCUT RDJCT ST STKR TNL TNLRD TNLS TNLN TRL')
_g('square', 'SQR MKT PRKGT')
_g('site', 'AMTH RUIN ARCH ANS HSTS MNMT CMTY PYR PYRS TMB CARN BTL BTLFLD OBPT PRK PRKHQ GDN GDNS ZOO STDM RECG CTRB BUR CMTY MN MNA OILF')
_g('building', 'RSTN BLDG BLDO BP BRKS CTHSE FRM FRMS FRMT FRMQ FRMR GOSP HSE HSEC HSP HSPC HSPD HSPL ML MLM MLSG MLSGQ MLWND MLWTR MUS OPRA PS PSH PO POST PP PPQ '
              'RET RHSE SCH SCHA SCHC SCHN SCHM SCHT SNTR STNC STNM STNR THTR TOWR UNIV USGE VETF LIBR LTHSE OBS ASTR AIRP AIRB AIRF AIRH AIRQ BANK BNKR EST ESTT ESTO ESTR '
              'ESTSG ESTSL ESTX FCL INDS FRMQ HMSD BDLD DAM DAMQ DAMSB DIKE PIER QUAY DCK DCKB DCKY JTY ASYL REST SWT SHSE LNDF PRN PRNJ PRNQ PRNN HUT HUTS PKLT '
              'WHRF STNB MFG MFGB MFGC MFGCU MFGLM MFGM MFGPH MFGQ MFGSG')
# the codes of the first letters when the code itself is not listed
_GN_CLASS = {'A': 'area', 'P': 'settlement', 'H': 'natural', 'T': 'natural', 'L': 'area', 'R': 'street', 'S': 'built', 'U': 'natural', 'V': 'natural'}
#: today's population thresholds for the size of a populated place (a proxy: the editions' own size is not known). 'city' from 50,000, 'town' from 5,000, 'village' below 1,000
POP_CITY, POP_TOWN, POP_VILLAGE = 50000, 5000, 1000


def gn_type(fcode, pop=None) -> Optional[str]:
    """Fine type of a GeoNames feature code. A populated place (PPL, PPLA3 ...) is a ``city`` for PPLC / PPLA / PPLA2 and for 50,000 inhabitants or more, a ``town`` from 5,000,
    a ``village`` below 1,000 (``pop`` known only), else the generic ``settlement``: GeoNames' administrative level says little about the size of the place in 1500."""
    f = fcode if isinstance(fcode, str) else ''
    f = f.strip().upper()
    if not f or f in ('NONE', 'NAN'):
        return None
    if f.startswith('PPL') and f not in ('PPLH', 'PPLQ', 'PPLW', 'PPLR', 'PPLX', 'PPLL', 'PPLF', 'PPLS', 'PPLCH', 'PPLC', 'PPLG'):
        try:
            p = float(pop) if pop is not None else None
        except (TypeError, ValueError):
            p = None
        if f in ('PPLA', 'PPLA2'):
            return 'city'
        if p is not None and p == p:
            if p >= POP_CITY:
                return 'city'
            if p >= POP_TOWN:
                return 'town'
            if 0 < p < POP_VILLAGE:
                return 'village'
        return 'settlement'
    t = _GN.get(f)
    if t:
        return t
    if f.startswith('PCL'):
        return 'country'
    return _GN_CLASS.get(f[0]) if f[:1] in _GN_CLASS else None


# ======================================================================================================================
# 5. Getty TGN place types
# ======================================================================================================================
_TGN = {
    'inhabited place': 'settlement', 'nation': 'country', 'country': 'country', 'former nation/state/empire': 'country', 'dependent state': 'country', 'state': 'province',
    'general region': 'region', 'historical region': 'region', 'ancient region': 'region', 'region (administrative division)': 'region', 'regional division': 'region',
    'former administrative division': 'region', 'former group of nations/states/cities': 'region', 'area': 'region', 'deserted settlement': 'settlement',
    'island': 'island', 'island group': 'island', 'river': 'river', 'stream': 'river', 'creek': 'river', 'canal': 'river', 'hill': 'mountain', 'mountain': 'mountain',
    'mountains': 'mountain', 'mountain range': 'mountain', 'mountain system': 'mountain', 'volcano': 'mountain', 'continent': 'continent', 'subcontinent': 'continent',
    'neighborhood': 'city_quarter', 'suburb': 'city_quarter', 'ortsteil': 'city_quarter', 'borough': 'city_quarter', 'rione': 'city_quarter', 'province': 'province',
    'autonomous community': 'province', 'third level subdivision': 'district', 'fourth level subdivision': 'district', 'second level subdivision': 'district',
    'national district': 'district', 'administrative division': 'district', 'canton': 'canton', 'county': 'county', 'fief': 'lordship', 'sea': 'sea', 'ocean': 'sea',
    'lake': 'lake', 'cape': 'cape', 'point': 'cape', 'peninsula': 'cape', 'inlet': 'bay', 'bay': 'bay', 'gulf': 'bay', 'pass': 'pass', 'forest': 'forest', 'gorge': 'valley',
    'plain': 'valley', 'valley': 'valley', 'castle': 'castle', 'palace': 'palace', 'fort': 'fort', 'abbey': 'monastery', 'religious center': 'settlement',
    'road': 'street', 'ruins': 'site', 'ancient site': 'site', 'farm': 'building', 'autonomous city': 'city', 'special city': 'city', 'city': 'city', 'town': 'town',
    'village': 'village', 'cave': 'natural', 'desert': 'natural', 'anchorage': 'bay', 'locality': 'settlement', 'church': 'church', 'bridge': 'bridge',
}


def tgn_type(s) -> Optional[str]:
    """Fine type of a Getty TGN place type ('inhabited place' -> settlement; a TGN record with several types ('river; stream') gives the first that is known)."""
    if not isinstance(s, str) or not s.strip():
        return None
    for part in re.split(r'[;|]', s):
        k = ' '.join(part.lower().split())
        if k in _TGN:
            return _TGN[k]
        t = wd_label_type(k)
        if t:
            return t
    return None


# ======================================================================================================================
# 6. Orbis Latinus (Graesse-Benedict 1909): the book's abbreviations and the parser's normalised types
# ======================================================================================================================
#: the abbreviations of the book (St. Stadt, Mfl. Marktflecken, D. Dorf, Fl. Fluss, Gfsch. Grafschaft ...) as the parser reads them (``parse_orbis_latinus.TYPE_CANON``)
ORBIS_ABBR = {
    'st': 'city', 'mfl': 'market_town', 'd': 'village', 'fl': 'river', 'nfl': 'river', 'qufl': 'river', 'zfl': 'river', 'ins': 'island', 'isl': 'island', 'insel': 'island',
    'ibl': 'island', 'islgr': 'island', 'inselgr': 'island', 'vgb': 'cape', 'bg': 'mountain', 'gbg': 'mountain', 'gb': 'mountain', 'bggr': 'mountain', 'hochebene': 'valley',
    'gschf': 'county', 'gfsch': 'county', 'lschf': 'region', 'lach': 'region', 'ld': 'country', 'kgr': 'kingdom', 'fstt': 'principality', 'hgt': 'duchy', 'prov': 'province',
    'bz': 'district', 'hschf': 'lordship', 'ldvgt': 'bailiwick', 'kanton': 'canton', 'schi': 'castle', 'schl': 'castle', 'schloss': 'castle', 'fst': 'fort', 'fstg': 'fort',
    'fat': 'fort', 'festung': 'fort', 'klst': 'monastery', 'abt': 'monastery', 'ru': 'site', 'hf': 'bay', 'mbs': 'bay', 'hbl': 'cape', 'hbisl': 'cape', 'oschf': 'settlement',
    'bd': 'town', 'vst': 'city_quarter', 'sttl': 'city_quarter', 'stt': 'city_quarter', 'lac': 'lake', 'see': 'lake', 'strandsee': 'lake', 'mtl': 'market_town',
    'mu': 'market_town', 'tal': 'valley', 'pass': 'pass', 'kanal': 'river', 'gau': 'region', 'id': 'island', 'pg': 'region', 'str': 'sea',
}
#: the normalised ``place_type`` column of data/orbis_latinus.tsv
ORBIS_TYPE = {
    'city': 'city', 'village': 'village', 'river': 'river', 'market_town': 'market_town', 'region': 'region', 'island': 'island', 'islands': 'island', 'county': 'county',
    'country_or_region': 'region', 'province': 'province', 'mountain': 'mountain', 'mountains': 'mountain', 'duchy': 'duchy', 'valley': 'valley', 'settlement': 'settlement',
    'cape': 'cape', 'lake': 'lake', 'canton': 'canton', 'pass': 'pass', 'gulf': 'bay', 'peninsula': 'cape', 'kingdom': 'kingdom', 'spa': 'town', 'strait': 'sea',
    'plateau': 'valley', 'city_district': 'city_quarter', 'principality': 'principality', 'canal': 'river', 'country': 'country', 'district': 'district',
    'castle': 'castle', 'fortress': 'fort', 'monastery': 'monastery', 'abbey': 'monastery', 'ruin': 'site', 'harbour': 'bay', 'suburb': 'city_quarter',
    'lordship': 'lordship', 'bailiwick': 'bailiwick', 'people': None,
}


def orbis_type(t) -> Optional[str]:
    """Fine type of an Orbis Latinus type: the normalised ``place_type`` of ``data/orbis_latinus.tsv`` ('market_town') or the book's own abbreviation ('Mfl.', 'Gfsch.')."""
    if not isinstance(t, str) or not t.strip():
        return None
    k = t.strip().lower()
    if k in ORBIS_TYPE:
        return ORBIS_TYPE[k]
    k = k.replace('.', '').replace('ä', 'a').replace('ö', 'o').replace('ü', 'u').replace(' ', '')
    return ORBIS_ABBR.get(k)


# ======================================================================================================================
# 7. the editions' own register types (listPlace @type, note_type of the tables)
# ======================================================================================================================
#: None = the value is a technical label of the harvest or says nothing ('place', 'editor_slug', 'from_text_forms')
_REG = {
    'place': None, 'editor_slug': None, 'standoff_slug': None, 'from_header_forms': None, 'from_text_forms': None, 'printed_register': None, 'document_origin': None,
    'document_sent_from': None, 'epithet_only': None, 'celestial': None, 'none': None, 'nan': None, 'unknown': None,
    'country': 'country', 'coun': 'country', 'countr': 'country', 'place_country': 'country', 'region': 'region', 'place_ea_region': 'region', 'historische region': 'region',
    'continent': 'continent', 'cont': 'continent', 'city': 'city', 'stadt': 'town', 'gemeinde': 'settlement', 'gemeindeteil': 'settlement', 'dorfteil': 'settlement',
    'siedlung': 'settlement', 'sett': 'settlement', 'settlement': 'settlement', 'town': 'town', 'dorf': 'village', 'vill': 'village', 'village': 'village', 'weiler': 'hamlet',
    'stadtteil': 'city_quarter', 'quarter': 'city_quarter', 'place_populatedarea': 'settlement', 'place_pa_settlement': 'settlement', 'place_pa_village': 'village',
    'place_pa_town': 'town', 'place_pa_city': 'city', 'place_pa_suburb': 'city_quarter', 'place_pa_ward': 'city_quarter', 'place_pa_liberty': 'district',
    'place_pa_manor': 'lordship', 'place_pa_parish': 'parish', 'kirchspiel': 'parish', 'place_county': 'shire', 'place_ea_county': 'shire', 'county': 'county',
    'place_ea_province': 'province', 'province': 'province', 'dist': 'district', 'district': 'district', 'landschaft': 'region', 'kanton': 'canton',
    'eidgenossischer ort': 'canton', 'eidgenossischer ort kanton': 'canton', 'herrschaft': 'lordship', 'gerichtsherrschaft': 'lordship', 'freiherrschaft': 'lordship',
    'landvogtei': 'bailiwick', 'vogtei': 'bailiwick', 'amt': 'bailiwick', 'gemeine herrschaft': 'bailiwick', 'landgericht': 'district', 'dekanat': 'district',
    'grafschaft': 'county', 'landgrafschaft': 'landgraviate', 'markgrafschaft': 'margraviate', 'herzogtum': 'duchy', 'furstentum': 'principality', 'kurfurstentum': 'electorate',
    'konigreich': 'kingdom', 'place_duchy': 'duchy', 'place_ea_diocese': 'bishopric', 'bistum': 'bishopric', 'erzbistum': 'bishopric', 'hochstift': 'bishopric',
    'place_ea_archdeaconry': 'bishopric', 'diocese': 'bishopric',
    'moun': 'mountain', 'mountain': 'mountain', 'berg': 'mountain', 'hugel': 'mountain', 'rive': 'river', 'river': 'river', 'fluss': 'river', 'flus': 'river', 'bach': 'river',
    'place_ea_river': 'river', 'lake': 'lake', 'see': 'lake', 'lac': 'lake', 'place_ea_lake': 'lake', 'sea': 'sea', 'place_ea_sea': 'sea', 'island': 'island', 'insel': 'island',
    'place_ea_island': 'island', 'place_ea_mountain': 'mountain', 'wald': 'forest', 'forest': 'forest', 'place_ea_forest': 'forest', 'valley': 'valley', 'tal': 'valley',
    'place_f_waterfeature': 'natural', 'graben': 'natural',
    'stre': 'street', 'street': 'street', 'strasse': 'street', 'weg': 'street', 'place_f_street': 'street', 'place_f_openarea': 'site', 'platz': 'square', 'place_f_market': 'square',
    'burg': 'castle', 'castle': 'castle', 'schloss': 'castle', 'place_f_residence': 'building', 'place_f_palace': 'palace', 'palace': 'palace', 'palast': 'palace',
    'place_f_church': 'church', 'kirche': 'church', 'church': 'church', 'capell': 'church', 'kapelle': 'church', 'kloster': 'monastery', 'monastery': 'monastery',
    'abbey': 'monastery', 'place_f_religioushouse': 'monastery', 'bruderhaus': 'monastery', 'johanniterkommende': 'monastery', 'chorherrenstift': 'monastery',
    'kanonissenstift': 'monastery', 'place_f_school': 'building', 'place_f_townhall': 'building', 'place_f_guildhall': 'building', 'place_f_victuallinghouse': 'inn',
    'place_f_inn': 'inn', 'place_f_playhouse': 'building', 'place_f_theatre': 'building', 'place_f_arena': 'building', 'place_f_property': 'building',
    'place_f_hospital': 'building', 'place_f_prison': 'building', 'place_f_placeofpunishment': 'building', 'place_f_innofcourt': 'building', 'place_f_storage': 'building',
    'place_f_other': 'built', 'place_f_gate': 'gate', 'gate': 'gate', 'tor': 'gate', 'place_f_bridge': 'bridge', 'bridge': 'bridge', 'brucke': 'bridge', 'verwaltungsgebaude': 'building',
    'mule': 'building', 'muhle': 'building', 'hof': 'building', 'meierhof': 'building', 'fronhof': 'building', 'gut': 'building', 'haus': 'building', 'spital': 'building',
    'building': 'building', 'brunnen': 'site', 'flurname': 'site', 'acker': 'site', 'wies weideland': 'site', 'fahrstelle': 'site',
}
_REG_SUFFIX = (('kloster', 'monastery'), ('stift', 'monastery'), ('kommende', 'monastery'), ('kirche', 'church'), ('burg', 'castle'), ('herrschaft', 'lordship'),
               ('grafschaft', 'county'), ('herzogtum', 'duchy'), ('furstentum', 'principality'), ('vogtei', 'bailiwick'))


def register_type(t, casebooks_type: Optional[str] = None) -> Optional[str]:
    """Fine type of the type an edition's register gives its entry (``note_type``: 'region', 'place_f_church', 'herrschaft, kastlanei', 'klarissenkloster'). A list of types
    ('herrschaft, amt, gemeine herrschaft, grafschaft') gives the first that is known; ``casebooks_type`` is the Casebooks directory's ``entry_type`` (parish, village ...)."""
    if isinstance(casebooks_type, str) and casebooks_type.strip():
        k = casebooks_type.strip().lower()
        v = {'parish': 'parish', 'village': 'village', 'town': 'town', 'hamlet': 'hamlet', 'estate': 'building', 'manor': 'lordship', 'district': 'district',
             'township': 'settlement', 'liberty': 'district'}.get(k)
        if v:
            return v
    if not isinstance(t, str) or not t.strip():
        return None
    k = fold(t.replace(',', ' , '))
    whole = fold(t)
    if whole in _REG:
        return _REG[whole]
    raw = t.strip().lower()
    if raw in _REG:
        return _REG[raw]
    for part in re.split(r'[,;/]', t):
        p = fold(part)
        if not p:
            continue
        if p in _REG and _REG[p] is not None:
            return _REG[p]
        for suf, typ in _REG_SUFFIX:
            if p.endswith(suf) and len(p) > len(suf):
                return typ
    return None


# ======================================================================================================================
# 8. the name itself
# ======================================================================================================================
#: qualifier words (folded): 'Böhmen, Königreich', 'Ballenstedt, Amt', 'Gaster, Landvogtei', 'Stolberg, Grafschaft' ... (a superset of ``polities.QUALIFIERS``)
QUALIFIER_WORDS = {
    'kingdom': 'konigreich koenigreich kingdom royaume regno reino regnum koninkrijk kongerige konungariket vizekonigreich viceroyalty sultanat khanat khanate sultanate',
    'duchy': 'herzogtum herzogthum duchy duche ducato ducado ducatus hertogdom grossherzogtum grand duche granducato archduchy erzherzogtum archiduche arciducato '
             'grossfurstentum',
    'county': 'grafschaft county comte contea condado comitatus graafschap freigrafschaft pfalzgrafschaft',
    'landgraviate': 'landgrafschaft landgraviate landgraviat',
    'margraviate': 'markgrafschaft markgraviate margraviate margraviat marquisat marchesato marca',
    'principality': 'furstentum furstenthum principality principaute principato principado principatus vorstendom hetmanat hetmanate',
    'electorate': 'kurfurstentum kurfurstenthum kurfurstenthumb electorate electorat elettorato electoratus kurpfalz',
    'republic': 'republik republic republique repubblica republica respublica commonwealth freistaat',
    'empire': 'kaiserreich empire imperium impero imperio',
    'bishopric': 'bistum bisthum erzbistum hochstift erzstift stift fursterzbistum furstbistum bishopric archbishopric eveche archeveche diocese bisdom episcopatus reichsabtei '
                 'fursterzstift',
    'canton': 'kanton canton cantone',
    'lordship': 'herrschaft freiherrschaft lordship seigneurie signoria senorio dominium gerichtsherrschaft honour',
    'bailiwick': 'amt amtshauptmannschaft landvogtei vogtei bailliage baliaggio',
    'district': 'bezirk contado kreis distretto district arrondissement hundred',
    'province': 'provinz province provincia gouvernement governorate woiwodschaft wojewodztwo voivodeship departement',
    'region': 'landschaft region regio',
    'republic_union': 'eidgenossenschaft confederation confederatio confederazione',
    'city': 'stadtstaat',
}
_QW: dict = {}
for _t_, _ws in QUALIFIER_WORDS.items():
    for _w in _ws.split():
        _QW.setdefault(_w, 'republic' if _t_ == 'republic_union' else _t_)
#: a qualifier that is also a plain word: 'Stadt' after a comma is a town / city, 'Insel' an island, 'Fluss' a river ('Elbe, Fluss')
_QW.update({'stadt': 'settlement', 'gebirge': 'mountain', 'insel': 'island', 'fluss': 'river', 'burg': 'castle', 'schloss': 'castle', 'festung': 'fort', 'kloster': 'monastery', 'berg': 'mountain',
            'halbinsel': 'cape', 'pass': 'pass', 'kirche': 'church', 'bergstadt': 'town', 'talstadt': 'town', 'vorstadt': 'city_quarter', 'stadtteil': 'city_quarter'})
#: words of the name (not after a comma) that fix the type; whole words only
_NW_ANY = {
    'kloster': 'monastery', 'abtei': 'monastery', 'priory': 'monastery', 'abbey': 'monastery', 'convent': 'monastery', 'monastery': 'monastery', 'nunnery': 'monastery',
    'schloss': 'castle', 'schlosz': 'castle', 'burg': 'castle', 'castle': 'castle', 'chateau': 'castle', 'castello': 'castle', 'castillo': 'castle', 'festung': 'fort',
    'fortress': 'fort', 'fort': 'fort', 'palace': 'palace', 'palast': 'palace', 'palazzo': 'palace', 'palais': 'palace', 'cathedral': 'church', 'church': 'church',
    'chapel': 'church', 'kirche': 'church', 'kapelle': 'church', 'inn': 'inn', 'tavern': 'inn', 'hospital': 'building', 'spital': 'building', 'college': 'building',
    'guildhall': 'building', 'playhouse': 'building', 'prison': 'building', 'gaol': 'building', 'rathaus': 'building', 'bridge': 'bridge', 'brucke': 'bridge',
    'gate': 'gate', 'street': 'street', 'strasse': 'street', 'gasse': 'street', 'lane': 'street', 'alley': 'street', 'square': 'square', 'platz': 'square',
    'market': 'square', 'wharf': 'site', 'cathedral': 'church', 'amt': 'bailiwick', 'herrschaft': 'lordship', 'grafschaft': 'county', 'landgrafschaft': 'landgraviate',
    'markgrafschaft': 'margraviate', 'herzogtum': 'duchy', 'furstentum': 'principality', 'kurfurstentum': 'electorate', 'konigreich': 'kingdom', 'kingdom': 'kingdom',
    'duchy': 'duchy', 'landvogtei': 'bailiwick', 'vogtei': 'bailiwick', 'hochstift': 'bishopric', 'erzstift': 'bishopric', 'bistum': 'bishopric', 'erzbistum': 'bishopric',
    'diocese': 'bishopric', 'shire': 'shire', 'county': 'county', 'province': 'province', 'provinz': 'province', 'kanton': 'canton', 'fluss': 'river', 'river': 'river',
    'bach': 'river', 'insel': 'island', 'island': 'island', 'isle': 'island', 'lake': 'lake', 'mount': 'mountain', 'mountain': 'mountain', 'gebirge': 'mountain',
    'halbinsel': 'cape', 'parish': 'parish',
}
#: 'Parish of St. Martin' is a church, 'St Paul's Cathedral' a church; whole-word 'bach' and 'berg' are NOT used (Offenbach, Hamburg are towns): see ``name_type``
_NW_FIRST = {'berg': 'mountain', 'monte': 'mountain', 'mons': 'mountain', 'mount': 'mountain', 'mt': 'mountain', 'lago': 'lake', 'lac': 'lake', 'rio': 'river',
             'fiume': 'river', 'riviere': 'river', 'fleuve': 'river', 'isola': 'island', 'ile': 'island', 'isla': 'island', 'bad': 'town'}
_NW_PLAIN_RIVERS = frozenset('thames themse rhine rhein rhin rhenus danube donau danubius elbe albis weser visurgis maas meuse mosa scheldt seine sequana loire liger rhone rhodanus '
                             'tiber tevere tagus tajo ebro vistula weichsel volga oder odera main mosel moselle'.split())
_NW_FILLER = frozenset('the der die das le la les l river fluss rivier fleuve flumen fluvius riviere'.split())
_ST = frozenset('st saint sainte sankt s san santo santa sant'.split())
_DE_PREP = frozenset('am im bei an ob zu unter uber'.split())


def qualifier_word(name, path=None) -> tuple:
    """(word, type) of the qualifier after the last comma, or (None, None): 'Böhmen, Königreich' -> ('konigreich', 'kingdom'); 'Askanien (Aschersleben), Grafschaft' -> ('grafschaft',
    'county'); 'Weston, Hertfordshire' -> (None, None). The qualifier may be several words ('Grafschaft bzw. Fürstentum (seit 1652)', 'Stadt und Republik'): the first word that
    names an area wins, else the first that names anything. A reviewed row (source ``qualifier``, key = the word) overrides the built-in table."""
    if not isinstance(name, str) or ',' not in name:
        return None, None
    base = _PAREN.sub(' ', name)
    parts = [p for p in base.split(',') if p.strip()]
    if len(parts) < 2:
        return None, None
    words = fold(parts[-1]).split()
    hits = []
    for w in words:
        found, t = _reviewed('qualifier', w, path)
        t = t if found else _QW.get(w)
        if t:
            hits.append((w, t))
    if not hits:
        return None, None
    for w, t in hits:
        if kind_of(t) == 'a' and t not in GENERIC:
            return w, t
    return hits[0]


def qualifier_type(name, path=None) -> Optional[str]:
    """Type named by the qualifier after the last comma: 'Böhmen, Königreich' -> kingdom, 'Weston, Hertfordshire' -> None (see ``qualifier_word``)."""
    return qualifier_word(name, path)[1]


def name_word(name, path=None) -> tuple:
    """(word, type) of the word of the name that fixes its type (not the qualifier after a comma), or (None, None): "St. Paul's Cathedral" -> ('cathedral', 'church'), 'Kloster Zinna'
    -> ('kloster', 'monastery'), 'The Thames' -> ('thames', 'river'), 'Monte Somma' -> ('monte', 'mountain'). Whole words only; 'Burg' and 'Berg' only as the first word ('Hamburg',
    'Offenbach', 'Rorbach' are not named by them); a river's name counts only when the name is the river and nothing else. A reviewed row (source ``name``) overrides the table."""
    if not isinstance(name, str) or not name.strip():
        return None, None
    toks = fold(name.replace("'", ' ').replace('-', ' ')).split()
    if not toks:
        return None, None

    def look(w, default):
        found, t = _reviewed('name', w, path)
        return t if found else default

    if len(toks) > 1 and toks[0] in _NW_FIRST:                       # 'Monte Rosa', 'Berg Sinai' (a bare 'Berg' or 'Mons' is a village or a town)
        t = look(toks[0], _NW_FIRST[toks[0]])
        if t:
            return toks[0], t
    if 'parish' in toks and any(t in _ST for t in toks):
        return 'parish of st', look('parish of st', 'church')
    rest = [t for t in toks if t not in _NW_FILLER]
    if len(rest) == 1 and rest[0] in _NW_PLAIN_RIVERS:
        return rest[0], 'river'
    german_prep = any(t in _DE_PREP for t in toks)
    # English names end in their head ('Whitehall Palace', 'Market Street'), German ones begin with it ('Kloster Zinna'): the last word first, then the others
    order = [toks[-1]] + toks[:-1] if len(toks) > 1 else toks
    for t in order:
        if t in _NW_ANY:
            if t == 'inn' and german_prep:                               # 'Rattenberg am Inn' is on the river
                continue
            ty = look(t, _NW_ANY[t])
            if ty:
                return t, ty
    return None, None


def name_type(name, path=None) -> Optional[str]:
    """Type a place NAME says by itself: the qualifier after a comma first ('Pommern, Herzogtum' -> duchy), else a word of the name (``name_word``); None for a plain name."""
    q = qualifier_type(name, path)
    return q or name_word(name, path)[1]


# ======================================================================================================================
# 9. the reviewed table
# ======================================================================================================================
DATA = Path(__file__).resolve().parents[2] / 'data'
REVIEWED_PATH = DATA / 'placetypes.tsv'
SOURCES = ('wd', 'gn', 'tgn', 'orbis', 'register', 'qualifier', 'name', 'casebooks')
#: ``type`` of a reviewed row that says 'looked at, deliberately no type' (a class like 'tourist destination'): the lookup returns None and does not fall back to the rules
NEUTRAL = '-'


@lru_cache(maxsize=4)
def load_reviewed(path=None) -> dict:
    """Read ``data/placetypes.tsv`` -> {(source, key): type_code or None}. Columns: source (wd gn tgn orbis register qualifier name casebooks), key (Wikidata QID, GeoNames
    code, TGN type, Orbis abbreviation or type, register type, folded qualifier or name word, Casebooks entry_type), label, type (a code of ``TYPES`` or ``-`` = reviewed, no
    type), n_mentions, n_places, status (``reviewed`` = read by a human against the counts, ``auto`` = the rule's answer, not yet read), note. A row with an unknown type is an
    error. Rows whose ``type`` is empty are ignored."""
    p = Path(path) if path else REVIEWED_PATH
    out: dict = {}
    if not p.exists():
        return out
    with open(p, newline='', encoding='utf-8') as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter='\t'), start=2):
            typ = (r.get('type') or '').strip()
            if not typ:
                continue
            src = (r.get('source') or '').strip()
            if src not in SOURCES:
                raise ValueError(f'{p}:{i}: unknown source {src!r}')
            if typ != NEUTRAL and typ not in TYPES:
                raise ValueError(f'{p}:{i}: unknown type {typ!r}')
            out[(src, (r.get('key') or '').strip())] = None if typ == NEUTRAL else typ
    return out


def _reviewed(source: str, key, path=None):
    """(found, type)."""
    tab = load_reviewed(str(path) if path else None)
    k = (source, str(key or '').strip())
    if k in tab:
        return True, tab[k]
    return False, None


def wd_type(qid: Optional[str] = None, label: Optional[str] = None, path=None) -> Optional[str]:
    """Fine type of a Wikidata class: the reviewed row of the QID first, else the label rules; None for a class that says nothing."""
    found, t = _reviewed('wd', qid, path)
    if found:
        return t
    return wd_label_type(label)


# ======================================================================================================================
# 10. classification of a place from all its sources
# ======================================================================================================================
#: weight of a vote by source (as ``placekind.kind5``: Wikidata and GeoNames first, the register and TGN a little less, the bare name least)
WEIGHTS = {'wd': 1.0, 'gn': 1.0, 'tgn': 0.8, 'register': 0.8, 'casebooks': 0.8, 'orbis': 0.7, 'qualifier': 1.2, 'name': 0.6, 'exonym': 1.0, 'label': 0.9, 'inherit': 0.5}
#: a vote for a generic type (a group) counts this fraction, and it lifts the specific types of its own group that other sources voted for
GENERIC_FACTOR = 0.3


@dataclass
class Evidence:
    """Everything the sources say about one place. All fields are optional."""
    wd_classes: Sequence = ()          # [(qid, label_en)] or ['Q515:city', ...] (the format of places_global.classes)
    wd_label_en: Optional[str] = None  # English label of the item itself ('Duchy of Savoy', 'Margraviate of Meissen'): its qualifier word is a vote
    gn_fcodes: Sequence = ()           # GeoNames feature codes: 'PPLA3' or ('PPLA3', population)
    gn_pop: Optional[float] = None
    tgn_types: Sequence = ()
    notes: Sequence = ()               # register types (note_type)
    casebooks_types: Sequence = ()
    names: Sequence = ()               # names of the place as written in the register (first three used)
    orbis_types: Sequence = ()
    exonym_kind: Optional[str] = None  # 'city' / 'area' of the exonym table
    inherited: Sequence = ()           # types of the places this one was propagated from
    kind: Optional[str] = None         # the coarse kind the atlas has already decided (c a n b u): the fine type is chosen inside it


@dataclass
class TypeResult:
    code: str
    source: str                        # the source that carried the winning type most strongly ('' for other)
    votes: dict = field(default_factory=dict)
    consistent: bool = True            # False when the unrestricted vote had another coarse kind than ``Evidence.kind`` (the fine type was then chosen inside the given kind)


def parse_classes(classes) -> list:
    """places_global.classes ('Q4022:river|Q573344:main stem') or a list of 'Q..:label' / (qid, label) -> [(qid, label)]."""
    if classes is None:
        return []
    if isinstance(classes, str):
        classes = classes.split('|')
    out = []
    for c in classes:
        if isinstance(c, (tuple, list)) and len(c) >= 2:
            out.append((str(c[0]), str(c[1])))
        elif isinstance(c, str) and c:
            q, _, lab = c.partition(':')
            out.append((q, lab))
    return out


def votes_of(ev: Evidence, path=None) -> dict:
    """{type: (weight, source)}-style votes: Counter-like dict type -> total weight, plus a parallel dict of the strongest source per type under key ``('src', type)``."""
    votes: Counter = Counter()
    best: dict = {}

    def add(t, src, w=None):
        if not t:
            return
        w = (WEIGHTS[src] if w is None else w) * (GENERIC_FACTOR if t in GENERIC else 1.0)
        votes[t] += w
        if w > best.get(t, (0, ''))[0]:
            best[t] = (w, src)

    for q, lab in parse_classes(ev.wd_classes):
        add(wd_type(q, lab, path), 'wd')
    if ev.wd_label_en:
        add(qualifier_type_of_label(ev.wd_label_en), 'label')
    for f in ev.gn_fcodes:
        code, pop = (f if isinstance(f, (tuple, list)) else (f, ev.gn_pop))
        add(gn_type(code, pop), 'gn')
    for t in ev.tgn_types:
        found, ty = _reviewed('tgn', fold(t) if isinstance(t, str) else '', path)
        add(ty if found else tgn_type(t), 'tgn')
    for n in ev.notes:
        found, ty = _reviewed('register', str(n).strip().lower() if n is not None else '', path)
        add(ty if found else register_type(n), 'register')
    for n in ev.casebooks_types:
        add(register_type(None, n), 'casebooks')
    for t in ev.orbis_types:
        add(orbis_type(t), 'orbis')
    for nm in list(ev.names)[:3]:
        q = qualifier_type(nm)
        if q:
            add(q, 'qualifier')
        else:
            add(name_type(nm), 'name')
    if ev.exonym_kind in ('city', 'area'):
        add('city' if ev.exonym_kind == 'city' else 'area', 'exonym')
    for t in ev.inherited:
        add(t, 'inherit')
    return {'votes': votes, 'best': best}


_LABEL_Q = re.compile(r'^(?:the\s+)?(?P<q>[a-z\- ]+?)\s+of\s+', re.I)


def qualifier_type_of_label(label) -> Optional[str]:
    """Type named by an English Wikidata item label of the form 'Duchy of Savoy', 'Margraviate of Meissen', 'Prince-Bishopric of Liège', 'County of Hainaut', 'Free Imperial
    City of Nuremberg'; None for 'Savoy'. Only polity and ecclesiastical words count (a label 'University of Padua' is not a type)."""
    if not isinstance(label, str):
        return None
    m = _LABEL_Q.match(label.strip())
    if not m:
        return None
    q = fold(m.group('q'))
    if q in ('free imperial city', 'imperial city', 'free city'):
        return 'city'
    if q in ('prince bishopric', 'prince archbishopric', 'archbishopric', 'bishopric', 'imperial abbey', 'prince abbey'):
        return 'bishopric'
    if q in ('grand duchy', 'archduchy'):
        return 'duchy'
    if q in ('principality', 'lordship', 'duchy', 'county', 'kingdom', 'republic', 'landgraviate', 'margraviate', 'electorate', 'canton', 'empire', 'bailiwick'):
        return q
    return None


def classify(ev: Evidence, path=None) -> TypeResult:
    """The fine type of a place from all its sources.

    Weighted vote (``WEIGHTS``; a generic type counts ``GENERIC_FACTOR``). A generic vote ('settlement') lifts the specific types of its group that other sources voted for.
    If ``ev.kind`` is given, only types of that coarse kind compete: the fine type refines the atlas's kind and never contradicts it ('consistent' is False when the free vote
    would have had another kind). With no vote of the right kind the answer is the generic type of the kind when the kind is c a n b, else ``other``.
    Ties: the more specific political or physical form first (``_ORDER``)."""
    v = votes_of(ev, path)
    votes: Counter = v['votes']
    best = v['best']
    if not votes:
        k = ev.kind if ev.kind in KINDS else None
        return TypeResult(GROUP_OF_KIND[k] if k and k != 'u' else 'other', '', {}, True)
    # a generic vote supports the specific types of its group that have their own votes
    adj = Counter(votes)
    for t, w in votes.items():
        if t in GENERIC:
            for t2 in votes:
                if t2 not in GENERIC and group_of(t2) == t:
                    adj[t2] += w * 0.5

    def pick(cands):
        if not cands:
            return None
        leaf = [t for t in cands if t not in GENERIC]
        pool = leaf or list(cands)
        top = max(adj[t] for t in pool)
        win = [t for t in pool if abs(adj[t] - top) < 1e-9]
        return min(win, key=lambda t: _RANK[t])

    free = pick(list(adj))
    consistent = True
    k = ev.kind if ev.kind in KINDS else None
    if k and k != 'u':
        cands = [t for t in adj if kind_of(t) == k]
        t = pick(cands)
        consistent = kind_of(free) == k
        if t is None:
            return TypeResult(GROUP_OF_KIND[k], '', dict(votes), consistent)
        return TypeResult(t, best.get(t, (0, ''))[1], dict(votes), consistent)
    if k == 'u':
        return TypeResult('other', '', dict(votes), kind_of(free) == 'u')
    return TypeResult(free, best.get(free, (0, ''))[1], dict(votes), True)


def classify_many(evs: Iterable[Evidence], weights: Optional[Iterable[float]] = None, kind: Optional[str] = None, path=None) -> TypeResult:
    """The type of a group of places (the members of a globe point): the weighted vote of their individual types, specific before generic."""
    tot: Counter = Counter()
    src: dict = {}
    ws = list(weights) if weights is not None else None
    for i, ev in enumerate(evs):
        r = classify(ev, path)
        w = ws[i] if ws is not None else 1.0
        tot[r.code] += w * (GENERIC_FACTOR if r.code in GENERIC else 1.0)
        src.setdefault(r.code, r.source)
    if kind in KINDS:
        tot = Counter({t: w for t, w in tot.items() if kind_of(t) == kind}) or tot
    if not tot:
        return TypeResult('other', '')
    leaf = [t for t in tot if t not in GENERIC and t != 'other'] or list(tot)
    top = max(tot[t] for t in leaf)
    win = min((t for t in leaf if abs(tot[t] - top) < 1e-9), key=lambda t: _RANK[t])
    return TypeResult(win, src.get(win, ''), dict(tot))


# ======================================================================================================================
# 11. analysis on the tables of a build (read-only): evidence of every local place, term inventory, coverage
# ======================================================================================================================
#: coord_source values of ``places_local_geo`` that mean 'located by a name' (the page's "located by name": propagation, GeoNames / TGN name match, exonym and polity table, Latin)
NAME_BASED_SOURCES = ('name_propagation', 'name_geonames', 'name_exonym', 'name_latin')
_KIND3 = {'city': 'c', 'area': 'a', 'other': 'u'}
_FC_EVID = re.compile(r'GeoNames \d+ .*? \(([A-Z0-9]{2,5}), [A-Z]{2}(?:, pop (\d+))?')
_EXO_EVID = re.compile(r'\((city|area), radius')
_ORBIS_EVID = re.compile(r'orbis ([^;,]+)')


def local_evidence_frame(final_dir, wd_entities=None, gn_index=None, orbis_tsv=None, casebooks_tsv=None, globe_places=None, point_kinds=None):
    """One row per local place of ``places_local_geo.parquet`` with the evidence of every source, as columns:
    ``classes`` ([(qid, label)] of the global place), ``wd_label_en``, ``fcode`` (GeoNames feature code: the local column, else the one in the geocoder's evidence, else the
    candidate's or the authority id's in ``gn_index``), ``gn_pop``, ``tgn_type``, ``note_type``, ``cb_type`` (Casebooks entry_type), ``orbis`` (types of the Orbis entries behind a
    Latin name), ``exo_kind``, ``cand_key`` (the local place a propagated point was copied from), ``kind`` (the coarse kind of the globe point, else the 3-way kind) and
    ``point`` (the id of the globe point). ``point_kinds``: {point id: kind} overriding the kind of the points (the page's core.json kinds). ``final_dir``: the directory with ``places_local_geo.parquet`` and ``places_global.parquet`` (and ``globe_places.json``)."""
    import json
    import pandas as pd
    fd = Path(final_dir)
    pl = pd.read_parquet(fd / 'places_local_geo.parquet')
    pg = pd.read_parquet(fd / 'places_global.parquet', columns=['id', 'globe_id', 'classes', 'qid', 'label_en'])
    gp = Path(globe_places) if globe_places else fd / 'globe_places.json'
    k_of_point = {}
    if gp.exists():
        k_of_point = {p['id']: p.get('k') for p in json.loads(gp.read_text('utf-8'))['places']}
    pg['point'] = pg['globe_id']
    pg['pk'] = pg['globe_id'].map(k_of_point)
    df = pl.merge(pg.rename(columns={'id': 'global_id', 'qid': 'g_qid', 'label_en': 'g_label_en'}), on='global_id', how='left')
    df['classes'] = [parse_classes(c) if isinstance(c, str) else [] for c in df['classes']]
    # English label of the Wikidata item
    labs = {}
    if wd_entities is not None:
        we = pd.read_parquet(wd_entities, columns=['qid', 'labels'])
        need = set(df['g_qid'].dropna()) | set(df['qid'].dropna())
        for q, l in zip(we['qid'], we['labels']):
            if q in need and isinstance(l, str):
                try:
                    labs[q] = json.loads(l).get('en')
                except ValueError:
                    pass
    df['wd_label_en'] = [labs.get(q) or labs.get(q2) for q, q2 in zip(df['g_qid'], df['qid'])]
    # GeoNames feature code
    ev = df['gc_evidence'].fillna('').str.extract(_FC_EVID)
    df['fcode'] = df['gn_fcode'].where(df['gn_fcode'].notna(), ev[0])
    if 'gn_pop' not in df:
        df['gn_pop'] = None
    pe = pd.to_numeric(ev[1], errors='coerce')
    df['gn_pop'] = df['gn_pop'].where(df['gn_pop'].notna() | df['gn_fcode'].notna(), pe)
    if gn_index is not None:
        gi = pd.read_parquet(gn_index, columns=['gid', 'fcode', 'pop', 'prim']).sort_values('prim', ascending=False).drop_duplicates('gid').set_index('gid')
        cand = df['gc_cand'].fillna('').str.extract(r'^gn:(\d+)$', expand=False)
        gid = df['geonames_id'].astype(str).str.split(';').str[0].where(df['geonames_id'].notna())
        for src in (cand, gid):
            fc = src.map(gi['fcode'])
            pop = src.map(gi['pop'])
            miss = df['fcode'].isna() & fc.notna()
            df.loc[miss, 'fcode'] = fc[miss]
            df.loc[miss & df['gn_pop'].isna(), 'gn_pop'] = pop[miss & df['gn_pop'].isna()]
    # Casebooks directory entry type
    df['cb_type'] = None
    if casebooks_tsv is not None and Path(casebooks_tsv).exists():
        cb = pd.read_csv(casebooks_tsv, sep='\t', dtype=str, keep_default_na=False)
        df['cb_type'] = df['cb_place_id'].map(dict(zip(cb['place_id'], cb['entry_type']))) if 'cb_place_id' in df else None
    # Orbis Latinus types behind a Latin name
    ob = {}
    if orbis_tsv is not None and Path(orbis_tsv).exists():
        from .latin_names import fold_key as _lfold
        o = pd.read_csv(orbis_tsv, sep='\t', dtype=str, keep_default_na=False)
        for k, t in zip(o['latin_name'], o['place_type']):
            ob.setdefault(_lfold(k), set()).add(t)
    lat = df['gc_source'].eq('name_latin')
    df['orbis'] = [sorted({t for w in _ORBIS_EVID.findall(e or '') for t in ob.get(_lfold(w.strip()), ())}) if m else [] for e, m in zip(df['gc_evidence'], lat)] if ob \
        else [[] for _ in range(len(df))]
    df['exo_kind'] = df['gc_evidence'].fillna('').str.extract(_EXO_EVID, expand=False).where(df['gc_source'].eq('name_exonym'))
    df['cand_key'] = df['gc_cand'].where(df['coord_source'].eq('name_propagation'))
    df['kind'] = df['pk'].where(df['pk'].isin(KINDS), df['place_kind'].map(_KIND3))
    if point_kinds:
        pk = df['point'].map(point_kinds)
        df['kind'] = pk.where(pk.isin(KINDS), df['kind'])
    return df


def evidence_of(row) -> Evidence:
    """Evidence of one row of ``local_evidence_frame``."""
    def nz(x):
        return x if isinstance(x, str) and x and x not in ('None', 'nan') else None
    return Evidence(wd_classes=row['classes'], wd_label_en=nz(row.get('wd_label_en')), gn_fcodes=[x for x in [nz(row.get('fcode'))] if x], gn_pop=row.get('gn_pop'),
                    tgn_types=[x for x in [nz(row.get('tgn_type'))] if x], notes=[x for x in [nz(row.get('note_type'))] if x],
                    casebooks_types=[x for x in [nz(row.get('cb_type'))] if x], names=[x for x in [nz(row.get('name'))] if x], orbis_types=list(row.get('orbis') or []),
                    exonym_kind=nz(row.get('exo_kind')), kind=nz(row.get('kind')))


def pool_evidence(evs: Sequence, kind: Optional[str] = None) -> Evidence:
    """One Evidence from the evidence of several local places of one place: every distinct term counts once (a class, a feature code with its population, a TGN type, a register
    type, a Casebooks type, an Orbis type), the names are the three most frequent distinct ones (``evs`` should be ordered by mentions, most mentioned first)."""
    seen_c, classes = set(), []
    gn, tgn, notes, cbt, orb, names, inh = [], [], [], [], [], [], []
    wd_label, exo = None, None
    for ev in evs:
        for q, lab in parse_classes(ev.wd_classes):
            if q not in seen_c:
                seen_c.add(q)
                classes.append((q, lab))
        for f in ev.gn_fcodes:
            f = f if isinstance(f, (tuple, list)) else (f, ev.gn_pop)
            if f not in gn:
                gn.append(f)
        for lst, src in ((tgn, ev.tgn_types), (notes, ev.notes), (cbt, ev.casebooks_types), (orb, ev.orbis_types), (inh, ev.inherited)):
            for x in src:
                if x not in lst:
                    lst.append(x)
        for nm in ev.names:
            if nm not in names:
                names.append(nm)
        wd_label = wd_label or ev.wd_label_en
        exo = exo or ev.exonym_kind
    return Evidence(wd_classes=classes, wd_label_en=wd_label, gn_fcodes=gn, tgn_types=tgn, notes=notes, casebooks_types=cbt, names=names[:3], orbis_types=orb,
                    exonym_kind=exo, inherited=inh, kind=kind)


def classify_frame(df, path=None):
    """Fine type of every local place, from the pooled evidence of the place it belongs to: the locals of one globe point (``point``), else of one global place (``global_id``), else
    the local place alone. Adds ``type`` (of the pool), ``type_source``, ``type_consistent``, ``type_local`` (from the local's own evidence only), ``pool``.
    A propagated point (``cand_key``) that is still generic inherits the type of the place it was copied from."""
    df = df.copy()
    evs = [evidence_of(r) for r in df.to_dict('records')]
    own = [classify(e, path) for e in evs]
    df['type_local'] = [r.code for r in own]
    pool_key = df['point'].where(df['point'].notna(), df['global_id']).where(lambda x: x.notna(), df['coll_slug'] + ':' + df['entity_key'])
    df['pool'] = pool_key.values
    order = df['n_in_period'].fillna(0).to_numpy().argsort(kind='stable')[::-1]
    members: dict = defaultdict(list)
    for i in order:
        members[pool_key.iat[i]].append(i)
    res: dict = {}
    for k, idx in members.items():
        kinds = Counter()
        for i in idx:
            if evs[i].kind:
                kinds[evs[i].kind] += max(int(df['n_in_period'].iat[i] or 0), 1)
        kind = kinds.most_common(1)[0][0] if kinds else None
        res[k] = classify(pool_evidence([evs[i] for i in idx], kind), path)
    df['type'] = [res[k].code for k in pool_key]
    df['type_source'] = [res[k].source for k in pool_key]
    df['type_consistent'] = [res[k].consistent for k in pool_key]
    key = dict(zip(df['coll_slug'] + ':' + df['entity_key'], df['type']))
    inh = df['cand_key'].map(key)
    for i in df.index[inh.notna() & df['type'].isin(list(GROUPS))]:
        pos = df.index.get_loc(i)
        e = pool_evidence([evs[pos]], evs[pos].kind)
        e.inherited = [inh[i]]
        r = classify(e, path)
        if r.code not in GROUPS or r.code != df.at[i, 'type']:
            df.at[i, 'type'], df.at[i, 'type_source'], df.at[i, 'type_consistent'] = r.code, r.source, r.consistent
    return df


def term_inventory(df) -> 'pd.DataFrame':
    """Every source term with the placed in-period mentions and places that carry it: columns source, key, label, n_places, n_mentions. Placed = ``coord_source != 'none'``.
    Wikidata classes are counted once per local place (a local place carries all the classes of its global place)."""
    import pandas as pd
    d = df[df['coord_source'] != 'none']
    acc: dict = defaultdict(lambda: [0, 0])
    labels: dict = {}

    def add(src, key, n, label=''):
        if key in (None, '') or (isinstance(key, float) and key != key):
            return
        a = acc[(src, str(key))]
        a[0] += 1
        a[1] += int(n)
        if label and (src, str(key)) not in labels:
            labels[(src, str(key))] = label

    for r in d.to_dict('records'):
        n = r['n_in_period']
        for q, lab in r['classes']:
            add('wd', q, n, lab)
        add('gn', r.get('fcode'), n)
        add('tgn', fold(r['tgn_type']) if isinstance(r.get('tgn_type'), str) else None, n, r.get('tgn_type') or '')
        nt = r.get('note_type')
        add('register', nt.strip().lower() if isinstance(nt, str) else None, n, nt if isinstance(nt, str) else '')
        add('casebooks', r.get('cb_type') if isinstance(r.get('cb_type'), str) and r.get('cb_type') else None, n)
        for t in r.get('orbis') or []:
            add('orbis', t, n)
        qw, q = qualifier_word(r.get('name'))
        if q:
            add('qualifier', qw, n, r['name'])
        else:
            nw, nt2 = name_word(r.get('name'))
            if nt2:
                add('name', nw, n, r['name'])
    rows = [dict(source=s, key=k, label=labels.get((s, k), ''), n_places=v[0], n_mentions=v[1]) for (s, k), v in acc.items()]
    return pd.DataFrame(rows).sort_values(['source', 'n_mentions'], ascending=[True, False]).reset_index(drop=True)


def rule_type_of_term(source: str, key: str, label: str = '') -> Optional[str]:
    """The built-in rule's answer for a term (without the reviewed table)."""
    if source == 'wd':
        return wd_label_type(label)
    if source == 'gn':
        return gn_type(key, None)
    if source == 'tgn':
        return tgn_type(label or key)
    if source == 'register':
        return register_type(label or key)
    if source == 'casebooks':
        return register_type(None, key)
    if source == 'orbis':
        return orbis_type(key)
    if source == 'qualifier':
        return _QW.get(key)
    if source == 'name':
        return {'parish of st': 'church'}.get(key) or _NW_ANY.get(key) or _NW_FIRST.get(key) or ('river' if key in _NW_PLAIN_RIVERS else None)
    return None


def coverage(df) -> dict:
    """Share of the placed in-period mentions that get a fine type, overall and by coordinate source / deciding source; mentions per type."""
    d = df[df['coord_source'] != 'none']
    tot = int(d['n_in_period'].sum())
    other = int(d.loc[d['type'] == 'other', 'n_in_period'].sum())
    generic = int(d.loc[d['type'].isin(GROUPS) & (d['type'] != 'other'), 'n_in_period'].sum())
    out = {'placed_mentions': tot, 'with_type': tot - other, 'share_with_type': round((tot - other) / max(tot, 1), 4),
           'with_specific_type': tot - other - generic, 'share_specific': round((tot - other - generic) / max(tot, 1), 4),
           'generic_only': generic, 'share_generic_only': round(generic / max(tot, 1), 4),
           'inconsistent_with_atlas_kind_mentions': int(d.loc[~d['type_consistent'], 'n_in_period'].sum())}
    out['by_coord_source'] = {s: {'mentions': int(g['n_in_period'].sum()), 'share_with_type': round(float(g.loc[g['type'] != 'other', 'n_in_period'].sum() / max(g['n_in_period'].sum(), 1)), 4),
                                  'share_specific': round(float(g.loc[~g['type'].isin(GROUPS), 'n_in_period'].sum() / max(g['n_in_period'].sum(), 1)), 4)}
                              for s, g in d.groupby('coord_source')}
    out['by_type_source'] = {str(s or 'none'): int(g['n_in_period'].sum()) for s, g in d.groupby(d['type_source'].fillna(''))}
    out['mentions_by_type'] = {t: int(v) for t, v in d.groupby('type')['n_in_period'].sum().sort_values(ascending=False).items()}
    out['places_by_type'] = {t: int(v) for t, v in d.groupby('type').size().sort_values(ascending=False).items()}
    return out


# ======================================================================================================================
# 12. maintaining data/placetypes.tsv, the command line
# ======================================================================================================================
REVIEWED_COLUMNS = ('source', 'key', 'label', 'type', 'n_mentions', 'n_places', 'status', 'note')
MIN_MENTIONS = 20                      # a term with this many placed mentions must have a reviewed row


def read_reviewed_rows(path=None) -> list:
    p = Path(path) if path else REVIEWED_PATH
    if not p.exists():
        return []
    with open(p, newline='', encoding='utf-8') as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter='\t')]


def write_reviewed_rows(rows: list, path=None) -> None:
    p = Path(path) if path else REVIEWED_PATH
    rows = sorted(rows, key=lambda r: (SOURCES.index(r['source']), -int(r.get('n_mentions') or 0), r['key']))
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEWED_COLUMNS, delimiter='\t', extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in REVIEWED_COLUMNS})
    load_reviewed.cache_clear()


def merge_inventory(inv, path=None, decisions: Optional[Mapping] = None) -> dict:
    """Bring ``data/placetypes.tsv`` up to date with a term inventory (``term_inventory``): counts are refreshed, a term that has no row gets one with the rule's answer and status
    ``reviewed`` if ``decisions`` (a {(source, key): (type, note)} mapping of a human's decisions) has it, else ``auto``. Existing rows keep their type and status. Returns counts.
    The rule's answer is blank for a term the rules do not know; such a row with 20 or more mentions is listed by ``uncovered``."""
    rows = {(r['source'], r['key']): r for r in read_reviewed_rows(path)}
    new = 0
    for r in inv.to_dict('records'):
        k = (r['source'], str(r['key']))
        if k in rows:
            rows[k].update(n_mentions=r['n_mentions'], n_places=r['n_places'])
            if not rows[k].get('label'):
                rows[k]['label'] = r['label']
            continue
        rule = rule_type_of_term(r['source'], r['key'], r['label'])
        typ, status, note = rule or '', 'auto', ''
        if decisions and k in decisions:
            typ, note = decisions[k]
            status = 'reviewed'
        elif rule and r['n_mentions'] >= MIN_MENTIONS and decisions is not None:
            status = 'reviewed'                  # read against the counts and left as the rule said
        rows[k] = dict(source=r['source'], key=r['key'], label=r['label'], type=typ, n_mentions=r['n_mentions'], n_places=r['n_places'], status=status, note=note)
        new += 1
    write_reviewed_rows(list(rows.values()), path)
    return {'rows': len(rows), 'new': new}


def uncovered(inv, path=None, min_mentions: int = MIN_MENTIONS) -> list:
    """Terms with ``min_mentions`` or more placed mentions that neither a reviewed row (with a type or ``-``) nor the built-in rules map: the list the next review starts from."""
    tab = load_reviewed(str(path) if path else None)
    out = []
    for r in inv.to_dict('records'):
        if r['n_mentions'] < min_mentions:
            continue
        if (r['source'], str(r['key'])) in tab:
            continue
        if rule_type_of_term(r['source'], r['key'], r['label']):
            continue
        out.append((r['source'], r['key'], r['label'], int(r['n_mentions'])))
    return out


def coverage_table(df, inv):
    """The coverage table (one DataFrame): ``summary`` rows (shares), ``type`` rows (places and mentions per fine type, with the coarse kind and the parent) and ``term`` rows
    (every source term with its counts and the type it maps to, ``covered`` = a reviewed row or a rule maps it)."""
    import pandas as pd
    cov = coverage(df)
    rows = []

    def add(section, source, key, label, typ, n_places, n_mentions, note=''):
        rows.append(dict(section=section, source=source, key=key, label=label, type=typ, n_places=n_places, n_mentions=n_mentions, note=note))

    add('summary', '', 'placed_mentions', 'placed in-period mentions (coord_source != none)', '', '', cov['placed_mentions'])
    add('summary', '', 'with_fine_type', 'mentions of a place whose fine type is not "other" (generic group types included)', '', '', cov['with_type'], f"{cov['share_with_type']:.4f}")
    add('summary', '', 'with_specific_type', 'mentions of a place whose fine type is a leaf (not "area", "settlement", "natural", "built", "other")', '', '', cov['with_specific_type'],
        f"{cov['share_specific']:.4f}")
    add('summary', '', 'generic_only', 'mentions of a place known only to its group', '', '', cov['generic_only'], f"{cov['share_generic_only']:.4f}")
    add('summary', '', 'decided_by_a_source_term', 'mentions whose type was decided by at least one source term (not by the atlas kind alone)', '', '',
        cov['placed_mentions'] - cov['by_type_source'].get('none', 0), f"{1 - cov['by_type_source'].get('none', 0) / max(cov['placed_mentions'], 1):.4f}")
    for s, v in cov['by_coord_source'].items():
        add('summary', s, 'share_with_type_by_coord_source', f"{v['mentions']} placed mentions", '', '', v['mentions'], f"with type {v['share_with_type']:.4f}; leaf {v['share_specific']:.4f}")
    for s, n in cov['by_type_source'].items():
        add('summary', s, 'type_decided_by', 'mentions whose winning type came from this source', '', '', n)
    add('summary', '', 'kind_disagreements', 'mentions whose free vote had another coarse kind than the atlas kind (fine type chosen inside the atlas kind)', '', '',
        cov['inconsistent_with_atlas_kind_mentions'])
    for t, n in cov['mentions_by_type'].items():
        add('type', '', t, TYPES[t].label, t, cov['places_by_type'].get(t, 0), n, f"kind {TYPES[t].kind}; parent {TYPES[t].parent or '-'}")
    tab = load_reviewed()
    for r in inv.to_dict('records'):
        k = (r['source'], str(r['key']))
        rule = rule_type_of_term(r['source'], r['key'], r['label'])
        typ = tab[k] if k in tab else rule
        covered = k in tab or bool(rule)
        add('term', r['source'], r['key'], r['label'], typ or ('-' if k in tab else ''), r['n_places'], r['n_mentions'],
            ('covered' if covered else 'NOT COVERED') + ('; reviewed row' if k in tab else '; rule only') + ('' if r['n_mentions'] >= MIN_MENTIONS else '; < 20 mentions'))
    return pd.DataFrame(rows)


def _main(argv=None) -> int:
    import argparse
    import json
    import sys
    ap = argparse.ArgumentParser(prog='python -m ratlas.placetypes', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    iv = sub.add_parser('inventory', help='read-only analysis of a build: term counts, coverage table; optionally update data/placetypes.tsv')
    iv.add_argument('--final', type=Path, required=True, help='directory with places_local_geo.parquet, places_global.parquet, globe_places.json')
    iv.add_argument('--wd-entities', type=Path)
    iv.add_argument('--gn-index', type=Path)
    iv.add_argument('--orbis', type=Path, default=DATA / 'orbis_latinus.tsv')
    iv.add_argument('--casebooks', type=Path, default=DATA / 'casebooks_place_map.tsv')
    iv.add_argument('--reviewed', type=Path, default=REVIEWED_PATH)
    iv.add_argument('--out', type=Path, required=True, help='coverage table (TSV)')
    iv.add_argument('--update', action='store_true', help='append the terms without a row to the reviewed table (status auto)')
    a = ap.parse_args(argv)
    df = local_evidence_frame(a.final, a.wd_entities, a.gn_index, a.orbis, a.casebooks)
    df = classify_frame(df, a.reviewed)
    inv = term_inventory(df)
    if a.update:
        print(json.dumps(merge_inventory(inv, a.reviewed)), file=sys.stderr)
        load_reviewed.cache_clear()
        df = classify_frame(df, a.reviewed)
    ct = coverage_table(df, inv)
    ct.to_csv(a.out, sep='\t', index=False)
    miss = uncovered(inv, a.reviewed)
    print(json.dumps(coverage(df), indent=1))
    print(f'{len(miss)} terms with {MIN_MENTIONS}+ placed mentions are not covered', file=sys.stderr)
    for m in miss:
        print('  ', m, file=sys.stderr)
    return 0 if not miss else 2


if __name__ == '__main__':
    raise SystemExit(_main())
