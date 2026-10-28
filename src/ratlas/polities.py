"""Historical polities and 'Name, Qualifier' forms (v2.3; pure functions, no I/O).

Why. The TEI editions of the Anhalt diaries, the Religionsfrieden sources, the Swiss and Prussian letters name polities that no gazetteer has under that
name, in the form 'Head, Qualifier' (Böhmen, Königreich; Hessen-Kassel, Landgrafschaft; Pommern, Herzogtum; Niederlande, Vereinigte (Republik der Sieben
Vereinigten Provinzen)) or 'Name (Other name)' (Löbau (Lubawa); Askanien (Aschersleben), Grafschaft; Candia (Kreta), Königreich). v2.2 left 110,867 mentions
unplaced, among them the 2,400 mentions of these forms. Three tools:

* ``parse_qualified(raw)``: splits a name into head, alternative names (the parentheses) and the qualifier type (the polity word after the comma,
  ``kingdom`` ``duchy`` ``county`` ``landgraviate`` ``margraviate`` ``principality`` ``electorate`` ``republic`` ``bishopric`` ``circle`` ``confederation`` ...).
  A qualifier of one of these types makes the entity an **area** (the kind follows the qualifier, not the head: 'Pommern, Herzogtum' is a duchy, not the
  village of Pommern on the Moselle).
* ``POLITIES``: a small **reviewed** table of historical polities with the Wikidata item (verified in the local ``wd_entities.parquet`` of 2026-10-17), a
  point and a radius. Coordinates are never invented: ``basis`` says where each one comes from, ``wd`` = the item's own P625 value in the local fetch, ``seat``
  = the well-known coordinate of the seat or capital, ``centroid`` = the mean of named member places, ``table`` = the point of the curated exonym table
  (``exonyms.py``) for the region of the same name. The table is consulted by the geocoder like the exonym table (``name_exonym``, an area, medium).
* ``QTYPE_RADIUS``: the radius of the disc drawn for a polity whose head the geocoder resolved by GeoNames (a county named after its seat is the seat's
  neighbourhood, not a point).

Every entry is meant to be read by a human before a release (``docs/CONTRACT.md`` section 19).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from .merge import _norm
from .namegate import small_letters

#: qualifier type -> normalised words (``merge._norm``: lower case, no diacritics, 'ß' -> 'ss')
QUALIFIERS = {
    'kingdom': 'konigreich koenigreich konigreiche kingdom royaume regno reino regnum koninkrijk kongerige konungariket',
    'duchy': 'herzogtum herzogthum duchy duche ducato ducado ducatus hertogdom hertogdomme',
    'grand duchy': 'grossherzogtum grossfurstentum grand duche granducato grand duchy',
    'archduchy': 'erzherzogtum archduchy archiduche arciducato',
    'county': 'grafschaft county comte comte contea condado comitatus graafschap',
    'landgraviate': 'landgrafschaft landgraviate landgraviat',
    'margraviate': 'markgrafschaft markgraviate margraviate margraviat marquisat marchesato marca',
    'principality': 'furstentum furstenthum principality principaute principato principado principatus vorstendom',
    'electorate': 'kurfurstentum kurfurstenthum kurfurstenthumb electorate electorat elettorato electoratus',
    'republic': 'republik republic republique repubblica republica respublica',
    'union': 'vereinigte vereinigten vereinigter vereinigtes united unies uniti unidas unitae foederatae',
    'empire': 'kaiserreich empire imperium impero imperio',
    'bishopric': 'bistum bisthum erzbistum hochstift fursterzbistum furstbistum bishopric archbishopric eveche archeveche diocese bisdom episcopatus',
    'circle': 'reichskreis',
    'confederation': 'eidgenossenschaft confederation confederatio confederazione',
    'hetmanate': 'hetmanat hetmanate',
    'region': 'landschaft region provinz province provincia gouvernement governorate',
    'lordship': 'herrschaft lordship seigneurie signoria senorio dominium',
    'district': 'amt amtshauptmannschaft landvogtei vogtei bezirk',
    'viceroyalty': 'vizekonigreich vizekoenigreich viceroyalty vice royaume',
    'captaincy': 'kapitanat captaincy capitania',
}
_QWORD = {}
for _t, _ws in QUALIFIERS.items():
    for _w in _ws.split():
        _QWORD.setdefault(_norm(_w), _t)

#: qualifier types that make the entity an area
AREA_QTYPES = frozenset(QUALIFIERS) - {'union'}
#: radius (km) of the disc drawn for a polity placed at the point of its head (a county named after its seat ...); all are areas (radius >= 60 draws as one)
QTYPE_RADIUS = {'kingdom': 150.0, 'duchy': 90.0, 'grand duchy': 150.0, 'archduchy': 120.0, 'county': 40.0, 'landgraviate': 60.0, 'margraviate': 70.0,
                'principality': 50.0, 'electorate': 100.0, 'republic': 120.0, 'empire': 400.0, 'bishopric': 60.0, 'circle': 200.0, 'confederation': 150.0,
                'hetmanate': 200.0, 'region': 100.0, 'lordship': 25.0, 'district': 15.0, 'viceroyalty': 300.0, 'captaincy': 300.0}


@dataclass
class Qualified:
    raw: str
    head: str = ''                         # normalised key of the head ('bohmen')
    head_raw: str = ''                     # the head as written, without parentheses ('Böhmen')
    qtype: Optional[str] = None            # 'kingdom' ... or None
    alts: list = field(default_factory=list)   # normalised keys of the parenthetical alternative names ('lubawa', 'aschersleben')
    full: str = ''                         # the whole name, normalised, parentheses flattened ('niederlande vereinigte republik der sieben vereinigten provinzen')

    @property
    def is_area(self) -> bool:
        return self.qtype in AREA_QTYPES


def is_qualifier(key: str) -> bool:
    """True when a normalised comma part is a polity qualifier ('konigreich', 'grafschaft bzw furstentum'): the part before it is the head, not a container."""
    return bool(key) and (key in _QWORD or any(w in _QWORD for w in key.split()))


_PAREN = re.compile(r'\(([^)]*)\)')


def _key(s) -> str:
    return _norm(small_letters(unicodedata.normalize('NFC', str(s or ''))).replace('’', "'"))


def parse_qualified(raw) -> Qualified:
    """'Böhmen, Königreich' -> head 'bohmen', qtype 'kingdom'; 'Askanien (Aschersleben), Grafschaft' -> head 'askanien', alts ['aschersleben'], qtype 'county';
    'Niederlande, Vereinigte (Republik der Sieben Vereinigten Provinzen)' -> head 'niederlande', qtype 'union', alts [the republic's full name];
    'Löbau (Lubawa)' -> head 'lobau', alts ['lubawa'], qtype None. The qualifier is the last comma part, parentheses taken out; only a known polity word counts
    ('Weston, Hertfordshire' has none)."""
    q = Qualified(raw=str(raw or ''))
    if not isinstance(raw, str) or not raw.strip():
        return q
    s = small_letters(unicodedata.normalize('NFC', raw)).replace('’', "'")
    q.full = _key(s.replace('(', ' ').replace(')', ' '))
    for p in _PAREN.findall(s):
        k = _key(p)
        if k and k not in q.alts:
            q.alts.append(k)
    base = _PAREN.sub(' ', s)
    parts = [p.strip() for p in base.split(',') if p.strip()]
    if len(parts) >= 2:
        qw = _key(parts[-1])
        # the qualifier may be several words ('Grafschaft bzw. Fürstentum (seit 1652)'): the first word that is a polity word decides
        qt = _QWORD.get(qw) or next((_QWORD[w] for w in qw.split() if w in _QWORD), None)
        if qt:
            q.qtype = qt
            q.head_raw = ', '.join(parts[:-1])
        else:
            q.head_raw = parts[0]
    else:
        q.head_raw = parts[0] if parts else ''
    q.head = _key(q.head_raw)
    # 'Hohnstein (Grafschaft)': the bracket is the polity word itself, not a second name
    if q.qtype is None:
        for a in list(q.alts):
            t = _QWORD.get(a) or next((_QWORD[w] for w in a.split() if w in _QWORD), None) if len(a.split()) <= 2 else None
            if t:
                q.qtype = t
                q.alts.remove(a)
                break
    return q


# ---------------------------------------------------------------------------------------------------------------------
# the reviewed table
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Polity:
    label: str            # English label (shown as the display name)
    qid: Optional[str]    # Wikidata item (verified in the local wd_entities.parquet), None where no local item says so
    lat: float
    lon: float
    radius: float
    basis: str            # where the point comes from: wd | seat:<town> | centroid:<members> | table (the exonym table's region of the same name)
    plain: tuple          # normalised forms that name the polity whatever the qualifier
    qualified: tuple = ()  # normalised heads that name the polity only with a polity qualifier ('Moldau, Fürstentum', not the river Moldau)


def _P(label, qid, lat, lon, radius, basis, plain, qualified=''):
    return Polity(label, qid, lat, lon, float(radius), basis, tuple(_key(x.replace('_', ' ')) for x in plain.split()),
                  tuple(_key(x.replace('_', ' ')) for x in qualified.split()))


# fmt: off
POLITIES = [
    _P('Holy Roman Empire', 'Q12548', 50.0, 10.0, 600, 'wd',
       'heiliges_romisches_reich heiliges_romisches_reich_deutscher_nation sacrum_romanum_imperium sacrum_imperium_romanum holy_roman_empire saint_empire '
       'saint_empire_romain_germanique sacro_romano_impero romisch_deutsches_reich imperium_romano_germanicum'),
    _P('Dutch Republic', 'Q170072', 52.08, 4.30, 160, 'wd',
       'niederlande_vereinigte vereinigte_niederlande verenigde_provincien provinces_unies united_provinces provinciae_unitae foederatae_belgii_provinciae '
       'generalstaaten vereinigte_provinzen sieben_vereinigte_provinzen republik_der_sieben_vereinigten_provinzen republik_der_vereinigten_niederlande '
       'united_netherlands provincias_unidas'),
    _P('Kingdom of Bohemia', 'Q42585', 50.0875, 14.4214, 130, 'wd',
       'konigreich_bohmen bohmen_konigreich bohmen_konigreich_krone bohmische_krone krone_bohmen kingdom_of_bohemia crown_of_bohemia royaume_de_boheme regnum_bohemiae '
       'regnum_boemiae'),
    _P('Muscovy', None, 55.7558, 37.6173, 500, 'seat:Moscow',
       'moskauer_reich moskowitisches_reich grossrussland moskauer_reich_grossrussland moskovien russisches_reich tsardom_of_russia tsardom_of_muscovy royaume_de_moscovie'),
    _P('Old Swiss Confederacy', 'Q351537', 46.7986, 8.2320, 150, 'wd (Switzerland, Q39)',
       'schweiz_eidgenossenschaft schweizerische_eidgenossenschaft confoederatio_helvetica old_swiss_confederacy corps_helvetique dreizehn_alte_orte'),
    _P('Five Catholic Cantons', None, 47.0164, 8.4980, 30, 'centroid: Lucerne, Altdorf, Schwyz, Stans, Zug',
       'funf_orte die_funf_orte funforte quinque_pagi quinque_pagi_catholici cinq_cantons_catholiques five_catholic_cantons funf_katholische_orte'),
    _P('Landgraviate of Hesse-Kassel', None, 51.3127, 9.4797, 70, 'seat:Kassel',
       'hessen_kassel landgrafschaft_hessen_kassel hesse_cassel hassia_casselana hessen_cassel'),
    _P('Landgraviate of Hesse-Marburg', None, 50.8021, 8.7667, 40, 'seat:Marburg', 'hessen_marburg hessen_marpurg'),
    _P('Landgraviate of Hesse-Darmstadt', None, 49.8728, 8.6512, 40, 'seat:Darmstadt', 'hessen_darmstadt hesse_darmstadt'),
    _P('Duchy of Pomerania', None, 54.0, 15.8, 120, 'table (Pomerania)', 'pomern pommern_herzogtum herzogtum_pommern pomerania_ducatus ducatus_pomeraniae pommerellen_herzogtum',
       ''),
    _P('Farther Pomerania', None, 54.1, 16.3, 100, 'centroid: Koszalin, Slupsk, Stargard (approximate)', 'hinterpommern hinterpommern_herzogtum ulterior_pomerania'),
    _P('Western Pomerania', None, 53.8, 13.5, 70, 'centroid: Stralsund, Greifswald, Anklam (approximate)', 'vorpommern vorpommern_herzogtum'),
    _P('Zaporozhian Sich', None, 47.82, 35.08, 150, 'seat:Khortytsia island (approximate)',
       'saporoger_sitsch saporoger_sitsch_hetmanat zaporozhian_sich zaporizhian_sich zaporogi saporoger_kosaken'),
    _P('Warmia', 'Q322620', 54.3583, 19.6833, 50, 'wd', 'ermland ermelandt ermeland warmia varmia ermlandia bistum_ermland ermelant'),
    _P('Masovia', 'Q643843', 52.9774, 19.0203, 100, 'wd', 'masovia masoviae masoviam masowien mazowsze mazovia masovien'),
    _P('Courland', 'Q185072', 57.0, 22.0, 100, 'wd', 'kurland curonia curlandia courland kurland_herzogtum herzogtum_kurland'),
    _P('Principality of Moldavia', 'Q10957559', 47.1585, 27.6014, 150, 'seat:Iasi', 'moldavia moldavie principatus_moldaviae', 'moldau moldova'),
    _P('Wallachia', 'Q171393', 44.43, 26.10, 130, 'wd', 'walachei wallachia valachia walachia wallachie'),
    _P('Duchy of Milan', 'Q153529', 45.4667, 9.1667, 70, 'wd', 'herzogtum_mailand ducato_di_milano ducatus_mediolanensis milanois milanais mailand_herzogtum'),
    _P('Valtellina', 'Q685105', 46.1667, 9.8667, 40, 'wd', 'valteline valtellina veltlin valtelina valle_telina vallis_telina vallis_tellina valtellinae'),
    _P('League of God\'s House', None, 46.8499, 9.5329, 50, 'seat:Chur', 'gotteshausbund gotteshaus_bund ligue_de_la_maison_de_dieu'),
    _P('Duchy of Bari', None, 41.1171, 16.8719, 40, 'seat:Bari', 'status_barensis ducatus_barensis ducato_di_bari'),
    _P('Polish-Lithuanian Commonwealth', 'Q172107', 50.05, 19.9333, 400, 'wd', 'polen_litauen polish_lithuanian_commonwealth respublica_polona rzeczpospolita'),
    _P('Electorate of Saxony', 'Q156199', 51.0333, 13.7333, 100, 'wd', 'kursachsen churfurstenthum_sachsen electoratus_saxoniae kursachsen_kurfurstentum'),
    _P('Duchy of Saxe-Weimar', 'Q315667', 50.9795, 11.3235, 50, 'seat:Weimar', 'sachsen_weimar sachsen_weimar_herzogtum'),
    _P('Upper Saxon Circle', 'Q429086', 51.0, 13.0, 250, 'table (Saxony)', 'obersachsischer_reichskreis saxonia_superior_reichskreis saxonia_superior obersachsischer_kreis circulus_saxoniae_superioris'),
    _P('Franconian Circle', 'Q970082', 49.8, 10.5, 150, 'table (Franconia)', 'fränkischer_reichskreis frankischer_reichskreis fränkischer_kreis'),
    _P('Swabian Circle', 'Q741966', 48.3, 9.5, 130, 'table (Swabia)', 'schwabischer_reichskreis schwäbischer_reichskreis schwabischer_kreis'),
    _P('Bavarian Circle', 'Q708812', 48.9, 11.8, 130, 'table (Bavaria)', 'bayerischer_reichskreis bayerischer_kreis'),
    # v2.4 (check of v2.3: GeoNames / TGN heads on the wrong locality). Reviewed points; the whole qualified name is the key, never the bare head
    _P('County of Stolberg', None, 51.5667, 10.95, 30, 'seat:Stolberg (Harz); county of Stolberg-Wernigerode (the Anhalt diaries and the Religionsfrieden mean the Harz, not Lippe)',
       'stolberg_grafschaft stolberg_im_harz_grafschaft stolberg_harz_grafschaft stolberg_wernigerode_grafschaft grafschaft_stolberg'),
    _P('Bailiwick of Gaster', None, 47.158, 9.044, 25, 'seat:Schanis (the Landvogtei Gaster of Schwyz and Glarus, canton St. Gallen; not the Gastere valley near Kandersteg)',
       'gaster_landvogtei landvogtei_gaster'),
    _P('Lordship of Oberstein', None, 49.713, 7.313, 25, 'seat:Oberstein (Idar-Oberstein)', 'oberstein_herrschaft herrschaft_oberstein'),
    _P('County of Hohenlohe', None, 49.199, 9.506, 40, 'seat:Ohringen (Hohenlohe-Neuenstein, Franconian-Wurttemberg Hohenlohe)',
       'hohenlohe_grafschaft grafschaft_hohenlohe hohenlohe_furstentum furstentum_hohenlohe'),
    _P('Lordship of Falkenberg (Niemodlin)', None, 50.643, 17.585, 25, 'seat:Niemodlin (Falkenberg in Upper Silesia)',
       'falkenberg_herrschaft_in_oberschlesien falkenberg_herrschaft_oberschlesien falkenberg_oberschlesien'),
]
# fmt: on

#: normalised form -> Polity (plain forms) / head -> Polity (forms that need a polity qualifier)
PLAIN: dict = {}
NEEDS_QUALIFIER: dict = {}
for _p in POLITIES:
    for _f in _p.plain:
        PLAIN.setdefault(_f, _p)
    for _f in _p.qualified:
        NEEDS_QUALIFIER.setdefault(_f, _p)
BY_LABEL = {_p.label: _p for _p in POLITIES}
BY_QID = {_p.qid: _p for _p in POLITIES if _p.qid}


def lookup(q: Qualified) -> Optional[Polity]:
    """The table's polity for a parsed name: the whole name, the head, the alternative names; heads that need a qualifier only with a polity qualifier."""
    for k in (q.full, q.head, *q.alts):
        if k in PLAIN:
            return PLAIN[k]
    if q.qtype in AREA_QTYPES or q.qtype == 'union':
        for k in (q.head, *q.alts):
            if k in NEEDS_QUALIFIER:
                return NEEDS_QUALIFIER[k]
    return None


def radius_for(qtype: Optional[str]) -> float:
    return QTYPE_RADIUS.get(qtype or '', 40.0)
