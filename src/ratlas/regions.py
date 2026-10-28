"""Regional priors of the name-based geocoder that come from the document itself (v2.2; pure functions).

Why. The sample check of v2.1 found name-based points far from the territory of their own file. Hearth Tax: the 61 files are returns of one county
each (the file's header and its ``placeName@county`` heading say West Riding, Essex, Norfolk ...), and 169 places were put in another county (Newton in
Cornwall for a West Riding township) because the doc cloud of the geocoder pooled the 61 files under five ``doc_id`` values and had no anchor at
all (no register points). MoEML (the London placeography) and Old Bailey: streets, inns, churches and houses of London were put in Dorset, Somerset
and Hertfordshire (Marshalsea, Petty France, Newgate Street, Rosemary Lane), or in Lyon (Lyon's Inn) and Savoie (Savoy Manor).

Three mechanisms (all in ``reconcile_geocode``):

* **document regions**: ``doc_region(surfaces)`` reads the county (or city) of a Hearth Tax file from its own headings and gives a disc
  (centre, radius); a name-based candidate for a place of that file must lie in the disc of one of the files the place occurs in (countries exempt).
* **London prior**: MoEML (every name-based point) and Old Bailey (names that look like London streets, inns, churches, houses ...) must lie in
  Greater London unless the feature is a country, a first-order region, a capital or has a million inhabitants. The register's own coordinates are never
  touched (only places without coordinates are geocoded), so "unless the register says so" is automatic; the MoEML placeholder
  ``PLACE OUTSIDE OF LONDON`` is not a place at all (``classify.is_placeholder``).
* **file consistency** (``file_outliers``): after the run, in the collections whose files are regional by nature (``FILE_CONSISTENCY_COLLECTIONS``: the Hearth Tax), a
  name-based point of 60 km radius or less that lies more than 100 km from the median of a regional file (most of its placed points within 150 km of the median) is
  taken back, unless some other file of the place supports it.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

import numpy as np

#: collections whose files are returns of one county (the file's own headings give the region)
DOC_REGION_COLLECTIONS = frozenset({'hearth_tax'})
#: collections whose files are regional by nature, for the file-consistency rule. Not the legal sources and the letters (Basel's oath records, the Swiss
#: law sources and the letters name Italy, Lorraine, Valais from one file: 150 of 150 points the rule took back in a trial on all editions were such)
FILE_CONSISTENCY_COLLECTIONS = frozenset({'hearth_tax'})
#: Greater London (lat0, lat1, lon0, lon1)
LONDON_BOX = (51.28, 51.70, -0.52, 0.34)
#: v2.3: the box reaches into Hertfordshire at its north-west corner (Potters Bar, Elstree, Borehamwood, Aldenham: 'Green Street' of Hertfordshire is at
#: 51.676 N, -0.274 E, 25 km from the City). London plus Middlesex ends at Stanmore / Barnet: a point in this rectangle is not in the jurisdiction.
LONDON_CUT = (51.625, 51.72, -0.52, -0.12)
#: 'all': every name-based candidate of the collection must be in London (plus Middlesex); v2.3: Old Bailey too (a jurisdiction prior for EVERY name of the
#: trial reports, not only for the names that look like London streets: Holburn was put in Northumberland, Petty-France in Gloucestershire, Gatehouse in
#: Scotland, Kingsland in Herefordshire, Hermitage in Berkshire, Hamstead on the Isle of Wight, Flushing in Cornwall)
LONDON_PRIOR = {'map_of_early_modern_london': 'all', 'old_bailey': 'all'}
#: the Old Bailey names a place outside the jurisdiction on purpose now and then. Two exemptions, both documented: (a) a name that is not British ('Flushing,
#: a Town in Holland') may be placed where it is; (b) a provincial town of 100,000 inhabitants or more (Bristol, Oxford, York ...) is not a homonym of a hamlet
FOREIGN_OK = frozenset({'old_bailey'})
PROVINCIAL_TOWN_POP = 100_000
BRITISH_CC = frozenset({'GB', 'IE', 'IM', 'JE', 'GG'})
#: British Isles (lat0, lat1, lon0, lon1) for a point that has no country code (the exonym table)
BRITISH_BOX = (49.8, 61.0, -11.0, 1.8)
REGION_MARGIN_KM = 15.0
FILE_OUTLIER_KM = 100.0
FILE_REGIONAL_P80_KM = 150.0
FILE_MIN_PLACED = 8
FILE_POINT_MAX_RADIUS_KM = 60.0       # a result with a larger radius is an area (a county, a country): legitimate anywhere

# (label, lat, lon, radius_km, regex on the normalised heading). The first county / city of the list that matches wins; the order is specific -> general.
COUNTY_REGIONS = [
    ('West Riding of Yorkshire', 53.85, -1.65, 85, r'\bwest rid'),
    ('East Riding of Yorkshire', 53.95, -0.60, 65, r'\beast rid'),
    ('North Riding of Yorkshire', 54.35, -1.15, 85, r'\bnorth rid'),
    ('Kingston upon Hull', 53.745, -0.335, 25, r'kingston upon hull|\bhull\b|hullshire'),
    ('York', 53.96, -1.08, 25, r'^york\b.*\bcity\b|city of york'),
    ('Yorkshire', 54.00, -1.20, 140, r'yorkshire|\byorks\b'),
    ('Bristol', 51.45, -2.60, 20, r'\bbristol\b'),
    ('Chester', 53.19, -2.89, 15, r'(?:^chester\b.*\bcity\b|city of chester)'),
    ('Coventry', 52.41, -1.51, 15, r'\bcoventry\b'),
    ('Norwich', 52.63, 1.30, 15, r'norwich\b.*\bcit|city of norwich'),
    ('Durham', 54.70, -1.80, 55, r'\bdurham\b'),
    ('Westmorland', 54.45, -2.70, 50, r'westm[eo]r'),
    ('Cheshire', 53.20, -2.60, 60, r'\bcheshire\b'),
    ('Essex', 51.80, 0.50, 75, r'\bessex\b'),
    ('Kent', 51.20, 0.80, 90, r'\bkent\b'),
    ('Hertfordshire', 51.80, -0.20, 40, r'hertfordshire|\bherts\b'),
    ('Lancashire', 53.90, -2.60, 90, r'\blancashire\b|\blancs\b'),
    ('Norfolk', 52.65, 1.00, 75, r'\bnorfolk\b'),
    ('Sussex', 50.95, -0.30, 80, r'\bsussex\b'),
    ('Warwickshire', 52.30, -1.55, 45, r'\bwarwick'),
    ('Wiltshire', 51.30, -1.95, 55, r'\bwiltshire\b|\bwilts\b'),
]
_COMPILED = [(lab, la, lo, r, re.compile(rx)) for lab, la, lo, r, rx in COUNTY_REGIONS]


def _key(s) -> str:
    s = unicodedata.normalize('NFKD', str(s or '').casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s).split())


def county_region(surface) -> Optional[tuple]:
    """(label, lat, lon, radius_km) of the English county, riding or city a heading names, else None."""
    k = _key(surface)
    if not k:
        return None
    for lab, la, lo, r, rx in _COMPILED:
        if rx.search(k):
            return (lab, la, lo, float(r))
    return None


def doc_region(county_surfaces: Iterable, header_surfaces: Iterable = (), other_surfaces: Iterable = ()) -> Optional[tuple]:
    """Region of one file from its own headings: the county headings first, then the header titles, then the city / borough headings.
    A file whose headings name two different regions (a certificate for Norfolk that lists Norwich) takes the first one in this order."""
    for group in (county_surfaces, header_surfaces, other_surfaces):
        found = [county_region(s) for s in group]
        found = [f for f in found if f]
        if found:
            # the most frequent one; ties to the first listed
            labs = [f[0] for f in found]
            best = max(dict.fromkeys(labs), key=labs.count)
            return next(f for f in found if f[0] == best)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# London
# ---------------------------------------------------------------------------------------------------------------------
_LONDON_WORDS = frozenset('''street streets lane alley row yard court square gate bridge market hospital inn tavern church chapel house hall palace
prison compter wharf dock stairs alehouse college school tower cross hill fields field green walk passage buildings place rents terrace gardens
cheap ward parish liberty precinct brewhouse coffeehouse playhouse theatre bear head lion crown angel swan star rose'''.split())
#: single names of London districts, prisons and gates (a bare 'Newgate' or 'Temple' is London in the London editions)
LONDON_NAMES = frozenset('''temple savoy strand smithfield holborn newgate ludgate aldgate bishopsgate moorgate aldersgate cripplegate cheapside southwark
westminster lambeth shoreditch whitechapel wapping stepney moorfields bridewell marshalsea clerkenwell finsbury tyburn bankside blackfriars whitefriars
charterhouse bedlam cornhill poultry pimlico guildhall billingsgate bartholomew islington hackney bethnal limehouse ratcliff rotherhithe deptford
borough lothbury'''.split())


def london_like(name) -> bool:
    """True for a name that is a London street, inn, church, house, ward ...: 'Rosemary-Lane', 'Newgate-street', 'St. Stephen (Coleman Street)', 'Lyon's
    Inn', 'Bedford House', 'The Lion (Shoreditch)', 'Temple'."""
    k = _key(str(name or '').replace('\u2019', "'").replace("'s", 's'))
    if not k:
        return False
    toks = k.split()
    if toks[0] in ('st', 'saint') and len(toks) > 1:
        return True
    return any(t in LONDON_NAMES for t in toks) or any(t in _LONDON_WORDS for t in toks[1:])


def in_london(lat, lon):
    """Greater London without the Hertfordshire corner (``LONDON_CUT``): the jurisdiction of the Old Bailey, London plus Middlesex."""
    lat, lon = np.asarray(lat, dtype='float64'), np.asarray(lon, dtype='float64')
    a, b, c, d = LONDON_BOX
    e, f, g, h = LONDON_CUT
    return (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d) & ~((lat >= e) & (lat <= f) & (lon >= g) & (lon <= h))


def london_applies(coll: str, name=None) -> bool:
    return LONDON_PRIOR.get(coll) == 'all'


def in_british_isles(lat, lon, cc=None):
    """True where the point is in the British Isles: by country code when the candidate has one, else by the box."""
    lat, lon = np.asarray(lat, dtype='float64'), np.asarray(lon, dtype='float64')
    if cc is not None:
        return np.isin(np.asarray(cc), list(BRITISH_CC))
    a, b, c, d = BRITISH_BOX
    return (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)


def london_keep(coll: str, lat, lon, pop, fcode, cc=None):
    """Boolean array: the candidates a London prior lets through. In London (plus Middlesex); a country, a capital or a populated place of a million
    (NOT an administrative unit: the Département du Nord has 2.6 million inhabitants and 'North' is a direction); for the collections in ``FOREIGN_OK`` also a
    populated place or a country outside the British Isles (Flushing = Vlissingen) and a provincial town of ``PROVINCIAL_TOWN_POP`` inhabitants."""
    lat, lon = np.atleast_1d(np.asarray(lat, dtype='float64')), np.atleast_1d(np.asarray(lon, dtype='float64'))
    pop = np.atleast_1d(np.asarray(pop, dtype='float64'))
    fc = [str(f) for f in np.atleast_1d(np.asarray(fcode))]
    is_ppl = np.array([f.startswith('PPL') for f in fc])
    is_pcl = np.array([f.startswith('PCL') for f in fc])
    keep = in_london(lat, lon) | is_pcl | np.array([f == 'PPLC' for f in fc]) | (is_ppl & (pop >= 1_000_000))
    if coll in FOREIGN_OK:
        abroad = ~in_british_isles(lat, lon, cc)
        unknown = np.array([f in ('', 'None', 'nan') for f in fc])                  # a propagated point: the feature code is not known
        keep = keep | (abroad & (is_ppl | is_pcl | unknown)) | (is_ppl & (pop >= PROVINCIAL_TOWN_POP))
    return keep


# ---------------------------------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------------------------------
def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


#: v2.4 (check of v2.3): the county discs overlap their neighbours (the disc of Norfolk reaches Colchester: The Hythe of a Norfolk file was put in Essex). A county disc
#: now lets a point through only when the point is, relatively to the radii, at least as close to this county's centre as to the centre of any other county of the list that is not
#: a part of it (a riding, a city) or the whole of it. The discs are not polygons: this is the tighter rule short of polygons, which are not available locally.
COUNTY_FAMILY = {'West Riding of Yorkshire': 'Yorkshire', 'East Riding of Yorkshire': 'Yorkshire', 'North Riding of Yorkshire': 'Yorkshire', 'Kingston upon Hull': 'Yorkshire',
                 'York': 'Yorkshire', 'Norwich': 'Norfolk', 'Chester': 'Cheshire', 'Coventry': 'Warwickshire'}
RIVAL_RULE = True
_BY_DISC = {(round(la, 4), round(lo, 4), float(r)): lab for lab, la, lo, r, _rx in COUNTY_REGIONS}


def _family(label: str) -> str:
    return COUNTY_FAMILY.get(label, label)


def rival_ok(lat, lon, region) -> np.ndarray:
    """Boolean array: for a region disc that is one of ``COUNTY_REGIONS``, the points that are not closer (relative to the radii) to another county's centre than to this one's."""
    lat, lon = np.atleast_1d(np.asarray(lat, dtype='float64')), np.atleast_1d(np.asarray(lon, dtype='float64'))
    lab = _BY_DISC.get((round(float(region[0]), 4), round(float(region[1]), 4), float(region[2])))
    if lab is None or not RIVAL_RULE:
        return np.ones(len(lat), dtype=bool)
    me = hav(lat, lon, region[0], region[1]) / float(region[2])
    ok = np.ones(len(lat), dtype=bool)
    for lab2, la, lo, r, _rx in COUNTY_REGIONS:
        if _family(lab2) == _family(lab):
            continue
        ok &= me <= hav(lat, lon, la, lo) / float(r)
    return ok


def in_disc(lat, lon, regions, margin: float = REGION_MARGIN_KM):
    """Boolean array: the point(s) lie in the disc of at least one of ``regions`` [(lat, lon, radius)] (and, for a county disc, are not closer to a rival county, ``rival_ok``)."""
    lat, lon = np.atleast_1d(np.asarray(lat, dtype='float64')), np.atleast_1d(np.asarray(lon, dtype='float64'))
    m = np.zeros(len(lat), dtype=bool)
    for rl, ro, rr in regions:
        m |= (hav(lat, lon, rl, ro) <= rr + margin) & rival_ok(lat, lon, (rl, ro, rr))
    return m


def file_median(lat, lon) -> tuple:
    return float(np.median(lat)), float(np.median(lon))


def is_regional(lat, lon) -> Optional[tuple]:
    """(median lat, median lon) when at least ``FILE_MIN_PLACED`` points have 80 % of them within ``FILE_REGIONAL_P80_KM`` of their median, else None."""
    lat, lon = np.asarray(lat, dtype='float64'), np.asarray(lon, dtype='float64')
    if len(lat) < FILE_MIN_PLACED:
        return None
    ml, mo = file_median(lat, lon)
    if float(np.quantile(hav(lat, lon, ml, mo), 0.8)) > FILE_REGIONAL_P80_KM:
        return None
    return ml, mo
