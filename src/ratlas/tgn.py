"""The local Getty Thesaurus of Geographic Names (v2.1): names and coordinates of the TGN ids the editions give.

Source: ``~/renascor-atlas-data/tgn/`` on Hypatia (built from the Getty Vocabularies download ``tgn_rel_0126`` with ``build_tgn.py``):

* ``tgn_places.parquet``  one row per subject: ``subject_id name place_type place_types record_type merged_stat lat lon parent_name hierarchy nation
  admin1 admin2`` ...
* ``tgn_names.parquet``   one row per (term, language): ``subject_id name name_norm language historical display_date is_preferred`` ...

Licence: Open Data Commons Attribution License 1.0. Attribution: **Getty Thesaurus of Geographic Names, The J. Paul Getty Trust.**
``vocab.getty.edu`` (robots.txt ``Disallow: /``) is never queried: only this local index is used.

Uses.
* **gates**: the names of a TGN subject (all languages, historical forms, Latin) are compared with the entry's names like the names of a GeoNames
  feature. Compatible: the id is accepted; not: rejected (``tgn_name_mismatch``); a subject the index does not hold stays suggested.
* **coordinates**: an accepted TGN id gives a point (``coord_source`` ``tgn``), after the register, Wikidata and GeoNames ids.
* **geocoder**: the TGN names of places that carry coordinates join the geocoder's name dictionary (``reconcile_geocode.add_tgn_names``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ATTRIBUTION = 'Getty Thesaurus of Geographic Names, The J. Paul Getty Trust (ODC-By 1.0)'

_AREA_TYPES = re.compile(r'\b(nation|country|general region|region|province|state|department|county|canton|autonomous|kingdom|duchy|'
                         r'principality|empire|continent|island group|archipelago|sea|ocean|gulf|bay|peninsula|district|governorate|'
                         r'prefecture|oblast|voivodeship|territory|dependent state|political entity|first level|second level)\b', re.I)
_CITY_TYPES = re.compile(r'\b(inhabited place|city|town|village|hamlet|suburb|neighborhood|neighbourhood|borough|settlement|parish|'
                         r'capital|market town|port|commune|municipality|locality)\b', re.I)


def kind_of_type(place_type) -> Optional[str]:
    """city | area | other from the TGN place type(s) (None when there is none)."""
    t = place_type if isinstance(place_type, str) else ''
    if not t:
        return None
    if _CITY_TYPES.search(t) and not _AREA_TYPES.search(t.split(';')[0]):
        return 'city'
    if _AREA_TYPES.search(t.split(';')[0]):
        return 'area'
    return 'other'


def load_subset(tgn_dir, ids: Iterable) -> dict:
    """{subject_id: {'name', 'names' (list, preferred first), 'lat', 'lon', 'type', 'types', 'nation', 'hierarchy', 'merged'}} for ``ids``.

    Reads the two parquet files with a row filter (the full index has 3 million subjects and 5 million names)."""
    tgn_dir = Path(tgn_dir)
    ids = sorted({str(i) for i in ids if i is not None and str(i) != ''})
    out: dict = {}
    if not ids:
        return out
    pcols = ['subject_id', 'name', 'place_type', 'place_types', 'merged_stat', 'lat', 'lon', 'hierarchy', 'nation', 'parent_id']
    P = pq.read_table(tgn_dir / 'tgn_places.parquet', columns=pcols, filters=[('subject_id', 'in', ids)]).to_pandas()
    pids = sorted({str(x) for x in P['parent_id'].dropna()})
    par = {}
    if pids:                                            # v2.2: the parent's point, for the hierarchy-vs-coordinates check
        PP = pq.read_table(tgn_dir / 'tgn_places.parquet', columns=['subject_id', 'name', 'place_type', 'lat', 'lon'],
                           filters=[('subject_id', 'in', pids)]).to_pandas()
        par = {str(r.subject_id): r for r in PP.itertuples(index=False)}
    N = pq.read_table(tgn_dir / 'tgn_names.parquet', columns=['subject_id', 'name', 'is_preferred'], filters=[('subject_id', 'in', ids)]).to_pandas()
    names: dict = {}
    for r in N.sort_values('is_preferred', ascending=False).itertuples(index=False):
        names.setdefault(r.subject_id, []).append(r.name)
    for r in P.itertuples(index=False):
        lat = float(r.lat) if r.lat == r.lat and r.lat is not None else None
        lon = float(r.lon) if r.lon == r.lon and r.lon is not None else None
        nm = list(dict.fromkeys(([r.name] if isinstance(r.name, str) else []) + names.get(r.subject_id, [])))
        pr = par.get(str(r.parent_id)) if r.parent_id is not None and r.parent_id == r.parent_id else None
        plat = float(pr.lat) if pr is not None and pr.lat == pr.lat and pr.lat is not None else None
        plon = float(pr.lon) if pr is not None and pr.lon == pr.lon and pr.lon is not None else None
        out[str(r.subject_id)] = {'name': r.name, 'names': nm, 'lat': lat, 'lon': lon, 'type': r.place_type, 'types': r.place_types,
                                  'nation': r.nation, 'hierarchy': r.hierarchy, 'merged': r.merged_stat,
                                  'parent_name': pr.name if pr is not None else None, 'parent_type': pr.place_type if pr is not None else None,
                                  'parent_lat': plat, 'parent_lon': plon}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# v2.2: hierarchy versus coordinates (sample check of v2.1: Aptera at 39.4 N 21.6 E is a deserted settlement of Khania in Crete, 500 km from its
# own parent; Lusitania has the longitude +6.3 for -6.3)
# ---------------------------------------------------------------------------------------------------------------------
#: largest plausible distance (km) between a subject and its preferred parent, by the parent's place type: the 99.9th percentile of the distance of
#: the 2.9 million subjects of the 2026-01 release that have a point and a parent with a point, rounded up (computed once, 2026-10-19).
PARENT_MAX_KM = {'regional division': 300, 'county': 350, 'unitary authority': 230, 'canton': 230, 'parish (political)': 125, 'borough': 105,
                 'special municipality': 115, 'national district': 320, 'prefecture': 1000, 'province': 960, 'governorate': 550,
                 'controlled region': 220, 'first level subdivision': 360, 'autonomous community': 220, 'voivodeship': 450,
                 'municipality': 780, 'region (administrative division)': 1250, 'oblast': 1300, 'state': 1500, 'republic': 1500}
PARENT_MAX_DEFAULT_KM = None        # nations, empires, continents, 'World', islands, unknown types: their point is a centroid, the distance says nothing
EXONYM_SLACK_KM = 200.0             # the curated exonym table (Latin and old names) against a TGN point: the point must lie within max(3 x radius, 200) km


def _hav(a, b, c, d) -> float:
    a, b, c, d = (np.radians(float(x)) for x in (a, b, c, d))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(min(1.0, x))))


def parent_limit_km(parent_type) -> Optional[float]:
    """Largest plausible distance to a parent of this type, None for the types whose point says nothing (continents, nations, empires ...)."""
    return PARENT_MAX_KM.get(str(parent_type or '').lower(), PARENT_MAX_DEFAULT_KM)


def coord_issue(rec: Optional[dict], name_keys: Iterable[str] = ()) -> Optional[str]:
    """None when the point of a TGN subject can be used, else why not: ``parent_far`` (the point is farther from the preferred parent's point than
    99.9 % of the TGN's own subjects of that parent type) or ``exonym_far`` (a name of the entry is in the curated exonym table and the point lies
    farther than max(3 x radius, 200 km) from it: Lusitania with the wrong sign of the longitude). The TGN id itself stays accepted."""
    if not rec or rec.get('lat') is None or rec.get('lon') is None:
        return None
    if rec.get('parent_lat') is not None and rec.get('parent_lon') is not None:
        lim = parent_limit_km(rec.get('parent_type'))
        d = _hav(rec['lat'], rec['lon'], rec['parent_lat'], rec['parent_lon'])
        if lim is not None and d > lim:
            return f"parent_far: {d:.0f} km from {rec.get('parent_name')} ({rec.get('parent_type')}, limit {lim:.0f} km)"
    from . import exonyms as EX
    for k in name_keys:
        h = EX.lookup(k)
        if h is None:
            continue
        lat, lon, rad, label, _kind = h
        d = _hav(rec['lat'], rec['lon'], lat, lon)
        if d > max(3 * rad, EXONYM_SLACK_KM):
            return f"exonym_far: {d:.0f} km from the exonym table's {label} ({lat:.2f}, {lon:.2f}; radius {rad:.0f} km)"
    return None
