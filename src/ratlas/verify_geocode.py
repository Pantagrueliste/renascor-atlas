"""Reconcile stage, part 5: audit of the name-based coordinates and the demotions that follow from it.

    python -m ratlas reconcile verify sea   --geo places_local_geo.parquet --zip allCountries.zip --out sea_check.parquet  (Hypatia)
    python -m ratlas reconcile verify rules --geo places_local_geo.parquet --out geocode_verify.tsv
    python -m ratlas reconcile build ... --geocode geocode_places.parquet --geocode-verify geocode_verify.tsv

Why. The held-out self-check of ``reconcile geocode run`` (places that already have coordinates, hidden and geocoded again by
name) reported ~95 % within 10 km for the high-confidence results. A read-through of 800 newly placed places against the TEI
passages (stratified by collection, method and confidence; ``verify/audit_*.tsv``) found a much lower precision for GeoNames
matches that no other source corroborates: the hidden places are famous places with several mentions, the unplaced ones are
mostly obscure single-candidate names (a Fribourg village taken for a French namesake, a common noun taken for a hamlet, a
Kent hundred taken for Preston in Lancashire). ``p`` is 1.0 whenever GeoNames knows only one candidate for the name, which
says nothing about the candidate being the place the editor meant.

What. This module does not move any point. It lists the places to take back from the globe (they stay in the ``gc_*`` columns
as low-confidence candidates, with the rule that dropped them in ``verify_rule``). The rules are the strata below the 85 %
accuracy bar of the audit, plus two clear bugs.

* ``gn_uncorroborated``  a GeoNames-only match (no propagation agrees with it) whose own certainty is low (``gc_p`` below ``GN_MIN_P``, the
  share of the softmax mass of the best cluster of candidates), unless the feature is a major one (country / first-order region / seat of a
  province or a populated place of 50 000 people or more; 49 of 56 audited cases were right). **v2.1**: the v1.1 rule took back every
  uncorroborated match in a stratum (collection x confidence) below 85 % in the audit: on the 373 labelled matches it took back 84 right
  ones for 35 wrong ones. ``GN_MIN_P`` is fitted on 70 % of the labels
  (the 765 audited places of v1.1 that the v2 geocoder still places, and the Casebooks places the directory matched with confidence high /
  medium) and reported on the other 30 % (``scripts/calibrate_rules.py``, ``data/calibration_v2_1.json``).
* ``casebooks_far``      Casebooks: the patients of Forman and Napier lived in London and in north Buckinghamshire; a village
  name resolved by GeoNames more than 60 km from both is a namesake (Lowton, Calverton, Wicken, Aspley, Eaton ...).
* ``channel_islands``    "St Mary", "Saint Peter" ... resolved to the parishes of Jersey / Guernsey (all of them English or
  Zürich parishes).
* ``london_microtoponym``  Map of Early Modern London: a ward, church, street, gate or wall of the City resolved to a namesake
  more than 30 km from London (The Wall -> Wall in Staffordshire, Faringdon Ward -> Faringdon, S. Margarets Church -> Chur).
* ``english_parish_abroad``  Casebooks "of X" (the patient's home parish) or any hearth-tax place (England and Wales only) resolved
  outside the British Isles (of Ringsteede -> Ringsted in Denmark, of Wels, of Naldwick, North Borough -> the Nord).
* ``prop_stratum``       propagation strata that did not reach 85 % (hearth tax: county homonyms; SSRQ medium: Swiss-French names
  copied from French namesakes).
* ``block_collection``   collections whose name-based coordinates were wrong in most audited cases (Hab diary: wrong register
  coordinates as donors).
* ``audit_manual``       single high-weight places the audit showed to be wrong (listed in ``MANUAL``).

``sea`` is a diagnostic, not a rule: 34 of 17 310 points lie more than 60 km from any populated place of GeoNames, all of them
country / sea centroids (Egypt, Libya, Ostsee, Kaspisches Meer ...); no settlement is in the sea.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NAME_SOURCES = ('name_propagation', 'name_geonames')

# v2.1: a GeoNames-only match is taken back when the geocoder's own certainty is below this (fitted, see the docstring). The v1.1 strata table
# (KEEP_GN_UNCORROBORATED) is gone.
GN_MIN_P = 0.8
DROP_PROP = {('ssrq_swisslaw', 'medium'), ('hearth_tax', 'high'), ('hearth_tax', 'medium')}
BLOCK_COLLECTIONS = {'selbstzeugnisse_hab_diarium'}
MAJOR_FCODES = frozenset('PCLI PCLD PCLS PCLF PCL PCLH ADM1 ADM1H ADM2 ADMD PPLC PPLA PPLA2 PPLG'.split())
POP_MAJOR = 50_000
CHANNEL_CC = frozenset({'JE', 'GG', 'IM'})
LONDON = (51.51, -0.10)
LONDON_KM = 30.0
LONDON_COLLECTIONS = {'map_of_early_modern_london'}
LONDON_MICRO = re.compile(r"(?i)\b(?:ward|church|wall|street|strete|lane|alley|gate|row|court|yard|inn|hospital|conduit|bridge|market|"
                          r"cross|abbey|priory|chapel|chappell|st\.?|saint|s\.|hill|fields|green)\b")
CASEBOOKS_CORE = ((52.0586, -0.7687), (51.5074, -0.1278))      # Great Linford (Napier), London (Forman)
CASEBOOKS_FAR_KM = 60.0
CASEBOOKS_FAR_POP = 250_000

# single high-weight places shown wrong by the audit of the 300 heaviest name-based places: (collection, name) -> reason
MANUAL: dict = {
    ('cidtc', 'Löbau (Lubawa)'): 'Lubawa in Prussia, not Löbau in Saxony (propagated from the Saxon namesake)',
    ('bullinger', 'Schmalkaldener'): 'the members of the Schmalkaldic League, not the town',
    ('ckcc_epistolarium', 'Batavis'): 'Batavi = the Dutch, not Batavia Castra (Passau)',
    ('bullinger', 'Wellenberg'): 'Herrschaft Wellenberg in the Thurgau, not the Wellenberg near Zürich',
    ('bullinger', 'Buchberg'): 'Buchberg west of Rheineck (St. Gallen), not Buchberg near Schaffhausen',
    ('ssrq_swisslaw', 'Nant'): 'Nant in the Vully (Fribourg), not Nantes',
    ('ckcc_epistolarium', 'Spae'): 'part of "Spaensche" (Spanish faction) after a line break, not Spa',
    ('hearth_tax', 'Alveston parish'): 'Alveston in Warwickshire (Warwick assessment), not Alveston in Gloucestershire',
    ('bullinger', 'Prussia'): 'Prussia (the duchy), not a hamlet of that name in Piedmont',
    ('sandrart', 'Pantheon'): 'the Pantheon in Rome, not the Panthéon quarter of Paris',
    ('cidtc', 'Rheden (Radzyn)'): 'Radzyn Chełmiński in Prussia, not Rheden in Lower Saxony',
}

#: v2.3: manual rules that apply only to the wrong point they were written for: (collection, name) -> longitude (the Saxon Löbau is at 14.7 E; Lubawa at 19.7 E is right)
MANUAL_ONLY_WEST_OF = {('cidtc', 'Löbau (Lubawa)'): 17.0}

_GN = re.compile(r"^(?:GeoNames|TGN) (\d+) .*? \((\w+), (\w*), pop (\d+)\)")


def _t(msg):
    print(f'[verify {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_evidence(ev) -> dict:
    """GeoNames feature code / country / population of a name_geonames evidence string, and whether a second method agrees."""
    ev = ev if isinstance(ev, str) else ''
    m = _GN.match(ev)
    t = re.search(r"key '[^']*' \((\w+),", ev)
    return {'fcode': m.group(2) if m else None, 'cc': m.group(3) if m else None,
            'pop': int(m.group(4)) if m else 0, 'corroborated': 'corroborated by' in ev, 'tier': t.group(1) if t else None}


def is_major(fcode, pop, cc) -> bool:
    if cc in CHANNEL_CC:
        return False
    return (fcode in MAJOR_FCODES) or ((pop or 0) >= POP_MAJOR and str(fcode).startswith('PPL'))


def in_channel_box(lat, lon):
    lat, lon = np.asarray(lat, dtype='float64'), np.asarray(lon, dtype='float64')
    jg = (lat >= 49.1) & (lat <= 49.8) & (lon >= -2.8) & (lon <= -2.0)
    iom = (lat >= 54.0) & (lat <= 54.45) & (lon >= -4.9) & (lon <= -4.2)
    return jg | iom


# ---------------------------------------------------------------------------------------------------------------------
def rules(geo: pd.DataFrame) -> pd.DataFrame:
    """Places to take back: one row per (collection, entity_key) with the first rule that applies."""
    g = geo[geo['coord_source'].isin(NAME_SOURCES) & geo['coord_conf'].isin(['high', 'medium']) & geo['lat'].notna()].copy()
    ev = [parse_evidence(e) if s == 'name_geonames' else {'fcode': None, 'cc': None, 'pop': 0,
                                                           'corroborated': isinstance(e, str) and 'corroborated by' in e}
          for e, s in zip(g['gc_evidence'], g['coord_source'])]
    g['fcode'] = [e['fcode'] for e in ev]
    g['cc'] = [e['cc'] for e in ev]
    g['pop'] = [e['pop'] for e in ev]
    g['corr'] = [e['corroborated'] for e in ev]
    gn = g['coord_source'] == 'name_geonames'
    g['major'] = [bool(a) and is_major(f, p, c) for a, f, p, c in zip(gn, g['fcode'], g['pop'], g['cc'])]
    rule = pd.Series('', index=g.index, dtype='object')
    detail = pd.Series('', index=g.index, dtype='object')

    def put(mask, name, why):
        mask = mask & (rule == '')
        rule[mask] = name
        detail[mask] = why if isinstance(why, str) else why[mask]

    # audit_manual
    mm = pd.Series(False, index=g.index)
    why = pd.Series('', index=g.index, dtype='object')
    for (c, nm), reason in MANUAL.items():
        k = (g['coll_slug'] == c) & (g['name'] == nm)
        if (c, nm) in MANUAL_ONLY_WEST_OF:               # v2.3: the audit's rule takes back the WRONG point only (head_alt now answers Lubawa)
            k &= g['lon'] < MANUAL_ONLY_WEST_OF[(c, nm)]
        mm |= k
        why[k] = reason
    put(mm, 'audit_manual', why)
    # block_collection
    put(g['coll_slug'].isin(BLOCK_COLLECTIONS), 'block_collection', 'most audited name-based coordinates of this collection were wrong')
    # channel_islands
    nm_ok = ~g['name'].str.contains(r'(?i)jersey|guernsey|isle of man|alderney', regex=True, na=False)
    chan = (g['cc'].isin(CHANNEL_CC) | pd.Series(in_channel_box(g['lat'], g['lon']), index=g.index)) & nm_ok
    put(chan, 'channel_islands', 'a parish name taken for the namesake parish of Jersey / Guernsey / Isle of Man')
    # prop_stratum
    ps = (g['coord_source'] == 'name_propagation') & pd.Series([(c, f) in DROP_PROP for c, f in zip(g['coll_slug'], g['coord_conf'])], index=g.index)
    put(ps, 'prop_stratum', 'stratum (collection x propagation x confidence) below 85 % in the audit')
    # london_microtoponym
    dl = pd.Series(hav(g['lat'], g['lon'], *LONDON), index=g.index)
    lm = g['coll_slug'].isin(LONDON_COLLECTIONS) & g['name'].str.contains(LONDON_MICRO, na=False) & (dl > LONDON_KM)
    put(lm, 'london_microtoponym', 'a ward / church / street of the City resolved to a namesake more than 30 km from London')
    # english_parish_abroad
    abroad = ~(g['lat'].between(49.8, 61.0) & g['lon'].between(-10.8, 2.0))
    ea = abroad & (((g['coll_slug'] == 'casebooks') & g['name'].str.match(r'(?i)^of\s')) | (g['coll_slug'] == 'hearth_tax'))
    put(ea, 'english_parish_abroad', 'a home parish of the English collections resolved outside the British Isles')
    # casebooks_far
    d = np.minimum(hav(g['lat'], g['lon'], *CASEBOOKS_CORE[0]), hav(g['lat'], g['lon'], *CASEBOOKS_CORE[1]))
    cf = (g['coll_slug'] == 'casebooks') & gn & ~g['corr'] & (pd.Series(d, index=g.index) > CASEBOOKS_FAR_KM) & (g['pop'] < CASEBOOKS_FAR_POP)
    put(cf, 'casebooks_far', 'more than 60 km from Great Linford and from London: a namesake of a Buckinghamshire / London place')
    # gn_uncorroborated
    gp = pd.to_numeric(g['gc_p'], errors='coerce').fillna(0.0) if 'gc_p' in g else pd.Series(0.0, index=g.index)
    gu = gn & ~g['corr'] & ~g['major'] & (gp < GN_MIN_P)
    put(gu, 'gn_uncorroborated', f'GeoNames-only match, not a major feature, certainty p below {GN_MIN_P}')
    out = g[rule != ''].copy()
    out['rule'] = rule[rule != '']
    out['detail'] = detail[rule != '']
    return out[['coll_slug', 'entity_key', 'name', 'coord_source', 'coord_conf', 'n_in_period', 'lat', 'lon', 'gc_cand_name',
                'fcode', 'pop', 'corr', 'rule', 'detail']]


# ---------------------------------------------------------------------------------------------------------------------
def sea_check(geo: pd.DataFrame, zpath: Path, out: Path) -> pd.DataFrame:
    """Distance of every name-based point to the nearest populated place (GeoNames class P) of allCountries.zip."""
    g = geo[geo['coord_source'].isin(NAME_SOURCES) & geo['coord_conf'].isin(['high', 'medium']) & geo['lat'].notna()].copy()
    _t(f'{len(g)} name-based points')
    lats, lons = [], []
    with zipfile.ZipFile(zpath) as z, z.open('allCountries.txt') as raw:
        for n, line in enumerate(io.TextIOWrapper(raw, encoding='utf-8', newline='')):
            p = line.split('\t', 8)
            if len(p) > 7 and p[6] == 'P':
                lats.append(float(p[4]))
                lons.append(float(p[5]))
    la, lo = np.array(lats), np.array(lons)
    _t(f'{len(la)} populated places')
    cell = 0.25
    ci = np.floor((la + 90) / cell).astype(np.int64)
    cj = np.floor((lo + 180) / cell).astype(np.int64)
    key = ci * 100000 + cj
    order = np.argsort(key, kind='stable')
    ks = key[order]
    uk, st, cnt = np.unique(ks, return_index=True, return_counts=True)
    where = {int(k): (int(a), int(a + c)) for k, a, c in zip(uk, st, cnt)}
    res = []
    for y, x in zip(g['lat'].to_numpy(), g['lon'].to_numpy()):
        best = np.inf
        for ring in (1, 2, 4, 8, 16):
            i0, j0 = int(np.floor((y + 90) / cell)), int(np.floor((x + 180) / cell))
            jw = int(ring / max(0.2, np.cos(np.radians(y)))) + 1
            idx = []
            for i in range(i0 - ring, i0 + ring + 1):
                for j in range(j0 - jw, j0 + jw + 1):
                    r = where.get(int(i) * 100000 + int(((j % 1440) + 1440) % 1440))
                    if r:
                        idx.append(order[r[0]:r[1]])
            if idx:
                ii = np.concatenate(idx)
                best = float(hav(y, x, la[ii], lo[ii]).min())
                if best <= ring * cell * 111.0 * 0.9 or ring == 16:
                    break
        res.append(best)
    g['d_populated_km'] = res
    o = g[['coll_slug', 'entity_key', 'name', 'coord_source', 'coord_conf', 'n_in_period', 'lat', 'lon', 'gc_cand_name', 'd_populated_km']]
    o.to_parquet(out, index=False)
    _t(f'{int((g["d_populated_km"] > REMOTE_KM).sum())} points more than {REMOTE_KM:.0f} km from a populated place -> {out}')
    return o


def summarize(dem: pd.DataFrame, geo: pd.DataFrame) -> dict:
    ip = geo['n_in_period'] > 0
    new = geo[geo['coord_source'].isin(NAME_SOURCES) & geo['coord_conf'].isin(['high', 'medium']) & geo['lat'].notna() & ip]
    d = dem[dem['n_in_period'] > 0]
    return {'name_based_in_period': {'distinct': int(len(new)), 'mentions': int(new['n_in_period'].sum())},
            'demoted_in_period': {'distinct': int(len(d)), 'mentions': int(d['n_in_period'].sum())},
            'by_rule': {k: {'distinct': int(len(g)), 'mentions': int(g['n_in_period'].sum())} for k, g in d.groupby('rule')}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile verify')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sea')
    p.add_argument('--geo', type=Path, required=True)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('rules')
    p.add_argument('--geo', type=Path, required=True, help='places_local_geo.parquet built with --geocode and WITHOUT --geocode-verify')
    p.add_argument('--out', type=Path, required=True, help='geocode_verify.tsv')
    a = ap.parse_args(argv)
    cols = ['coll_slug', 'entity_key', 'name', 'n_in_period', 'coord_source', 'coord_conf', 'lat', 'lon', 'gc_evidence', 'gc_cand_name', 'gc_p']
    geo = pd.read_parquet(a.geo, columns=cols)
    if a.cmd == 'sea':
        sea_check(geo, a.zip, a.out)
        return 0
    dem = rules(geo)
    dem.to_csv(a.out, sep='\t', index=False)
    rep = summarize(dem, geo)
    a.out.with_suffix('.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
