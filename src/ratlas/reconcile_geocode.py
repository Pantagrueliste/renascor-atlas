"""Reconcile stage, part 4: name-based geocoding of places that have no coordinates (run on Hypatia through Slurm).

    python -m ratlas reconcile geocode index --local places_local.parquet --zip allCountries.zip --out gn_index.parquet [--tgn-dir DIR]
    python -m ratlas reconcile geocode run   --geo places_local_geo.parquet --local places_local.parquet \
        --mentions mentions.parquet --index gn_index.parquet --out DIR [--eval-n 1000 --eval-seed 2]

A new, separate coordinate source. Coordinates from edition registers, Wikidata and GeoNames ids are never touched: only
places with ``coord_source == 'none'`` are geocoded.

Methods (``coord_source`` of the result)
* ``name_propagation``  the cleaned name of an unplaced place matches places that already carry coordinates elsewhere in
  the atlas. Accepted when the placed candidates agree (>= 90% of their mentions within 50 km of the heaviest one); the
  candidates of the same collection family (language area) are preferred when all of them do not agree.
* ``name_geonames``     the cleaned name (and its Latin / early-modern variants) is looked up in GeoNames (names and
  alternate names, classes P and A; H, T, L only for places the edition tags as rivers, seas, regions ...) and ranked by
  population, feature type, a regional prior per collection and the distance to the placed places of the same
  documents. Ambiguous homonyms (Newport, Neustadt, Frankfurt without qualifier) are left unplaced unless the context
  decides.

v2.1: the names of the local Getty TGN (Latin, historical and foreign forms: Ratisbona, Tigurum, Dantiscum) join the name dictionary
(``add_tgn_names``): a TGN name that GeoNames lacks becomes an alternate name of the GeoNames feature next to the TGN point (8 km), or a TGN-only
candidate (``TGN <id>``). They are ranked like every GeoNames candidate (population of the aliased feature, regional prior of the edition, hard
region boxes) and never override a register or authority coordinate: only places with no coordinates are geocoded. A result that rests on a
TGN name has ``gc_tgn`` True.

Confidence ``high`` / ``medium`` go on the globe, ``low`` does not. Everything is recorded in the output: clean key,
rule, tier, candidate id, evidence, radius_km.

Two passes: pass 1 uses only coordinates from registers / authorities as anchors; pass 2 adds the high / medium results
of pass 1 to the regional priors and to the documents' context (the ambiguous names get resolved by the places that
surround them).

Precision self-check: ``--eval-n N`` hides N places that have coordinates and geocodes them again by name (both
methods separately and combined).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import exonyms as EX
from . import geotext as GT
from . import latin_geo as LG
from . import latin_homonyms as LH
from . import latin_names as LN
from . import placeclass as PCL
from . import polities as POL
from . import regions as RGN
from .merge import _norm

GN_COLS = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature_class', 'feature_code',
           'country_code', 'cc2', 'admin1', 'admin2', 'admin3', 'admin4', 'population', 'elevation', 'dem', 'timezone',
           'modification']
CLASSES = frozenset('PAHTL')
MIN_P = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _t(msg):
    print(f'[geocode {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------------------------------------------------
# collection families (language areas) and regional seeds (ISO country weights), used when a collection has too few
# placed places to derive its own prior
# ---------------------------------------------------------------------------------------------------------------------
FAMILY = {}
for fam, slugs in {
    'swiss_german': 'bullinger ssrq_swisslaw st_galler_missiven urfehdebuecher_basel_x koenigsfelden cosuiza',
    'german': ('abacus controversia_confessio religionsfrieden_quellenedition fruchtbringende_briefe germanc '
               'hainhofer_reisen hainhofer_stammbuch kaiserin_eleonora_magdalena luene_nonnenbriefe '
               'magdeburger_zenturien_methode_hab karlstadt_kritische_gesamtausgabe leibniz_reihe_viii dibilit_german '
               'dipko_schwarzenhorn_1649 dta_1600 sandrart selbstzeugnisse_hab_diarium selbstzeugnisse_hab_sz2 '
               'tagebuecher_christian_ii_anhalt regensburger_wappen_portraetbuecher ferdinand_i ethica_complementoria '
               'rubens_online'),
    'english': ('casebooks hearth_tax old_bailey map_of_early_modern_london reed pceec2 devonshire_ms emmo_folger rup '
                'h15corpus dracor_eng grand_tour'),
    'irish': 'celt',
    'low_countries': 'ckcc_epistolarium',
    'french': ('epistemon_bvh freem_bvh freem_other elan_crenum elan_fnvr elan_polygrenaille lettres_calvin '
               'corpus17_french obvil_critique guicciardini_si'),
    'italian': 'biblioteca_italiana paesi_novamente suriano cavriana poliziano_sappho dracor_ita',
    'iberian': 'postscriptum bdpn gongora_epistolario gongora_soledades letradas disco_sonnets polemicagongorina_obvil',
    'polish': 'cidtc rejestr_poznan',
    'neo_latin': ('camena_cera camena_historicapolitica camena_poemata camena_thesaurus croala corpus_corporum '
                  'dasypodius_dictionarium_latinogermanicum sozinianische_briefwechsel'),
}.items():
    for s in slugs.split():
        FAMILY[s] = fam
SEEDS = {
    'swiss_german': {'CH': 1.0, 'LI': 0.9, 'DE': 0.7, 'AT': 0.6, 'FR': 0.5, 'IT': 0.5},
    'german': {'DE': 1.0, 'AT': 0.8, 'CH': 0.7, 'CZ': 0.5, 'PL': 0.5, 'NL': 0.5, 'FR': 0.5, 'IT': 0.5, 'BE': 0.4, 'LU': 0.4,
               'HU': 0.4, 'SK': 0.4, 'SI': 0.4},
    'english': {'GB': 1.0, 'IE': 0.5},
    'irish': {'IE': 1.0, 'GB': 0.8},
    'low_countries': {'NL': 1.0, 'BE': 1.0, 'DE': 0.6, 'FR': 0.6, 'GB': 0.5},
    'french': {'FR': 1.0, 'BE': 0.7, 'CH': 0.6, 'IT': 0.5, 'DE': 0.4, 'ES': 0.4},
    'italian': {'IT': 1.0, 'FR': 0.5, 'CH': 0.5, 'DE': 0.4, 'ES': 0.4, 'VA': 0.8, 'SM': 0.5},
    'iberian': {'ES': 1.0, 'PT': 1.0, 'MX': 0.6, 'IT': 0.4, 'FR': 0.4},
    'polish': {'PL': 1.0, 'DE': 0.6, 'LT': 0.5, 'UA': 0.4, 'RU': 0.4, 'CZ': 0.4, 'IT': 0.4, 'BY': 0.4},
}
SEED_OVERRIDE = {'bdpn': {'MX': 1.0, 'ES': 0.7, 'GT': 0.4, 'US': 0.3}}

# feature code weights and indicative radii (km)
TYPE_W = {'PPLC': 1.25, 'PPLA': 1.1, 'PPLA2': 1.05, 'PPLA3': 1.0, 'PPLA4': 0.95, 'PPLA5': 0.9, 'PPL': 1.0, 'PPLH': 0.8,
          'PPLQ': 0.4, 'PPLX': 0.4, 'PPLF': 0.6, 'PPLL': 0.8, 'PPLS': 0.8, 'PPLW': 0.3, 'STLMT': 0.5, 'PCLI': 1.2,
          'PCLH': 1.0, 'PCLS': 0.9, 'PCLD': 0.9, 'PCLF': 0.9, 'PCL': 0.9, 'TERR': 0.5, 'ADM1': 1.0, 'ADM2': 0.85,
          'ADM3': 0.6, 'ADM4': 0.45, 'ADM5': 0.3, 'ADMD': 0.7, 'ADM1H': 0.8, 'ADM2H': 0.6, 'ADM3H': 0.4, 'ADM4H': 0.3,
          'RGN': 0.9, 'AREA': 0.5, 'SEA': 1.0, 'OCN': 1.0, 'STM': 0.8, 'LK': 0.8, 'GULF': 0.9, 'BAY': 0.7, 'ISL': 0.9,
          'ISLS': 0.8, 'MT': 0.7, 'MTS': 0.8, 'PEN': 0.7, 'CAPE': 0.5, 'STRT': 0.7, 'CHN': 0.6, 'LKS': 0.7}
RADIUS = {'PPLC': 15, 'PPLA': 10, 'PPLA2': 8, 'PPLA3': 6, 'PPLA4': 5, 'PPL': 5, 'PPLH': 6, 'PPLX': 3, 'PCLI': 350,
          'PCLH': 250, 'PCLS': 150, 'PCLD': 150, 'PCL': 250, 'ADM1': 120, 'ADM2': 40, 'ADM3': 20, 'ADM4': 10, 'ADMD': 40,
          'ADM1H': 120, 'ADM2H': 40, 'RGN': 150, 'AREA': 60, 'SEA': 300, 'OCN': 1000, 'STM': 100, 'LK': 20, 'GULF': 150,
          'BAY': 60, 'ISL': 40, 'ISLS': 80, 'MT': 15, 'MTS': 80, 'PEN': 100, 'CAPE': 10, 'STRT': 100}

TH = dict(T=0.35, w_pop=1.6, w_type=0.8, w_match=0.7, w_pc=2.0, w_pd=2.5, w_pf=1.0, sigma_f=40.0, high_p=0.88, med_p=0.62, low_p=0.38,
          amb_ctx=0.35, S_high=2.8, S_med=2.6, pc_med=0.2, tol_cap=400.0, dens_mass=30.0, pf_min=0.1, pop_local=50000, cluster_km=25.0, agree_km=50.0, agree_share=0.90, sigma_c=150.0, sigma_d=120.0)

# ---------------------------------------------------------------------------------------------------------------------
# v2 barriers (audit 2026-10-16, rapport section 2, point 9)
# ---------------------------------------------------------------------------------------------------------------------
# Strict regional prior: GeoNames candidates (and propagated points) outside these boxes are rejected unless the feature is major
# (a million inhabitants, a country, a first-order region, a capital). (lat0, lat1, lon0, lon1). Only for corpora whose places are local:
# not for cidtc (Dantiscus wrote from Spain, Flanders and Rome) nor st_galler_missiven (letters to Nuremberg, Frankfurt ...), which keep the
# soft prior (collection and document clouds) and the paren-first / exonym rules.
HARD_REGIONS = {
    'ssrq_swisslaw': [(45.3, 48.3, 5.4, 11.0)],
    'koenigsfelden': [(46.6, 48.4, 6.5, 10.5)],
    'cosuiza': [(35.5, 44.5, -10.0, 4.5)],
    'rejestr_poznan': [(51.0, 53.8, 15.0, 19.0)],
    'hearth_tax': [(49.8, 56.0, -6.5, 2.0)],
    'casebooks': [(49.8, 56.0, -6.5, 2.0)],
    'old_bailey': [(49.8, 56.0, -6.5, 2.0)],
    'luene_nonnenbriefe': [(52.2, 54.7, 8.3, 12.3)],
}
MAJOR_FCODES = frozenset('PCLI PCLD PCLS PCLF PCL PCLH ADM1 PPLC'.split())
#: words that qualify a bare place name inside the same edition's register ("Bergen op Zoom", "Frankfurt an der Oder", "Newcastle upon Tyne", "Villeneuve sur Yonne")
QUALIFIER_WORDS = frozenset('op aan an am upon on sur sous bei ob im'.split())
MAJOR_POP = 1_000_000
MAJOR_AREA_FCODES = frozenset('PCLI PCLD PCLS PCLF PCL ADM1'.split())
REGION_FCODES = frozenset('ADM1 ADM2 ADM1H ADM2H RGN RGNH AREA TERR LCTY'.split())
MAJOR_AREA_POP = 500_000
# register type -> GeoNames feature classes a match must have ('moun' is a mountain, not a village of that name)
TYPE_EXPECT = {'moun': 'T', 'mountain': 'T', 'berg': 'T', 'mons': 'T', 'rive': 'H', 'river': 'H', 'fluss': 'H', 'flumen': 'H',
               'fleuve': 'H', 'coun': 'AL', 'country': 'AL', 'region': 'AL', 'dist': 'AL'}
# register types of micro-toponyms that a plain name match can only get wrong (fields, paths, streets)
NO_GEOCODE_TYPES = frozenset({'flurname', 'acker', 'weg', 'stre', 'place_f_street', 'place_f_openarea', 'place_f_property'})
PAREN_FIRST = frozenset({'cidtc'})          # 'Schoneck (Skarszewy)': the bracketed modern name is the first key
#: v2.3 (path C): English collections whose small homonyms (Weston, Broughton, Milton ...) are placed when exactly one candidate lies in the prior region
REGIONAL_UNIQUE = frozenset({'hearth_tax', 'casebooks'})
CASEBOOKS_CONFLICT_KM = 40.0          # a name-based guess this far from the directory's suggestion is only a candidate
CASEBOOKS_SUGGESTION_KM = 15.0        # a Casebooks place is placed by name only next to the directory's own (unverified) suggestion
UNIQUE_CLUSTER_KM = 3.0               # candidates closer than this are one place
ORBIS_MIN_POP = 15_000                # the Orbis Latinus alone places a settlement of this size or more; a smaller one needs a second source
LATIN_HOMONYMS = frozenset(LN.fold_key(x) for x in 'lugdunum traiectum augusta isara aquae novum castrum nova civitas burgum monasterium alba mons pons portus marchia'.split(' ') if x)
LATIN_HOMONYM_CTX = 0.35              # a known Latin homonym needs this much support from the places of its own documents
LATIN_FAR_CTX = 0.05                  # ... one family alone needs at least this much support from its documents, unless the place has LATIN_PROMINENT mentions in the atlas
LATIN_PROMINENT = 30.0
#: v2.4 (check of v2.3): the head of a polity qualifier ('Stolberg, Grafschaft') is a GeoNames administrative unit or region, never a TGN-only locality (LCTY)
POLITY_HEAD_FCODES = frozenset('ADM1 ADM2 ADM1H ADM2H RGN RGNH AREA TERR'.split())
SURFACE_OK_MIN = 0.5                  # v2.4: the Latin path needs the mentions' surface to be the name's form (not a different word the register standardised: 'Tagum' read as 'Fagum')
LATIN_ADJECTIVE_END = ('inus', 'ensis', 'ense', 'anus', 'icus', 'eus', 'alis', 'aris', 'ilis')        # masculine adjective endings (the feminine and neuter ones are also place names: Messina, Ariminum)
LOCATOR_KM = 120.0                    # v2.4: 'Lucena (Córdoba)': a head within this distance of the larger bracketed place is that place's village
LOCATOR_FCODES = frozenset('PPLA PPLA2 PPLC ADM1 ADM2 PPL'.split())
ORBIS_LOW_FACTOR = 0.95               # an Orbis Latinus entry answered by GeoNames with p >= 0.9 and the book's country but a low score: the rule score weighs a little less


def in_box(boxes, lat, lon):
    lat, lon = np.asarray(lat, dtype='float64'), np.asarray(lon, dtype='float64')
    m = np.zeros(np.broadcast(lat, lon).shape, dtype=bool)
    for a, b, c, d in boxes:
        m |= (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)
    return m


def region_keep(coll, lat, lon, pop, fcode):
    """Hard regional prior: True where a candidate is inside the collection's region or major."""
    boxes = HARD_REGIONS.get(coll)
    lat = np.atleast_1d(lat)
    if not boxes:
        return np.ones(len(lat), dtype=bool)
    major = (np.asarray(pop) >= MAJOR_POP) | np.isin(np.asarray(fcode), list(MAJOR_FCODES))
    return in_box(boxes, lat, np.atleast_1d(lon)) | major


HINT_H = ('river', 'fluss', 'flumen', 'fluvius', 'stream', 'sea', 'meer', 'mare', 'lake', 'see', 'lacus', 'ocean', 'bay', 'gulf',
          'fleuve', 'riviere', 'rivière', 'strait', 'canal', 'water')
HINT_T = ('mountain', 'berg', 'mons', 'montagne', 'island', 'insel', 'insula', 'isle', 'cape', 'peninsula', 'range', 'hill',
          'alp', 'gebirge', 'volcano', 'forest', 'wald')
HINT_L = ('region', 'country', 'coun', 'kingdom', 'duchy', 'province', 'county', 'land', 'territory', 'landscape', 'empire',
          'reich', 'regnum', 'regio', 'provincia', 'dist', 'shire', 'principality', 'republic', 'bishopric', 'canton', 'kanton',
          'geogname', 'area')


# ---------------------------------------------------------------------------------------------------------------------
# stage 1: compact GeoNames name index
# ---------------------------------------------------------------------------------------------------------------------
def needed_keys(local: Path, extra_names=None):
    """The keys the GeoNames index must answer: every name and variant of the atlas's places, plus (v2.3) ``extra_names``, the German-spelled modern names of
    the Orbis Latinus (the Latin path looks them up)."""
    t = pq.read_table(local, columns=['name', 'variants']).to_pandas()
    need: set = set()
    for name, var in zip(t['name'], t['variants']):
        for s in [name] + (list(var)[:25] if var is not None else []):
            need |= GT.all_keys(GT.forms(s))
    for s in (extra_names or ()):
        need |= GT.all_keys(GT.forms(s))
    need = {k for k in need if len(k) >= 3}
    return need


def build_index(local: Path, zpath: Path, out: Path, tgn_dir: Optional[Path] = None, extra_names=None):
    t0 = time.time()
    need = needed_keys(local, extra_names)
    need_skel = {GT.skeleton(k) for k in need}
    _t(f'{len(need)} exact keys, {len(need_skel)} skeletons wanted')
    nc = lru_cache(maxsize=3_000_000)(_norm)
    rows = []
    feats = {k: [] for k in ('lat', 'lon', 'gid', 'name', 'fcode', 'cc', 'pop')} if tgn_dir else None
    seen = 0
    with zipfile.ZipFile(zpath) as z, z.open('allCountries.txt') as raw:
        for n, line in enumerate(io.TextIOWrapper(raw, encoding='utf-8', newline='')):
            p = line.rstrip('\n').split('\t')
            if len(p) < 15 or p[6] not in CLASSES:
                continue
            seen += 1
            hit: dict = {}
            names = [(p[1], True), (p[2], True)] + [(a, False) for a in p[3].split(',') if a]
            for nm, prim in names:
                k = nc(nm)
                if len(k) < 3:
                    continue
                if k in need:
                    hit[k] = hit.get(k, False) or prim
                else:
                    sk = GT.skeleton(k)
                    if sk in need_skel:
                        hit[k] = hit.get(k, False) or prim
            pop = int(p[14]) if p[14].isdigit() else 0
            if hit:
                for k, prim in hit.items():
                    rows.append((k, p[0], prim, p[1], float(p[4]), float(p[5]), p[6], p[7], p[8], p[10], pop))
            if feats is not None and p[6] in ('P', 'A'):
                feats['lat'].append(float(p[4]))
                feats['lon'].append(float(p[5]))
                feats['gid'].append(p[0])
                feats['name'].append(p[1])
                feats['fcode'].append(p[7])
                feats['cc'].append(p[8])
                feats['pop'].append(pop)
            if n % 2_000_000 == 0:
                _t(f'{n} lines, {seen} P/A/H/T/L, {len(rows)} index rows ({time.time() - t0:.0f}s)')
    df = pd.DataFrame(rows, columns=['key', 'gid', 'prim', 'name', 'lat', 'lon', 'fclass', 'fcode', 'cc', 'adm1', 'pop'])
    df['src'] = 'geonames'
    if tgn_dir:
        df = add_tgn_names(df, feats, Path(tgn_dir), need, need_skel, nc)
    df['skel'] = [GT.skeleton(k) for k in df['key']]
    df['exact'] = df['key'].isin(need)
    df = df.sort_values(['key', 'gid']).reset_index(drop=True)
    df.to_parquet(out, index=False)
    _t(f'index: {len(df)} rows, {df["gid"].nunique()} features -> {out} ({time.time() - t0:.0f}s)')
    return df


# TGN place type -> (GeoNames class, feature code) of a TGN-only candidate
def tgn_fcode(place_type) -> tuple:
    from . import tgn as TGN
    k = TGN.kind_of_type(place_type)
    t = (place_type or '') if isinstance(place_type, str) else ''
    if k == 'city':
        return 'P', 'PPL'
    if k == 'area':
        return 'A', ('PCLI' if t.startswith('nation') else 'ADM1')
    return 'L', 'LCTY'


def nearest_features(feats: dict, lat, lon, qkeys=None, nc=None, max_km: float = 8.0):
    """Index (into ``feats``) of the nearest GeoNames P / A feature within ``max_km`` of each (lat, lon) whose primary name has the key ``qkeys[i]``
    (the TGN subject's own name: the same place under its modern name), -1 when none."""
    fl, fo = np.asarray(feats['lat'], dtype='float64'), np.asarray(feats['lon'], dtype='float64')
    cell = 0.2
    ci = np.floor((fl + 90) / cell).astype(np.int64)
    cj = np.floor((fo + 180) / cell).astype(np.int64)
    key = ci * 100000 + cj
    order = np.argsort(key, kind='stable')
    ks = key[order]
    out = np.full(len(lat), -1, dtype=np.int64)
    dist = np.full(len(lat), np.inf)
    for q, (y, x) in enumerate(zip(lat, lon)):
        i0, j0 = int(np.floor((y + 90) / cell)), int(np.floor((x + 180) / cell))
        jw = int(max_km / (111.0 * cell * max(0.2, np.cos(np.radians(y))))) + 1
        cand = []
        for i in range(i0 - 1, i0 + 2):
            for j in range(j0 - jw, j0 + jw + 1):
                kk = i * 100000 + j
                a = np.searchsorted(ks, kk, 'left')
                b = np.searchsorted(ks, kk, 'right')
                if b > a:
                    cand.append(order[a:b])
        if cand:
            c = np.concatenate(cand)
            d = hav(y, x, fl[c], fo[c])
            near = np.flatnonzero(d <= max_km)
            if qkeys is not None:
                near = np.array([j for j in near if nc(feats['name'][c[j]]) == qkeys[q]], dtype=int)
            if len(near):
                m = near[int(np.argmin(d[near]))]
                out[q], dist[q] = c[m], d[m]
    return out, dist


def add_tgn_names(df: pd.DataFrame, feats: dict, tgn_dir: Path, need: set, need_skel: set, nc) -> pd.DataFrame:
    """Add the TGN names that the entities of the atlas ask for to the GeoNames name index (see the module docstring)."""
    t0 = time.time()
    N = pq.read_table(tgn_dir / 'tgn_names.parquet', columns=['subject_id', 'name', 'is_preferred']).to_pandas()
    uniq = pd.Series(N['name'].dropna().unique())
    kmap = {u: nc(u) for u in uniq}
    N['key'] = N['name'].map(kmap)
    ok = N['key'].str.len().ge(3)
    m = ok & (N['key'].isin(need) | N['key'].map(lambda k: GT.skeleton(k) in need_skel if isinstance(k, str) and len(k) >= 3 else False))
    H = N[m].drop_duplicates(['subject_id', 'key']).copy()
    _t(f'TGN: {len(N)} names, {len(H)} (subject, key) hits ({time.time() - t0:.0f}s)')
    P = pq.read_table(tgn_dir / 'tgn_places.parquet', columns=['subject_id', 'name', 'place_type', 'lat', 'lon', 'merged_stat'],
                      filters=[('subject_id', 'in', sorted(set(H['subject_id'])))]).to_pandas()
    H = H.merge(P.rename(columns={'name': 'tgn_name'}), on='subject_id', how='inner')
    H = H[H['lat'].notna() & H['lon'].notna() & (H['lat'].abs() <= 90) & (H['lon'].abs() <= 180)].reset_index(drop=True)
    near, dist = nearest_features(feats, H['lat'].to_numpy(), H['lon'].to_numpy(), H['tgn_name'].map(lambda x: nc(x)).to_numpy(), nc, 8.0)
    fgid = np.asarray(feats['gid'], dtype=object)
    have = set(zip(df['key'], df['gid']))
    add = []
    n_alias = n_only = n_dup = 0
    for r, f, d in zip(H.itertuples(index=False), near, dist):
        if f >= 0:                                  # the TGN point is next to the GeoNames feature of the same name: another name of that feature
            gid = str(fgid[f])
            if (r.key, gid) in have:
                n_dup += 1
                continue
            have.add((r.key, gid))
            fcode = feats['fcode'][f]
            add.append((r.key, gid, False, feats['name'][f], feats['lat'][f], feats['lon'][f], 'P' if str(fcode).startswith('PPL') else 'A', fcode,
                        feats['cc'][f], '', int(feats['pop'][f]), 'tgn'))
            n_alias += 1
        else:                                       # nothing of GeoNames within 8 km: a TGN-only candidate
            gid = 'tgn' + str(r.subject_id)
            if (r.key, gid) in have:
                continue
            have.add((r.key, gid))
            fc, code = tgn_fcode(r.place_type)
            add.append((r.key, gid, bool(r.is_preferred), r.tgn_name, float(r.lat), float(r.lon), fc, code, '', '', 0, 'tgn'))
            n_only += 1
    if add:
        a = pd.DataFrame(add, columns=['key', 'gid', 'prim', 'name', 'lat', 'lon', 'fclass', 'fcode', 'cc', 'adm1', 'pop', 'src'])
        df = pd.concat([df, a], ignore_index=True)
    _t(f'TGN names added to the index: {n_alias} aliases of GeoNames features, {n_only} TGN-only candidates, {n_dup} already known '
       f'({time.time() - t0:.0f}s); {ATTRIBUTION_TGN}')
    return df


ATTRIBUTION_TGN = 'Getty Thesaurus of Geographic Names, The J. Paul Getty Trust (ODC-By 1.0)'


class GN:
    """Sorted arrays over the index with O(1) key lookup."""

    def __init__(self, df: pd.DataFrame):
        keys = df['key'].to_numpy()
        o = np.argsort(keys, kind='stable')
        d = df.iloc[o].reset_index(drop=True)
        self.d = d
        self.gid = d['gid'].to_numpy()
        self.prim = d['prim'].to_numpy()
        self.lat = d['lat'].to_numpy('float64')
        self.lon = d['lon'].to_numpy('float64')
        self.fclass = d['fclass'].to_numpy()
        self.fcode = d['fcode'].to_numpy()
        self.cc = d['cc'].to_numpy()
        self.pop = d['pop'].to_numpy('int64')
        self.name = d['name'].to_numpy()
        self.tgn = (d['src'].to_numpy() == 'tgn') if 'src' in d else np.zeros(len(d), dtype=bool)
        self.exact_idx = self._ix(d['key'].to_numpy())
        sk = d['skel'].to_numpy()
        self.sk_order = np.argsort(sk, kind='stable')
        self.skel_idx = self._ix(sk[self.sk_order])

    @staticmethod
    def _ix(arr):
        u, start, cnt = np.unique(arr, return_index=True, return_counts=True)
        return {k: (s, s + c) for k, s, c in zip(u.tolist(), start.tolist(), cnt.tolist())}

    def rows_exact(self, key):
        r = self.exact_idx.get(key)
        return np.arange(r[0], r[1]) if r else np.empty(0, dtype=int)

    def rows_skel(self, sk):
        r = self.skel_idx.get(sk)
        return self.sk_order[r[0]:r[1]] if r else np.empty(0, dtype=int)


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def load_places(geo: Path, local: Path) -> pd.DataFrame:
    gn_ = pq.ParquetFile(geo).schema_arrow.names
    extra = [c for c in ('coord_kind', 'coord_radius_km', 'place_kind', 'cb_county', 'cb_match', 'cb_lat', 'cb_lon', 'cb_geo_conf') if c in gn_]      # v2.3
    g = pq.read_table(geo, columns=['coll_slug', 'entity_key', 'name', 'norm_name', 'n_mentions', 'n_in_period', 'tier',
                                    'coord_source', 'lat', 'lon', 'qid'] + extra).to_pandas()
    lcols = ['coll_slug', 'entity_key', 'key_type', 'note_type', 'variants']
    names = pq.ParquetFile(local).schema_arrow.names
    if 'person_class' in names:
        lcols.append('person_class')
    l = pq.read_table(local, columns=lcols).to_pandas()
    if 'person_class' not in l:
        l['person_class'] = None
    p = g.merge(l, on=['coll_slug', 'entity_key'], how='left', validate='1:1')
    p['place_class'] = [PCL.classify(n, t, c) for n, t, c in zip(p['name'], p['note_type'], p['person_class'])]
    p['family'] = p['coll_slug'].map(FAMILY).fillna('other')
    p['src0'] = p['coord_source'].where(p['lat'].notna(), 'none')
    return p.reset_index(drop=True)


def doc_codes(m: pd.DataFrame) -> pd.Series:
    """Document code of each mention row. ``doc_id`` is the root xml:id or the file stem; a few editions give every file the same one (Hearth Tax: 61 files,
    five ``doc_id`` values), which pooled their places into one "document" in v2.1: where one ``doc_id`` spans several files the file is the document."""
    sp = m['source_path'].fillna('')
    did = m['doc_id'].fillna('')
    n_sp = sp.groupby([m['coll_slug'], did]).transform('nunique')
    key = np.where(n_sp.to_numpy() > 1, (m['coll_slug'] + '|' + sp + '|' + did).to_numpy(), (m['coll_slug'] + '|' + did).to_numpy())
    return pd.Series(key, index=m.index).astype('category').cat.codes


def load_pairs(mentions: Path, places: pd.DataFrame):
    """(doc, row) distinct pairs, the tag hints per place row, and the region of each file whose own headings name one (``regions.doc_region``)."""
    mcols = ['coll_slug', 'doc_id', 'source_path', 'kind', 'tag', 'zone', 'local_key', 'surface', 'canonical']
    mnames = pq.ParquetFile(mentions).schema_arrow.names
    if 'excluded' in mnames:
        mcols.append('excluded')
    if 'lang' in mnames:                                  # v2.3: the language of the mention, for the Latin path
        mcols.append('lang')
    t = pq.read_table(mentions, columns=mcols)
    t = t.filter(pc.and_(pc.equal(t.column('canonical'), True), pc.equal(t.column('kind'), 'place')))
    m = t.to_pandas()
    if 'excluded' in m:                                   # v2: commentary, footnotes, repositories ... are not context
        m = m[~m['excluded'].fillna(False).astype(bool)]
    uniq = m['surface'].fillna('').unique()
    nm = {s: _norm(s) for s in uniq}
    n = m['surface'].fillna('').map(nm)
    key = np.where(m['local_key'].notna(), m['local_key'], 'surf:' + n)
    keep = (m['local_key'].notna() | (n != '')).to_numpy()
    m = m.assign(entity_key=key)[keep]
    ix = pd.Series(np.arange(len(places)), index=pd.MultiIndex.from_arrays([places['coll_slug'], places['entity_key']]))
    ix = ix[~ix.index.duplicated()]
    row = ix.reindex(pd.MultiIndex.from_arrays([m['coll_slug'], m['entity_key']])).to_numpy()
    m = m.assign(row=row)
    m = m[m['row'].notna()]
    m['row'] = m['row'].astype('int64')
    m['doc'] = doc_codes(m)
    pairs = m[['doc', 'row']].drop_duplicates()
    doc_regions = file_regions(m)
    # tag hints: which extra GeoNames classes the edition's tags allow
    tg = m['tag'].fillna('').str.lower()
    geo = tg.str.contains('geogname', regex=False)
    m = m.assign(h=geo | tg.map(lambda s: any(w in s for w in HINT_H)), t=geo | tg.map(lambda s: any(w in s for w in HINT_T)),
                 l=geo | tg.map(lambda s: any(w in s for w in HINT_L)))
    hint = m.groupby('row')[['h', 't', 'l']].mean()
    if 'lang' in m:
        kn = m['lang'].map(LN.is_latin_language)
        known = kn.notna()
        ls = pd.DataFrame({'row': m['row'], 'known': known.astype('int64'), 'latin': (kn == True).astype('int64')}).groupby('row')[['known', 'latin']].sum()   # noqa: E712
        hint = hint.join(ls.rename(columns={'known': 'n_known', 'latin': 'n_latin'}), how='left')
    _t(f'{len(pairs)} (doc, place) pairs in {pairs["doc"].nunique()} documents, {len(hint)} places with tags, '
       f'{len(doc_regions)} documents with a region of their own')
    return pairs, hint, doc_regions


def surface_compatible(name, surface) -> bool:
    """The surface of a mention is the form of the place's name: the first word starts alike (an inflected ending changes the end of a Latin word, never its first letters)
    or one is the beginning of the other (an abbreviation: 'Bat.'). 'Tagum' (the Tagus) under the register name 'Fagum' is not (v2.4, cidtc)."""
    a = LN.fold_key(str(name or '')).replace('w', 'u').split()
    b = LN.fold_key(str(surface or '')).replace('w', 'u').split()
    if not a or not b:
        return True
    x, y = a[0], b[0]
    return x[:3] == y[:3] or x.startswith(y) or y.startswith(x)


def load_mention_aux(mentions: Path, places: pd.DataFrame, log=_t) -> dict:
    """{row: {'windows': [passages], 'surf_ok': share}} for the places the v2.4 guards need: the passages of the mentions of a place whose name is in a family of Latin homonyms
    (``latin_homonyms.FAMILIES``), and the share of a place's mentions whose surface is the form of its name (``surface_compatible``; stored only where it is below 1)."""
    names = places['name'].tolist()
    fam_rows = {i for i, n in enumerate(names) if LH.family_of(n) is not None}
    mcols = ['coll_slug', 'kind', 'tag', 'local_key', 'surface', 'canonical']
    mnames = pq.ParquetFile(mentions).schema_arrow.names
    if 'context' in mnames:
        mcols.append('context')
    if 'excluded' in mnames:
        mcols.append('excluded')
    t = pq.read_table(mentions, columns=mcols)
    t = t.filter(pc.and_(pc.equal(t.column('canonical'), True), pc.equal(t.column('kind'), 'place')))
    m = t.to_pandas()
    if 'excluded' in m:
        m = m[~m['excluded'].fillna(False).astype(bool)]
    uniq = m['surface'].fillna('').unique()
    nm = {x: _norm(x) for x in uniq}
    n = m['surface'].fillna('').map(nm)
    key = np.where(m['local_key'].notna(), m['local_key'], 'surf:' + n)
    keep = (m['local_key'].notna() | (n != '')).to_numpy()
    m = m.assign(entity_key=key)[keep]
    ix = pd.Series(np.arange(len(places)), index=pd.MultiIndex.from_arrays([places['coll_slug'], places['entity_key']]))
    ix = ix[~ix.index.duplicated()]
    row = ix.reindex(pd.MultiIndex.from_arrays([m['coll_slug'], m['entity_key']])).to_numpy()
    m = m.assign(row=row)
    m = m[m['row'].notna()]
    m['row'] = m['row'].astype('int64')
    out: dict = {}
    g = m.assign(_s=m['surface'].fillna('')).groupby(['row', '_s']).size().reset_index(name='n')
    pn = places['name'].to_numpy()
    comp = {}
    for r, sfc, c in zip(g['row'], g['_s'], g['n']):
        d = comp.setdefault(int(r), [0, 0])
        d[1] += int(c)
        if surface_compatible(pn[r], sfc):
            d[0] += int(c)
    for r, (ok, tot) in comp.items():
        if tot and ok < tot:
            out.setdefault(r, {})['surf_ok'] = ok / tot
    if fam_rows and 'context' in m:
        sub = m[m['row'].isin(fam_rows)]
        for r, ctxs in sub.groupby('row')['context']:
            out.setdefault(int(r), {})['windows'] = [c for c in ctxs.dropna().tolist() if c][:400]
    log(f'mention aux: {sum(1 for v in out.values() if "windows" in v)} places with passages (Latin homonyms), {sum(1 for v in out.values() if "surf_ok" in v)} places whose surface differs from the name')
    return out


def file_regions(m: pd.DataFrame) -> dict:
    """{doc code: (lat, lon, radius_km)} for the files of the collections whose files are returns of one county (Hearth Tax): the region comes from the
    file's own headings (``placeName@county``, the header title, the city / borough headings), see ``regions.doc_region``."""
    out: dict = {}
    sub = m[m['coll_slug'].isin(RGN.DOC_REGION_COLLECTIONS)]
    for d, g in sub.groupby('doc'):
        tag = g['tag'].fillna('').str.lower()
        cty = g.loc[tag.eq('placename@county'), 'surface'].dropna().tolist()
        hdr = g.loc[g['zone'].eq('header') & tag.eq('placename'), 'surface'].dropna().tolist()
        oth = g.loc[tag.isin(['placename@city', 'placename@borough']), 'surface'].dropna().tolist()
        r = RGN.doc_region(cty, hdr, oth)
        if r:
            out[int(d)] = (r[1], r[2], r[3])
    return out


# ---------------------------------------------------------------------------------------------------------------------
def sat(x, k=4.0):
    return 1.0 - np.exp(-k * np.clip(x, 0, None))


def cluster_points(lat, lon, score, km):
    """Greedy clusters by decreasing score; returns cluster id per point."""
    order = np.argsort(-score, kind='stable')
    cid = np.full(len(lat), -1)
    reps = []
    for i in order:
        for c, r in enumerate(reps):
            if hav(lat[i], lon[i], lat[r], lon[r]) <= km:
                cid[i] = c
                break
        else:
            cid[i] = len(reps)
            reps.append(i)
    return cid, reps


class Geocoder:
    def __init__(self, P: pd.DataFrame, gn: GN, pairs: pd.DataFrame, hint: pd.DataFrame, doc_regions: Optional[dict] = None, resources: Optional[dict] = None):
        """``resources`` (v2.3): ``orbis`` (``latin_geo.OrbisLatinus``), ``wdlatin`` (``latin_geo.WikidataLatin``): the Latin path (path A) is on when either is given."""
        self.P = P
        self.gn = gn
        self.pairs = pairs
        self.doc_regions = doc_regions or {}
        self.row_regions = self._row_regions(pairs, self.doc_regions)
        n = len(P)
        self.res = resources or {}
        self.orbis = self.res.get('orbis')
        self.wdlatin = self.res.get('wdlatin')
        self.latin_on = bool(self.orbis) or bool(self.wdlatin)
        self.aux = self.res.get('mention_aux') or {}          # v2.4: passages of the Latin homonyms, surface check
        self.country_cc = self._countries(gn)
        self.area_cc = self._area_only(gn)
        self.name_forms = [self.adjust(GT.forms(s)) for s in P['name']]
        self.var_forms = []
        for nm, var in zip(P['name'], P['variants']):
            seen, out = set(), []
            for s in (list(var) if var is not None else [])[:20]:
                if s == nm:
                    continue
                f = self.adjust(GT.forms(s))
                ks = f.keys(('t0',))
                if f.junk or not ks or ks[0] in seen:
                    continue
                seen.add(ks[0])
                out.append((s, f))
                if len(out) >= 8:
                    break
            self.var_forms.append(out)
        for i, c in enumerate(P['coll_slug'].to_numpy()):          # the bracketed modern name first (cidtc)
            if c in PAREN_FIRST:
                f = self.name_forms[i]
                if f.paren:
                    f.t0 = list(f.paren) + [x for x in f.t0 if x not in f.paren]
        h = hint.reindex(np.arange(n)).fillna(0.0)
        self.qp = [POL.parse_qualified(x) for x in P['name']]                      # v2.3: head, alternative names and polity qualifier of every name
        self.latin_level = self._latin_levels(P, h)
        nt = P['note_type'].fillna('').str.lower().to_numpy()
        names = P['name'].fillna('').str.lower().to_numpy()
        self.allow_h = np.array([(h.at[i, 'h'] >= 0.5) or any(w in nt[i] for w in HINT_H) or any(w in names[i] for w in HINT_H[:8])
                                 for i in range(n)])
        self.allow_t = np.array([(h.at[i, 't'] >= 0.5) or any(w in nt[i] for w in HINT_T) for i in range(n)])
        self.allow_l = np.array([(h.at[i, 'l'] >= 0.5) or any(w in nt[i] for w in HINT_L) for i in range(n)])
        self.weight = np.maximum(P['n_mentions'].to_numpy('float64'), 1.0)
        self.pclass = np.array([c if isinstance(c, str) else None for c in (P['place_class'] if 'place_class' in P else [None] * n)], dtype=object)
        ntl = P['note_type'].fillna('').str.lower().to_numpy()
        self.expect = [TYPE_EXPECT.get(t, '') for t in ntl]
        self.no_geocode = np.array([t in NO_GEOCODE_TYPES for t in ntl])
        self.why: dict = {}
        self.coll = P['coll_slug'].to_numpy()
        self.fam = P['family'].to_numpy()
        self.pdocs = pairs
        self.cells_docs = None

    @staticmethod
    def _countries(gn: GN) -> dict:
        m = np.isin(gn.fcode, ['PCLI', 'PCLD', 'PCLS', 'PCLF', 'PCL', 'PCLIX']) | ((gn.fcode == 'ADM1') & (gn.pop >= 1_000_000))
        out: dict = defaultdict(set)
        keys = gn.d['key'].to_numpy()
        for k, c in zip(keys[m], gn.cc[m]):
            out[k].add(c)
        return {k: frozenset(v) for k, v in out.items()}

    @staticmethod
    def _area_only(gn: GN) -> dict:
        """v2.3: keys that name a province or region and no settlement of any size ('Guipúzcoa', 'Gipuzkoa', 'Aragón'): {key: country codes}. In 'Guipúzcoa, Irún' such a
        first part is the container of the place that follows, not its head (the first part of 'Würzburg, Kloster' names a city and stays the head)."""
        is_area = np.isin(gn.fcode, ['ADM1', 'ADM2', 'RGN', 'AREA', 'TERR', 'ADM1H', 'ADM2H', 'RGNH'])
        is_p = (gn.fclass == 'P') & (gn.pop >= 3000)
        keys = gn.d['key'].to_numpy()
        has_p = set(keys[is_p])
        out: dict = defaultdict(set)
        for k, c in zip(keys[is_area], gn.cc[is_area]):
            if k not in has_p:
                out[k].add(c)
        return {k: frozenset(v) for k, v in out.items()}

    def _latin_levels(self, P: pd.DataFrame, h: pd.DataFrame) -> list:
        """'latin' / 'mixed' / 'non_latin' / 'unknown' for each place: the language of its own mentions when most of them say one (``lang``), else the edition's level
        (``latin_names.latin_context_level``)."""
        out = []
        nk = h['n_known'].to_numpy() if 'n_known' in h else np.zeros(len(P))
        nl = h['n_latin'].to_numpy() if 'n_latin' in h else np.zeros(len(P))
        for c, k, l in zip(P['coll_slug'], nk, nl):
            lvl = LN.latin_context_level(c)
            if k >= 2:
                sh = l / k
                lvl = 'latin' if sh >= 0.6 else ('non_latin' if sh <= 0.15 else ('mixed' if lvl in ('latin', 'mixed') else 'non_latin'))
            out.append(lvl)
        return out

    @staticmethod
    def _row_regions(pairs: pd.DataFrame, doc_regions: dict) -> dict:
        """{row: [(lat, lon, radius)]}: the regions of the files a place occurs in, for the places all of whose files have one (a place that also occurs in a
        file without a region is not constrained: nothing says where it is)."""
        if not doc_regions or pairs is None or not len(pairs):
            return {}
        has = pairs['doc'].isin(list(doc_regions))
        g = pairs.assign(_h=has).groupby('row')
        ok = g['_h'].all()
        rows = ok.index[ok.to_numpy()]
        sub = pairs[pairs['row'].isin(rows)]
        out: dict = {}
        for r, docs in sub.groupby('row')['doc']:
            out[int(r)] = sorted({tuple(doc_regions[int(d)]) for d in docs})
        return out

    def adjust(self, f: GT.Forms) -> GT.Forms:
        """'España, Valladolid': the leading part is a country, the place is the last part (country = qualifier).

        v2.3, compound strings. 'Province, Town' ('Guipúzcoa, Irún', 'Guipúzcoa, Behobia'), 'Europa, Holanda, Amesterdão', 'España, Zaragoza, Bubierca': the LAST part is
        the place and everything before it is its container. v2.2 kept the middle parts as fallback keys and the first part as a comma key: Bubierca was put on Zaragoza,
        Irún on the centroid of Guipúzcoa, Amsterdam on the point of Europe. Now the container parts only qualify (they are the anchor of the leaf: the leaf must be near
        the container, ``qual_anchor``); when the leaf is not found the place stays unplaced. The first part is a container when it is a country, a region of the exonym
        table (Europa) or a province / region that names no settlement (``area_cc``)."""
        if len(f.parts) >= 2:
            first, last = f.parts[0], f.parts[-1]
            cc = self.country_cc.get(first)
            is_area = False
            if cc is None:
                cc = self.area_cc.get(first)
                is_area = cc is not None
            if cc is None:
                h = EX.lookup(first)
                if h is not None and h[4] == 'area':
                    cc = frozenset()
            if cc is not None and first != last and not GT.is_junk_key(last) and not POL.is_qualifier(last):
                f.qcc = cc
                f.t0 = [(last, 'comma_last')] + [(k, r) for k, r in f.t0 if k not in f.parts]
                f.t1 = []
                f.leaf_only = True
                mids = [m for m in f.parts[1:-1] if not GT.is_junk_key(m)]
                f.qual = mids[-1] if mids else (first if is_area else '')       # the nearest container anchors the leaf (a country does by its country code)
                f.container = first
                f.container_generous = True
        return f

    # -- reference index of placed places --------------------------------------------------------------------------
    def set_anchors(self, lat0, lon0):
        """lat0/lon0: coordinates of the places that count as placed (register / authority); NaN elsewhere."""
        self.lat0, self.lon0 = lat0, lon0
        placed = ~np.isnan(lat0)
        self.placed = placed
        R: dict = defaultdict(list)
        Rv: dict = defaultdict(list)
        for i in np.flatnonzero(placed):
            f = self.name_forms[i]
            for k in f.keys(('t0', 't1', 'paren')):
                R[k].append(i)
            for _, fv in self.var_forms[i]:
                for k in fv.keys(('t0',)):
                    Rv[k].append(i)
        self.R = {k: np.array(v) for k, v in R.items()}
        self.Rv = {k: np.array(v) for k, v in Rv.items()}
        # v2.2: the placed places of an edition whose name is "Head <qualifier word> Something" (Bergen op Zoom, Frankfurt an der Oder, Newcastle upon Tyne)
        QL: dict = defaultdict(list)
        for i in np.flatnonzero(placed):
            ks = self.name_forms[i].keys(('t0',))
            toks = ks[0].split() if ks else []
            if len(toks) >= 3 and toks[1] in QUALIFIER_WORDS and len(toks[0]) >= 3:
                QL[(self.coll[i], toks[0])].append(i)
        self.QL = {k: np.array(v) for k, v in QL.items()}
        Rs: dict = defaultdict(list)
        for k, v in self.R.items():
            Rs[GT.skeleton(k)].extend(v.tolist())
        self.Rs = {k: np.unique(v) for k, v in Rs.items()}
        self._latin_anchors(placed)
        # v2.3: the centre of the places a regional collection has placed, for a place without a county or a suggestion (path C)
        self.region_cloud = {}
        for coll in REGIONAL_UNIQUE:
            if coll == 'hearth_tax':
                continue
            m = placed & (self.coll == coll)
            if int(m.sum()) >= 20:
                w = self.weight[m]
                self.region_cloud[coll] = (float(np.median(lat0[m])), float(np.median(lon0[m])), 100.0)

    def _latin_anchors(self, placed):
        """v2.3: for the Latin path, (a) the point of every Wikidata item that a PLACED local carries (a hidden place of the hold-out cannot vouch for itself), (b) the
        names of the placed places of the Latin editions."""
        self.wd_points: dict = {}
        self.RL: dict = {}
        if not self.latin_on:
            return
        qids = self.P['qid'].to_numpy(dtype=object) if 'qid' in self.P else np.full(len(self.P), None, dtype=object)
        ck = self.P['coord_kind'].to_numpy(dtype=object) if 'coord_kind' in self.P else np.full(len(self.P), None, dtype=object)
        cr = pd.to_numeric(self.P['coord_radius_km'], errors='coerce').to_numpy('float64') if 'coord_radius_km' in self.P else np.full(len(self.P), np.nan)
        RL: dict = defaultdict(list)
        for i in np.flatnonzero(placed):
            q = qids[i]
            if isinstance(q, str) and (q not in self.wd_points or self.weight[i] > self.wd_points[q][4]):
                rad = float(cr[i]) if ck[i] == 'area' and np.isfinite(cr[i]) else None
                self.wd_points[q] = (float(self.lat0[i]), float(self.lon0[i]), rad, int(i), float(self.weight[i]))
            sh = LN.edition_latin_share(self.coll[i])
            if sh is not None and sh >= 0.5:
                for k in self.name_forms[i].keys(('t0',)):
                    RL[LN.fold_key(k)].append(int(i))
        self.RL = {k: np.array(v) for k, v in RL.items()}

    def set_context(self, a_lat, a_lon, a_w):
        """Regional prior per collection and document clouds from the anchors (a_*: arrays over rows, NaN = none)."""
        P = self.P
        ok = ~np.isnan(a_lat)
        cells = pd.DataFrame({'coll': self.coll[ok], 'ci': np.floor(a_lat[ok] * 2).astype(int), 'cj': np.floor(a_lon[ok] * 2).astype(int),
                              'w': a_w[ok]})
        g = cells.groupby(['coll', 'ci', 'cj'])['w'].sum().reset_index()
        self.prior_cells = {}
        for c, d in g.groupby('coll'):
            self.prior_cells[c] = ((d['ci'].to_numpy() + 0.5) / 2, (d['cj'].to_numpy() + 0.5) / 2, d['w'].to_numpy('float64'))
        self.prior_mass = {c: float(v[2].sum()) for c, v in self.prior_cells.items()}
        # document clouds
        pr = self.pairs
        A = pr[ok[pr['row'].to_numpy()]].copy()
        A['lat'] = a_lat[A['row'].to_numpy()]
        A['lon'] = a_lon[A['row'].to_numpy()]
        na = A.groupby('doc')['row'].transform('size')
        A['wt'] = 1.0 / np.sqrt(na)
        A['ci'] = np.floor(A['lat']).astype(int)
        A['cj'] = np.floor(A['lon']).astype(int)
        dc = A.groupby(['doc', 'ci', 'cj'])['wt'].sum().reset_index().sort_values('doc')
        self._dc_ci, self._dc_cj, self._dc_w = dc['ci'].to_numpy(), dc['cj'].to_numpy(), dc['wt'].to_numpy()
        u, st, cnt = np.unique(dc['doc'].to_numpy(), return_index=True, return_counts=True)
        self._doc_slices = {int(d): (int(a), int(a + c)) for d, a, c in zip(u, st, cnt)}
        self.cloud_cache: dict = {}
        if not hasattr(self, '_rows_docs'):
            self._rows_docs = self.pairs.groupby('row')['doc'].apply(np.array).to_dict()

    def cloud(self, i):
        """(lats, lons, weights) of the anchored places that share a document with place i."""
        if i in self.cloud_cache:
            return self.cloud_cache[i]
        docs = self._rows_docs.get(i)
        res = None
        if docs is not None and len(docs):
            acc: dict = defaultdict(float)
            for d in docs[:3000]:
                sl = self._doc_slices.get(int(d))
                if sl is None:
                    continue
                for ci, cj, w in zip(self._dc_ci[sl[0]:sl[1]], self._dc_cj[sl[0]:sl[1]], self._dc_w[sl[0]:sl[1]]):
                    acc[(int(ci), int(cj))] += w
            if acc:
                items = sorted(acc.items(), key=lambda kv: -kv[1])[:80]
                res = (np.array([k[0] + 0.5 for k, _ in items]), np.array([k[1] + 0.5 for k, _ in items]),
                       np.array([w for _, w in items]))
        self.cloud_cache[i] = res
        return res

    def prior_coll(self, coll, fam, lat, lon, cc=None):
        """(coarse prior, fine prior) in [0, 1]: share of the collection's placed mass near the candidate (150 km / 40 km
        kernels, from the collection's anchors) and, for the coarse one, the family's country seeds."""
        out = np.zeros(len(lat))
        fine = np.zeros(len(lat))
        pcell = self.prior_cells.get(coll)
        if pcell is not None and self.prior_mass.get(coll, 0) >= 5:
            la, lo, w = pcell
            d = hav(lat[:, None], lon[:, None], la[None, :], lo[None, :])
            out = sat((np.exp(-d / TH['sigma_c']) * w[None, :]).sum(axis=1) / w.sum(), 4.0)
            fine = sat((np.exp(-d / TH['sigma_f']) * w[None, :]).sum(axis=1) / w.sum(), 6.0)
        seeds = SEED_OVERRIDE.get(coll) or SEEDS.get(fam)
        if seeds is not None and cc is not None:
            sv = np.array([seeds.get(c, 0.0) for c in cc]) * 0.7
            out = np.maximum(out, sv)
        return out, fine

    def prior_doc(self, i, lat, lon):
        cl = self.cloud(i)
        if cl is None:
            return np.zeros(len(lat))
        la, lo, w = cl
        mass = w.sum()
        if mass < 1.5:
            return np.zeros(len(lat))
        d = hav(lat[:, None], lon[:, None], la[None, :], lo[None, :])
        dens = (np.exp(-d / TH['sigma_d']) * w[None, :]).sum(axis=1) / mass
        return sat(dens, 3.0) * min(1.0, mass / 4.0)

    # -- propagation ---------------------------------------------------------------------------------------------------
    def agree(self, rows):
        w = self.weight[rows]
        lat, lon = self.lat0[rows], self.lon0[rows]
        a = int(np.argmax(w))
        d = hav(lat[a], lon[a], lat, lon)
        inc = d <= TH['agree_km']
        share = float(w[inc].sum() / w.sum())
        return dict(ok=share >= TH['agree_share'], share=share, a=rows[a], lat=float(lat[a]), lon=float(lon[a]),
                    n_ent=int(len(rows)), n_coll=int(len(set(self.coll[rows]))), rad=float(max(5.0, d[inc].max())),
                    all_in=bool(inc.all()))

    def pool_lookup(self, f: GT.Forms, own: int):
        """First tier with placed candidates: returns (tier, key, rows) or None."""
        R, Rs, Rv = self.R, self.Rs, self.Rv
        for tier, tiers in (('t0', ('t0',)), ('t1', ('t1',)), ('paren', ('paren',))):
            ks = [k for k in f.keys(tiers) if k in R]
            rows = [R[k] for k in ks]
            if rows:
                r = np.unique(np.concatenate(rows))
                r = r[r != own]
                if len(r):
                    return tier, ks[0], r
        ks = f.keys(('t0', 't1'))
        mk = []
        for k in ks:
            mk.extend(GT.morph_keys(k))
        mk = [k for k in mk if k in R]
        rows = [R[k] for k in mk]
        if rows:
            r = np.unique(np.concatenate(rows))
            r = r[r != own]
            if len(r):
                return 'morph', mk[0], r
        sk = [(k, GT.skeleton(k)) for k in ks]
        sk = [(k, x) for k, x in sk if x in Rs]
        rows = [Rs[x] for _, x in sk]
        if rows:
            r = np.unique(np.concatenate(rows))
            r = r[r != own]
            if len(r):
                return 'skel', sk[0][0], r
        vk = [k for k in ks if k in Rv]
        rows = [Rv[k] for k in vk]
        if rows:
            r = np.unique(np.concatenate(rows))
            r = r[r != own]
            if len(r):
                return 'var', vk[0], r
        return None

    @staticmethod
    def coarse(f: GT.Forms, key: str) -> bool:
        """The key is the province / country part of a 'Country, Province, Place' name: a coarse fallback."""
        if len(f.parts) < 2:
            return False
        if f.rule_of(key) == 'comma_mid':
            return True
        return bool(f.qcc) and key == f.parts[0]

    def propagate(self, i, f: GT.Forms, homonym: bool = False):
        hit = self.pool_lookup(f, i)
        if hit is None:
            return None
        tier, key, rows = hit
        if RGN.london_applies(self.coll[i]):                 # v2.3: a London church is the church of that name AND that qualifier
            rows = self.church_filter(i, rows)
            if not len(rows):
                self.why[i] = 'parish_qualifier'
                return dict(method='name_propagation', conf=None, reason='same name, other parish (qualifier differs)', tier=tier, key=key)
        fam = self.fam[i]
        allr = self.agree(rows)
        res = None
        if allr['ok']:
            a = allr
            conf = 'medium'
            if tier in ('t0', 't1') and a['n_ent'] >= 2 and a['all_in'] and (not homonym or a['n_coll'] >= 3):
                conf = 'high'
            res = (a, conf, 'all')
        else:
            if fam != 'other':
                fr = rows[self.fam[rows] == fam]
                if len(fr):
                    a = self.agree(fr)
                    if a['ok']:
                        res = (a, 'medium' if a['n_ent'] >= 2 and tier in ('t0', 't1') else 'low', 'family')
        if res is None:
            return dict(method='name_propagation', conf=None, reason=f'candidates disagree ({allr["n_ent"]} entities, '
                        f'share {allr["share"]:.2f})', tier=tier, key=key)
        a, conf, pool = res
        if not self.in_region(i, a['lat'], a['lon']):
            self.why[i] = 'outside_region'
            return dict(method='name_propagation', conf=None, reason='outside the collection region', tier=tier, key=key)
        # intrinsic confidence: two or more placed namesakes that agree = high; everything else needs a second opinion
        if tier in ('morph', 'skel', 'var', 'paren') or pool == 'family' or not a['all_in'] or (homonym and a['n_ent'] < 2):
            if tier in ('morph', 'skel', 'var', 'paren') or homonym and a['n_ent'] < 2:
                conf = 'low'
            elif not a['all_in']:                          # a few far namesakes among many agreeing ones
                conf = 'medium' if (a['share'] >= 0.97 and a['n_ent'] >= 5 and pool == 'all') else 'low'
            else:
                conf = 'medium'
        elif a['n_ent'] < 2:
            conf = 'medium'
        if self.coarse(f, key):
            conf = 'low'
        rep = a['a']
        ev = (f"{a['n_ent']} placed entities named '{key}' ({tier}) in {a['n_coll']} collections agree within {a['rad']:.0f} km "
              f"(share {a['share']:.2f}, pool {pool}); heaviest {self.coll[rep]}:{self.P['name'].iat[rep]} [{self.P['src0'].iat[rep]}]")
        return dict(method='name_propagation', conf=conf, lat=a['lat'], lon=a['lon'], radius=a['rad'], p=a['share'],
                    cand=f"{self.coll[rep]}:{self.P['entity_key'].iat[rep]}", cand_name=self.P['name'].iat[rep], tier=tier, key=key,
                    evidence=ev, n_cands=a['n_ent'], homonym=homonym, diag=dict(
                        n_ent=a['n_ent'], n_coll=a['n_coll'], share=a['share'], all_in=a['all_in'], pool=pool,
                        src=str(self.P['src0'].iat[rep]), same_coll=int(np.sum(self.coll[rows] == self.coll[i])),
                        same_fam=int(np.sum(self.fam[rows] == self.fam[i]))))

    # -- v2.3: London parishes: a saint's name plus its qualifier ---------------------------------------------------------------------------------
    CHURCH_TOKS = frozenset('st saint sainte sankt sanct sancti hallows trinity church parish chapel inn tavern alehouse'.split())     # v2.4: inns too (Cross Keys Inn x 3)
    NOISE_TOKS = frozenset('parish of church the in at st saint sainte sankt sanct s chapel de'.split())

    def church_sig(self, j):
        """The tokens of a name that tell one parish church from another with the same saint: ('margaret', 'westminster') against ('margaret', 'lothbury'). None when
        the name is not a church / parish name."""
        f = self.name_forms[j]
        ks = f.keys(('t0',))
        base = ks[0] if ks else ''
        toks = set(base.split())
        if not (toks & self.CHURCH_TOKS):
            return None
        toks |= set((f.qual or '').split())
        return frozenset(toks - self.NOISE_TOKS)

    def church_filter(self, i, rows):
        """MoEML copied the point of 'St. Margaret Lothbury' to 'Parish of St Margaret (Westminster)' (4 km off), St Martin Vintry to Ludgate, All Hallows Honey Lane to
        London Wall: the bare saint's name is the key of all of them. A London church takes the point of a placed church only when the dedication AND the qualifier
        (the tokens that are not the saint's) are the same."""
        sig = self.church_sig(i)
        if sig is None:
            return rows
        return np.array([r for r in rows if self.church_sig(r) == sig], dtype=rows.dtype)

    def qual_anchor(self, f: GT.Forms):
        """(lat, lon) of the populated place named by the qualifier of 'X (Y)' / 'X, Y', or None. v2.3: for a compound 'Province, Town' / 'Country, Province, Town'
        (``leaf_only``) the qualifier is the nearest container and is read with the radii of a region (a province is named like its capital: Bubierca is 120 km from
        the city of Zaragoza and in the province of that name); an administrative feature of the name comes before the populated place."""
        q = f.qual
        if not q or len(q) < 4:
            return None
        rows = self.gn.rows_exact(q)
        if not len(rows):
            return None
        rows = rows[np.isin(self.gn.fclass[rows], ['P', 'A'])]
        if not len(rows):
            return None
        k = rows[np.argmax(self.gn.pop[rows])]
        if f.container_generous:
            adm = rows[np.isin(self.gn.fcode[rows], ['ADM1', 'ADM2', 'RGN', 'AREA', 'TERR', 'PCLI', 'PCLD', 'PCLS', 'PCL'])]
            if len(adm):
                k = adm[np.argmax(self.gn.pop[adm])]
            return float(self.gn.lat[k]), float(self.gn.lon[k]), (150.0, 250.0, 300.0)
        fc = str(self.gn.fcode[k])
        if self.gn.pop[k] < 1000 and not fc.startswith(('PPLA', 'PPLC', 'ADM', 'PCL')):
            return None
        region = fc.startswith(('ADM', 'PCL', 'RGN'))
        return float(self.gn.lat[k]), float(self.gn.lon[k]), (150.0, 250.0, 300.0) if region else (50.0, 80.0, 100.0)

    def in_region(self, i, lat, lon) -> bool:
        return bool(self.row_keep(i, [lat], [lon], [0], [''])[0])

    def row_keep(self, i, lat, lon, pop, fcode, hard: bool = True, cc=None):
        """The regional priors of place ``i`` on candidate points: the hard box of its collection (``HARD_REGIONS``), the London prior of MoEML / Old Bailey
        (``regions.LONDON_PRIOR``, ``regions.london_keep``) and the region of the files it occurs in (``regions.doc_region``). Countries, first-order regions, capitals and cities of
        a million inhabitants are exempt from the boxes (not from the file region, which exempts only countries). The London prior (v2.3) lets through a
        country, a capital and a populated place of a million, NOT an administrative unit (the Département du Nord is not 'North'); for Old Bailey also the
        names that are not British ('Flushing, a Town in Holland') and the provincial towns of 100,000 inhabitants."""
        coll = self.coll[i]
        lat, lon = np.atleast_1d(np.asarray(lat, dtype='float64')), np.atleast_1d(np.asarray(lon, dtype='float64'))
        fcode = np.atleast_1d(np.asarray(fcode))
        keep = region_keep(coll, lat, lon, pop, fcode) if hard else np.ones(len(lat), dtype=bool)
        if RGN.london_applies(coll):
            keep = keep & RGN.london_keep(coll, lat, lon, pop, fcode, cc)
        regs = self.row_regions.get(int(i))
        if regs:
            keep = keep & (RGN.in_disc(lat, lon, regs) | np.array([str(f).startswith('PCL') for f in fcode]))
        return keep

    # -- GeoNames ------------------------------------------------------------------------------------------------------
    def gn_rows(self, i, f: GT.Forms, classes=None):
        """First tier with GeoNames rows: (tier, key, row indices)."""
        gn = self.gn
        ok_cls = {'P', 'A'}
        if self.allow_h[i]:
            ok_cls.add('H')
        if self.allow_t[i]:
            ok_cls.add('T')
        if self.allow_l[i]:
            ok_cls.add('L')
        if self.expect[i]:                                  # the register says mountain / river / country
            ok_cls = set(self.expect[i])
        if classes is not None:                             # v2.3: the caller knows the kind of the place (Orbis Latinus type, polity qualifier)
            ok_cls = set(classes)

        def sel(idx):
            if not len(idx):
                return idx
            m = np.isin(gn.fclass[idx], list(ok_cls))
            return idx[m]
        def fallback_only(rows):
            """v2.1: a name that only the Getty TGN gives a place (a Latin or historical form) is a fallback: where GeoNames has candidates of the
            tier, the TGN rows are dropped (a TGN variant such as 'Evesham' for Epsom must not outrank the real Evesham)."""
            t = gn.tgn[rows]
            return rows[~t] if (t.any() and (~t).any()) else rows
        for tier, tiers in (('t0', ('t0',)), ('t1', ('t1',)), ('paren', ('paren',))):
            ks = f.keys(tiers)
            idx = [(k, sel(gn.rows_exact(k))) for k in ks]
            idx = [(k, x) for k, x in idx if len(x)]
            if idx:
                return tier, idx[0][0], fallback_only(np.unique(np.concatenate([x for _, x in idx])))
        ks = f.keys(('t0', 't1'))
        mk = []
        for k in ks:
            mk.extend(GT.morph_keys(k))
        idx = [(k, sel(gn.rows_exact(k))) for k in mk]
        idx = [(k, x) for k, x in idx if len(x)]
        if idx:
            return 'morph', idx[0][0], fallback_only(np.unique(np.concatenate([x for _, x in idx])))
        idx = []
        for k in ks:
            idx.append((k, sel(gn.rows_skel(GT.skeleton(k)))))
        idx = [(k, x) for k, x in idx if len(x)]
        if idx:
            return 'skel', idx[0][0], fallback_only(np.unique(np.concatenate([x for _, x in idx])))
        return None

    def gn_rank(self, i, f: GT.Forms, classes=None, ccs=None):
        hit = self.gn_rows(i, f, classes)
        if hit is None:
            return None
        tier, key, idx = hit
        gn = self.gn
        # aggregate by geonameid
        gid = gn.gid[idx]
        order = np.argsort(gid, kind='stable')
        idx, gid = idx[order], gid[order]
        first = np.r_[True, gid[1:] != gid[:-1]]
        grp = np.cumsum(first) - 1
        prim = np.zeros(grp.max() + 1, dtype=bool)
        np.logical_or.at(prim, grp, gn.prim[idx])
        tgo = np.ones(grp.max() + 1, dtype=bool)               # every matching row of the feature comes from the Getty TGN (GeoNames lacks the name)
        np.logical_and.at(tgo, grp, gn.tgn[idx])
        rep = idx[first]
        lat, lon, pop = gn.lat[rep], gn.lon[rep], gn.pop[rep]
        fcode, cc = gn.fcode[rep], gn.cc[rep]
        if len(rep) > 400:                                  # keep the most populated
            top = np.argsort(-pop, kind='stable')[:400]
            rep, lat, lon, pop, fcode, cc, prim, tgo = rep[top], lat[top], lon[top], pop[top], fcode[top], cc[top], prim[top], tgo[top]
        if ccs:                                             # v2.3: the country the Orbis entry names
            ok = np.isin(cc, list(ccs))
            if not ok.any():
                self.why[i] = 'latin_country'
                return None
            rep, lat, lon, pop, fcode, cc, prim, tgo = rep[ok], lat[ok], lon[ok], pop[ok], fcode[ok], cc[ok], prim[ok], tgo[ok]
        keep = self.row_keep(i, lat, lon, pop, fcode, cc=cc)       # strict regional priors: collection box, London prior, region of the file
        if not keep.all():
            if not keep.any():
                self.why[i] = 'outside_region'
                return None
            rep, lat, lon, pop, fcode, cc, prim, tgo = rep[keep], lat[keep], lon[keep], pop[keep], fcode[keep], cc[keep], prim[keep], tgo[keep]
        pop_s = np.log10(pop + 1.0) / 7.0
        type_w = np.array([TYPE_W.get(c, 0.3) for c in fcode])
        mq = np.where(prim, 1.0, 0.75) * {'t0': 1.0, 't1': 0.95, 'paren': 0.8, 'morph': 0.6, 'skel': 0.5}[tier]
        coll, fam = self.coll[i], self.fam[i]
        pcv, pfv = self.prior_coll(coll, fam, lat, lon, cc)
        pdv = self.prior_doc(i, lat, lon)
        S = (TH['w_pop'] * pop_s + TH['w_type'] * type_w + TH['w_match'] * mq + TH['w_pc'] * pcv + TH['w_pd'] * pdv
             + TH['w_pf'] * pfv)
        if f.qcc:
            S = S + np.where(np.isin(cc, list(f.qcc)), 0.8, -1.5)
        S = S - np.where(fcode == 'PPLX', 1.0, 0.0)
        qa = self.qual_anchor(f)                             # 'Neukloster (Buxtehude)': the head name is searched near the qualifier
        dq = None
        if qa is not None:
            dq = hav(lat, lon, qa[0], qa[1])
            near, far, _rej = qa[2]
            S = S + np.where(dq <= near, 1.2, np.where(dq > far, -1.0, 0.0))
        cid, reps = cluster_points(lat, lon, S, TH['cluster_km'])
        cs = np.array([S[cid == c].max() for c in range(len(reps))])
        o = np.argsort(-cs)
        e = np.exp((cs - cs.max()) / TH['T'])
        pr = e / e.sum()
        best_c = o[0]
        b = reps[best_c]
        # notable clusters (what makes a name a homonym)
        notable = 0
        for c in range(len(reps)):
            m = cid == c
            if pop[m].max() >= 5000 or any(x in ('ADM1', 'PCLI', 'PPLA', 'PPLC', 'ADM2', 'PPLA2') for x in fcode[m]):
                notable += 1
        ctx = TH['w_pc'] * pcv + TH['w_pd'] * pdv + TH['w_pf'] * pfv
        ctx_margin = 0.0
        if len(reps) > 1:
            second = o[1]
            ctx_margin = float(ctx[b] - ctx[reps[second]])
        return dict(tier=tier, key=key, b=int(b), p=float(pr[best_c]), n_clusters=len(reps), notable=notable, S=float(S[b]),
                    lat=float(lat[b]), lon=float(lon[b]), fcode=str(fcode[b]), cc=str(cc[b]), pop=int(pop[b]),
                    prim=bool(prim[b]), tgn=bool(tgo[b]), gid=str(gn.gid[rep[b]]), nm=str(gn.name[rep[b]]), pc=float(pcv[b]), pd=float(pdv[b]), pf=float(pfv[b]),
                    ctx_margin=ctx_margin, ext=int(np.sum(cid == best_c)), qual_km=(float(dq[b]) if dq is not None else None),
                    qual_rej=(qa[2][2] if qa is not None else None),
                    rad=float(RADIUS.get(str(fcode[b]), 25)), fclass=str(gn.fclass[rep[b]]))

    def gn_decide(self, i, f: GT.Forms, classes=None, ccs=None):
        r = self.gn_rank(i, f, classes, ccs)
        if r is None:
            return None
        amb = r['notable'] >= 2
        conf = None
        has_prior = (self.coll[i] in SEED_OVERRIDE) or (self.fam[i] in SEEDS)
        pc_ok = (r['pc'] >= TH['pc_med']) or not has_prior
        # v2.3: a country or first-order region of half a million inhabitants or more, found under its exact name and with no equal namesake, is not "a village far from
        # everything the collection talks about": Cataluña in the letters of the Low Countries is Catalonia (an area, 549 mentions)
        if (not pc_ok and r['fcode'] in MAJOR_AREA_FCODES and r['pop'] >= MAJOR_AREA_POP and r['tier'] in ('t0', 't1') and not amb and r['p'] >= 0.9):
            pc_ok = True
        if r['p'] >= TH['low_p']:
            conf = 'low'
        if (r['p'] >= TH['med_p'] and r['S'] >= TH['S_med'] and pc_ok
                and (not amb or r['ctx_margin'] >= TH['amb_ctx'] or r['p'] >= TH['high_p'])):
            conf = 'medium'
        if (r['p'] >= TH['high_p'] and r['S'] >= TH['S_high'] and not amb and r['prim'] and r['tier'] in ('t0', 't1')
                and (r['pc'] >= 0.05 or not has_prior) and r['fclass'] in 'PA'):
            conf = 'high'
        if r['tier'] in ('morph', 'skel', 'paren') or r['fcode'] == 'PPLX' or self.coarse(f, r['key']):
            conf = 'low' if conf else None
        if conf and r.get('qual_km') is not None and r['qual_km'] > r['qual_rej']:   # 'Neukloster (Buxtehude)' is not in Mecklenburg
            conf = None
            self.why[i] = 'qualifier_far'
        # a village far from everything the collection (and the document) talks about
        if (conf in ('high', 'medium') and r['pop'] < TH['pop_local'] and self.prior_mass.get(self.coll[i], 0) >= TH['dens_mass']
                and r['pf'] < TH['pf_min'] and r['pd'] < TH['pf_min']):
            conf = 'medium' if conf == 'high' else 'low'
        if conf in ('high', 'medium') and not r['prim'] and r['pop'] == 0 and r['pc'] + r['pd'] < 0.2:
            conf = 'low'
        src = f"TGN {r['gid'][3:]}" if r['gid'].startswith('tgn') else f"GeoNames {r['gid']}"
        ev = (f"{src} {r['nm']} ({r['fcode']}, {r['cc']}, pop {r['pop']}); key '{r['key']}' ({r['tier']}, "
              f"{'name' if r['prim'] else 'alternate'}{', Getty TGN name' if r['tgn'] else ''}); p={r['p']:.2f}, {r['n_clusters']} candidate places"
              f"{', homonym' if amb else ''}; prior coll={r['pc']:.2f}/{r['pf']:.2f} doc={r['pd']:.2f}")
        return dict(method='name_geonames', conf=conf, lat=r['lat'], lon=r['lon'], radius=r['rad'], p=r['p'],
                    cand=('tgn:' + r['gid'][3:]) if r['gid'].startswith('tgn') else 'gn:' + r['gid'], cand_name=r['nm'], tier=r['tier'], key=r['key'],
                    evidence=ev, tgn=r['tgn'],
                    n_cands=r['n_clusters'], homonym=amb, diag={k: r[k] for k in (
                        'S', 'pc', 'pd', 'pf', 'pop', 'notable', 'fcode', 'fclass', 'cc', 'prim', 'ctx_margin', 'n_clusters')})

    # -- one place: both methods, name then variants ------------------------------------------------------------------------
    def resolve_one(self, i, f: GT.Forms):
        gr = self.gn_decide(i, f)
        pr = self.propagate(i, f, bool(gr and gr.get('homonym')))
        return pr, gr

    def _exonym_hit(self, f: GT.Forms, is_var: bool, coll: str = None):
        """First exonym-table hit of one form: the head name only, not the parenthetical locator ('Ehrenberg (Tirol)'); a leading
        comma part only when it is a city ('Lima, Perú'), never a country or region ('España, Navarra, Pamplona'); a variant only
        when it is a city (a variant that names a country does not make a country of the entity)."""
        for key in f.keys(('t0', 't1')):
            if (f.qual and key == f.qual) or f.rule_of(key) in ('slash', 'slash+strip'):
                continue
            lead = f.rule_of(key) in ('comma', 'comma+strip')
            hit = EX.lookup(key, coll)
            if hit and (lead or is_var) and hit[4] != 'city':
                continue
            if hit:
                return key, hit
        return None

    def edition_qualified(self, i, f: GT.Forms):
        """The bare name of an edition whose own register has the qualified form and no other place of that bare name: Bergen in the Descartes letters is Bergen
        op Zoom (the register has it, 354 mentions; the bare Bergen of GeoNames is Norway's second city). The editors register the places their text names,
        so the one qualified namesake is the reading of the bare form. One namesake only (all qualified places of the edition within 30 km of each other),
        no placed place of the bare name in the same edition, the bare name a single word; medium confidence."""
        ks = f.keys(('t0',))
        if not ks or len(ks[0].split()) != 1 or len(ks[0]) < 3 or not len(self.QL):
            return None
        k0, coll = ks[0], self.coll[i]
        rows = self.QL.get((coll, k0))
        if rows is None or not len(rows):
            return None
        if any(self.coll[r] == coll for r in self.R.get(k0, ())):            # the edition has a placed place of exactly that name: it decides
            return None
        lat, lon = self.lat0[rows], self.lon0[rows]
        if float(np.max(hav(lat[0], lon[0], lat, lon))) > 30.0:
            return None
        a = int(np.argmax(self.weight[rows]))
        r = int(rows[a])
        if not self.in_region(i, lat[a], lon[a]):
            return None
        ev = f"edition register: '{self.P['name'].iat[r]}' [{self.P['src0'].iat[r]}] is the only place of the edition that begins with '{k0}' + a qualifier word"
        return dict(method='name_propagation', conf='medium', lat=float(lat[a]), lon=float(lon[a]), radius=5.0, p=1.0,
                    cand=f"{coll}:{self.P['entity_key'].iat[r]}", cand_name=self.P['name'].iat[r], tier='edition_qualified', key=k0, evidence=ev,
                    n_cands=int(len(rows)), homonym=False, diag={'n_ent': int(len(rows)), 'pool': 'edition'})

    def exonym(self, i):
        """The curated table of Latin / old exonyms (and peoples named after their country) comes before any lookup. The name itself
        decides; the variants of the entity vote only when the name has no hit: the label with at least two variants and more than
        any other wins ('Neuenburg' = Nowe by 'Nowe' and 'castri Novensis' against one stray 'Norimbergae')."""
        coll = self.coll[i]
        h = self._exonym_hit(self.name_forms[i], False, coll)
        if h is None:
            votes: dict = {}
            for _, fv in self.var_forms[i]:
                v = self._exonym_hit(fv, True, coll)
                if v is not None:
                    votes.setdefault(v[1][3], []).append(v)
            if votes:
                ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
                if len(ranked) == 1 or len(ranked[0][1]) > len(ranked[1][1]):
                    if len(ranked[0][1]) >= 2 or sum(len(v) for v in votes.values()) == 1:
                        h = ranked[0][1][0]
        if h is None:
            return None
        key, (lat, lon, rad, label, kind) = h
        ev = f"exonym table: '{key}' = {label} ({kind}, radius {rad:.0f} km)"
        return dict(method='name_exonym', conf='high' if kind == 'city' else 'medium', lat=lat, lon=lon, radius=rad,
                    p=1.0, cand='exonym:' + label, cand_name=label, tier='exonym', key=key, evidence=ev,
                    n_cands=1, homonym=False, diag={'kind': kind})

    # -- v2.3 path B: polities and 'Name, Qualifier' forms ----------------------------------------------------------------------------------------
    @staticmethod
    def _rank(r) -> int:
        return {'high': 3, 'medium': 2, 'low': 1}.get((r or {}).get('conf'), 0)

    def _poly_result(self, key, label, lat, lon, rad, ev, qid=None, tier='polity'):
        return dict(method='name_exonym', conf='medium', lat=float(lat), lon=float(lon), radius=float(rad), p=1.0, cand='exonym:' + label, cand_name=label, tier=tier,
                    key=key, evidence=ev, n_cands=1, homonym=False, qid=qid, diag={'kind': 'area', 'qid': qid})

    def polity(self, i):
        """The reviewed polity table (``polities.POLITIES``) on the whole name, its head and its alternative names; then, for a name that carries a polity qualifier
        (Königreich, Herzogtum, Grafschaft, Landgrafschaft, Fürstentum, Republik, Eidgenossenschaft ...), the exonym table on the head and the alternative names. The
        qualifier sets the kind: the result is an area (medium), whatever the head is called elsewhere ('Pommern, Herzogtum' is the duchy, not the village on the Moselle)."""
        q = self.qp[i]
        if not q.head and not q.full:
            return None
        pol = POL.lookup(q)
        if pol is not None:
            ev = (f"polity table: '{q.full}' = {pol.label}{' (' + pol.qid + ')' if pol.qid else ''}; point {pol.basis}; radius {pol.radius:.0f} km"
                  f"{'; qualifier ' + q.qtype if q.qtype else ''}")
            return self._poly_result(q.head or q.full, pol.label, pol.lat, pol.lon, pol.radius, ev, pol.qid)
        if q.qtype in POL.AREA_QTYPES or q.qtype == 'union':
            for k in (q.head, *q.alts):
                h = EX.lookup(k, self.coll[i]) if k else None
                if h:
                    lat, lon, rad, label, kind = h
                    r = max(rad, POL.radius_for(q.qtype)) if kind != 'area' else max(rad, 60.0)
                    ev = f"exonym table: '{k}' = {label} ({kind}, radius {rad:.0f} km); the qualifier '{q.qtype}' makes the entry an area (radius {r:.0f} km)"
                    return self._poly_result(k, label, lat, lon, r, ev, None, 'polity_exonym')
        return None

    def latin_homonym(self, i, fam: 'LH.Family'):
        """v2.4: a Latin name of several places (``latin_homonyms.FAMILIES``). The passages of the place's mentions name a candidate ('Batavorum', 'Rhodani') or the places of
        its documents support one clearly (the Dutch cloud of a letter says Leiden); Lyon, Maastricht, Turin ... only with a cue of their own. Mixed cues, no support: nothing is
        placed. Medium at most, a point of the table (city) or an area (the sea)."""
        aux = self.aux.get(int(i)) or {}
        ctx = {c.label: self.doc_ctx_at(i, c.lat, c.lon) for c in fam.cands}
        d = LH.decide(fam, aux.get('windows') or [], ctx)
        if d.cand is None:
            self.why[i] = 'latin_homonym_' + d.reason
            return None
        c = d.cand
        if not self.in_region(i, c.lat, c.lon):
            self.why[i] = 'outside_region'
            return None
        cues = ', '.join(f'{k} {v}' for k, v in d.n_cue.items())
        ev = (f"Latin '{self.P['name'].iat[i]}' is a name of several places ({', '.join(x.label for x in fam.cands)}): {d.reason}; passages naming a candidate: {cues} of {d.n_windows}; "
              f"support of the documents' places: " + ', '.join(f'{k} {v:.2f}' for k, v in d.ctx.items()))
        return dict(method='name_exonym', conf='medium', lat=float(c.lat), lon=float(c.lon), radius=float(c.radius), p=1.0, cand='exonym:' + c.label, cand_name=c.label,
                    tier='latin_homonym', key=LN.fold_key(self.P['name'].iat[i]), evidence=ev, n_cands=len(fam.cands), homonym=True,
                    diag={'kind': 'area' if c.kind in ('area', 'sea') else 'city', 'cue': d.n_cue, 'reason': d.reason})

    def polity_gn(self, i):
        """A qualified name that the tables do not know: the head (areas only) or an alternative name (any place: a county is named after its seat, 'Askanien
        (Aschersleben), Grafschaft') through GeoNames. Medium at most; the result is an area with the radius of the qualifier."""
        q = self.qp[i]
        if q.qtype not in POL.AREA_QTYPES:
            return None
        found = []
        for role, k, cls in [('head', q.head, 'AL')] + [('alt', a, 'PAL') for a in q.alts[:2]]:
            if len(k) < 3 or GT.is_junk_key(k):
                continue
            gd = self.gn_decide(i, GT.Forms(t0=[(k, role)]), classes=cls)
            # the head of 'X, Grafschaft' is a region, not a commune of the same name (Hohnstein in Saxony is not the county of the Harz; Raab in Upper Austria is not the see of Győr)
            if gd and role == 'head' and not (str(gd['diag'].get('fcode')) in POLITY_HEAD_FCODES or str(gd['diag'].get('fcode')).startswith('PCL')):
                gd = None
            # v2.4: ... and a GeoNames feature, never a TGN-only locality (Stolberg in Lippe, Gaster near Kandersteg, Hohenlohe in the Upper Palatinate): the reviewed table or nothing
            if gd and (str(gd.get('cand', '')).startswith('tgn:') or gd.get('tgn') or str(gd['diag'].get('fcode')) == 'LCTY'):
                gd = None
            if gd and gd.get('conf') in ('high', 'medium'):
                found.append((role, k, gd))
        if not found:
            return None
        pts = np.array([[g['lat'], g['lon']] for _, _, g in found])
        if len(found) > 1 and float(np.max(hav(pts[0, 0], pts[0, 1], pts[:, 0], pts[:, 1]))) > max(TH['agree_km'], 100.0):
            self.why[i] = 'polity_head_alt_conflict'
            return None
        role, k, gd = found[0]
        r = dict(gd)
        r['conf'] = 'medium'
        r['radius'] = max(float(gd['radius']), POL.radius_for(q.qtype))
        r['tier'] = 'polity_' + role
        r['evidence'] = f"{role} '{k}' of a name with the polity qualifier '{q.qtype}' (an area, radius {r['radius']:.0f} km): " + gd['evidence']
        return r

    def ctx_at(self, i, lat, lon, cc=None) -> float:
        """How well the collection's and the document's places support a point (the context terms of the GeoNames score)."""
        la, lo = np.array([lat], dtype='float64'), np.array([lon], dtype='float64')
        pcv, pfv = self.prior_coll(self.coll[i], self.fam[i], la, lo, None if cc is None else np.array([cc]))
        pdv = self.prior_doc(i, la, lo)
        return float(TH['w_pc'] * pcv[0] + TH['w_pd'] * pdv[0] + TH['w_pf'] * pfv[0])

    def doc_ctx_at(self, i, lat, lon) -> float:
        """Support for a point from the places of the SAME documents only (not from the collection's cloud, which a homonym's own placed namesakes can inflate)."""
        return float(TH['w_pd'] * self.prior_doc(i, np.array([lat], dtype='float64'), np.array([lon], dtype='float64'))[0])

    def _is_area_result(self, fin) -> bool:
        """A result that stands for a country, a region or another area: a radius of 60 km or more, a region / country feature, or a propagated copy of an area."""
        if float(fin.get('radius') or 0) >= 60:
            return True
        fc = str((fin.get('diag') or {}).get('fcode'))
        if fc in REGION_FCODES or fc.startswith('PCL'):
            return True
        cand = fin.get('cand')
        if fin.get('method') == 'name_propagation' and isinstance(cand, str) and ':' in cand and 'coord_kind' in self.P:
            coll, key = cand.split(':', 1)
            m = (self.P['coll_slug'].to_numpy() == coll) & (self.P['entity_key'].to_numpy() == key)
            if m.any():
                j = int(np.flatnonzero(m)[0])
                if self.P['coord_kind'].iat[j] == 'area' or (pd.notna(self.P['coord_radius_km'].iat[j]) and float(self.P['coord_radius_km'].iat[j]) >= 60):
                    return True
        return False

    def head_alt(self, i):
        """'Löbau (Lubawa)', 'Cracow (Kraków)', 'Askanien (Aschersleben)': the head name and the bracketed alternative name are resolved SEPARATELY (v2.2 flattened
        them into one key and a propagated Saxon Löbau drowned Lubawa in Prussia). One confident answer wins; two that agree corroborate each other; two that
        disagree are decided by the places of the edition and of the document (the regional prior), else nothing is placed."""
        q = self.qp[i]
        if not q.alts or q.qtype in POL.AREA_QTYPES:
            return None
        out = []
        for role, k in [('head', q.head)] + [('alt', a) for a in q.alts[:2]]:
            if len(k) < 3 or GT.is_junk_key(k):
                continue
            pr, gr = self.resolve_one(i, GT.Forms(t0=[(k, role)]))
            fin = self.combine(i, pr, gr)
            if fin and fin.get('conf') in ('high', 'medium'):
                out.append((role, k, fin))
        if not out:
            return None
        if len(out) == 1:
            role, k, fin = out[0]
            if role == 'alt' and self._is_area_result(fin):
                self.why[i] = 'alt_is_qualifier'            # 'Molsheim (Frankreich)', 'Cascante (Navarra)': the bracket is the country / region of a place the head names, not the place
                return None
            r = dict(fin)
            r['conf'] = 'medium' if r['conf'] == 'high' else r['conf']
            r['evidence'] = f"{role} '{k}' of '{self.P['name'].iat[i]}' alone: " + fin['evidence']
            r['tier'] = 'head_alt_' + role
            return r
        pts = np.array([[f['lat'], f['lon']] for _, _, f in out])
        d = hav(pts[0, 0], pts[0, 1], pts[:, 0], pts[:, 1])
        tol = max(TH['agree_km'], min(max(f['radius'] for _, _, f in out), TH['tol_cap']))
        if float(d.max()) <= tol:
            role, k, fin = max(out, key=lambda x: self._rank(x[2]))
            r = dict(fin)
            r['conf'] = 'high' if all(f['conf'] in ('high', 'medium') for _, _, f in out) and any(f['conf'] == 'high' for _, _, f in out) else 'medium'
            r['evidence'] = f"head and alternative name of '{self.P['name'].iat[i]}' agree ({d.max():.0f} km): " + fin['evidence']
            r['tier'] = 'head_alt_agree'
            return r
        # v2.4 (Lucena (Córdoba)): the bracket after a village name is, more often than not, the province or the city that LOCATES it, and the province capital of the same name
        # is not the village: when the head is a settlement of its own that lies within LOCATOR_KM of the bracketed larger place, the head is the answer
        hd = [x for x in out if x[0] == 'head']
        al = [x for x in out if x[0] == 'alt']
        if hd and al and self.coll[i] not in PAREN_FIRST:
            hf, af = hd[0][2], al[0][2]
            dh = float(hav(hf['lat'], hf['lon'], af['lat'], af['lon']))
            hfc = str((hf.get('diag') or {}).get('fcode', 'PPL'))           # a propagated head (the same name placed elsewhere in the atlas) carries no feature code: a place
            if dh <= LOCATOR_KM and str((af.get('diag') or {}).get('fcode')) in LOCATOR_FCODES and int((af.get('diag') or {}).get('pop', 0)) > int((hf.get('diag') or {}).get('pop', 0)) \
                    and (hfc.startswith('PPL') or hfc == 'None') and not self._is_area_result(hf):
                r = dict(hf)
                r['conf'] = 'medium'
                r['evidence'] = (f"head '{hd[0][1]}' of '{self.P['name'].iat[i]}' ({dh:.0f} km from the bracketed '{al[0][1]}', a larger place that locates it): the bracket is the locator, "
                                 f"not the place: " + hf['evidence'])
                r['tier'] = 'head_alt_head_locator'
                return r
        ctx = [self.ctx_at(i, f['lat'], f['lon'], (f.get('diag') or {}).get('cc')) for _, _, f in out]
        o = np.argsort(ctx)[::-1]
        alt_ix = [k for k, (role, _, _) in enumerate(out) if role == 'alt']
        if self.coll[i] in PAREN_FIRST and alt_ix:
            # the edition's own convention: the bracketed name is the modern name of the place ('Löbau (Lubawa)'); a German namesake elsewhere does not outvote it
            role, k, fin = out[alt_ix[0]]
            r = dict(fin)
            r['conf'] = 'medium'
            r['evidence'] = (f"head and alternative name disagree ({float(d.max()):.0f} km); {self.coll[i]} gives the modern name in brackets: the alternative '{k}' "
                             f"decides: " + fin['evidence'])
            r['tier'] = 'head_alt_alt'
            return r
        if ctx[o[0]] - ctx[o[1]] >= 0.8:
            role, k, fin = out[int(o[0])]
            r = dict(fin)
            r['conf'] = 'medium'
            r['evidence'] = (f"head and alternative name disagree ({float(d.max()):.0f} km); the places of the edition favour the {role} '{k}' "
                             f"(context {ctx[o[0]]:.2f} against {ctx[o[1]]:.2f}): " + fin['evidence'])
            r['tier'] = 'head_alt_' + role
            return r
        self.why[i] = 'head_alt_conflict'
        return None

    # -- v2.3 path C: English small homonyms (Hearth Tax, Casebooks) --------------------------------------------------------------------------------
    def _cb_suggestion(self, i):
        P = self.P
        if 'cb_lat' not in P:
            return None
        la, lo = pd.to_numeric(P['cb_lat'].iat[i], errors='coerce'), pd.to_numeric(P['cb_lon'].iat[i], errors='coerce')
        gc = P['cb_geo_conf'].iat[i] if 'cb_geo_conf' in P else None
        if not (np.isfinite(la) and np.isfinite(lo)) or gc not in ('high', 'medium'):
            return None
        return float(la), float(lo)

    def regional_region(self, i):
        """(discs, margin, what) of the region the document gives for a place of ``REGIONAL_UNIQUE``: the county of the Hearth Tax file; for the Casebooks the 15 km round the
        directory's own suggestion, else (no suggestion) the 100 km round the centre of the places the collection has placed (the cloud of Forman's and Napier's patients)."""
        coll = self.coll[i]
        if coll == 'hearth_tax':
            regs = self.row_regions.get(int(i))
            return (regs, RGN.REGION_MARGIN_KM, 'the county of its file') if regs else None
        sg = self._cb_suggestion(i)
        if sg is not None:
            return [(sg[0], sg[1], CASEBOOKS_SUGGESTION_KM)], 0.0, f"{CASEBOOKS_SUGGESTION_KM:.0f} km of the Casebooks directory's suggestion"
        cl = self.region_cloud.get(coll)
        return ([cl], 0.0, f'{cl[2]:.0f} km of the centre of the places the collection has placed') if cl else None

    def regional_candidates(self, i, f: GT.Forms):
        """The settlements named like the place that lie in its region: -> (number of distinct places, best index, evidence parts) or None when the rule does not apply.
        Candidates carry the exact name (not a skeleton) and pass the regional priors; places closer than ``UNIQUE_CLUSTER_KM`` are one."""
        coll = self.coll[i]
        if coll not in REGIONAL_UNIQUE or f.junk or f.leaf_only:
            return None
        keys = f.keys(('t0',))[:2]
        rr = self.regional_region(i)
        if not keys or rr is None:
            return None
        regs, margin, what = rr
        gn = self.gn
        parts = [gn.rows_exact(k) for k in keys]
        idx = np.unique(np.concatenate(parts)) if any(len(x) for x in parts) else np.empty(0, dtype=int)
        if not len(idx):
            return (0, None, what, keys)
        idx = idx[(gn.fclass[idx] == 'P') & ~gn.tgn[idx]]
        if len(idx):
            idx = idx[RGN.in_disc(gn.lat[idx], gn.lon[idx], regs, margin)]
        if len(idx):
            idx = idx[self.row_keep(i, gn.lat[idx], gn.lon[idx], gn.pop[idx], gn.fcode[idx], cc=gn.cc[idx])]
        if not len(idx):
            return (0, None, what, keys)
        gid = gn.gid[idx]
        o = np.argsort(gid, kind='stable')
        idx, gid = idx[o], gid[o]
        first = np.r_[True, gid[1:] != gid[:-1]]
        grp = np.cumsum(first) - 1
        prim = np.zeros(grp.max() + 1, dtype=bool)
        np.logical_or.at(prim, grp, gn.prim[idx])
        rep = idx[first]
        cid, reps = cluster_points(gn.lat[rep], gn.lon[rep], gn.pop[rep].astype('float64'), UNIQUE_CLUSTER_KM)
        if len(reps) != 1:
            return (len(reps), None, what, keys)
        b = reps[0]
        return (1, (int(rep[b]), bool(prim.any())), what, keys)

    def regional_unique(self, i, f: GT.Forms):
        """A bare English village name (Weston, Broughton, Milton: 400-500 mentions each in the Casebooks and the Hearth Tax) is placed only when EXACTLY ONE settlement
        of that name lies in the region the document gives (``regional_region``); several namesakes in the region: nothing. Medium when the GeoNames name is the primary
        name or the place has inhabitants, else low."""
        rc = self.regional_candidates(i, f)
        if rc is None:
            return None
        n, best, what, keys = rc
        if n >= 2:
            self.why[i] = 'regional_ambiguous'
            return None
        if n == 0:
            return None
        k, prim = best
        gn = self.gn
        conf = 'medium' if (prim or int(gn.pop[k]) > 0) else 'low'
        ev = (f"GeoNames {gn.gid[k]} {gn.name[k]} ({gn.fcode[k]}, {gn.cc[k]}, pop {int(gn.pop[k])}) is the only settlement named '{keys[0]}' in {what}"
              f"{'; the directory points there too' if self.coll[i] == 'casebooks' and self._cb_suggestion(i) is not None else ''}")
        return dict(method='name_geonames', conf=conf, lat=float(gn.lat[k]), lon=float(gn.lon[k]), radius=float(RADIUS.get(str(gn.fcode[k]), 5.0)), p=1.0,
                    cand='gn:' + str(gn.gid[k]), cand_name=str(gn.name[k]), tier='regional_unique', key=keys[0], evidence=ev, n_cands=1, homonym=False,
                    diag={'fcode': str(gn.fcode[k]), 'pop': int(gn.pop[k]), 'prim': bool(prim)})

    # -- v2.3 path A: Latin names ---------------------------------------------------------------------------------------------------------------
    _ETH_LAT = frozenset(LN.fold_key(w) for w in list(PCL.ETHNONYM) + 'hispanus gallus romanus germanus anglus italus polonus scotus sueonus danus batavus belga turca persa graecus'.split())

    def orbis_point(self, i, e):
        """The Orbis Latinus gives a German-spelled modern name, a type and a country; GeoNames gives the point. The class follows the type, the country the region
        text; a confident answer (or a near-certain one for a name the book dates and places) is a hit."""
        gd = self.gn_decide(i, GT.forms(e.modern), classes=e.classes, ccs=e.ccs)
        if gd is None:
            return None
        if gd.get('conf') in ('high', 'medium') or (gd.get('conf') == 'low' and gd['p'] >= 0.9 and e.ccs):
            return gd
        return None

    def latin(self, i):
        """Path A. The name of a place in a Latin context is de-inflected to nominative candidates (``latin_names.lemmatize``) that are looked up in the Wikidata Latin
        labels of the atlas's own items, the Orbis Latinus (through GeoNames), the exonym and polity tables and the names of the Latin editions' placed places.
        ``latin_geo.combine`` decides (homonym, agreement, strength of the rule), the regional prior of the edition has the last word."""
        if not self.latin_on:
            return None
        lvl = self.latin_level[i]
        if lvl not in ('latin', 'mixed'):
            return None
        name = self.P['name'].iat[i]
        if not isinstance(name, str) or any(c in name for c in '(),/[]'):
            return None
        aux = self.aux.get(int(i)) or {}
        if float(aux.get('surf_ok', 1.0)) < SURFACE_OK_MIN:               # v2.4: the text says 'Tagum', the register standardised 'Fagum': the name is not what the mentions say
            self.why[i] = 'latin_surface_mismatch'
            return None
        coll = self.coll[i]
        cands = [c for c in LN.lemmatize(name, min_score=0.4 if lvl == 'latin' else 0.6) if LN.fold_key(c.lemma) not in LG.NOT_PLACES]      # v2.4: 'Palatinus' is an adjective
        hits = []
        H = LG.Hit
        for c in cands:
            lk = LN.fold_key(c.lemma)
            if lk in self._ETH_LAT:
                continue
            if self.wdlatin is not None:
                for q in self.wdlatin.get(c.lemma):
                    pt = self.wd_points.get(q)
                    if pt:
                        hits.append(H('wikidata', c.score, pt[0], pt[1], pt[2], q, c.lemma, pt[3], str(self.P['name'].iat[pt[3]]), pt[4]))
            h = EX.lookup(_norm(c.lemma), coll)
            if h:
                hits.append(H('exonym', c.score, h[0], h[1], h[2] if h[4] == 'area' else None, None, c.lemma, None, h[3]))
            pol = POL.PLAIN.get(POL._key(c.lemma))
            if pol:
                hits.append(H('polity', c.score, pol.lat, pol.lon, pol.radius, pol.qid, c.lemma, None, pol.label))
            rows = self.RL.get(lk)
            if rows is not None and len(rows):
                ag = self.agree(rows)
                if ag['ok']:
                    a = int(ag['a'])
                    qa = self.P['qid'].iat[a] if 'qid' in self.P else None
                    hits.append(H('atlas', c.score, ag['lat'], ag['lon'], ag['rad'] if ag['rad'] >= 60 else None, qa if isinstance(qa, str) else None, c.lemma, a,
                                  str(self.P['name'].iat[a]), float(self.weight[rows].sum())))
            if self.orbis is not None:
                es = self.orbis.get(c.lemma)
                if es:
                    gd = self.orbis_point(i, es[0])
                    if gd:                                   # a near-certain answer (low, but p >= 0.9 with the book's country) weighs a little less than a confident one
                        sc = c.score if gd.get('conf') in ('high', 'medium') else c.score * ORBIS_LOW_FACTOR
                        if int(gd.get('diag', {}).get('pop', 0)) < ORBIS_MIN_POP or es[0].classes != 'P':      # a village named by the book alone ('Vianam' -> Weinstetten) or a
                            sc = min(sc, LG.WEAK_RULE - 0.01)                                                    # region ('Marchia' -> Steiermark) is a suggestion
                        hits.append(H('orbis', sc, gd['lat'], gd['lon'], gd['radius'] if gd['radius'] >= 60 else None, None, c.lemma, None, es[0].modern))

        def prefer(clusters):
            """Two places carry the Latin name: the edition's and the document's places decide (margin ``TIEBREAK_CTX``), or a place the atlas mentions twenty times
            more often with no context against it."""
            ctx = [max(self.ctx_at(i, h.lat, h.lon) for h in cl_) for cl_ in clusters]
            wt = [max([h.weight for h in cl_] + [0.0]) for cl_ in clusters]
            o = np.argsort(ctx)[::-1]
            a_, b_ = int(o[0]), int(o[1])
            if ctx[a_] - ctx[b_] >= LG.TIEBREAK_CTX:
                return a_
            w = np.argsort(wt)[::-1]
            if wt[int(w[1])] > 0 and wt[int(w[0])] / wt[int(w[1])] >= LG.TIEBREAK_RATIO and ctx[int(w[0])] - ctx[int(w[1])] >= 0.1:
                return int(w[0])
            return None
        nk = LN.fold_key(name)
        boost = any(h.weight >= LATIN_PROMINENT and h.score >= LG.STRONG_RULE and LN.fold_key(h.lemma) != nk for h in hits if h.src in ('wikidata', 'atlas'))
        conf, reason, chosen = LG.combine(hits, lvl, prefer, boost)
        if conf is None:
            if reason != 'latin_no_match':
                self.why[i] = reason
            return None
        lat, lon, rad = chosen.lat, chosen.lon, chosen.radius
        if not self.in_region(i, lat, lon):
            self.why[i] = 'outside_region'
            return None
        # the edition has to be able to mean it: a Latin name that is a known homonym (Lugdunum = Lyon or Leiden) and, for one family alone, a place far from every place
        # the edition and the document talk about (Vallesia in Bullinger is Valais, not Wales), is a suggestion unless the place is one of the atlas's most mentioned
        fam_set = {LG.FAMILY.get(h.src, h.src) for h in hits}
        fam_n = len(fam_set)
        if conf in ('high', 'medium') and chosen.src != 'polity':
            cx = self.doc_ctx_at(i, lat, lon)
            cl = self.cloud(i)
            has_cloud = cl is not None and float(cl[2].sum()) >= 1.5            # the documents name other placed places, which can contradict the answer
            # v2.4: ... also for two families that are not a curated table (Nordovico: the atlas's Norwich and the Orbis agree, the letters are Dutch: Noordwijk)
            far = has_cloud and cx < LATIN_FAR_CTX and chosen.weight < LATIN_PROMINENT and (fam_n < 2 or 'table' not in fam_set)
            # v2.4: the Orbis Latinus alone (OCR-noisy, no coordinates of its own) places a name only with a regular nominative, no adjective ending and the support of the
            # documents' places: Coronia -> Landskrona, Becam -> Brakel, AEgea -> Vostitza, Romuleae -> Sevilla
            orbis_alone = fam_set == {'orbis'} and (chosen.score < LG.STRONG_RULE or cx < LATIN_HOMONYM_CTX or LN.fold_key(chosen.lemma).endswith(LATIN_ADJECTIVE_END))
            if (LN.fold_key(chosen.lemma) in LATIN_HOMONYMS and cx < LATIN_HOMONYM_CTX) or far or orbis_alone:
                conf = 'low'
                if orbis_alone:
                    self.why[i] = 'latin_orbis_alone'
        if conf in ('high', 'medium') and chosen.src != 'polity':
            # the surface itself names a placed place elsewhere ('Monaco' is the principality and the ablative of Monacum = Munich): the two readings disagree, a suggestion
            ks = [k for k in self.name_forms[i].keys(('t0',)) if k in self.R]
            if ks:
                rows = self.R[ks[0]]
                rows = rows[rows != i]
                if len(rows):
                    ag = self.agree(rows)
                    if ag['ok'] and float(hav(ag['lat'], ag['lon'], lat, lon)) > max(100.0, float(rad or 0.0)):
                        conf = 'low'
                        self.why[i] = 'latin_vs_vernacular'
        row = chosen.row
        label = chosen.label or chosen.lemma
        fams = sorted({LG.FAMILY.get(h.src, h.src) for h in hits})
        cand = f"{self.coll[row]}:{self.P['entity_key'].iat[row]}" if row is not None else ('exonym:' + label if chosen.src in ('exonym', 'polity') else 'latin:' + chosen.src)
        rule = next((c.rule for c in cands if c.lemma == chosen.lemma), '?')
        ev = (f"Latin '{name}' -> '{chosen.lemma}' (rule {rule}, score {chosen.score:.2f}, {lvl} context); sources: "
              + '; '.join(f"{h.src} {h.lemma}" for h in hits[:6]) + f"; the point is {chosen.src}" + ('; two places carry the name, the edition decides' if reason == 'ok_tiebreak' else ''))
        return dict(method=LG.METHOD, conf=conf, lat=float(lat), lon=float(lon), radius=float(rad) if rad else 10.0, p=float(chosen.score), cand=cand, cand_name=str(label),
                    tier='latin_' + '+'.join(fams) + ('+tiebreak' if reason == 'ok_tiebreak' else ''), key=LN.fold_key(chosen.lemma), evidence=ev, n_cands=len(hits),
                    homonym=False, qid=chosen.qid, diag={'sources': fams, 'rule_score': float(chosen.score)})

    def resolve(self, i):
        """-> dict(prop=..., gn=..., final=...) for place i."""
        cls = self.pclass[i]
        if cls in PCL.SKIP_CLASSES or self.no_geocode[i]:
            self.why[i] = f'class_{cls}' if cls else 'micro_toponym'
            return dict(prop=None, gn=None, final=None, via='skip')
        fam = LH.family_of(self.P['name'].iat[i])
        if fam is not None:                   # v2.4: Lugdunum = Lyon or Leiden ...: the passage and the document decide, else nothing
            lh = self.latin_homonym(i, fam)
            return dict(prop=None, gn=None, final=lh, via='latin_homonym')
        pol = self.polity(i)                  # v2.3: the reviewed polity table, then the exonym table on the head of a qualified name
        if pol is not None:
            return dict(prop=None, gn=None, final=pol, via='polity')
        ex = self.exonym(i)
        if ex is not None and ex['diag'].get('kind') == 'city' and not self.row_keep(i, [ex['lat']], [ex['lon']], [0], ['PPL'], hard=False)[0]:
            ex = None                       # v2.2: Lyon's Inn is not Lyon, Savoy Manor not Savoie: London and the file's region apply to the exonym table's cities
                                            # (not the collections' hard boxes: Rome is Rome in the Lüneburg nuns' letters)
            self.why[i] = 'outside_region'
        if ex is not None:
            return dict(prop=None, gn=None, final=ex, via='exonym')
        if PCL.is_ethnonym_surface(self.P['name'].iat[i]):
            self.why[i] = 'class_ethnonym'
            return dict(prop=None, gn=None, final=None, via='skip')
        if self.name_forms[i].admin:
            self.why[i] = 'administrative_area'
            return dict(prop=None, gn=None, final=None, via='skip')
        f = self.name_forms[i]
        pr = gr = None
        via = 'name'
        eq = None if f.junk else self.edition_qualified(i, f)
        if eq is not None:
            return dict(prop=eq, gn=None, final=eq, via='edition_qualified')
        if not f.junk:
            pr, gr = self.resolve_one(i, f)
        if (pr is None or pr.get('conf') is None) and (gr is None or gr.get('conf') is None):
            # junk name or nothing found: vote with the variants
            vp, vg = [], []
            for s, fv in self.var_forms[i]:
                p1, g1 = self.resolve_one(i, fv)
                if p1 and p1.get('conf') and p1['conf'] != 'low':
                    vp.append((s, p1))
                if g1 and g1.get('conf'):
                    vg.append((s, g1))
            pv = self.vote(i, vp)
            gv = self.vote(i, vg)
            if pv is not None or gv is not None:
                pr, gr, via = (pv or pr), (gv or gr), 'variants'
        final = self.combine(i, pr, gr)
        # v2.3: 'X, Grafschaft' / 'X, Herzogtum': the polity qualifier makes the entity an area, so a GeoNames commune or village of the head's name is not the answer
        if final is not None and final.get('method') == 'name_geonames' and self.qp[i].qtype in POL.AREA_QTYPES and final.get('conf') in ('high', 'medium'):
            fc = str((final.get('diag') or {}).get('fcode'))
            if not (fc in POLITY_HEAD_FCODES or fc.startswith('PCL')) or str(final.get('cand', '')).startswith('tgn:'):
                self.why[i] = 'polity_head_not_a_region'
                final = dict(final, conf='low', evidence=final['evidence'] + ' | a polity qualifier makes the entity an area: a settlement of that name is not the answer')
        # v2.3, path C: in a regional collection a GeoNames guess among several namesakes of the region is no placement
        if self.coll[i] in REGIONAL_UNIQUE and (final is None or final.get('method') in ('name_geonames', 'name_propagation')):
            rc = self.regional_candidates(i, f)
            if rc is not None and rc[0] >= 2 and final is not None and final.get('method') == 'name_geonames':
                self.why[i] = 'regional_ambiguous'
                final = dict(final, conf='low', evidence=final['evidence'] + f' | {rc[0]} settlements of that name in {rc[2]}')
            elif rc is not None and rc[0] == 1 and final is not None and final.get('conf') in ('high', 'medium', 'low'):
                # the one settlement of the region: a guess that lies elsewhere gives way to it
                ru = self.regional_unique(i, f)
                if ru is not None and float(hav(ru['lat'], ru['lon'], final['lat'], final['lon'])) > 10.0 and self._rank(ru) >= 2:
                    final = dict(ru, evidence=ru['evidence'] + f" | not the guess {final.get('cand_name')} ({float(hav(ru['lat'], ru['lon'], final['lat'], final['lon'])):.0f} km away)")
        # ... and a Casebooks place whose directory entry (a suggestion, matched 'low') lies far from the guess is not placed by the guess
        if final is not None and self.coll[i] == 'casebooks' and final.get('conf') in ('high', 'medium'):
            sg = self._cb_suggestion(i)
            if sg is not None:
                dsg = float(hav(sg[0], sg[1], final['lat'], final['lon']))
                if dsg > CASEBOOKS_CONFLICT_KM:
                    self.why[i] = 'casebooks_conflict'
                    final = dict(final, conf='low', evidence=final['evidence'] + f" | the Casebooks directory suggests a place {dsg:.0f} km away")
        # v2.3: the paths for what the generic lookup could not settle (nothing, or a suggestion only). A better answer replaces a worse one, never the reverse.
        if self._rank(final) < 2:
            for fn, name in ((self.polity_gn, 'polity_gn'), (self.head_alt, 'head_alt'), (lambda k: self.regional_unique(k, f), 'regional_unique'), (self.latin, 'latin')):
                try:
                    alt = fn(i)
                except Exception as e:                       # one odd name must not stop the run
                    alt, self.why[i] = None, f'error_{name}'
                    print(f'[geocode] {name} failed on {self.P["name"].iat[i]!r}: {e!r}', file=sys.stderr)
                if alt is not None and self._rank(alt) > self._rank(final):
                    final, via = alt, name
                    if self._rank(final) >= 2:
                        self.why.pop(i, None)
                        break
        return dict(prop=pr, gn=gr, final=final, via=via)

    def vote(self, i, lst):
        """Several variants of one entity (la Haye / Hagae Comitis / The Hague): they count as evidence only when they
        are not spellings of one name (same skeleton) and agree on the place. Never raises a confidence."""
        if not lst:
            return None
        pts = np.array([[r['lat'], r['lon']] for _, r in lst])
        rk = {'high': 3, 'medium': 2, 'low': 1}
        w = np.array([rk.get(r['conf'], 0) for _, r in lst], dtype=float)
        sk = [GT.skeleton(r['key']) for _, r in lst]
        best, bw, members = None, -1.0, None
        for a in range(len(lst)):
            d = hav(pts[a, 0], pts[a, 1], pts[:, 0], pts[:, 1])
            m = d <= TH['agree_km']
            wt = sum(max(w[j] for j in np.flatnonzero(m) if sk[j] == g) for g in {sk[j] for j in np.flatnonzero(m)})
            if wt > bw:
                best, bw, members = a, wt, m
        s, r = lst[best]
        r = dict(r)
        groups = {sk[j] for j in np.flatnonzero(members)}
        n = len(groups)
        if n >= 2 and not (len(lst) >= 3 and members.sum() < 0.6 * len(lst)):
            r['conf'] = 'high' if (n >= 3 and r['conf'] == 'high') else ('medium' if r['conf'] in ('high', 'medium') else 'low')
        else:
            r['conf'] = 'low'
        r['evidence'] = f"variant '{s}' ({n} distinct names of {len(set(sk))} agree); " + r['evidence']
        r['via'] = 'variants'
        return r

    def combine(self, i, pr, gr):
        """Two independent methods: agreement within the feature's radius corroborates, disagreement abstains."""
        rank = {'high': 3, 'medium': 2, 'low': 1}
        P = pr if pr is not None and pr.get('conf') else None
        Gd = gr if gr is not None and gr.get('conf') else None
        if P is None and Gd is None:
            return None
        if P is None:
            return Gd
        if Gd is None:
            if P['conf'] == 'high':
                return P
            best = dict(P)
            best['conf'] = 'low'
            return best
        d = float(hav(P['lat'], P['lon'], Gd['lat'], Gd['lon']))
        tol = max(TH['agree_km'], min(Gd['radius'], TH['tol_cap']))
        if d <= tol:
            sm = rank[P['conf']] + rank[Gd['conf']]
            best = dict(P if rank[P['conf']] >= rank[Gd['conf']] else Gd)
            other = Gd if best['method'] == 'name_propagation' else P
            best['conf'] = 'high' if (P['conf'] == 'high' or sm >= 5) else 'medium'
            best['evidence'] += f" | corroborated by {'GeoNames' if other is Gd else 'propagation'} ({d:.0f} km apart; {other['conf']}): {other['evidence']}"
            best['agree'] = True
            return best
        if P['conf'] == 'high':
            best = dict(P)
            best['conf'] = 'medium'
        else:
            best = dict(P if rank[P['conf']] >= rank[Gd['conf']] else Gd)
            best['conf'] = 'low'
        best['conflict_km'] = d
        best['evidence'] += f" | conflicts with {'GeoNames' if best['method'] == 'name_propagation' else 'propagation'} ({d:.0f} km)"
        return best


# ---------------------------------------------------------------------------------------------------------------------
def run_pipeline(G: Geocoder, hidden: np.ndarray, log=_t) -> dict:
    """Two passes. ``hidden``: row ids whose coordinates are masked (precision test). Returns {row: result}."""
    P = G.P
    lat_true = P['lat'].to_numpy('float64')
    lon_true = P['lon'].to_numpy('float64')
    lat0 = lat_true.copy()
    lon0 = lon_true.copy()
    lat0[hidden] = np.nan
    lon0[hidden] = np.nan
    live = P['n_mentions'].to_numpy() > 0                 # v2: n_mentions counts the atlas (non-excluded) mentions
    targets = np.flatnonzero(np.isnan(lat0) & live)
    G.set_anchors(lat0, lon0)
    wt = np.log1p(G.weight)
    out: dict = {}
    for ps in (1, 2):
        if ps == 1:
            a_lat, a_lon, a_w = lat0, lon0, wt
        else:
            a_lat, a_lon = lat0.copy(), lon0.copy()
            for i, r in out.items():
                f = r['final']
                if f and f['conf'] in ('high', 'medium'):
                    a_lat[i], a_lon[i] = f['lat'], f['lon']
            a_w = wt
        G.set_context(a_lat, a_lon, a_w)
        todo = targets if ps == 1 else np.array([i for i in targets if not (i in out and out[i]['final']
                                                                           and out[i]['final']['conf'] in ('high', 'medium'))])
        t0 = time.time()
        for n, i in enumerate(todo):
            try:
                r = G.resolve(int(i))
            except Exception as e:                                  # one odd name must not stop the run
                r = dict(prop=None, gn=None, final=None, via='error', error=repr(e))
            r['pass'] = ps
            if ps == 2 and i in out:
                old = out[i]
                if not (r['final'] and r['final']['conf'] in ('high', 'medium')) and old['final']:
                    r = old                                         # keep the pass-1 low-confidence result
            out[int(i)] = r
            if n and n % 10000 == 0:
                log(f'pass {ps}: {n}/{len(todo)} ({time.time() - t0:.0f}s)')
        got = sum(1 for i in todo if out[int(i)]['final'] and out[int(i)]['final']['conf'] in ('high', 'medium'))
        log(f'pass {ps}: {len(todo)} places, {got} high/medium ({time.time() - t0:.0f}s)')
    return out


def file_outliers(G: Geocoder, out: dict, log=_t) -> dict:
    """v2.2, per-file consistency: a name-based point (60 km radius or less) of a place that lies more than ``regions.FILE_OUTLIER_KM`` (100 km) from the
    median of a REGIONAL file (at least 8 placed places, 80 % of them within 150 km of their median) of a collection whose files are regional by nature
    (``regions.FILE_CONSISTENCY_COLLECTIONS``) is taken back to a low-confidence candidate when it is an outlier in every file of the place (a place that
    some other file supports is kept). Returns {row: reason}; ``out`` is modified
    (``final['conf']`` = 'low', ``r['dropped']`` = 'file_outlier'). Sample check of v2.1: 169 Hearth Tax places in another county than their file."""
    lat, lon = G.lat0.copy(), G.lon0.copy()
    for i, r in out.items():
        f = r['final']
        if f and f['conf'] in ('high', 'medium'):
            lat[i], lon[i] = f['lat'], f['lon']
    pr = G.pairs[['doc', 'row']]
    pr = pr[np.isin(G.coll[pr['row'].to_numpy()], list(RGN.FILE_CONSISTENCY_COLLECTIONS))]          # the collections whose files are regional by nature
    ok = ~np.isnan(lat[pr['row'].to_numpy()])
    med: dict = {}
    for d, rows in pr[ok].groupby('doc')['row']:
        rows = rows.to_numpy()
        if len(rows) < RGN.FILE_MIN_PLACED:
            continue
        m = RGN.is_regional(lat[rows], lon[rows])
        if m is not None:
            med[int(d)] = m
    if not med:
        return {}
    docs_of = pr[pr['doc'].isin(list(med))].groupby('row')['doc'].apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
    all_docs = pr.groupby('row')['doc'].nunique().to_dict()
    dropped: dict = {}
    for i, r in out.items():
        f = r['final']
        if not f or f['conf'] not in ('high', 'medium') or f['method'] == 'name_exonym' or float(f['radius']) > RGN.FILE_POINT_MAX_RADIUS_KM:
            continue
        ds = docs_of.get(int(i))
        if not ds:
            continue
        far = [float(hav(f['lat'], f['lon'], *med[d])) > RGN.FILE_OUTLIER_KM for d in ds]
        if all(far) and len(ds) == all_docs.get(int(i), len(ds)):          # an outlier in every file of the place, and no file without a verdict
            d0 = ds[0]
            f['evidence'] = (f.get('evidence') or '') + (f"; taken back: {float(hav(f['lat'], f['lon'], *med[d0])):.0f} km from the median of its regional file(s) "
                                                           f"(limit {RGN.FILE_OUTLIER_KM:.0f} km)")
            f['conf'] = 'low'
            r['dropped'] = 'file_outlier'
            G.why[int(i)] = 'file_outlier'
            dropped[int(i)] = 'file_outlier'
    log(f'file consistency: {len(med)} regional files, {len(dropped)} name-based points taken back')
    return dropped


def reason_of(G: Geocoder, i: int, r: dict) -> str:
    """Why a place stays unplaced (or only has a low-confidence candidate)."""
    if r.get('dropped'):
        return r['dropped']
    f = r.get('final')
    if r.get('via') == 'error':
        return 'error'
    if r.get('via') == 'skip' or (f is None and i in G.why):
        return G.why.get(i, 'skipped')
    if f and f['conf'] == 'low':
        return 'low_confidence'
    nf = G.name_forms[i]
    pr, gr = r.get('prop'), r.get('gn')
    if nf.junk and not G.var_forms[i]:
        return 'junk_or_generic_name'
    if pr is None and gr is None:
        return 'no_match_in_atlas_or_geonames'
    if gr is not None and gr.get('homonym'):
        return 'ambiguous_homonym'
    if pr is not None and pr.get('conf') is None and (gr is None or gr.get('conf') is None):
        return 'placed_candidates_disagree' if gr is None else 'ambiguous_low_score'
    if gr is not None and gr.get('conf') is None:
        return 'ambiguous_low_score'
    return 'other'


def unresolved_rows(G: Geocoder, out: dict) -> pd.DataFrame:
    P = G.P
    rows = []
    for i, r in out.items():
        f = r['final']
        if f and f['conf'] in ('high', 'medium'):
            continue
        rows.append(dict(coll_slug=P['coll_slug'].iat[i], entity_key=P['entity_key'].iat[i], gc_reason=reason_of(G, i, r)))
    return pd.DataFrame(rows)


def result_rows(G: Geocoder, out: dict) -> pd.DataFrame:
    P = G.P
    rows = []
    for i, r in out.items():
        f = r['final']
        if not f:
            continue
        rows.append(dict(
            coll_slug=P['coll_slug'].iat[i], entity_key=P['entity_key'].iat[i], name=P['name'].iat[i],
            gc_lat=f['lat'], gc_lon=f['lon'], gc_source=f['method'], gc_conf=f['conf'], gc_radius_km=float(f['radius']),
            gc_cand=f['cand'], gc_cand_name=f['cand_name'], gc_key=f['key'], gc_tier=f['tier'], gc_p=float(f['p']),
            gc_evidence=f['evidence'], gc_pass=r['pass'], gc_via=f.get('via', r.get('via')),
            gc_agree=bool(f.get('agree', False)), gc_conflict_km=f.get('conflict_km'),
            gc_prop_conf=(r['prop'] or {}).get('conf'), gc_gn_conf=(r['gn'] or {}).get('conf'),
            gc_tgn=bool(f.get('tgn', False) and f.get('method') == 'name_geonames'), gc_qid=f.get('qid')))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# precision self-check
# ---------------------------------------------------------------------------------------------------------------------
def sample_holdout(P: pd.DataFrame, n: int, seed: int, exclude: Optional[set] = None, per_coll_cap: int = 140) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ok = (P['lat'].notna() & (P['n_in_period'] > 0)).to_numpy()
    cand = np.flatnonzero(ok)
    if exclude:
        cand = np.array([c for c in cand if c not in exclude])
    cand = rng.permutation(cand)
    take, cnt = [], Counter()
    for c in cand:
        k = P['coll_slug'].iat[c]
        if cnt[k] < per_coll_cap:
            take.append(c)
            cnt[k] += 1
        if len(take) >= n:
            break
    return np.array(sorted(take))


def evaluate(G: Geocoder, out: dict, hidden: np.ndarray) -> tuple:
    P = G.P
    recs = []
    for i in hidden:
        r = out.get(int(i))
        if r is None:
            continue
        tl, tn = P['lat'].iat[i], P['lon'].iat[i]
        base = dict(row=int(i), coll_slug=P['coll_slug'].iat[i], family=P['family'].iat[i], name=P['name'].iat[i],
                    n_in_period=int(P['n_in_period'].iat[i]), truth_src=P['src0'].iat[i], lat=tl, lon=tn)
        for tag, x in (('prop', r['prop']), ('gn', r['gn']), ('final', r['final'])):
            if x and x.get('conf'):
                d = float(hav(tl, tn, x['lat'], x['lon']))
                base[f'{tag}_conf'] = x['conf']
                base[f'{tag}_km'] = d
                base[f'{tag}_lat'], base[f'{tag}_lon'] = x['lat'], x['lon']
                base[f'{tag}_rad'] = x.get('radius')
                base[f'{tag}_method'] = x.get('method')
                base[f'{tag}_homonym'] = x.get('homonym')
                for k, v in (x.get('diag') or {}).items():
                    base[f'{tag}_d_{k}'] = v
                base[f'{tag}_tier'] = x.get('tier')
                base[f'{tag}_key'] = x.get('key')
            else:
                base[f'{tag}_conf'] = None
                base[f'{tag}_km'] = np.nan
        recs.append(base)
    df = pd.DataFrame(recs)
    return df, summarise(df)


def _share(d, col, lim):
    d = d[d[col].notna()]
    return (round(float((d[col] <= lim).mean()), 4), int(len(d))) if len(d) else (None, 0)


def summarise(df: pd.DataFrame) -> dict:
    res: dict = {'n_hidden': int(len(df))}
    for tag in ('prop', 'gn', 'final'):
        blk: dict = {}
        c = df[f'{tag}_conf']
        km = df[f'{tag}_km']
        blk['answered_any'] = int(c.notna().sum())
        for lev in ('high', 'medium', 'low', 'high+medium'):
            m = c.isin(['high', 'medium']) if lev == 'high+medium' else (c == lev)
            d = df[m]
            blk[lev] = {'n': int(m.sum()), 'coverage': round(float(m.mean()), 4),
                        'within10': _share(d, f'{tag}_km', 10)[0], 'within50': _share(d, f'{tag}_km', 50)[0],
                        'within200': _share(d, f'{tag}_km', 200)[0],
                        'within_max50_or_radius': (round(float((d[f'{tag}_km'] <= np.maximum(50.0, d[f'{tag}_rad'].fillna(0).clip(upper=400))).mean()), 4)
                                                   if len(d) else None),
                        'median_km': round(float(d[f'{tag}_km'].median()), 1) if len(d) else None}
            w = d['n_in_period']
            if len(d):
                blk[lev]['within50_weighted_by_mentions'] = round(float(np.average(d[f'{tag}_km'] <= 50, weights=w)), 4)
        res[tag] = blk
    # per family and per collection, final, high+medium
    fam = {}
    for k, d in df[df['final_conf'].isin(['high', 'medium'])].groupby('family'):
        tot = int((df['family'] == k).sum())
        fam[k] = {'hidden': tot, 'answered_hm': int(len(d)), 'within10': _share(d, 'final_km', 10)[0],
                  'within50': _share(d, 'final_km', 50)[0]}
    res['by_family_final_high_medium'] = fam
    famm = {}
    for tag in ('prop', 'gn'):
        for k, d in df[df[f'{tag}_conf'].isin(['high', 'medium'])].groupby('family'):
            famm.setdefault(k, {})[tag] = {'n': int(len(d)), 'within10': _share(d, f'{tag}_km', 10)[0],
                                          'within50': _share(d, f'{tag}_km', 50)[0]}
    res['by_family_method_high_medium'] = famm
    col = {}
    for k, d in df[df['final_conf'].isin(['high', 'medium'])].groupby('coll_slug'):
        if len(d) >= 8:
            col[k] = {'n': int(len(d)), 'within10': _share(d, 'final_km', 10)[0], 'within50': _share(d, 'final_km', 50)[0]}
    res['by_collection_final_high_medium'] = col
    return res


# ---------------------------------------------------------------------------------------------------------------------
# precision self-check of the Latin path (v2.3)
# ---------------------------------------------------------------------------------------------------------------------
def evaluate_latin(G: 'Geocoder', n: int, seed: int, per_coll_cap: int = 250) -> tuple:
    """Hide ``n`` placed places of Latin / mixed context (their coordinates and their Wikidata items vouch for nothing: the anchors and the Wikidata points are built from the
    others), run ONLY the Latin path on them and compare with the truth. Returns (DataFrame, summary). ``surface_differs``: the name is not its own nominative (an
    inflected form), the case the path is for."""
    P = G.P
    rng = np.random.default_rng(seed)
    lv = np.array(G.latin_level)
    ok = (P['lat'].notna() & (P['n_in_period'] > 0)).to_numpy() & np.isin(lv, ['latin', 'mixed'])
    cand = rng.permutation(np.flatnonzero(ok))
    take, cnt = [], Counter()
    for c in cand:
        k = P['coll_slug'].iat[c]
        if cnt[k] < per_coll_cap:
            take.append(int(c))
            cnt[k] += 1
        if len(take) >= n:
            break
    hidden = np.array(sorted(take), dtype=int)
    lat0, lon0 = P['lat'].to_numpy('float64').copy(), P['lon'].to_numpy('float64').copy()
    lat0[hidden] = np.nan
    lon0[hidden] = np.nan
    G.set_anchors(lat0, lon0)
    G.set_context(lat0, lon0, np.log1p(G.weight))
    recs = []
    for i in hidden:
        r = G.latin(int(i))
        tl, tn = P['lat'].iat[i], P['lon'].iat[i]
        base = dict(row=int(i), coll_slug=P['coll_slug'].iat[i], name=P['name'].iat[i], level=G.latin_level[i], n_in_period=int(P['n_in_period'].iat[i]))
        if r is None:
            base.update(conf=None, km=np.nan, reason=G.why.get(int(i)))
        else:
            base.update(conf=r['conf'], km=float(hav(tl, tn, r['lat'], r['lon'])), radius=r['radius'], tier=r['tier'], key=r['key'], cand_name=r['cand_name'],
                        evidence=r['evidence'], surface_differs=bool(LN.fold_key(P['name'].iat[i]) != r['key']))
        recs.append(base)
    df = pd.DataFrame(recs)
    summ: dict = {'n_hidden': int(len(df)), 'answered': int(df['conf'].notna().sum())}
    for lev in ('high', 'medium', 'low', 'high+medium'):
        m = df['conf'].isin(['high', 'medium']) if lev == 'high+medium' else (df['conf'] == lev)
        d = df[m]
        tol = np.maximum(50.0, d['radius'].fillna(0).clip(upper=400)) if len(d) else d['km']
        summ[lev] = {'n': int(m.sum()), 'within10': round(float((d['km'] <= 10).mean()), 4) if len(d) else None,
                     'within50': round(float((d['km'] <= 50).mean()), 4) if len(d) else None,
                     'within_max50_or_radius': round(float((d['km'] <= tol).mean()), 4) if len(d) else None,
                     'inflected_only': int((d['surface_differs'] == True).sum()) if len(d) else 0}          # noqa: E712
    if 'tier' in df:
        summ['by_tier_high_medium'] = {str(t): {'n': int(len(g)), 'within50': round(float((g['km'] <= 50).mean()), 4)}
                                       for t, g in df[df['conf'].isin(['high', 'medium'])].groupby('tier')}
    return df, summ


def load_resources(a) -> dict:
    """The Latin path's lexicons (v2.3), when the options name them: Orbis Latinus and the Wikidata Latin labels of the place items."""
    res: dict = {}
    if getattr(a, 'orbis', None):
        res['orbis'] = LG.OrbisLatinus.load(a.orbis)
        _t(f'Orbis Latinus: {len(res["orbis"])} Latin keys')
    if getattr(a, 'wd_entities', None):
        res['wdlatin'] = LG.WikidataLatin.load(a.wd_entities)
        _t(f'Wikidata Latin labels: {len(res["wdlatin"])} keys')
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile geocode')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('index')
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tgn-dir', type=Path, default=None, help='local Getty TGN index: its names join the name dictionary (v2.1)')
    p.add_argument('--orbis', type=Path, default=None, help='data/orbis_latinus.tsv: its modern names join the keys the index answers (v2.3, Latin path)')
    p = sub.add_parser('run')
    p.add_argument('--geo', type=Path, required=True)
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--mentions', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--eval-n', type=int, default=0)
    p.add_argument('--eval-seeds', default='1,2', help='comma list: first seed = tuning sample, last = reported sample')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--set', action='append', default=[], help='TH override, key=value')
    p.add_argument('--orbis', type=Path, default=None, help='data/orbis_latinus.tsv (v2.3, Latin path)')
    p.add_argument('--wd-entities', type=Path, default=None, dest='wd_entities', help='wd_entities.parquet: Latin labels and aliases of the atlas\'s place items (v2.3)')
    p.add_argument('--eval-latin', type=int, default=0, help='hold-out of the Latin path: N hidden places of Latin context (v2.3)')
    a = ap.parse_args(argv)
    if a.cmd == 'index':
        extra = LG.OrbisLatinus.load(a.orbis).modern_names() if a.orbis else None
        build_index(a.local, a.zip, a.out, a.tgn_dir, extra)
        return 0
    for kv in a.set:
        k, v = kv.split('=')
        TH[k] = float(v)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = load_places(a.geo, a.local)
    _t(f'{len(P)} places, {int(P["lat"].notna().sum())} placed')
    pairs, hint, doc_regions = load_pairs(a.mentions, P)
    gn = GN(pd.read_parquet(a.index))
    _t(f'GeoNames index {len(gn.d)} rows ({time.time() - t0:.0f}s)')
    res = load_resources(a)
    res['mention_aux'] = load_mention_aux(a.mentions, P)
    G = Geocoder(P, gn, pairs, hint, doc_regions, res)
    _t(f'forms ready ({time.time() - t0:.0f}s)')
    if a.eval_latin and G.latin_on:
        for seed in (1, 2):
            df, summ = evaluate_latin(G, a.eval_latin, seed)
            df.to_csv(a.out / f'holdout_latin_seed{seed}.tsv', sep='\t', index=False)
            (a.out / f'holdout_latin_seed{seed}.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), 'utf-8')
            _t(f'Latin hold-out seed {seed}: high+medium {summ["high+medium"]}; low {summ["low"]["n"]}')
    if a.eval_n:
        used: set = set()
        for seed in [int(s) for s in a.eval_seeds.split(',')]:
            hid = sample_holdout(P, a.eval_n, seed, used)
            used |= set(hid.tolist())
            out = run_pipeline(G, hid)
            df, summ = evaluate(G, out, hid)
            df.to_csv(a.out / f'holdout_seed{seed}.tsv', sep='\t', index=False)
            (a.out / f'holdout_seed{seed}.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), 'utf-8')
            _t(f'holdout seed {seed}: final high+medium {summ["final"]["high+medium"]}')
    if a.eval_only:
        return 0
    out = run_pipeline(G, np.array([], dtype=int))
    file_outliers(G, out)
    res = result_rows(G, out)
    ur = unresolved_rows(G, out)
    res = res.merge(ur, how='outer', on=['coll_slug', 'entity_key'])       # reasons for the unresolved and the low ones
    res.to_parquet(a.out / 'geocode_places.parquet', index=False)
    _t(f'{len(res)} results: {res["gc_conf"].value_counts().to_dict()} ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
