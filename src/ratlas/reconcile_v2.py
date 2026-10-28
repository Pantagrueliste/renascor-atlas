"""Reconcile stage v2: gates, arbitration by names, kinds, globe (run on Hypatia through Slurm).

    python -m ratlas reconcile build2 --v2 DIR --rdir DIR --dest DIR [--namecheck FILE] [--geocode FILE] [--geocode-verify FILE]
                                      [--tgn-dir DIR] [--casebooks-map FILE]

Differences from ``reconcile build`` (v1.1):

* **Counts are the atlas counts**: excluded mentions (commentary, footnotes, regests, repositories, characters, placeholders,
  celestial bodies ...) are left out of ``n_mentions``, ``n_in_period``, the quarters and the globe. ``n_mentions_all`` and
  ``n_in_period_all`` keep the canonical total.
* **Gates** (``reconcile_gates``): an authority id is used only if its class is right and its names agree with the entry's (or a second
  independent id agrees); the rest is ``suggested`` (kept in ``ids_suggested`` / ``qid_suggested``, never used) or ``rejected``.
  Every non-accepted pair is in ``gate_log.tsv`` with its reason.
* **Arbitration by names**: a register-versus-authority conflict (> 50 km) is decided by the distance to the nearest GeoNames
  feature of the same name (``reconcile namecheck``) AND by the name gate. A name-compatible authority (exact / fold / stem
  match) keeps its QID and ids even when its coordinate is not used (rivers, countries and regions that distance alone had
  rejected: 13 cases); ties with no same-name feature and a conflict over 100 km go to the authority; conflicts over 100 km are
  flagged (``flag_100km``, ``coord_conflicts.tsv``).
* **Areas**: the register coordinates of countries (note_type country / coun, CKCC's regions) are the capital's: dropped;
  an area takes the GeoNames feature point of an accepted id, else the Wikidata point flagged ``coord_kind = area``, else none.
* **Kind** (``place_kind`` city / area / other) from the Wikidata classes of the accepted item, the GeoNames feature code and the
  register type; ``place_class`` (celestial, ethnonym, common noun ...) from ``placeclass``.
* ``globe_places.json`` gets ``k`` (kind) and ``cq`` (quarter counts per collection).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import exonyms as EX
from . import generic_terms as GEN
from . import manual_overrides as MO
from . import namegate as NG
from . import placekind as PK
from . import placeclass as PCL
from . import polities as POL
from . import reconcile_build as B
from . import reconcile_gates as RG
from . import tgn as TGN
from .merge import QCOLS, ROLE_COLS, _norm, _year_bin
from .reconcile_build import (CONFLICT_KM, GC_COLS, NAME_SOURCES, PERSON_FALLBACK, PLACE_FALLBACK, SRC_CODE, TIER_RANK, _blank,
                              _js, _score, _write_global, assign_clusters, dominant, edition_id, haversine_km, load_namecheck,
                              load_overrides, merge_points, valid_coord)

COUNT_COLS = ['n_mentions', 'n_in_period', 'n_undated', 'pre1450', 'post1700', 'undated_year'] + QCOLS + \
    [f'n_{r}' for r in ROLE_COLS] + ['n_other', 'n_generic_with_id']
SUM_COLS_V2 = COUNT_COLS + ['n_mentions_all', 'n_in_period_all']
AREA_DROP_TYPES = {'country', 'coun'}                       # register points of countries are the capital's
AREA_DROP_COLLS = {'ckcc_epistolarium'}                     # ... and its regions (Scotland on Edinburgh)
AREA_NOTE = {'country', 'coun', 'region', 'dist', 'district', 'province', 'landscape', 'kingdom', 'duchy', 'place_county',
             'place_ea_diocese'}
CITY_NOTE = {'city', 'stadt', 'vill', 'dorf', 'gemeinde', 'place_pa_settlement', 'gemeindeteil', 'settlement', 'town', 'village'}
OTHER_NOTE = {'moun', 'rive', 'stre', 'weg', 'acker', 'flurname', 'burg', 'mühle', 'hof', 'gut', 'place_f_church', 'place_f_street',
              'place_f_residence', 'place_f_openarea', 'place_f_school', 'place_f_property', 'place_f_victuallinghouse',
              'place_f_religioushouse', 'place_f_guildhall', 'place_pa_ward'}
AREA_RADIUS = {'country': 350.0, 'region': 150.0}
# v2.1: a place can also be placed by the Casebooks directory (the Forman / Napier patients' parishes; curated map PLACE id -> directory entry)
DIRECT_SOURCES = ('casebooks_directory',)
# v2.5: 'curated' = a point set by hand after the homonym audit (ratlas.curation): ours, not the editors', so it is a free source too
FREE_SOURCES = NAME_SOURCES + DIRECT_SOURCES + ('curated',)        # coordinates that no register or authority gives: lowest priority, never override one
CB_COLS = ['cb_place_id', 'cb_entry', 'cb_county', 'cb_parish', 'cb_match', 'cb_geo_conf', 'cb_geo_method', 'cb_lat', 'cb_lon', 'cb_radius_km',
           'cb_used', 'cb_gc_km']
CB_NAME_CONFLICT_KM = 25.0                          # a name-based point this far from the matched directory entry is not kept when the entry's own point is not used
CB_AREA_KM = 20.0                                   # a directory point with a radius of 20 km or more (county medians ...) is an area
STRONG = ('exact', 'fold', 'stem', 'manual_review')      # a decision confirmed by hand counts as a strong name match


def _t(msg):
    print(f'[build2 {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------------------------------------------------
_COUNTRY_LIKE = re.compile(r'\b(country|sovereign state|kingdom|empire|duchy|principality|republic|dominion|colony|protectorate|'
                           r'confederation|historical country|former country|grand duchy|electorate|margraviate|landgraviate|'
                           r'archduchy|bishopric|archbishopric|prince-bishopric|imperial abbey|free imperial city|viceroyalty|satrapy|sultanate|emirate|khanate|caliphate)\b', re.I)
_CITY_LIKE = re.compile(r'\b(city|town|village|municipality|commune|settlement|hamlet|borough|locality|parish|suburb|quarter|ward|'
                        r'capital|market|human settlement|urban|civil parish|township|metropolis|seat)\b', re.I)
_AREA_LIKE = re.compile(r'\b(region|province|state|county|canton|department|prefecture|territory|landscape|continent|sea|ocean|gulf|'
                        r'bay|strait|archipelago|shire|hundred|administrative|division|district|'
                        r'former (?:country|state|kingdom|territory|province|region)|historical (?:region|area|territory|country|state)|geographic(?:al)? region|'
                        r'cultural (?:region|area)|lordship|county of|mountain range|peninsula|unit)\b', re.I)


# an administrative state / region of a country (not a polity of its own): Bavaria is a "federated state of Germany" and a "ward"
_STATE_LIKE = re.compile(r'\b(federated state|free state|constituent state|state of [a-z]+|voivodeship|autonomous (community|region|province)|'
                         r'chartered community|province of [a-z]+|governorate|oblast|krai|department of [a-z]+|region of [a-z]+|'
                         r'county of [a-z]+|canton of [a-z]+|land of [a-z]+|prefecture of [a-z]+)\b', re.I)
_CITYSTATE = re.compile(r'\b(free imperial city|imperial city|city[- ]state|free city|hanseatic city|prince-bishopric city)\b', re.I)
_CITY_STRONG = re.compile(r'\b(city|big city|town|village|municipality|commune|human settlement|hamlet|borough|urban)\b', re.I)


# labels that say nothing about the kind of a place (Wikidata gives them to towns whose class was never refined: Ballenstedt is a "geographic location")
_GENERIC_CLASS = frozenset(['geographic location', 'geographical location', 'geographic object', 'geographical object', 'geographic entity',
                            'geographical feature', 'geographic feature', 'location', 'place', 'object', 'entity', 'toponym', 'territorial entity',
                            'physical object', 'topographical object', 'tourist destination', 'wikimedia disambiguation page'])
_OF = re.compile(r'\s+(?:of|in|within|on|at|from)\s+.*$', re.I)


def class_head(label) -> str:
    """'municipality with town privileges in the Czech Republic' -> 'municipality with town privileges'; 'city of United Kingdom' -> 'city'.

    The v2.1 patterns searched the whole label and took the country in it for the kind of the item: every town 'of the Czech Republic' or 'of the United
    Kingdom' (Prague, Plzen, Cambridge, Canterbury: 275 places, 7,602 mentions) was a "republic" or a "kingdom", hence an area."""
    return _OF.sub('', str(label or '')).strip()


def _singular(text: str) -> str:
    """The same words in the singular ('regions' -> 'region', 'duchies' -> 'duchy', 'counties' -> 'county')."""
    out = []
    for w in str(text).split():
        if len(w) > 4 and w.endswith('ies'):
            w = w[:-3] + 'y'
        elif len(w) > 4 and w.endswith(('ses', 'xes', 'ches', 'shes')):
            w = w[:-2]
        elif len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us', 'is')):
            w = w[:-1]
        out.append(w)
    return ' '.join(out)


def kind_of_classes(labels) -> Optional[str]:
    """city | area | other from the English labels of the P31 classes (None if there are none, or only generic ones).

    An item that is a state / region of a country is an area, unless it is also a city (Hamburg, Berlin: a point). ``ward`` or ``seat`` alone
    do not make a federal state a city (v2 audit, check 5: Bavaria was a point). v2.2: the patterns read the head of each label (before ``of`` / ``in``:
    a town of the Czech Republic is not a republic) and labels that say nothing (``geographic location``) are ignored."""
    labels = [l for l in labels if l and str(l).strip().lower() not in _GENERIC_CLASS]
    if not labels:
        return None
    s = ' | '.join(labels)                                  # the whole labels: 'state of Germany', 'county of ...' need their complement
    h = ' | '.join(class_head(l) for l in labels)           # the heads: 'town in the Czech Republic' is a town
    h = h + ' | ' + _singular(h)                            # 'Ukrainian historical regions', 'Duchies of Silesia'
    if _CITYSTATE.search(h) and _CITY_STRONG.search(h):
        return 'city'                  # Nuremberg (free imperial city), Hamburg, Bremen: a city with a political status, drawn as a point
    if _COUNTRY_LIKE.search(h):
        return 'area'
    if _STATE_LIKE.search(s):
        return 'city' if _CITY_STRONG.search(h) else 'area'
    if _CITY_LIKE.search(h):
        return 'city'
    if _AREA_LIKE.search(h):
        return 'area'
    # 'other' means a natural feature or a built one (river, castle, bridge ...); labels that say neither ('municipal part', 'cadastral area') say nothing
    # and leave the kind to the GeoNames feature, the TGN type and the register (Mährisch Weißwasser is a PPL, not an 'other')
    if any(PK.head_kind(l) in ('n', 'b') for l in labels):
        return 'other'
    return None


def polity_classes(labels) -> bool:
    """True when the head of one of the classes is a country-like polity (kingdom, duchy, principality, republic ...)."""
    return any(_COUNTRY_LIKE.search(class_head(l)) for l in labels if l)


def kind_of_fcode(fcode) -> Optional[str]:
    f = fcode if isinstance(fcode, str) else ''
    if not f or f in ('None', 'nan'):
        return None
    if f.startswith('PCL') or f in ('ADM1', 'ADM2', 'ADM1H', 'ADM2H', 'ADMD', 'RGN', 'AREA', 'CONT', 'TERR', 'SEA', 'OCN', 'GULF',
                                    'BAY', 'STRT', 'PEN', 'ISLS', 'MTS', 'RGNH', 'LCTY', 'DSRT'):
        return 'area'
    if f.startswith('PPL') or f in ('ADM3', 'ADM4', 'ADM5', 'ADM3H', 'ADM4H', 'STLMT', 'PRSH'):
        return 'city'
    return 'other'


def kind_of_regtype(nt) -> Optional[str]:
    t = str(nt or '').lower()
    if t in AREA_NOTE:
        return 'area'
    if t in CITY_NOTE:
        return 'city'
    if t in OTHER_NOTE:
        return 'other'
    return None


# ---------------------------------------------------------------------------------------------------------------------
# default counts
# ---------------------------------------------------------------------------------------------------------------------
def default_counts(v2: Path, kinds=('person', 'place')) -> pd.DataFrame:
    """Per (coll_slug, kind, entity_key): the counts of the canonical, non-excluded mentions (the atlas counts)."""
    cols = ['coll_slug', 'kind', 'canonical', 'excluded', 'local_key', 'surface', 'in_period', 'doc_year', 'doc_year_to', 'role']
    if 'generic_with_id' in pq.ParquetFile(v2 / 'mentions.parquet').schema_arrow.names:         # v2.2
        cols.append('generic_with_id')
    t = pq.read_table(v2 / 'mentions.parquet', columns=cols)
    t = t.filter(pc.and_(pc.equal(t.column('canonical'), True), pc.is_in(t.column('kind'), pa.array(list(kinds)))))
    d = t.to_pandas()
    d = d[~d['excluded'].fillna(False).astype(bool)]
    surf = d['surface'].fillna('')
    nm = {s: _norm(s) for s in surf.unique()}
    n = surf.map(nm)
    d['entity_key'] = np.where(d['local_key'].notna(), d['local_key'], 'surf:' + n)
    d = d[(d['local_key'].notna() | (n != '')).to_numpy()]
    d['_ip'] = (d['in_period'] == True).astype('int64')      # noqa: E712
    d['_un'] = d['in_period'].isna().astype('int64')
    d['_q'] = _year_bin(d['doc_year'])
    d['_r'] = d['role'].where(d['role'].isin(ROLE_COLS), 'other')
    key = ['coll_slug', 'kind', 'entity_key']
    g = d.groupby(key, sort=False)
    out = g.size().rename('n_mentions').to_frame()
    # the latest year the entity can be cited in: the end of the date range (a wide range has no doc_year), default mentions only. The chronology
    # gate (born after the last document) was blind to the persons of wide-range texts: DISCO's sonnets, dated 1501-1700 from the author's lifespan.
    d['_y'] = d['doc_year_to'].fillna(d['doc_year'])
    out['last_year'] = g['_y'].max()
    out['n_in_period'] = g['_ip'].sum()
    out['n_undated'] = g['_un'].sum()
    # v2.2: default mentions whose surface is a common noun that carries an authority id (kept and linked, but the noun is not the entity's name)
    d['_gid'] = d['generic_with_id'].fillna(False).astype('int64') if 'generic_with_id' in d else 0
    out['n_generic_with_id'] = d.groupby(key, sort=False)['_gid'].sum()
    ct = pd.crosstab([d['coll_slug'], d['kind'], d['entity_key']], d['_q'])
    for c in ['pre1450'] + QCOLS + ['post1700', 'undated_year']:
        out[c] = ct[c].reindex(out.index).fillna(0).astype('int64') if c in ct else 0
    cr = pd.crosstab([d['coll_slug'], d['kind'], d['entity_key']], d['_r'])
    for r in ROLE_COLS + ('other',):
        out[f'n_{r}'] = cr[r].reindex(out.index).fillna(0).astype('int64') if r in cr else 0
    return out.reset_index()


def excluded_report(v2: Path) -> dict:
    """Canonical mentions by exclusion reason (a mention has the reasons joined by ';'; the first one is counted) and kind, all and in period."""
    t = pq.read_table(v2 / 'mentions.parquet', columns=['canonical', 'excluded', 'excluded_reason', 'kind', 'in_period'])
    d = t.to_pandas()
    d = d[d['canonical'].fillna(True).astype(bool)]
    exc = d['excluded'].fillna(False).astype(bool)
    first = d['excluded_reason'].fillna('').str.split(';').str[0].where(exc, '')
    ip = (d['in_period'] == True)         # noqa: E712
    out = {'mentions_canonical': int(len(d)), 'excluded': int(exc.sum()), 'excluded_in_period': int((exc & ip).sum()),
           'default': int((~exc).sum()), 'default_in_period': int((~exc & ip).sum()),
           'by_first_reason': {k: int(v) for k, v in first[exc].value_counts().items()},
           'by_first_reason_in_period': {k: int(v) for k, v in first[exc & ip].value_counts().items()},
           'by_first_reason_and_kind': {f'{r}:{k}': int(n) for (r, k), n in d[exc].assign(_r=first[exc]).groupby(['_r', 'kind']).size().items()}}
    return out


def apply_default_counts(loc: pd.DataFrame, dc: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Replace the count columns of a local table by the default counts (``_all`` keeps the old totals)."""
    loc = loc.copy()
    loc['n_mentions_all'] = loc['n_mentions']
    loc['n_in_period_all'] = loc['n_in_period']
    drop = [c for c in COUNT_COLS + ['last_year'] if c in loc.columns]
    loc = loc.drop(columns=drop)
    d = dc[dc['kind'] == kind].drop(columns=['kind'])
    loc = loc.merge(d, how='left', on=['coll_slug', 'entity_key'])
    for c in COUNT_COLS:
        loc[c] = loc[c].fillna(0).astype('int64')
    return loc


# ---------------------------------------------------------------------------------------------------------------------
# accepted ids -> wide table
# ---------------------------------------------------------------------------------------------------------------------
def wide_ids(ids: pd.DataFrame, kind: str, schemes) -> pd.DataFrame:
    v = ids[(ids['kind'] == kind) & ids['scheme'].isin(schemes)]
    v = v.sort_values(['coll_slug', 'entity_key', 'scheme', 'n_mentions', 'id'], ascending=[True, True, True, False, True])
    v = v.drop_duplicates(['coll_slug', 'entity_key', 'scheme'])
    w = v.pivot(index=['coll_slug', 'entity_key'], columns='scheme', values='id')
    w.columns = ['id_' + c for c in w.columns]
    for s in schemes:
        if 'id_' + s not in w:
            w['id_' + s] = None
    return w.reset_index()


def suggested_ids(pairs: pd.DataFrame, kind: str) -> pd.DataFrame:
    s = pairs[(pairs['kind'] == kind) & pairs['status'].isin(['suggested', 'rejected'])]
    s = s.drop_duplicates(['coll_slug', 'entity_key', 'scheme', 'id'])
    if s.empty:
        return pd.DataFrame(columns=['coll_slug', 'entity_key', 'ids_suggested', 'n_ids_suggested'])
    s = s.assign(t=s['scheme'] + ':' + s['id'] + '[' + s['status'].str[:3] + ':' + s['reason'].astype(str) + ']')
    g = s.groupby(['coll_slug', 'entity_key'])['t'].agg(lambda x: ' ; '.join(list(x)[:6]))
    n = s.groupby(['coll_slug', 'entity_key']).size()
    return pd.DataFrame({'ids_suggested': g, 'n_ids_suggested': n}).reset_index()


def best_reason(pairs: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per entity: the strongest name relation of its accepted pairs (exact > fold > stem > ...)."""
    rank = {'manual_review': 0, 'exact': 0, 'fold': 1, 'stem': 2, 'ok': 3, 'ok_weak': 4, 'substring': 5, 'token': 6, 'title': 7, 'second_id': 9}
    a = pairs[(pairs['kind'] == kind) & (pairs['status'] == 'accepted')].copy()
    if a.empty:
        return pd.DataFrame(columns=['coll_slug', 'entity_key', 'auth_name_level'])
    a['r'] = a['reason'].map(lambda x: rank.get(x, 8))
    a = a.sort_values(['coll_slug', 'entity_key', 'r']).drop_duplicates(['coll_slug', 'entity_key'])
    return a[['coll_slug', 'entity_key', 'reason']].rename(columns={'reason': 'auth_name_level'})


# ---------------------------------------------------------------------------------------------------------------------
# arbitration by names
# ---------------------------------------------------------------------------------------------------------------------
def arbitrate_v2(pl: pd.DataFrame, nc: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Register-versus-authority coordinate conflicts (> 50 km). The coordinate with a same-name GeoNames feature nearby wins;
    a name-compatible authority keeps its identity (QID, ids) even when its coordinate loses; a tie with no same-name feature at
    all and a conflict over 100 km goes to the authority when the names agree strongly. Without a namecheck file nothing is
    decided (the register wins, the conflict is flagged)."""
    pl['namecheck_reg_km'] = np.nan
    pl['namecheck_auth_km'] = np.nan
    pl['arbitration'] = None
    pl['auth_rejected'] = False             # the authority's coordinate is not used
    pl['identity_dropped'] = False          # ... and its identity (QID, ids) is dropped too
    pl['reg_rejected'] = False
    strong = pl['auth_name_level'].isin(STRONG).to_numpy()
    conf = pl['coord_conflict'].to_numpy()
    if nc is None or not len(nc):
        pl['arbitration'] = np.where(conf, 'none_no_namecheck', None)
        return pl
    key = pd.MultiIndex.from_arrays([pl['coll_slug'], pl['entity_key']])
    d = nc.reindex(key)
    pl['namecheck_reg_km'] = d['d_reg'].to_numpy()
    pl['namecheck_auth_km'] = d['d_auth'].to_numpy()
    sr, sa = _score(pl['namecheck_reg_km']), _score(pl['namecheck_auth_km'])
    km = pl['coord_conflict_km'].to_numpy()
    rej_auth = conf & (sr > sa)
    rej_reg = conf & (sa > sr)
    tie = conf & (sr == sa)
    tie_far = tie & (sr == 1) & (km > 100) & strong        # no same-name feature for either point, names agree
    rej_reg = rej_reg | tie_far
    pl.loc[rej_auth, 'arbitration'] = 'register'
    pl.loc[rej_reg & ~tie_far, 'arbitration'] = 'authority'
    pl.loc[tie_far, 'arbitration'] = 'authority_tie_100km'
    pl.loc[tie & ~tie_far, 'arbitration'] = 'tie_register'
    pl['auth_rejected'] = rej_auth
    pl['reg_rejected'] = rej_reg
    pl['identity_dropped'] = rej_auth & ~strong
    pl.loc[rej_auth, ['has_wd', 'has_gn']] = False
    pl.loc[rej_reg, 'has_reg'] = False
    drop = ['qid', 'qid_via'] + [c for c in pl.columns if c.startswith('id_')]
    pl.loc[pl['identity_dropped'], drop] = None
    kept = rej_auth & strong
    _t(f'arbitration: {int(rej_auth.sum())} authority coordinates unused ({int(kept.sum())} keep their identity: names agree), '
       f'{int(rej_reg.sum())} register coordinates rejected ({int(tie_far.sum())} by the 100 km tie rule), '
       f'{int((tie & ~tie_far).sum())} ties kept as register')
    return pl


# ---------------------------------------------------------------------------------------------------------------------
def build(v2: Path, rdir: Path, dest: Path, namecheck: Optional[Path] = None, geocode: Optional[Path] = None,
          verify: Optional[Path] = None, tgn_dir: Optional[Path] = None, casebooks: Optional[Path] = None) -> dict:
    t0 = time.time()
    dest.mkdir(parents=True, exist_ok=True)
    rep: dict = {'generated': time.strftime('%Y-%m-%d %H:%M:%S')}
    auth = RG.AuthData(rdir, tgn_dir)
    coll = pd.read_csv(v2 / 'collections.tsv', sep='\t', usecols=['slug', 'licence', 'tier', 'renascor_ids'])
    # ------------------------------------------------------------------ counts, votes, gates
    dc = default_counts(v2)
    _t(f'default counts: {len(dc)} (kind, entity) rows ({time.time() - t0:.0f}s)')
    votes = RG.id_votes(v2)
    pl = apply_default_counts(pq.read_table(v2 / 'places_local.parquet').to_pandas(), dc, 'place')
    pe = apply_default_counts(pq.read_table(v2 / 'persons_local.parquet').to_pandas(), dc, 'person')
    gp = RG.run(votes, pl, auth, 'place', _t)['pairs']
    gq = RG.run(votes, pe, auth, 'person', _t)['pairs']
    ido = MO.load_id_overrides(rdir)           # decisions confirmed by hand (manual review): they survive regeneration
    wd_ok = {q for q, e in auth.wd.items() if not e.get('missing')}
    gp, rp = MO.apply_id_overrides(gp, ido, pl, 'place', wd_ok, _t)
    gq, rq_ = MO.apply_id_overrides(gq, ido, pe, 'person', wd_ok, _t)
    verify = MO.effective_verify(verify, rdir, dest, _t)
    pairs = pd.concat([gp, gq], ignore_index=True)
    lg = RG.write_log(pairs, dest / 'gate_log.tsv')
    pairs.drop(columns=[c for c in ('qid_raw_note',) if c in pairs]).astype({c: 'string' for c in pairs.columns if pairs[c].dtype == object}).to_parquet(
        dest / 'gate_pairs.parquet', index=False)       # every pair, accepted ones too (regression tests, audits)
    cp, cq = RG.choose(gp, votes), RG.choose(gq, votes)
    rep['gates'] = gate_report(pairs)
    rep['excluded_mentions'] = excluded_report(v2)
    rep['manual_overrides'] = {'place': rp, 'person': rq_, 'rows': int(len(ido))}
    # ------------------------------------------------------------------ places
    pl = _places(pl, cp, gp, votes, auth, rdir, namecheck, geocode, verify, dest, rep, casebooks)
    place_global, coll_sets, pl = _place_globals(pl, dest, auth)
    _write_global(place_global, dest / 'places_global.parquet')
    write_place_tables(pl, dest)
    # ------------------------------------------------------------------ persons
    pe, ap = _persons(pe, cq, gq, auth, dest)
    # ------------------------------------------------------------------ report and globe
    rep.update(report_numbers(pl, pe, place_global, ap, lg))
    globe = write_globe(place_global, coll, dest / 'globe_places.json', rep, pl)
    rep['globe'] = globe
    # v2.5: the homonym check of every build (ratlas.curation): rule hits not decided by an entry, and per-edition outliers with a nearer
    # same-name place; read by a person, listed in homonym_check.tsv
    from . import curation as CUR
    sn = rdir / 'gn_samename.parquet'
    chk = CUR.check(CUR.norm_frame(pd.read_parquet(dest / 'places_local_geo.parquet')), samename=pd.read_parquet(sn) if sn.exists() else None)
    chk.to_csv(dest / 'homonym_check.tsv', sep='\t', index=False)
    rep['homonym_check'] = {f'{c}:{"covered" if v else "open"}': int(n) for (c, v), n in chk.groupby(['check', 'covered']).size().items()}
    _t(f'homonym check: {rep["homonym_check"]}')
    (dest / 'reconcile_report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=_js), 'utf-8')
    _t(f'done in {time.time() - t0:.0f}s')
    return rep


def write_place_tables(pl: pd.DataFrame, dest: Path) -> None:
    """coord_conflicts.tsv (every register-versus-authority conflict, > 100 km first), unplaced_by_collection.tsv, unplaced_top_names.tsv."""
    cf = pl[pl['coord_conflict'] & (pl['n_mentions'] > 0)].sort_values(['flag_100km', 'n_in_period'], ascending=[False, False])
    cols = ['coll_slug', 'entity_key', 'name', 'n_in_period', 'reg_lat', 'reg_lon', 'auth_source', 'auth_lat', 'auth_lon',
            'coord_conflict_km', 'flag_100km', 'arbitration', 'auth_name_level', 'auth_rejected', 'identity_dropped', 'reg_rejected',
            'namecheck_reg_km', 'namecheck_auth_km', 'qid', 'coord_source']
    cf[cols].head(5000).to_csv(dest / 'coord_conflicts.tsv', sep='\t', index=False)
    live = pl[(pl['n_in_period'] > 0)]
    un = live[live['coord_source_geo'] == 'none']
    byc = live.groupby('coll_slug').apply(lambda d: pd.Series({
        'places_in_period': len(d), 'mentions_in_period': int(d['n_in_period'].sum()),
        'placed_distinct': int((d['coord_source_geo'] != 'none').sum()),
        'placed_mentions': int(d.loc[d['coord_source_geo'] != 'none', 'n_in_period'].sum()),
        'unplaced_distinct': int((d['coord_source_geo'] == 'none').sum()),
        'unplaced_mentions': int(d.loc[d['coord_source_geo'] == 'none', 'n_in_period'].sum()),
        'name_placed_mentions': int(d.loc[d['coord_source_geo'].isin(NAME_SOURCES), 'n_in_period'].sum()),
        'directory_placed_mentions': int(d.loc[d['coord_source_geo'].isin(DIRECT_SOURCES), 'n_in_period'].sum()),
        'skipped_by_class_mentions': int(d.loc[d['place_class'].notna() & (d['coord_source_geo'] == 'none'), 'n_in_period'].sum()),
        'tier': d['tier'].iloc[0]}), include_groups=False)
    byc['placed_share'] = (byc['placed_mentions'] / byc['mentions_in_period']).round(3)
    byc.sort_values('unplaced_mentions', ascending=False).reset_index().to_csv(dest / 'unplaced_by_collection.tsv', sep='\t', index=False)
    top = un.groupby('norm_name').agg(name=('name', 'first'), mentions=('n_in_period', 'sum'), n_colls=('coll_slug', 'nunique'),
                                      colls=('coll_slug', lambda s: ','.join(sorted(set(s))[:4])),
                                      place_class=('place_class', 'first'),
                                      reason=('gc_reason', lambda s: s.dropna().mode().iat[0] if s.notna().any() else '')).reset_index()
    top.sort_values('mentions', ascending=False).head(3000).to_csv(dest / 'unplaced_top_names.tsv', sep='\t', index=False)


def gate_report(pairs: pd.DataFrame) -> dict:
    out: dict = {}
    g = pairs[pairs['status'].isin(['suggested', 'rejected', 'ignored'])]
    out['by_status_reason'] = {f'{k[0]}:{k[1]}:{k[2]}': int(v) for k, v in g.groupby(['kind', 'status', 'reason']).size().items()}
    out['by_edition'] = {c: {f'{r[0]}:{r[1]}': int(n) for r, n in d.groupby(['status', 'reason']).size().items()}
                         for c, d in g.groupby('coll_slug')}
    out['accepted_pairs'] = {k: int(v) for k, v in pairs[pairs['status'] == 'accepted'].groupby('kind').size().items()}
    out['accepted_by_reason'] = {f'{k[0]}:{k[1]}': int(v) for k, v in pairs[pairs['status'] == 'accepted']
                                 .groupby(['kind', 'reason']).size().items()}
    return out


def load_casebooks_map(path: Path) -> pd.DataFrame:
    """The curated map PLACE id -> Casebooks directory entry (RQ/derived/atlas/casebooks_place_map.tsv)."""
    m = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    m['entity_key'] = 'casebooks:' + m['place_id']
    for c in ('lat', 'lon', 'geo_radius_km', 'n_mentions'):
        m[c] = pd.to_numeric(m[c], errors='coerce')
    return m.drop_duplicates('entity_key')


def apply_casebooks(pl: pd.DataFrame, path: Optional[Path], rep: Optional[dict] = None) -> pd.DataFrame:
    """v2.1: coordinates of the Casebooks places from the curated directory map (``casebooks_place_map.tsv``): the match PLACE id -> directory entry has a
    confidence high (about 98-100 % right) / medium (90-95 %) / low (65-75 %) / none. High and medium give the coordinate (``coord_source``
    ``casebooks_directory``; ``coord_conf`` high only when the directory's own geocoding is high / medium too, else medium; a point with a radius of
    20 km or more, a county median, is an area); low stays a suggestion (``cb_lat`` / ``cb_lon``, ``cb_used`` False). It comes before the name-based
    geocoder (which it replaces for the place and keeps in ``gc_*``) and never overrides a register or authority coordinate."""
    for c in CB_COLS:
        pl[c] = None
    if path is None or not Path(path).exists():
        return pl
    m = load_casebooks_map(path).set_index('entity_key')
    isc = (pl['coll_slug'] == 'casebooks').to_numpy()
    key = pl['entity_key'].to_numpy()
    hit = isc & np.isin(key, m.index.to_numpy())
    mm = m.reindex(key[hit])
    ix = pl.index[hit]
    pl.loc[ix, 'cb_place_id'] = mm['place_id'].to_numpy()
    pl.loc[ix, 'cb_entry'] = mm['directory_entry'].to_numpy()
    pl.loc[ix, 'cb_county'] = mm['county'].to_numpy()
    pl.loc[ix, 'cb_parish'] = mm['parish'].to_numpy()
    pl.loc[ix, 'cb_match'] = mm['confidence'].to_numpy()
    pl.loc[ix, 'cb_geo_conf'] = mm['geo_confidence'].to_numpy()
    pl.loc[ix, 'cb_geo_method'] = mm['geo_method'].to_numpy()
    pl.loc[ix, 'cb_lat'] = mm['lat'].to_numpy()
    pl.loc[ix, 'cb_lon'] = mm['lon'].to_numpy()
    pl.loc[ix, 'cb_radius_km'] = mm['geo_radius_km'].to_numpy()
    conf = pl['cb_match'].to_numpy(dtype=object)
    lat = pd.to_numeric(pl['cb_lat'], errors='coerce')
    free = pl['coord_source_geo'].isin(('none',) + NAME_SOURCES).to_numpy()
    use = hit & np.isin(conf, ['high', 'medium']) & lat.notna().to_numpy() & free
    # v2.3: a directory point whose OWN geocoding is low (a guess from a street name: 'St George's Lane' put in Kentish Town) is not a precise point: it stays a suggestion
    # (cb_lat / cb_lon, cb_used False) unless the directory itself gives it a radius of an area (a county median, 20 km or more)
    rad0 = pd.to_numeric(pl['cb_radius_km'], errors='coerce').fillna(5.0).to_numpy(dtype=float)
    low_pt = use & (pl['cb_geo_conf'].to_numpy(dtype=object) == 'low') & (rad0 < CB_AREA_KM)
    use = use & ~low_pt
    # ... and the name-based guess that then fills the place must agree with the directory's (unverified but matched) entry: 'Charterhouse' (the London one) must not become
    # the Charterhouse of Somerset 51 km away because the directory's point was not used
    n_conflict = 0
    if low_pt.any():
        nb = low_pt & pl['coord_source_geo'].isin(NAME_SOURCES).to_numpy()
        if nb.any():
            dd = np.asarray(haversine_km(pd.to_numeric(pl.loc[nb, 'cb_lat']), pd.to_numeric(pl.loc[nb, 'cb_lon']), pd.to_numeric(pl.loc[nb, 'geo_lat']), pd.to_numeric(pl.loc[nb, 'geo_lon'])))
            idx = pl.index[nb][dd > CB_NAME_CONFLICT_KM]
            n_conflict = len(idx)
            if n_conflict:
                pl.loc[idx, ['coord_source_geo', 'coord_conf', 'coord_kind', 'gc_reason']] = ['none', None, None, 'casebooks_directory_conflict']
                pl.loc[idx, ['geo_lat', 'geo_lon', 'coord_radius_km']] = np.nan
    if 'gc_lat' in pl:             # distance between the directory and the name-based result it replaces
        g = pd.to_numeric(pl['gc_lat'], errors='coerce').notna().to_numpy() & use
        if g.any():
            d = haversine_km(pd.to_numeric(pl.loc[g, 'cb_lat']), pd.to_numeric(pl.loc[g, 'cb_lon']),
                             pd.to_numeric(pl.loc[g, 'gc_lat']), pd.to_numeric(pl.loc[g, 'gc_lon']))
            pl.loc[pl.index[g], 'cb_gc_km'] = np.asarray(d)
    pl['cb_used'] = False
    ui = pl.index[use]
    pl.loc[ui, 'cb_used'] = True
    pl.loc[ui, 'coord_source_geo'] = 'casebooks_directory'
    pl.loc[ui, 'geo_lat'] = pd.to_numeric(pl.loc[ui, 'cb_lat'])
    pl.loc[ui, 'geo_lon'] = pd.to_numeric(pl.loc[ui, 'cb_lon'])
    gconf = pl.loc[ui, 'cb_geo_conf'].to_numpy(dtype=object)
    pl.loc[ui, 'coord_conf'] = np.where((conf[use] == 'high') & np.isin(gconf, ['high', 'medium']), 'high', 'medium')
    rad = pd.to_numeric(pl.loc[ui, 'cb_radius_km'], errors='coerce').fillna(5.0).to_numpy(dtype=float)
    area = rad >= CB_AREA_KM
    pl.loc[ui, 'coord_kind'] = np.where(area, 'area', 'point')
    pl.loc[ui, 'coord_radius_km'] = np.where(area, rad, np.nan)
    pl.loc[ui, 'place_kind'] = pl.loc[ui, 'place_kind'].where(pl.loc[ui, 'place_kind'].notna(), pd.Series(np.where(area, 'area', 'city'), index=ui))
    if rep is not None:
        rep['casebooks_directory'] = {
            'map_rows_matched': int(hit.sum()), 'placed_distinct': int(use.sum()),
            'placed_by_match_confidence': {k: int(((conf == k) & use).sum()) for k in ('high', 'medium')},
            'suggested_only_low': int((hit & (conf == 'low') & lat.notna().to_numpy()).sum()),
            'geo_low_point_not_used': int(low_pt.sum()), 'name_based_point_dropped_against_directory': int(n_conflict), 'geo_low_point_not_used_mentions_in_period': int(pl.loc[pl.index[low_pt], 'n_in_period'].sum()),
            'replaced_name_based': int((use & pl['coord_source_pre'].eq('none').to_numpy() & pl['gc_lat'].notna().to_numpy()).sum())
            if 'gc_lat' in pl else 0,
            'areas': int(area.sum())}
    return pl


def casebooks_low_candidates(pl: pd.DataFrame) -> pd.DataFrame:
    """v2.4 (decision pending of the owner): the Casebooks places whose directory entry is matched (high / medium) but whose directory point has a geocoding of its own rated
    ``low`` and a radius under ``CB_AREA_KM`` (a street guessed in Kentish Town): they stay unplaced, as in v2.3 (1,944 mentions of 315 places), and are listed here with the
    entry, the county and the point, so that they can later be turned into areas of London level (``london``: the entry is in London, Middlesex, Surrey or Southwark;
    ``suggested_radius_km``) if the owner decides so. Never on the globe."""
    cols = ['coll_slug', 'entity_key', 'name', 'n_in_period', 'cb_place_id', 'cb_entry', 'cb_county', 'cb_parish', 'cb_match', 'cb_geo_conf', 'cb_geo_method', 'cb_lat', 'cb_lon',
            'cb_radius_km', 'coord_source_geo', 'geo_lat', 'geo_lon']
    if 'cb_geo_conf' not in pl or not len(pl):
        return pd.DataFrame(columns=cols + ['london', 'suggested_radius_km'])
    lat = pd.to_numeric(pl['cb_lat'], errors='coerce')
    rad = pd.to_numeric(pl['cb_radius_km'], errors='coerce').fillna(5.0)
    m = (pl['coll_slug'] == 'casebooks') & pl['cb_match'].isin(['high', 'medium']) & lat.notna() & (pl['cb_geo_conf'] == 'low') & (rad < CB_AREA_KM) \
        & ~pl['cb_used'].fillna(False).astype(bool)
    out = pl.loc[m, [c for c in cols if c in pl]].copy()
    cty = out['cb_county'].fillna('').astype(str).str.lower()
    out['london'] = cty.isin(['london', 'middlesex', 'surrey', 'southwark', 'westminster']) | out['cb_entry'].fillna('').astype(str).str.contains('London', case=False)
    out['suggested_radius_km'] = np.where(out['london'], 15.0, np.nan)
    return out.sort_values('n_in_period', ascending=False).reset_index(drop=True)


def exonym_keys(name, variants=None) -> list:
    """Normalised keys of an entry's NAME (whole, before a comma, before a parenthesis) for the exonym table. The variants are not read: the table maps
    'Batavis' to the Dutch because GeoNames lists it as a name of Passau, so a variant would make a TGN Passau "far" from the Netherlands."""
    out = []
    if isinstance(name, str) and name:
        for c in (name, name.split(',')[0], name.split('(')[0]):
            k = _norm(c)
            if k and k not in out:
                out.append(k)
    return out


def _places(pl, cp, gp, votes, auth: RG.AuthData, rdir, namecheck, geocode, verify, dest, rep, casebooks=None) -> pd.DataFrame:
    key = ['coll_slug', 'entity_key']
    pl = pl.merge(cp['qid'], how='left', on=key).merge(cp['suggested'], how='left', on=key)
    wide = wide_ids(cp['ids'], 'place', PLACE_FALLBACK + ('wikipedia',))
    pl = pl.merge(wide, how='left', on=key)
    for s in PLACE_FALLBACK + ('wikipedia',):
        if 'id_' + s not in pl:
            pl['id_' + s] = None
    pl = pl.merge(suggested_ids(gp, 'place'), how='left', on=key).merge(best_reason(gp, 'place'), how='left', on=key)
    pl['qid_conflict'] = pl['qid_n_cand'].fillna(0) > 1
    # place classes (celestial, ethnonym, common noun ...)
    pl['place_class'] = [PCL.classify(n, t, c) for n, t, c in zip(pl['name'], pl['note_type'], pl['person_class'])]
    # --- the authority's data
    wq = pl['qid'].map(lambda q: auth.wd.get(q) if isinstance(q, str) else None)
    pl['wd_lat'] = [e['lat'] if e and not e.get('missing') else np.nan for e in wq]
    pl['wd_lon'] = [e['lon'] if e and not e.get('missing') else np.nan for e in wq]
    pl['wd_label'] = [e['label'] if e and not e.get('missing') else None for e in wq]
    pl['wd_kind'] = [kind_of_classes([auth.classlab.get(c, '') for c in e['classes']]) if e and not e.get('missing') else None for e in wq]
    pl['wd_polity'] = [polity_classes([auth.classlab.get(c, '') for c in e['classes']]) if e and not e.get('missing') else False for e in wq]
    gnf = pl['id_geonames'].map(lambda g: auth.gn.get(g) if isinstance(g, str) else None)
    pl['gn_lat'] = [f['lat'] if f else np.nan for f in gnf]
    pl['gn_lon'] = [f['lon'] if f else np.nan for f in gnf]
    pl['gn_fcode'] = [f['fcode'] if f else None for f in gnf]
    pl['gn_pop'] = [f['pop'] if f else np.nan for f in gnf]
    pl['gn_country'] = [f['cc'] if f else None for f in gnf]
    tgf = pl['id_tgn'].map(lambda g: auth.tgn.get(str(g)) if isinstance(g, str) else None)          # accepted TGN ids only (gates)
    pl['tgn_lat'] = [t['lat'] if t and t['lat'] is not None else np.nan for t in tgf]
    pl['tgn_lon'] = [t['lon'] if t and t['lon'] is not None else np.nan for t in tgf]
    pl['tgn_type'] = [t['type'] if t else None for t in tgf]
    # v2.2: a TGN point that contradicts the subject's own hierarchy (500 km from its parent) or the curated exonym table (the wrong sign of a
    # longitude) is not used; the id itself stays accepted
    pl['tgn_coord_issue'] = [TGN.coord_issue(t, exonym_keys(n, v)) if t else None for t, n, v in zip(tgf, pl['name'], pl['variants'])]
    bad_pt = pl['tgn_coord_issue'].notna()
    if bad_pt.any():
        pl.loc[bad_pt, ['tgn_lat', 'tgn_lon']] = np.nan
        pl.loc[bad_pt, ['coll_slug', 'entity_key', 'name', 'id_tgn', 'n_mentions', 'tgn_coord_issue']].to_csv(
            Path(dest) / 'tgn_coord_check.tsv', sep='\t', index=False)
        _t(f'TGN points not used (hierarchy / exonym check): {int(bad_pt.sum())}')
    # --- the register's coordinates
    reg_ok = pl['lat'].notna() & pl['lon'].notna() & valid_coord(pl['lat'], pl['lon'])
    pl['reg_lat'] = pl['lat'].where(reg_ok)
    pl['reg_lon'] = pl['lon'].where(reg_ok)
    pl['reg_source'] = pl['coord_source'].where(reg_ok)
    pl['reg_invalid'] = pl['lat'].notna() & ~reg_ok
    ov = load_overrides(rdir)
    if len(ov):
        ix = pl.merge(ov, how='left', on=key)
        hit = ix['ov_lat'].notna().to_numpy()
        pl.loc[hit, 'reg_lat'] = ix.loc[hit, 'ov_lat'].to_numpy()
        pl.loc[hit, 'reg_lon'] = ix.loc[hit, 'ov_lon'].to_numpy()
        pl.loc[hit, 'reg_source'] = 'override'
        drop = (ix['ov_drop'].fillna(False).astype(bool) & ~ix['ov_lat'].notna()).to_numpy() if 'ov_drop' in ix else np.zeros(len(ix), dtype=bool)
        pl.loc[drop, ['reg_lat', 'reg_lon', 'reg_source']] = [np.nan, np.nan, None]          # v2.2: the register point came from a wrong id
        _t(f'coordinate overrides applied: {int(hit.sum())}, register coordinates dropped by hand: {int(drop.sum())}')
    nt = pl['note_type'].fillna('').str.lower()
    drop_area = (nt.isin(AREA_DROP_TYPES) | (pl['coll_slug'].isin(AREA_DROP_COLLS) & nt.isin(AREA_NOTE))) & pl['reg_lat'].notna()
    pl['reg_area_dropped'] = drop_area
    pl.loc[drop_area, ['reg_lat', 'reg_lon', 'reg_source']] = [np.nan, np.nan, None]
    _t(f'register coordinates of countries / regions dropped (capital points): {int(drop_area.sum())}')
    # --- kind
    rk = pl['note_type'].map(kind_of_regtype)
    gk = pl['gn_fcode'].map(kind_of_fcode)
    tk = pl['tgn_type'].map(TGN.kind_of_type)
    pl['place_kind'] = pl['wd_kind'].fillna(gk).fillna(tk).fillna(rk)
    pl['kind_source'] = np.where(pl['wd_kind'].notna(), 'wikidata', np.where(gk.notna(), 'geonames', np.where(tk.notna(), 'tgn',
                                                                                                         np.where(rk.notna(), 'register', None))))
    # a GeoNames country / first-order region is an area, whatever Wikidata's class list says (Guernsey is an island and a Crown dependency),
    # unless the item is also a city (Hamburg, Istanbul, Solothurn: a point); v2 audit, check 5
    polity = pl['gn_fcode'].astype(str).str.match(r'^(PCL|ADM1$|TERR$)') & (pl['wd_kind'] != 'city')
    pl.loc[polity & (pl['place_kind'] != 'area'), 'kind_source'] = 'geonames'
    pl.loc[polity, 'place_kind'] = 'area'
    # v2.2: ... and a GeoNames settlement (PPL, 1,000 inhabitants or more) that Wikidata calls a region, province or district without a polity class is the
    # town of that name (Mantua is a PPLA2; its item Q16204 is the Province of Mantua): the entry's own id says it is the town. A register type that says
    # region / country keeps the area.
    town_gn = pl['gn_fcode'].astype(str).str.match(r'^PPL(?!X)') & (pd.to_numeric(pl['gn_pop'], errors='coerce').fillna(0) >= 1000)
    weak_area = (pl['place_kind'] == 'area') & (pl['wd_kind'] == 'area') & ~pl['wd_polity'] & ~pl['note_type'].fillna('').str.lower().isin(AREA_NOTE)
    pl.loc[town_gn & weak_area, 'kind_source'] = 'geonames'
    pl.loc[town_gn & weak_area, 'place_kind'] = 'city'
    # ... and the other way round: a register that types the entry 'country' (Hessen, Grafschaft Schwarzburg) outranks a TGN 'inhabited place' or a GeoNames
    # settlement of the same name, unless Wikidata says it is a city: an area (the TGN point is not used for it either, see ``tgn_bad_type``)
    reg_country = pl['note_type'].fillna('').str.lower().isin(AREA_DROP_TYPES) & (pl['place_kind'] == 'city') & pl['kind_source'].isin(['tgn', 'geonames']) \
        & (pl['wd_kind'] != 'city')
    pl.loc[reg_country, 'kind_source'] = 'register'
    pl.loc[reg_country, 'place_kind'] = 'area'
    is_area = (pl['place_kind'] == 'area')
    # v2.3: the register point of an entry the editor types 'region' / 'canton' / 'province' ... whose accepted authority is an area too is usually the seat's (Bullinger's
    # Solothurn canton, Q11929, sat on the city); the authority's centroid is the point of an area. Only within 100 km (a larger gap is the arbitration's business).
    if 'reg_lat' in pl:
        area_note = nt.isin(AREA_NOTE - AREA_DROP_TYPES).to_numpy()
        a_lat = np.where(is_area & pl['gn_lat'].notna(), pl['gn_lat'], np.where(pl['wd_lat'].notna() & pl['qid'].notna(), pl['wd_lat'], np.nan))
        a_lon = np.where(is_area & pl['gn_lat'].notna(), pl['gn_lon'], np.where(pl['wd_lat'].notna() & pl['qid'].notna(), pl['wd_lon'], np.nan))
        ok_a = is_area.to_numpy() & area_note & pl['reg_lat'].notna().to_numpy() & np.isfinite(a_lat)
        dd = np.full(len(pl), np.nan)
        if ok_a.any():
            dd[ok_a] = haversine_km(pl.loc[ok_a, 'reg_lat'], pl.loc[ok_a, 'reg_lon'], a_lat[ok_a], a_lon[ok_a])
        swap = ok_a & (dd > 2.0) & (dd < 100.0)
        pl['reg_area_replaced'] = swap
        pl.loc[swap, ['reg_lat', 'reg_lon', 'reg_source']] = [np.nan, np.nan, None]
        _t(f'register points of areas replaced by the authority centroid: {int(swap.sum())}')
    # --- authority coordinates: GeoNames centroid first for areas, Wikidata first for the rest
    pl['has_reg'] = pl['reg_lat'].notna()
    wd_ok = pl['wd_lat'].notna() & pl['qid'].notna()
    gn_ok = pl['gn_lat'].notna()
    use_gn = gn_ok & (is_area | ~wd_ok)
    pl['has_wd'] = wd_ok & ~use_gn
    pl['has_gn'] = gn_ok & ~pl['has_wd']
    pl['auth_lat'] = np.where(pl['has_wd'], pl['wd_lat'], pl['gn_lat'])
    pl['auth_lon'] = np.where(pl['has_wd'], pl['wd_lon'], pl['gn_lon'])
    pl['auth_source'] = np.where(pl['has_wd'], 'wikidata', np.where(pl['has_gn'], 'geonames', None))
    both = pl['has_reg'] & (pl['has_wd'] | pl['has_gn'])
    pl['coord_conflict_km'] = np.nan
    pl.loc[both, 'coord_conflict_km'] = haversine_km(pl.loc[both, 'reg_lat'], pl.loc[both, 'reg_lon'],
                                                     pl.loc[both, 'auth_lat'], pl.loc[both, 'auth_lon'])
    pl['coord_conflict'] = pl['coord_conflict_km'] > CONFLICT_KM
    pl = arbitrate_v2(pl, load_namecheck(rdir) if namecheck is None else _load_nc(namecheck))
    # v2.1: an accepted TGN id gives a point when nothing else does (register > Wikidata > GeoNames id > TGN)
    # a register type 'country' with a TGN subject that is not an area (a town called Hessen, an island): no point (audit check 5)
    tgn_bad_type = nt.isin(AREA_DROP_TYPES).to_numpy() & (tk != 'area').to_numpy()
    pl['has_tgn'] = pl['tgn_lat'].notna() & ~pl['has_reg'] & ~pl['has_wd'] & ~pl['has_gn'] & valid_coord(pl['tgn_lat'].fillna(0), pl['tgn_lon'].fillna(0)) \
        & (pl['tgn_lat'].fillna(0) != 0) & ~tgn_bad_type
    pl['coord_source_geo'] = np.where(pl['has_reg'], 'register', np.where(pl['has_wd'], 'wikidata',
                                      np.where(pl['has_gn'], 'geonames', np.where(pl['has_tgn'], 'tgn', 'none'))))
    pl['geo_lat'] = np.where(pl['has_reg'], pl['reg_lat'], np.where(pl['has_wd'], pl['wd_lat'], np.where(pl['has_gn'], pl['gn_lat'], pl['tgn_lat'])))
    pl['geo_lon'] = np.where(pl['has_reg'], pl['reg_lon'], np.where(pl['has_wd'], pl['wd_lon'], np.where(pl['has_gn'], pl['gn_lon'], pl['tgn_lon'])))
    pl.loc[pl['coord_source_geo'] == 'none', ['geo_lat', 'geo_lon']] = np.nan
    # coarse points: areas whose point comes from an authority
    pl['coord_kind'] = np.where(pl['coord_source_geo'] == 'none', None, np.where(is_area, 'area', 'point'))
    pl['coord_radius_km'] = np.where(pl['coord_kind'] == 'area', np.where(pl['gn_fcode'].astype(str).str.startswith('PCL')
                                                                         | (pl['tgn_type'].fillna('').astype(str).str.contains('nation') & pl['has_tgn']),
                                                                         350.0, 150.0), np.nan)
    rad_auth = pl['coord_radius_km'].to_numpy()
    ck_auth = pl['coord_kind'].to_numpy(dtype=object)
    pl = B.apply_geocode(pl.reset_index(drop=True), geocode, verify).reset_index(drop=True)
    named = pl['coord_source_geo'].isin(NAME_SOURCES).to_numpy()
    pl['coord_radius_km'] = np.where(named, pd.to_numeric(pl['coord_radius_km'], errors='coerce'), rad_auth)
    pl['coord_kind'] = np.where(named, None, ck_auth)
    # a result of the geocoder carries its own radius / kind
    free = pl['coord_source_geo'].isin(NAME_SOURCES)
    # a name-based result carries its own radius; a polity (place_kind area: countries, regions, duchies, bishoprics, districts) is an area even
    # when the point was copied from a same-name place (name_propagation has no radius): 350 km for a country, 150 km otherwise
    gr = pd.to_numeric(pl.loc[free, 'gc_radius_km'], errors='coerce')
    # a copy of another entity's point is an area when that entity's point is one (Polen, Königreich copied from Poland; Galliam from Gallia)
    src_kind = {k: (ck, rd) for k, ck, rd in zip(zip(pl['coll_slug'], pl['entity_key']), ck_auth, rad_auth)}
    inh = [_cand_kind(c, src_kind) for c in pl.loc[free, 'gc_cand']]
    inherited = pd.Series([x[0] if x else None for x in inh], index=gr.index, dtype='object')
    inh_rad = np.array([x[1] if x and x[0] == 'area' and x[1] == x[1] else np.nan for x in inh], dtype=float)
    country = pl.loc[free, 'note_type'].fillna('').str.lower().isin(AREA_DROP_TYPES) | pl.loc[free, 'gn_fcode'].astype(str).str.startswith('PCL')
    # a register type 'country' is an area whatever the name-based point was copied from (Terçeira, an island typed country: audit check 5)
    # v2.3: a polity (the reviewed table, a qualified name resolved by its head) is an area of its own radius, however small (the Five Catholic Cantons: 30 km; a county: 40 km)
    polity_t = pl.loc[free, 'gc_tier'].astype(str).str.startswith('polity') & (pl.loc[free, 'gc_radius_km'].notna())
    as_area = (gr >= 60) | (pl.loc[free, 'place_kind'] == 'area') | (inherited == 'area') | pl.loc[free, 'note_type'].fillna('').str.lower().isin(AREA_DROP_TYPES) | polity_t
    pl.loc[free, 'coord_kind'] = np.where(as_area, 'area', 'point')
    radius_free = np.where(gr >= 60, gr, np.where(polity_t, gr, np.where(np.isfinite(inh_rad), inh_rad, np.where(country, 350.0, 150.0))))
    pl.loc[free, 'coord_radius_km'] = np.where(as_area, radius_free, np.nan)
    pl = apply_casebooks(pl.reset_index(drop=True), casebooks, rep)
    cbl = casebooks_low_candidates(pl)
    cbl.to_csv(Path(dest) / 'casebooks_low_candidates.tsv', sep='\t', index=False)
    rep['casebooks_low_candidates'] = {'places': int(len(cbl)), 'mentions_in_period': int(cbl['n_in_period'].sum()) if len(cbl) else 0, 'london': int(cbl['london'].sum()) if len(cbl) else 0}
    pl = apply_qualifier_kind(pl, rep)
    pl['flag_100km'] = (pl['coord_conflict_km'] > 100).fillna(False)
    pl = attach_exonym_qids(pl, auth, rep)
    pl = apply_continents(pl, rep)
    pl = apply_kind_overrides(pl, rdir, rep)
    # v2.5: the homonym audit (data/curation/homonym_fixes.json) and its two guard rules, last: they override every source above
    from . import curation as CUR
    pl = CUR.apply_reconcile(pl.reset_index(drop=True), rep, log=_t)
    return pl.reset_index(drop=True)


def _cand_kind(cand, src_kind: dict):
    """``gc_cand`` of a propagated point is ``collection:entity_key`` of the entity it was copied from: -> its (coord_kind, radius)."""
    if not isinstance(cand, str) or ':' not in cand:
        return None
    coll, rest = cand.split(':', 1)
    return src_kind.get((coll, rest))


def _load_nc(path: Path) -> pd.DataFrame:
    n = pd.read_csv(path, sep='\t', dtype={'coll_slug': str, 'entity_key': str})
    return n.drop_duplicates(['coll_slug', 'entity_key']).set_index(['coll_slug', 'entity_key'])


def _place_globals(pl: pd.DataFrame, dest: Path, auth: RG.AuthData):
    live = pl['n_mentions'] > 0
    pl['generic_name'] = [GEN.place_generic(n) is not None for n in pl['name']]         # v2.2: a common noun is not a display name
    pl['place_kind_final'] = pl['place_kind']
    # kind of a name-placed result: the GeoNames feature of the match
    ev = pl['gc_evidence'].astype(str).str.extract(r'^(?:GeoNames|TGN) \d+ .*? \((\w+),')[0] if 'gc_evidence' in pl else None
    pl['gc_fcode'] = ev if ev is not None else None          # v2.2: the feature code of a name-based match, for the five-way kind
    if ev is not None:
        kk = ev.map(kind_of_fcode)
        pl['place_kind_final'] = pl['place_kind_final'].fillna(kk)
    pl['place_kind_final'] = pl['place_kind_final'].fillna(pd.Series(np.where(pl['coord_kind'].eq('area'), 'area', None), index=pl.index, dtype=object))
    # a point of an unknown class (a register place of type 'place', a propagated name) is a settlement far more often than not
    unk = pl['place_kind_final'].isna() & (pl['coord_source_geo'] != 'none')
    pl.loc[unk, 'kind_source'] = 'default'
    pl.loc[unk, 'place_kind_final'] = 'city'
    cl = assign_clusters(pl[live].assign(qid=pl.loc[live, 'qid_cluster']) if 'qid_cluster' in pl else pl[live], PLACE_FALLBACK)     # v2.3: exonym results join the item's cluster
    pl['global_id'] = None
    pl['basis'] = None
    pl['scope'] = None
    ix = pl.index[live]
    pl.loc[ix, 'global_id'] = np.where(cl['gid'].notna().to_numpy(), ('pl:' + cl['gid'].astype(str)).to_numpy(), None)
    pl.loc[ix, 'basis'] = cl['basis'].to_numpy()
    pl.loc[ix, 'scope'] = np.where(cl['gid'].notna().to_numpy(), 'authority', 'edition')
    ed = live & pl['global_id'].isna()
    pl.loc[ed, 'global_id'] = [edition_id('pl', c, k) for c, k in zip(pl.loc[ed, 'coll_slug'], pl.loc[ed, 'entity_key'])]
    pl.loc[~live, 'scope'] = 'excluded'
    _t(f'place clusters: {pl.loc[live, "global_id"].nunique()} global ids for {int(live.sum())} live locals ({int((~live).sum())} only excluded)')
    loc_cols = ['coll_slug', 'entity_key', 'renascor_id', 'name', 'norm_name', 'n_mentions', 'n_in_period', 'n_mentions_all',
                'n_in_period_all', 'tier', 'licence', 'note_type', 'place_class', 'place_kind', 'kind_source', 'lat', 'lon',
                'coord_source', 'has_coords', 'generic_name', 'n_generic_with_id', 'tgn_coord_issue']
    geo = pl[loc_cols + ['qid', 'qid_via', 'qid_n_cand', 'qid_conflict', 'qid_suggested', 'qid_suggested_reason', 'ids_suggested',
                         'n_ids_suggested', 'auth_name_level', 'id_geonames', 'id_tgn', 'id_gnd', 'reg_invalid', 'reg_area_dropped',
                         'coord_source_geo', 'geo_lat', 'geo_lon', 'coord_kind', 'auth_source', 'auth_lat', 'auth_lon',
                         'coord_conflict_km', 'coord_conflict', 'flag_100km', 'arbitration', 'namecheck_reg_km', 'namecheck_auth_km',
                         'auth_rejected', 'identity_dropped', 'reg_rejected', 'global_id', 'basis', 'scope', 'coord_conf',
                         'coord_radius_km', 'wd_label', 'gn_fcode', 'gn_pop', 'gn_country', 'tgn_lat', 'tgn_lon', 'tgn_type'] + GC_COLS
                        + ['verify_rule'] + CB_COLS + ['qid_exo', 'reg_area_replaced', 'kind5_override']
                        + [c for c in ('curation', 'curation_why') if c in pl]].rename(
        columns={'lat': 'register_lat', 'lon': 'register_lon', 'coord_source': 'register_coord_source',
                 'has_coords': 'register_has_coords', 'coord_source_geo': 'coord_source', 'geo_lat': 'lat', 'geo_lon': 'lon',
                 'id_geonames': 'geonames_id', 'id_tgn': 'tgn_id', 'id_gnd': 'gnd_id'})
    geo['has_coords'] = geo['lat'].notna()
    geo.to_parquet(dest / 'places_local_geo.parquet', index=False)
    # ---- global places
    L = pl[live]
    ag = B.aggregate_global(L, 'pl', SUM_COLS_V2)
    for s in ('geonames', 'tgn', 'pleiades', 'gnd', 'hls', 'viaf', 'wikipedia'):
        ag[s] = dominant(L, 'id_' + s).reindex(ag.index)
    ag['qid'] = dominant(L, 'qid').reindex(ag.index)
    if 'qid_cluster' in L:                                      # v2.3: a cluster made of exonym results only still has its item
        ag['qid'] = ag['qid'].fillna(L.dropna(subset=['qid_cluster']).groupby('global_id')['qid_cluster'].first().reindex(ag.index))
    ag['place_kind'] = dominant(L, 'place_kind_final').reindex(ag.index)
    ag['place_class'] = dominant(L, 'place_class').reindex(ag.index)
    tg1 = L[L['has_tgn']].sort_values('n_mentions', ascending=False).drop_duplicates('global_id').set_index('global_id')
    wd1 = L[L['has_wd']].drop_duplicates('global_id').set_index('global_id')
    gn1 = L[L['has_gn']].sort_values('n_mentions', ascending=False).drop_duplicates('global_id').set_index('global_id')
    rg = L[L['has_reg']].sort_values('n_mentions', ascending=False).drop_duplicates('global_id').set_index('global_id')
    lat = pd.Series(np.nan, index=ag.index)
    lon = pd.Series(np.nan, index=ag.index)
    src = pd.Series('none', index=ag.index, dtype=object)
    ck = pd.Series(None, index=ag.index, dtype=object)
    rad = pd.Series(np.nan, index=ag.index)
    for name, tab, la, lo in (('tgn', tg1, 'tgn_lat', 'tgn_lon'), ('register', rg, 'reg_lat', 'reg_lon'), ('geonames', gn1, 'gn_lat', 'gn_lon'),
                              ('wikidata', wd1, 'wd_lat', 'wd_lon')):      # later overrides earlier (the TGN point comes last in priority)
        ix = ag.index.intersection(tab.index)
        lat[ix] = tab.loc[ix, la]
        lon[ix] = tab.loc[ix, lo]
        src[ix] = name
        ck[ix] = tab.loc[ix, 'coord_kind']
        rad[ix] = tab.loc[ix, 'coord_radius_km']
    ag['lat'], ag['lon'], ag['coord_source'], ag['coord_kind'], ag['coord_radius_km'] = lat, lon, src, ck, rad
    for qd, (_lab, la_, lo_, rd_) in CONTINENTS.items():        # v2.4: a continent has one point and one radius, not the P625 of its Wikidata item (Europe: a Württemberg district)
        mc = (ag['qid'] == qd) & (ag['coord_source'] != 'none')
        if mc.any():
            ag.loc[mc, ['lat', 'lon', 'coord_radius_km']] = [la_, lo_, rd_]
            ag.loc[mc, 'coord_kind'] = 'area'
    ag['coord_conf'] = pd.Series(np.where(src != 'none', 'authority', None), index=ag.index, dtype=object)
    ag['coord_cand'] = None
    ag['coord_evidence'] = None
    nmr = L[L['coord_source_geo'].isin(FREE_SOURCES) & L['coord_source_pre'].eq('none')].copy()
    if len(nmr):
        nmr['_r'] = nmr['coord_conf'].map({'high': 2, 'medium': 1})
        nmr = nmr.sort_values(['_r', 'n_mentions', 'entity_key'], ascending=[False, False, True]).drop_duplicates('global_id').set_index('global_id')
        ixs = [g for g in nmr.index if g in ag.index and ag.at[g, 'coord_source'] == 'none']
        for g in ixs:
            r = nmr.loc[g]
            ag.at[g, 'lat'], ag.at[g, 'lon'], ag.at[g, 'coord_source'] = r['geo_lat'], r['geo_lon'], r['coord_source_geo']
            ag.at[g, 'coord_conf'], ag.at[g, 'coord_radius_km'], ag.at[g, 'coord_kind'] = r['coord_conf'], r['coord_radius_km'], r['coord_kind']
            ag.at[g, 'coord_cand'], ag.at[g, 'coord_evidence'] = r['gc_cand'], r['gc_evidence']
        _t(f'global places: {len(ixs)} clusters placed by name or by the Casebooks directory')
    ag['n_register_coords'] = L.groupby('global_id')['has_reg'].sum().reindex(ag.index).astype('int64')
    ag['coord_conflict'] = L.groupby('global_id')['coord_conflict'].any().reindex(ag.index).fillna(False)
    ag['flag_100km'] = L.groupby('global_id')['flag_100km'].any().reindex(ag.index).fillna(False)
    ag['qid_conflict'] = L.groupby('global_id')['qid_conflict'].any().reindex(ag.index).fillna(False)
    rr = ag.index.intersection(rg.index)
    dd = haversine_km(rg.loc[rr, 'reg_lat'], rg.loc[rr, 'reg_lon'], ag.loc[rr, 'lat'], ag.loc[rr, 'lon'])
    ag['coord_conflict_km'] = pd.Series(dd, index=rr).reindex(ag.index)
    ag.loc[ag['coord_source'] == 'register', 'coord_conflict_km'] = np.nan
    ag['coord_conflict'] = ag['coord_conflict'] | (ag['coord_conflict_km'] > CONFLICT_KM)
    q = ag['qid']
    for l in ('en', 'fr', 'de', 'it'):
        ag['label_' + l] = q.map(lambda x, l=l: (auth.wd.get(x) or {}).get('labels', {}).get(l) if isinstance(x, str) else None)
    # v2.2: a global place whose only edition names are common nouns ('this towne' of the Casebooks, Palast, Rue ...) is named by its authority: the Wikidata
    # label, else the GeoNames or TGN name of its id, else the Casebooks directory entry; the noun stays in variants_seen and the flag generic_with_id
    alt = pd.Series([_alt_name(auth, r_gn, r_tg, r_cb) for r_gn, r_tg, r_cb in zip(ag['geonames'], ag['tgn'], ag.index.map(_cb_entries(L)))], index=ag.index)
    ag['label'] = ag['label_en'].fillna(ag['label_fr']).fillna(ag['label_de']).fillna(ag['label_it'])
    authn = ag['label'].fillna(alt)
    swap = ag['name_is_generic'] & authn.notna()
    ag.loc[swap, 'name'] = authn[swap]
    ag.loc[swap, 'name_is_generic'] = False
    ag['label'] = ag['label'].fillna(alt).fillna(ag['name'])
    ag.loc[ag['name_is_generic'], 'label'] = None            # a noun with no authority behind it is not a display name: the reader shows nothing, or the surface in context
    ag['generic_with_id'] = ag['n_generic_with_id'] > 0
    ex = ag['coord_cand'].astype(str).str.startswith('exonym:')       # Galli, Gallia, France: one point called France
    for c in ('label', 'label_en'):
        ag.loc[ex, c] = ag.loc[ex, 'coord_cand'].str[len('exonym:'):]
    ag['classes'] = q.map(lambda x: '|'.join(f'{c}:{auth.classlab.get(c, "")}' for c in (auth.wd.get(x) or {}).get('classes', []))
                          if isinstance(x, str) else None)
    ag['country'] = None
    ag = ag.reset_index().rename(columns={'global_id': 'id'})
    ag['globe_id'] = merge_points(ag, point_unions(ag, pl))
    coll_sets = L.groupby('global_id')['coll_slug'].agg(lambda x: sorted(set(x)))
    return ag, coll_sets, pl


def _cb_entries(L: pd.DataFrame) -> dict:
    """{global_id: Casebooks directory entry used for a local place of the cluster} (the entry names the parish a patient's 'this towne' is)."""
    if 'cb_entry' not in L:
        return {}
    d = L[L['cb_used'].fillna(False).astype(bool) & L['cb_entry'].notna()].sort_values('n_mentions', ascending=False).drop_duplicates('global_id')
    return dict(zip(d['global_id'], d['cb_entry']))


def _alt_name(auth: RG.AuthData, geonames, tgn, cb) -> Optional[str]:
    """The authority's name of a place: GeoNames, then the local TGN, then the Casebooks directory entry (None if there is none)."""
    for g in str(geonames or '').split(';'):
        f = auth.gn.get(g.strip())
        if f and f.get('name'):
            return f['name']
    t = auth.tgn.get(str(tgn)) if tgn is not None and tgn == tgn else None
    if t and t.get('name'):
        return t['name']
    return cb if isinstance(cb, str) and cb.strip() else None


def _persons(pe, cq, gq, auth: RG.AuthData, dest: Path):
    key = ['coll_slug', 'entity_key']
    pe = pe.merge(cq['qid'], how='left', on=key).merge(cq['suggested'], how='left', on=key)
    wide = wide_ids(cq['ids'], 'person', PERSON_FALLBACK + ('wikipedia',))
    pe = pe.merge(wide, how='left', on=key)
    for s in PERSON_FALLBACK:
        if 'id_' + s not in pe:
            pe['id_' + s] = None
    pe = pe.merge(suggested_ids(gq, 'person'), how='left', on=key)
    pe['qid_conflict'] = pe['qid_n_cand'].fillna(0) > 1
    live = pe['n_mentions'] > 0
    pe = pe.reset_index(drop=True)
    pe['generic_name'] = [GEN.person_generic(n) is not None for n in pe['name']]        # v2.2: Vater, Kayser, Gemahlin are not display names
    L = pe[live]
    cl = assign_clusters(L, PERSON_FALLBACK)
    pe['global_id'] = None
    pe['basis'] = None
    pe['scope'] = 'excluded'
    ix = pe.index[live]
    pe.loc[ix, 'global_id'] = np.where(cl['gid'].notna().to_numpy(), ('pe:' + cl['gid'].astype(str)).to_numpy(), None)
    pe.loc[ix, 'basis'] = cl['basis'].to_numpy()
    pe.loc[ix, 'scope'] = np.where(cl['gid'].notna().to_numpy(), 'authority', 'edition')
    ed = live & pe['global_id'].isna()
    pe.loc[ed, 'global_id'] = [edition_id('pe', c, k) for c, k in zip(pe.loc[ed, 'coll_slug'], pe.loc[ed, 'entity_key'])]
    _t(f'person clusters: {pe.loc[live, "global_id"].nunique()} global ids for {int(live.sum())} live locals; with QID {int(L["qid"].notna().sum())}')
    pe[['coll_slug', 'entity_key', 'name', 'norm_name', 'n_mentions', 'n_in_period', 'n_mentions_all', 'n_in_period_all', 'tier',
        'licence', 'person_class', 'note_type', 'qid', 'qid_via', 'qid_n_cand', 'qid_conflict', 'qid_suggested',
        'qid_suggested_reason', 'ids_suggested', 'n_ids_suggested', 'global_id', 'basis', 'scope', 'generic_name', 'n_generic_with_id']].to_parquet(
        dest / 'persons_local_global.parquet', index=False)
    L = pe[live]
    ap = B.aggregate_global(L, 'pe', SUM_COLS_V2)
    for s in PERSON_FALLBACK + ('wikipedia',):
        ap[s] = dominant(L, 'id_' + s).reindex(ap.index)
    ap['qid'] = dominant(L, 'qid').reindex(ap.index)
    ap['qid_conflict'] = L.groupby('global_id')['qid_conflict'].any().reindex(ap.index).fillna(False)
    ap['note_type'] = dominant(L, 'note_type').reindex(ap.index)
    ap['person_class'] = dominant(L, 'person_class').reindex(ap.index)
    ap['reg_birth'] = dominant(L, 'birth').reindex(ap.index)
    ap['reg_death'] = dominant(L, 'death').reindex(ap.index)
    q = ap['qid']
    ents = {x: auth.wd.get(x) for x in set(q.dropna())}

    def get(x, f, default=None):
        e = ents.get(x)
        return f(e) if e and not e.get('missing') else default
    for l in ('en', 'fr', 'de', 'it'):
        ap['label_' + l] = q.map(lambda x, l=l: get(x, lambda e: e['labels'].get(l)))
    ap['label'] = ap['label_en'].fillna(ap['label_fr']).fillna(ap['label_de']).fillna(ap['label_it'])
    sw = ap['name_is_generic'] & ap['label'].notna()         # v2.2: a kinship noun or a title with a QID is named by the person's Wikidata label
    ap.loc[sw, 'name'] = ap.loc[sw, 'label']
    ap.loc[sw, 'name_is_generic'] = False
    ap['label'] = ap['label'].fillna(ap['name'])
    ap.loc[ap['name_is_generic'], 'label'] = None            # Vater, Kayser with no Wikidata label: not a display name
    ap['generic_with_id'] = ap['n_generic_with_id'] > 0
    ap['birth'] = q.map(lambda x: get(x, lambda e: e['birth']))
    ap['death'] = q.map(lambda x: get(x, lambda e: e['death']))
    ap['birth_precision'] = q.map(lambda x: get(x, lambda e: e['birth_prec'])).astype('Int64')
    ap['death_precision'] = q.map(lambda x: get(x, lambda e: e['death_prec'])).astype('Int64')
    occ = q.map(lambda x: get(x, lambda e: e.get('occupations', []), []))
    ap['occupation_qids'] = occ.map(lambda v: list(v)[:3])
    ap['occupations'] = occ.map(lambda v: [auth.classlab.get(o, o) for o in list(v)[:3]])
    ap['wd_human'] = q.map(lambda x: get(x, lambda e: str(NG.HUMAN in e['classes']), None))
    ap['wd_category'] = q.map(lambda x: get(x, lambda e: NG.person_category(e['classes'], auth.classlab), None))
    ap['wd_person_class'] = ap['wd_category'].map(NG.WD_PERSON_CLASS)
    # the register's class wins; an unmarked person whose accepted item is a god, a legendary figure or a family takes the item's class
    pcf = ap['person_class'].where(ap['person_class'].notna() & ap['person_class'].ne('human'), None)
    wdc = ap['wd_person_class'].where(ap['wd_person_class'].isin(['mythological', 'dynasty', 'people']), None)
    ap['person_class_final'] = pcf.fillna(wdc).fillna(ap['person_class']).fillna('human')
    ap = ap.reset_index().rename(columns={'global_id': 'id'})
    _write_global(ap, dest / 'persons_global.parquet')
    return pe, ap


# ---------------------------------------------------------------------------------------------------------------------
# v2.3: polity points that were split, kind overrides
# ---------------------------------------------------------------------------------------------------------------------
#: the Wikidata item of a country of the exonym table (France = Galli, Gallia, Franckreich ...): the placed locals join the item's cluster, so one entity has one count.
#: Every pair is verified against the English label of the item in the local Wikidata table; a mismatch attaches nothing.
EXONYM_QID = {'France': 'Q142', 'Spain': 'Q29', 'Italy': 'Q38', 'Germany': 'Q183', 'England': 'Q21', 'Scotland': 'Q22', 'Ireland': 'Q27', 'Poland': 'Q36',
              'Sweden': 'Q34', 'Denmark': 'Q35', 'Norway': 'Q20', 'Switzerland': 'Q39', 'Austria': 'Q40', 'Hungary': 'Q28', 'Portugal': 'Q45', 'Greece': 'Q41',
              'Egypt': 'Q79', 'Bohemia': 'Q39193', 'Pomerania': 'Q104520', 'Mecklenburg': 'Q109057', 'Silesia': 'Q81720', 'Hesse': 'Q1199',
              'Europe': 'Q46', 'Asia': 'Q48', 'Africa': 'Q15', 'Americas': 'Q828'}      # v2.4: the continents of the exonym table join the item of the continent


def attach_exonym_qids(pl: pd.DataFrame, auth: 'RG.AuthData', rep: Optional[dict] = None) -> pd.DataFrame:
    """``qid_cluster``: the accepted QID of a local, else the item of the polity / country a name-based exonym result stands for (``gc_cand`` = ``exonym:<label>``,
    the polity table's QID in ``gc_qid``). A local that already has an accepted QID keeps it."""
    pl['qid_exo'] = None
    cand = pl['gc_cand'].astype(str) if 'gc_cand' in pl else pd.Series('', index=pl.index)
    gq_col = pl['gc_qid'] if 'gc_qid' in pl else pd.Series([None] * len(pl), index=pl.index)
    src_ok = pl['coord_source_geo'].isin(['name_exonym', 'name_latin'])
    isx = src_ok & pl['qid'].isna() & (cand.str.startswith('exonym:') | gq_col.map(lambda x: isinstance(x, str) and bool(x)))
    ok_label = {}

    def verified(q, label):
        e = auth.wd.get(q) if isinstance(q, str) else None
        if not e or e.get('missing'):
            return False
        if (q, label) not in ok_label:
            ok_label[(q, label)] = str((e.get('labels') or {}).get('en', '')).strip().lower() == str(label).strip().lower()
        return ok_label[(q, label)]
    out = []
    for lab, gq, x in zip(cand.str[len('exonym:'):], gq_col, isx):
        q = None
        if x:
            if isinstance(gq, str) and gq and auth.wd.get(gq) and not auth.wd[gq].get('missing'):
                q = gq                                              # the polity table's item or the Latin label's item (verified when the table was reviewed / by the gates)
            elif lab in EXONYM_QID and verified(EXONYM_QID[lab], lab):
                q = EXONYM_QID[lab]
        out.append(q)
    pl['qid_exo'] = out
    pl['qid_cluster'] = pl['qid'].where(pl['qid'].notna(), pl['qid_exo'])
    if rep is not None:
        rep['exonym_qid_attached'] = {'locals': int(pl['qid_exo'].notna().sum()), 'mentions_in_period': int(pl.loc[pl['qid_exo'].notna(), 'n_in_period'].sum()),
                                      'by_item': {str(k): int(v) for k, v in pl['qid_exo'].value_counts().items()}}
    return pl


#: v2.4: the continents. Wikidata's P625 of a continent is a nonsense point (Europe at 48.69 N 9.14 E, a Württemberg district) and the exonym table's point is the middle of the
#: continent with a radius of 1,500-4,000 km: one point and one radius for every local of the item (the page does not pin a continent: ``extra.json`` types it 'continent').
CONTINENTS = {'Q46': ('Europe', 50.0, 15.0, 1500.0), 'Q48': ('Asia', 34.0, 90.0, 3000.0), 'Q15': ('Africa', 5.0, 20.0, 3500.0), 'Q828': ('Americas', 5.0, -75.0, 4000.0)}


def apply_continents(pl: pd.DataFrame, rep: Optional[dict] = None) -> pd.DataFrame:
    q = pl['qid'].where(pl['qid'].notna(), pl['qid_exo']) if 'qid_exo' in pl else pl['qid']
    m = q.isin(list(CONTINENTS)) & (pl['coord_source_geo'] != 'none')
    if m.any():
        pt = q[m].map(CONTINENTS)
        pl.loc[m, 'geo_lat'] = [x[1] for x in pt]
        pl.loc[m, 'geo_lon'] = [x[2] for x in pt]
        pl.loc[m, 'coord_radius_km'] = [x[3] for x in pt]
        pl.loc[m, 'coord_kind'] = 'area'
        pl.loc[m, 'place_kind'] = 'area'
    if rep is not None:
        rep['continents'] = {'locals': int(m.sum()), 'mentions_in_period': int(pl.loc[m, 'n_in_period'].sum()), 'by_item': {str(k): int(v) for k, v in q[m].value_counts().items()}}
    return pl


def apply_qualifier_kind(pl: pd.DataFrame, rep: Optional[dict] = None) -> pd.DataFrame:
    """v2.3: the qualifier sets the kind. 'Lingen, Grafschaft', 'Pommern, Herzogtum', 'Böhmen, Königreich' name a county, a duchy, a kingdom: an area, whatever the head is (the
    town of Lingen). A point that rests on a name (not on the register or an authority id) is then drawn as an area with the radius of the qualifier."""
    qt = pl['name'].map(lambda n: POL.parse_qualified(n).qtype if isinstance(n, str) and (',' in n or '(' in n) else None)
    pl['name_qtype'] = qt
    area_q = qt.isin(POL.AREA_QTYPES)
    chg = area_q & (pl['place_kind'] != 'area')
    pl.loc[chg, 'kind_source'] = 'qualifier'
    pl.loc[chg, 'place_kind'] = 'area'
    named = area_q & pl['coord_source_geo'].isin(NAME_SOURCES + DIRECT_SOURCES)
    if named.any():
        rad = qt[named].map(POL.radius_for).to_numpy(dtype='float64')
        cur = pd.to_numeric(pl.loc[named, 'coord_radius_km'], errors='coerce').fillna(0.0).to_numpy(dtype='float64')
        from_table = (pl.loc[named, 'coord_source_geo'] == 'name_exonym').to_numpy()       # the polity / exonym table's own radius is the reviewed one
        pl.loc[named, 'coord_kind'] = 'area'
        pl.loc[named, 'coord_radius_km'] = np.where(from_table, np.maximum(cur, rad), rad)
    if rep is not None:
        rep['qualifier_kind'] = {'area_by_qualifier': int(area_q.sum()), 'kind_changed': int(chg.sum()), 'name_based_points_made_areas': int(named.sum()),
                                 'by_qualifier': {str(k): int(v) for k, v in qt[area_q].value_counts().items()}}
    return pl


def load_kind_overrides(rdir: Path) -> dict:
    """``kind_overrides.tsv`` (coll_slug, entity_key, kind c|a|n|b|u, note): the five-way kind of the point of an edition place, corrected by hand."""
    f = Path(rdir) / 'kind_overrides.tsv'
    if not f.exists():
        return {}
    o = MO.read_tsv(f)
    bad = o[~o['kind'].isin(PK.KINDS)]
    if len(bad):
        raise ValueError(f'kind_overrides: unknown kind {sorted(set(bad["kind"]))}')
    return {(a, b): k for a, b, k in zip(o['coll_slug'], o['entity_key'], o['kind'])}


def apply_kind_overrides(pl: pd.DataFrame, rdir: Path, rep: Optional[dict] = None) -> pd.DataFrame:
    ov = load_kind_overrides(rdir)
    pl['kind5_override'] = [ov.get(k) for k in zip(pl['coll_slug'], pl['entity_key'])]
    if rep is not None:
        rep['kind_overrides'] = {'rows': len(ov), 'matched': int(pl['kind5_override'].notna().sum())}
    return pl


def point_unions(ag: pd.DataFrame, pl: pd.DataFrame) -> list:
    """Pairs of global places that are the same point of the globe for a reason other than the name check of ``merge_points`` (v2.3: 505 pairs of the same name within
    5 km in v2.2 -- Ballenstedt the town and the Amt, Zerbst, Utrecht, Westminster; France three times):

    1. the same candidate (``coord_cand``): two places that the geocoder put on the same GeoNames feature / TGN subject / exonym entry are the same place;
    2. a name-based place and the cluster of the GeoNames id its candidate is (``gn:123`` and a place whose ``geonames`` id is 123);
    3. a name-propagated place and the cluster of the entity it was copied from (``gc_cand`` = ``collection:entity_key``), when the points agree (5 km; 150 km for an area);
    4. an exonym place and the Wikidata cluster of the same English label within the radius (France, England, Sweden, Poland);
    5. the same display name (the English label, else the head of the edition name) within 5 km (an area within half its radius)."""
    d = ag[ag['lat'].notna() & (ag['n_in_period'] >= 1)]
    ids = set(d['id'])
    pairs: list = []
    cand = d[d['coord_cand'].notna()]
    first: dict = {}
    for i_, c in zip(cand['id'], cand['coord_cand']):
        if c in first and first[c] != i_:
            pairs.append((first[c], i_))
        first.setdefault(c, i_)
    gnid: dict = {}
    for i_, g in zip(d['id'], d['geonames']):
        for x in str(g).split(';') if isinstance(g, str) else ():
            gnid.setdefault(x.strip(), i_)
    for i_, c in zip(cand['id'], cand['coord_cand']):
        if isinstance(c, str) and c.startswith('gn:') and gnid.get(c[3:]) in ids and gnid[c[3:]] != i_:
            pairs.append((i_, gnid[c[3:]]))
    loc2g = dict(zip(zip(pl['coll_slug'], pl['entity_key']), pl['global_id']))
    pt = {i_: (la, lo, r if isinstance(r, (int, float)) and r == r else 0.0, k) for i_, la, lo, r, k in
          zip(d['id'], d['lat'], d['lon'], d['coord_radius_km'], d['coord_kind'])}
    for i_, c in zip(cand['id'], cand['coord_cand']):
        if not isinstance(c, str) or ':' not in c or c.startswith(('gn:', 'tgn:', 'exonym:', 'latin:')):
            continue
        coll, rest = c.split(':', 1)
        j = loc2g.get((coll, rest))
        if j in ids and j != i_:
            la, lo, r, k = pt[j]
            lim = 150.0 if (k == 'area' or r >= 60) else 5.0
            if float(haversine_km(pt[i_][0], pt[i_][1], la, lo)) <= lim:
                pairs.append((i_, j))
    lab = {}
    for i_, q, le, k, r in zip(d['id'], d['qid'], d['label_en'], d['coord_kind'], d['coord_radius_km']):
        if isinstance(q, str) and isinstance(le, str):
            lab.setdefault(le.strip().lower(), []).append(i_)
    for i_, c in zip(cand['id'], cand['coord_cand']):
        if isinstance(c, str) and c.startswith('exonym:'):
            for j in lab.get(c[len('exonym:'):].strip().lower(), ()):
                if j != i_ and float(haversine_km(pt[i_][0], pt[i_][1], pt[j][0], pt[j][1])) <= max(MERGE_KM_V23, pt[i_][2], pt[j][2]):
                    pairs.append((i_, j))
    # 5. the same display name within 5 km
    groups: dict = {}
    for i_, nm, le, lb in zip(d['id'], d['name'], d['label_en'], d['label']):
        base = le if isinstance(le, str) and le.strip() else (lb if isinstance(lb, str) and lb.strip() else nm)
        if not isinstance(base, str):
            continue
        k = _norm(re.sub(r'\([^)]*\)', ' ', base.split(',')[0]))
        if len(k) >= 3 and k not in B.GENERIC_NAME:
            groups.setdefault(k, []).append(i_)
    for k, mem in groups.items():
        if len(mem) < 2:
            continue
        cells: dict = {}
        for i_ in mem:
            la, lo, _r, _k = pt[i_]
            cells.setdefault((int(la // 0.1), int(lo // 0.1)), []).append(i_)
        for (ci, cj), grp in cells.items():
            near = [x for dx in (-1, 0, 1) for dy in (-1, 0, 1) for x in cells.get((ci + dx, cj + dy), ())]
            for a_ in grp:
                for b_ in near:
                    if a_ < b_:
                        lim = max(5.0, 0.5 * max(pt[a_][2], pt[b_][2]))
                        if float(haversine_km(pt[a_][0], pt[a_][1], pt[b_][0], pt[b_][1])) <= lim:
                            pairs.append((a_, b_))
    # a province and its capital are not one point (Bullinger's Solothurn canton and city, linked through propagated copies of the name): only places of a compatible kind
    kind = dict(zip(d['id'], d['place_kind'])) if 'place_kind' in d else {}
    known = ('area', 'city')

    # ... except an administrative area named after its seat ('Zerbst', 'Anhalt-Zerbst, Fürstentum'; 'Ballenstedt(-Hoym), Amt'), which is the seat's point by construction
    seat_areas = set(pl.loc[pl['name_qtype'].isin(POL.AREA_QTYPES), 'global_id']) if 'name_qtype' in pl else set()

    def compat(a_, b_):
        ka, kb = kind.get(a_), kind.get(b_)
        return a_ in seat_areas or b_ in seat_areas or not (ka in known and kb in known and ka != kb)
    return [(a_, b_) for a_, b_ in pairs if compat(a_, b_)]


MERGE_KM_V23 = 20.0


# ---------------------------------------------------------------------------------------------------------------------
def report_numbers(pl, pe, pg, pgp, lg) -> dict:
    r: dict = {}
    live = pl['n_mentions'] > 0
    inp = pl['n_in_period'] > 0
    placed = pl['coord_source_geo'] != 'none'
    r['places_local'] = {
        'rows': int(len(pl)), 'rows_with_default_mentions': int(live.sum()), 'distinct_in_period': int(inp.sum()),
        'mentions_in_period': int(pl['n_in_period'].sum()), 'mentions_in_period_all': int(pl['n_in_period_all'].sum()),
        'placed_distinct_in_period': int((inp & placed).sum()), 'placed_mentions_in_period': int(pl.loc[placed, 'n_in_period'].sum()),
        'with_qid_distinct_in_period': int((inp & pl['qid'].notna()).sum()),
        'with_qid_mentions_in_period': int(pl.loc[pl['qid'].notna(), 'n_in_period'].sum()),
        'qid_conflict': int(pl['qid_conflict'].sum()), 'qid_suggested_only': int((pl['qid'].isna() & pl['qid_suggested'].notna()).sum()),
        'coord_conflicts_50km': int(pl['coord_conflict'].sum()), 'flag_100km': int(pl['flag_100km'].sum()),
        'register_area_points_dropped': int(pl['reg_area_dropped'].sum()),
        'authority_coordinate_unused_identity_kept': int((pl['auth_rejected'] & ~pl['identity_dropped']).sum()),
        'authority_identity_dropped_by_arbitration': int(pl['identity_dropped'].sum()),
    }
    for s in ('register', 'wikidata', 'geonames', 'tgn', 'casebooks_directory', 'name_propagation', 'name_geonames', 'name_exonym', 'name_latin', 'none'):
        m = pl['coord_source_geo'] == s
        r['places_local'][f'source_{s}_distinct_in_period'] = int((inp & m).sum())
        r['places_local'][f'source_{s}_mentions_in_period'] = int(pl.loc[m, 'n_in_period'].sum())
    r['places_local']['by_kind_placed_mentions_in_period'] = {str(k): int(v) for k, v in
                                                              pl[placed].groupby(pl['place_kind_final'].fillna('unknown'))['n_in_period'].sum().items()}
    r['places_local']['by_class_mentions_in_period'] = {str(k): int(v) for k, v in
                                                        pl[pl['place_class'].notna()].groupby('place_class')['n_in_period'].sum().items()}
    gi = pg['n_in_period'] > 0
    r['places_global'] = {
        'rows': int(len(pg)), 'rows_in_period': int(gi.sum()), 'authority_clusters': int((pg['scope'] == 'authority').sum()),
        'with_qid': int(pg['qid'].notna().sum()), 'with_coords': int(pg['lat'].notna().sum()),
        'with_coords_in_period': int((pg['lat'].notna() & gi).sum()),
        'mentions_in_period_with_coords': int(pg.loc[pg['lat'].notna(), 'n_in_period'].sum()),
        'mentions_in_period_total': int(pg['n_in_period'].sum()),
        'coord_source_in_period': {k: int(v) for k, v in pg[gi]['coord_source'].value_counts().items()},
        'kind_in_period': {str(k): int(v) for k, v in pg[gi]['place_kind'].fillna('unknown').value_counts().items()},
        'coord_conflict': int(pg['coord_conflict'].sum()), 'flag_100km': int(pg['flag_100km'].sum()),
    }
    live_p = pe['n_mentions'] > 0
    ip = pgp['n_in_period'] > 0
    idc = [c for c in pe.columns if c.startswith('id_')]
    r['persons_local'] = {'rows': int(len(pe)), 'live': int(live_p.sum()), 'with_accepted_qid': int(pe['qid'].notna().sum()),
                          'with_any_accepted_id': int((pe['qid'].notna() | pe[idc].notna().any(axis=1)).sum()),
                          'live_with_accepted_qid': int((live_p & pe['qid'].notna()).sum()),
                          'in_period_with_accepted_qid': int(((pe['n_in_period'] > 0) & pe['qid'].notna()).sum()),
                          'mentions_in_period_with_accepted_qid': int(pe.loc[pe['qid'].notna(), 'n_in_period'].sum()),
                          'with_suggested_qid_only': int((pe['qid'].isna() & pe['qid_suggested'].notna()).sum()),
                          'qid_conflict': int(pe['qid_conflict'].sum())}
    r['persons_global'] = {
        'rows': int(len(pgp)), 'rows_in_period': int(ip.sum()), 'authority_clusters': int((pgp['scope'] == 'authority').sum()),
        'with_qid': int(pgp['qid'].notna().sum()), 'with_qid_mentions_in_period': int(pgp.loc[pgp['qid'].notna(), 'n_in_period'].sum()),
        'mentions_in_period_total': int(pgp['n_in_period'].sum()),
        'non_human_qid': int((pgp['wd_human'] == 'False').sum()),
        'by_category': {str(k): int(v) for k, v in pgp['wd_category'].fillna('none').value_counts().items()},
        'by_wd_person_class': {str(k): int(v) for k, v in pgp['wd_person_class'].fillna('none').value_counts().items()},
        'by_person_class': {str(k): int(v) for k, v in pgp['person_class'].fillna('human').value_counts().items()},
        'by_person_class_final': {str(k): int(v) for k, v in pgp['person_class_final'].value_counts().items()},
    }
    r['gate_log_rows'] = int(len(lg))
    return r


def point_kind5(base, prows, evidence) -> str:
    """Five-way kind of a globe point (``placekind.kind5``) from the member places of the point: the English labels of the Wikidata classes of each member
    (``classes`` = 'Q123:label|Q456:label'), the GeoNames feature codes (of the accepted ids and of name-based matches), the TGN types, the register
    types, and the exonym table's own kind for a curated feature ('Alps', 'Thermopylae')."""
    labels, gnf, tgt, notes, exo = [], [], [], [], None
    for ev in evidence:
        if ev is not None and 'kind5_override' in ev:                 # v2.3: a kind corrected by hand (data/kind_overrides.tsv) is final
            ko = [x for x in ev['kind5_override'] if isinstance(x, str) and x in PK.KINDS]
            if ko:
                return ko[0]
    for r, ev in zip(prows, evidence):
        c = r.get('classes') if hasattr(r, 'get') else None
        if isinstance(c, str):
            labels += [x.split(':', 1)[1] for x in c.split('|') if ':' in x and x.split(':', 1)[1]]
        cand = r.get('coord_cand') if hasattr(r, 'get') else None
        if isinstance(cand, str) and cand.startswith('exonym:'):
            exo = EX.FEATURE_KIND.get(cand[len('exonym:'):], exo)
        if ev is not None:
            gnf += [x for x in list(ev.get('gn_fcode', [])) + list(ev.get('gc_fcode', [])) if isinstance(x, str)]
            tgt += [x for x in ev.get('tgn_type', []) if isinstance(x, str)]
            notes += [x for x in ev.get('note_type', []) if isinstance(x, str)]
    names = [r.get('name') for r in prows if hasattr(r, 'get') and isinstance(r.get('name'), str)]
    return PK.kind5(base, list(dict.fromkeys(labels)), gnf, tgt, notes, exo, names)


def write_globe(pg: pd.DataFrame, coll: pd.DataFrame, path: Path, rep: dict, pl: pd.DataFrame, max_bytes: int = 12_000_000) -> dict:
    """Same format as v1.1 plus ``k`` (city / area / other) and ``cq`` ({collection: [10 quarter counts]}) per point."""
    d = pg[pg['lat'].notna() & (pg['n_in_period'] >= 1)].copy()
    d = d.sort_values(['n_in_period', 'id'], ascending=[False, True]).reset_index(drop=True)
    qarr = d[QCOLS].to_numpy()
    by_gid: dict = {}
    for i, g in enumerate(d['globe_id'].tolist()):
        by_gid.setdefault(g, []).append(i)
    # per-collection quarters of the live locals, by global place id
    Lq = pl[(pl['n_mentions'] > 0) & pl['global_id'].isin(set(d['id']))]
    evid = {gid: g for gid, g in Lq[[c for c in ('global_id', 'gn_fcode', 'gc_fcode', 'tgn_type', 'note_type', 'kind5_override') if c in Lq]].groupby('global_id')}
    ncoll = Lq.groupby('global_id')['coll_slug'].agg(lambda x: set(x))
    cqs: dict = {}
    for (gid, c), g in Lq.groupby(['global_id', 'coll_slug']):
        a = g[QCOLS].to_numpy().sum(axis=0)
        if a.sum() > 0:
            cqs.setdefault(gid, {})[c] = [int(x) for x in a]
    items = []
    for gid, mem in by_gid.items():
        lead = next((i for i in mem if d.at[i, 'coord_source'] not in FREE_SOURCES), mem[0])
        r = d.iloc[lead]
        nm = r['name'] if isinstance(r['name'], str) and r['name'] else ''
        if bool(r.get('name_is_generic')):
            nm = ''                                  # v2.2: a common noun with no authority behind it is not a name
        if str(r['coord_cand']).startswith('exonym:'):
            nm = str(r['coord_cand'])[len('exonym:'):]
        q = qarr[mem].sum(axis=0)
        cnt: Counter = Counter()
        cq: dict = {}
        allc: set = set()
        for i in mem:
            allc |= ncoll.get(d.at[i, 'id'], set())
            for c, k in zip(d.at[i, 'top_collections'], d.at[i, 'top_collection_counts']):
                cnt[c] += int(k)
            for c, a in cqs.get(d.at[i, 'id'], {}).items():
                cq[c] = [x + y for x, y in zip(cq[c], a)] if c in cq else list(a)
        kinds = [d.at[i, 'place_kind'] for i in mem if isinstance(d.at[i, 'place_kind'], str)]
        kind = r['place_kind'] if isinstance(r['place_kind'], str) else (Counter(kinds).most_common(1)[0][0] if kinds else None)
        if r['coord_kind'] == 'area' and kind in (None, 'city'):
            kind = 'area'
        kind = point_kind5(kind, [d.iloc[i] for i in mem], [evid.get(d.at[i, 'id']) for i in mem])      # v2.2: c a n b u
        qid = next((d.at[i, 'qid'] for i in mem if isinstance(d.at[i, 'qid'], str)), None)
        it = {'id': gid, 'name': nm[:60], 'lat': round(float(r['lat']), 4), 'lon': round(float(r['lon']), 4),
              'n': int(d.loc[mem, 'n_in_period'].sum()), 'q': [int(x) for x in q], 'c': int(len(allc) or 1),
              'cols': [c for c, _ in cnt.most_common(3)], 'qid': qid,
              't': int(d.loc[mem, 'tier_worst'].max()), 's': SRC_CODE.get(r['coord_source'], str(r['coord_source'])[0]),
              'k': kind, 'cq': cq}
        if r['coord_kind'] == 'area' and pd.notna(r['coord_radius_km']):
            it['r'] = int(r['coord_radius_km'])
        if len(mem) > 1:
            it['m'] = len(mem)
        items.append(it)
    items.sort(key=lambda x: (-x['n'], x['id']))
    used = Counter()
    for it in items:
        for c in it['cols']:
            used[c] += 1
    cl = coll.sort_values('slug')
    header = {
        'title': 'Renascor Atlas, places tagged by editors (v2.3: gated authority ids, atlas counts, Casebooks directory, local TGN, name-based coordinates of the rest, Latin names, historical polities)',
        'generated': time.strftime('%Y-%m-%d'), 'period': [1450, 1700], 'quarters': [1450 + 25 * i for i in range(10)],
        'fields': {'n': 'in-period mentions (atlas counts: commentary, footnotes, regests, repositories, characters, placeholders and celestial bodies left out)',
                   'q': 'mentions per quarter-century by document year (sum can be below n: documents dated across a boundary)',
                   'cq': 'the same per collection: {collection: [10 counts]}', 'c': 'collections',
                   'cols': 'top 3 collections', 'k': 'kind of place (v2.2, five-way): c = city, town, village, city quarter; a = country, region, territory; '
                                                   'n = river, lake, sea, island, mountain, valley, pass, cape, forest; b = building, castle, monastery, street, square, gate, bridge, site; '
                                                   'u = other. From the Wikidata classes of the accepted item, the GeoNames feature code, the TGN place type and the register type',
                   'r': 'for areas: radius in km of the coarse point', 't': 'worst licence tier among all contributing collections (1=T1, 2=T2, 3=T3: internal use only)',
                   's': 'coordinate source: r=register, w=wikidata, g=geonames id, t=Getty TGN id, c=Casebooks directory, p=name propagation, '
                        'n=GeoNames / TGN name match, e=curated table of Latin / old exonyms and historical polities, l=Latin name de-inflected and found in a lexicon (Wikidata Latin labels, '
                        'Orbis Latinus, exonym tables, Latin editions); c, p, n, e and l are high / medium confidence only; p, n, e and l are located by name',
                   'm': 'number of authority / edition places merged into this point (same name within 20 km); absent = 1'},
        'attribution': ['Wikidata (CC0)', 'GeoNames, https://www.geonames.org (CC BY 4.0)',
                        'Getty Thesaurus of Geographic Names, The J. Paul Getty Trust (ODC-By 1.0)'],
        'coordinate_sources_note': 'p, n, e and l points are inferred from names, not from the editions; c points come from the curated Casebooks '
                                   'directory map; see places_local_geo.parquet (gc_* and cb_* columns).',
        'places': len(items), 'mentions': int(sum(x['n'] for x in items)),
        'collections': [{'slug': r['slug'], 'licence': r['licence'], 'tier': r['tier'], 'renascor_ids': r['renascor_ids'],
                         'places_in_top3': int(used.get(r['slug'], 0))} for _, r in cl.iterrows()],
        'trimmed': None,
    }

    def dump(its, hd):
        return json.dumps({'header': hd, 'places': its}, ensure_ascii=False, separators=(',', ':'))
    s = dump(items, header)
    if len(s.encode('utf-8')) > max_bytes:
        header['trimmed'] = 'per-collection quarters only for the 10000 largest points'
        for it in items[10000:]:
            it['cq'] = {}
        s = dump(items, header)
    if len(s.encode('utf-8')) > max_bytes:
        header['trimmed'] = 'only places with n>=2'
        items = [x for x in items if x['n'] >= 2]
        header['places'] = len(items)
        s = dump(items, header)
    path.write_text(s, 'utf-8')
    nb = len(s.encode('utf-8'))
    _t(f'globe: {len(items)} points ({len(d)} places merged), {nb} bytes')
    return {'places': len(items), 'bytes': nb, 'trimmed': header['trimmed'], 'mentions': int(sum(x['n'] for x in items)),
            'places_eligible': int(len(d)), 'merged_points': int(sum(1 for x in items if x.get('m', 1) > 1)),
            'by_kind': dict(Counter(x['k'] for x in items)), 'by_source': dict(Counter(x['s'] for x in items))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile build2')
    ap.add_argument('--v2', type=Path, required=True)
    ap.add_argument('--rdir', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--namecheck', type=Path, default=None)
    ap.add_argument('--geocode', type=Path, default=None)
    ap.add_argument('--geocode-verify', type=Path, default=None, dest='verify')
    ap.add_argument('--tgn-dir', type=Path, default=None, help='local Getty TGN index (tgn_places.parquet, tgn_names.parquet)')
    ap.add_argument('--casebooks-map', type=Path, default=None, help='casebooks_place_map.tsv (PLACE id -> directory entry)')
    a = ap.parse_args(argv)
    rep = build(a.v2, a.rdir, a.dest, a.namecheck, a.geocode, a.verify, a.tgn_dir, a.casebooks_map)
    print(json.dumps(rep, indent=1, default=_js)[:6000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
