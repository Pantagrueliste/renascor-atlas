"""Reconcile stage, part 3: coordinates, QIDs, global clusters and the globe file (run on Hypatia through Slurm).

    python -m ratlas reconcile build --v1 DIR --rdir DIR --dest DIR [--geonames FILE]

Inputs   v1/{mentions,entities,places_local,persons_local}.parquet, collections.tsv
         rdir/{ids,wd_map,wd_places,wd_persons}.tsv (from ``reconcile fetch``), geonames_lookup.tsv (optional)
Outputs  places_local_geo.parquet, persons_local_global.parquet, places_global.parquet, persons_global.parquet,
         globe_places.json, reconcile_report.json, unplaced_by_collection.tsv, unplaced_top_names.tsv,
         coord_conflicts.tsv

Rules
* Local (edition-scoped) place coordinates: register first (coord_source 'register'), then the Wikidata coordinates of
  the place's QID ('wikidata'), then the GeoNames coordinates of its GeoNames id ('geonames'), else none. A register
  coordinate more than 50 km from the authority coordinate is flagged (coord_conflict). With a ``namecheck_local.tsv``
  (``reconcile namecheck``: distance of each coordinate to a GeoNames feature of the SAME NAME) the conflict is
  arbitrated: the coordinate with a same-name feature within 30 km (else 100 km) wins; a rejected authority also
  loses its QID and ids (cavriana's TGN ids map to Illinois for Bordeaux), a rejected register coordinate is ignored
  (selbstzeugnisse_hab_diarium has copy errors). Ties keep the register. ``coord_overrides.tsv`` (rdir) corrects
  confirmed typos of a source register (Speyer 9.3 N instead of 49.3 N).
* Name-based geocoding (``--geocode geocode_places.parquet`` from ``reconcile geocode run``): a separate, last-resort
  coordinate source for places that have none (coord_source 'name_propagation' / 'name_geonames', confidence
  high / medium / low). Register, Wikidata and GeoNames-id coordinates are never overwritten. Only high / medium
  confidence fill ``lat`` / ``lon`` and the globe; a low-confidence candidate stays in the ``gc_*`` columns.
* Globe: places of the same name within 20 km are merged into one point (an edition-scoped register place and the
  Wikidata cluster of the same town used to give two points); ``globe_id`` in places_global.parquet.
* QID of a local entity: the QID with the largest weighted vote of the ids found on its mentions and register entry
  (explicit Wikidata id, else QIDs found through the identifier properties). Place QIDs of a class that is not a
  place (human, company, disambiguation page, taxon, work...) are dropped; Wikimedia duplicate items lose to the real one.
* Global clusters: same QID; else same GeoNames / TGN / Pleiades / GND / HLS / VIAF id (places) or GND / VIAF / ULAN /
  ISNI... id (persons), attached to an existing QID cluster when that id is carried by exactly one QID cluster. The
  rest stays edition-scoped (scope 'edition', one row per local entity).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

import re
import unicodedata

from .merge import QCOLS, _norm
from .reconcile_ids import PRIORITY, _long_ids, normalise_long

CONFLICT_KM = 50.0
REL = {'wikidata': 3.0, 'geonames': 2.0, 'tgn': 2.0, 'pleiades': 2.0, 'gnd': 2.0, 'hls': 1.5, 'viaf': 1.5, 'ulan': 1.5,
       'isni': 1.5, 'loc': 1.5, 'cerl': 1.5, 'odnb': 1.5, 'deutsche-biographie': 1.5, 'bnf': 1.5, 'wikipedia': 1.0}
PLACE_FALLBACK = ('geonames', 'tgn', 'pleiades', 'gnd', 'hls', 'viaf')
PERSON_FALLBACK = ('gnd', 'viaf', 'ulan', 'isni', 'loc', 'cerl', 'odnb', 'deutsche-biographie', 'bnf', 'hls')
HUMAN = 'Q5'
# Wikidata classes that can never be a place: a QID of one of them found through a GND / GeoNames / TGN id of a place
# is a wrong match (Genève -> Thomson Reuters (Switzerland), Corinthus -> disambiguation page, Tartessus -> a genus).
NON_PLACE_CLASSES = {'Q5', 'Q4167410', 'Q16521', 'Q783794', 'Q4830453', 'Q11424', 'Q7725634', 'Q571', 'Q13442814',
                     'Q101352', 'Q202444'}


def is_non_place(classes: str) -> bool:
    return any(c.split(':', 1)[0] in NON_PLACE_CLASSES for c in (classes or '').split('|') if c)
EXTRA_Q = ['pre1450', 'post1700', 'undated_year']
SUM_COLS = ['n_mentions', 'n_in_period', 'n_undated'] + EXTRA_Q + QCOLS + [
    'n_author', 'n_sender', 'n_addressee', 'n_sent_from', 'n_sent_to', 'n_signer', 'n_cast']
TIER_RANK = {'T1': 1, 'T2': 2, 'T3': 3}


def _t(msg):
    print(f'[build {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def valid_coord(lat, lon):
    lat = pd.to_numeric(lat, errors='coerce')
    lon = pd.to_numeric(lon, errors='coerce')
    ok = lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))
    return ok


# ----------------------------------------------------------------------------------------------------------------------
# Wikidata tables
# ----------------------------------------------------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)


def load_wd(rdir: Path):
    m = _read(rdir / 'wd_map.tsv')
    mapping = pd.DataFrame({'scheme': m['scheme'], 'id': m['id'], 'qid': m['qids'].str.split('|')}).explode('qid')
    mapping = mapping[mapping['qid'].notna() & (mapping['qid'] != '')]
    mapping['k'] = mapping.groupby(['scheme', 'id'])['qid'].transform('size')
    p = _read(rdir / 'wd_places.tsv')
    p['lat'] = pd.to_numeric(p['lat'], errors='coerce')
    p['lon'] = pd.to_numeric(p['lon'], errors='coerce')
    p['human'] = p['classes'].map(lambda s: any(c.split(':', 1)[0] == HUMAN for c in s.split('|') if c))
    q = _read(rdir / 'wd_persons.tsv')
    q['human'] = q['classes'].map(lambda s: any(c.split(':', 1)[0] == HUMAN for c in s.split('|') if c))
    return mapping, p, q


def load_geonames(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=['geonameid', 'name', 'latitude', 'longitude', 'feature_class', 'feature_code',
                                     'country_code', 'population'])
    g = pd.read_csv(path, sep='\t', dtype=str, comment='#', keep_default_na=False)
    g['latitude'] = pd.to_numeric(g['latitude'], errors='coerce')
    g['longitude'] = pd.to_numeric(g['longitude'], errors='coerce')
    return g


# ----------------------------------------------------------------------------------------------------------------------
# id votes per local entity
# ----------------------------------------------------------------------------------------------------------------------
def id_votes(v1: Path, kinds=('person', 'place')) -> pd.DataFrame:
    """(coll_slug, kind, entity_key, scheme, id) -> n, n_ip; entity_key as in merge.local_table."""
    t = pq.read_table(v1 / 'mentions.parquet', columns=['coll_slug', 'kind', 'canonical', 'local_key', 'surface',
                                                        'in_period', 'auth'])
    t = t.filter(pc.and_(pc.equal(t.column('canonical'), True), pc.is_in(t.column('kind'), pa.array(list(kinds)))))
    _t(f'mentions {t.num_rows}')
    lk = t.column('local_key').to_pandas()
    surf = t.column('surface').to_pandas()
    uniq = surf.fillna('').unique()
    nm = {s: _norm(s) for s in uniq}
    n = surf.fillna('').map(nm)
    key = np.where(lk.notna(), lk, 'surf:' + n)
    keep = (lk.notna() | (n != '')).to_numpy()
    t = t.append_column('entity_key', pa.array(key, type=pa.string()))
    t = t.append_column('ip', pc.fill_null(pc.equal(t.column('in_period'), True), False))
    t = t.filter(pa.array(keep))
    d = normalise_long(_long_ids(t, {'coll_slug': None, 'kind': None, 'entity_key': None, 'ip': None}))
    g = d.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id']).agg(
        n=('ip', 'size'), n_ip=('ip', 'sum')).reset_index()
    _t(f'mention id votes: {len(g)} rows')
    e = pq.read_table(v1 / 'entities.parquet', columns=['coll_slug', 'kind', 'local_key', 'auth'])
    e = e.filter(pc.is_in(e.column('kind'), pa.array(list(kinds))))
    e = e.append_column('entity_key', e.column('local_key'))
    de = normalise_long(_long_ids(e, {'coll_slug': None, 'kind': None, 'entity_key': None}))
    ge = de.drop_duplicates().assign(n=0, n_ip=0)
    out = pd.concat([g, ge[g.columns]], ignore_index=True)
    out = out.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id'], as_index=False).agg(
        n=('n', 'sum'), n_ip=('n_ip', 'sum'))
    _t(f'id votes incl. register: {len(out)} rows')
    return out


DUP_CLASSES = ('Q17362920', 'Q21286738')       # Wikimedia duplicated page / permanent duplicate item


def resolve_qids(votes: pd.DataFrame, mapping: pd.DataFrame, non_place_qids: set, kind: str,
                 dup_qids: Optional[set] = None) -> pd.DataFrame:
    """One row per local entity with its dominant QID: coll_slug, entity_key, qid, qid_via, qid_n_cand, qid_score."""
    v = votes[votes['kind'] == kind]
    rel = v['scheme'].map(REL).fillna(1.0)
    v = v.assign(w=(v['n'] + 0.01) * rel)
    ex = v[v['scheme'] == 'wikidata'].assign(qid=lambda d: d['id'])[['coll_slug', 'entity_key', 'scheme', 'qid', 'w']]
    mp = v[v['scheme'] != 'wikidata'].merge(mapping, on=['scheme', 'id'], how='inner')
    mp = mp.assign(w=mp['w'] / mp['k'])[['coll_slug', 'entity_key', 'scheme', 'qid', 'w']]
    c = pd.concat([ex, mp], ignore_index=True)
    if dup_qids:                                # keep a duplicate item only when it is the sole candidate
        c = c.assign(w=np.where(c['qid'].isin(dup_qids), c['w'] * 0.01, c['w']))
    if kind == 'place' and non_place_qids:
        c = c[~c['qid'].isin(non_place_qids)]
    if c.empty:
        return pd.DataFrame(columns=['coll_slug', 'entity_key', 'qid', 'qid_via', 'qid_n_cand', 'qid_score'])
    c['qnum'] = c['qid'].str[1:].astype('int64')
    g = c.groupby(['coll_slug', 'entity_key', 'qid', 'qnum'], as_index=False).agg(
        qid_score=('w', 'sum'), qid_via=('scheme', lambda s: '+'.join(sorted(set(s)))))
    g['qid_n_cand'] = g.groupby(['coll_slug', 'entity_key'])['qid'].transform('size')
    g = g.sort_values(['coll_slug', 'entity_key', 'qid_score', 'qnum'], ascending=[True, True, False, True])
    g = g.drop_duplicates(['coll_slug', 'entity_key'])
    return g[['coll_slug', 'entity_key', 'qid', 'qid_via', 'qid_n_cand', 'qid_score']]


def dominant_ids(votes: pd.DataFrame, kind: str, schemes) -> pd.DataFrame:
    """Per local entity and scheme, the id with most mentions -> wide table id_<scheme>."""
    v = votes[(votes['kind'] == kind) & votes['scheme'].isin(schemes)]
    v = v.sort_values(['coll_slug', 'entity_key', 'scheme', 'n', 'id'], ascending=[True, True, True, False, True])
    v = v.drop_duplicates(['coll_slug', 'entity_key', 'scheme'])
    w = v.pivot(index=['coll_slug', 'entity_key'], columns='scheme', values='id')
    w.columns = ['id_' + c for c in w.columns]
    for s in schemes:
        if 'id_' + s not in w:
            w['id_' + s] = None
    return w.reset_index()


# ----------------------------------------------------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------------------------------------------------
def assign_clusters(loc: pd.DataFrame, fallback: tuple[str, ...]) -> pd.DataFrame:
    """loc has qid and id_<scheme> columns. Returns gid (authority key or None) and basis."""
    gid = pd.Series([None] * len(loc), index=loc.index, dtype=object)
    basis = pd.Series([None] * len(loc), index=loc.index, dtype=object)
    hq = loc['qid'].notna()
    gid[hq] = loc.loc[hq, 'qid']
    basis[hq] = 'wikidata'
    tok: dict = {}
    for s in fallback:
        col = 'id_' + s
        sub = loc.loc[hq & loc[col].notna(), [col, 'qid']].drop_duplicates()
        cnt = sub.groupby(col)['qid'].nunique()
        uniq = cnt[cnt == 1].index
        for i, q in sub[sub[col].isin(uniq)].itertuples(index=False):
            tok[(s, i)] = q
    rest = loc.index[~hq]
    cols = {s: loc['id_' + s] for s in fallback}
    for ix in rest:
        for s in fallback:
            i = cols[s].at[ix]
            if isinstance(i, str):
                q = tok.get((s, i))
                if q:
                    gid.at[ix], basis.at[ix] = q, 'wikidata'
                else:
                    gid.at[ix], basis.at[ix] = f'{s}:{i}', s
                break
    return pd.DataFrame({'gid': gid, 'basis': basis})


def edition_id(pref: str, coll: str, key: str) -> str:
    return f'{pref}:e:{coll}:{hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]}'


# ----------------------------------------------------------------------------------------------------------------------
def _blank(s: pd.Series) -> pd.Series:
    s = s.astype(object)
    return s.where(s.notna() & (s != ''), None)


def dominant(df: pd.DataFrame, col: str) -> pd.Series:
    """Per global_id, the most frequent non-null value of ``col`` (ties: smallest), vectorised."""
    d = df[['global_id', col]].dropna()
    c = d.groupby(['global_id', col], sort=False).size().rename('_n').reset_index()
    c = c.sort_values(['global_id', '_n', col], ascending=[True, False, True]).drop_duplicates('global_id')
    return c.set_index('global_id')[col]


def _first_nonnull_mode(s: pd.Series):
    s = s.dropna()
    return s.value_counts().index[0] if len(s) else None


def aggregate_global(loc: pd.DataFrame, pref: str, sum_cols=None) -> pd.DataFrame:
    """loc has global_id, scope, coll_slug, renascor_id, name, variants, tier, licence + SUM_COLS. -> one row per global."""
    g = loc.groupby('global_id', sort=False)
    out = g[list(sum_cols or SUM_COLS)].sum()
    out['n_locals'] = g.size()
    out['n_collections'] = g['coll_slug'].nunique()
    out['scope'] = g['scope'].first()
    out['basis'] = g['basis'].first()
    # editions
    ed = loc[['global_id', 'renascor_id']].dropna().copy()
    ed['renascor_id'] = ed['renascor_id'].str.split(';')
    ed = ed.explode('renascor_id')
    out['n_editions'] = ed.groupby('global_id')['renascor_id'].nunique().reindex(out.index).fillna(0).astype('int64')
    # top collections
    cc = loc.groupby(['global_id', 'coll_slug'], as_index=False)['n_mentions'].sum()
    cc = cc.sort_values(['global_id', 'n_mentions', 'coll_slug'], ascending=[True, False, True])
    top = cc.groupby('global_id').head(3)
    out['top_collections'] = top.groupby('global_id')['coll_slug'].apply(list).reindex(out.index)
    out['top_collection_counts'] = top.groupby('global_id')['n_mentions'].apply(lambda s: [int(x) for x in s]).reindex(out.index)
    # name = most frequent first variant; v2.2: a common noun (alhier, Stadt, Vater, this towne) is never preferred to a real name
    named = loc[loc['name'].notna() & (loc['name'] != '')]
    gen = named['generic_name'].fillna(False).astype(bool) if 'generic_name' in named else pd.Series(False, index=named.index)
    nm = named.assign(_g=gen.astype('int64')).groupby(['global_id', 'name', '_g'], as_index=False)['n_mentions'].sum()
    nm = nm.sort_values(['global_id', '_g', 'n_mentions', 'name'], ascending=[True, True, False, True]).drop_duplicates('global_id')
    out['name'] = nm.set_index('global_id')['name'].reindex(out.index)
    out['name_is_generic'] = nm.set_index('global_id')['_g'].reindex(out.index).fillna(0).astype(bool)
    # variants seen
    vv = loc[['global_id', 'variants', 'n_mentions']].copy()
    vv = vv.explode('variants').dropna(subset=['variants'])
    vv['_k'] = vv['variants'].str.casefold()
    vv = vv.sort_values(['global_id', 'n_mentions'], ascending=[True, False]).drop_duplicates(['global_id', '_k'])
    out['variants_seen'] = vv.groupby('global_id').head(12).groupby('global_id')['variants'].apply(list).reindex(out.index)
    out['variants_seen'] = out['variants_seen'].map(lambda v: v if isinstance(v, list) else [])
    # licence / tier
    t = loc[['global_id', 'tier', 'licence']].dropna(subset=['tier']).drop_duplicates()
    out['tiers'] = t.groupby('global_id')['tier'].apply(lambda s: ','.join(sorted(set(s)))).reindex(out.index)
    out['licences'] = t.groupby('global_id')['licence'].apply(lambda s: ';'.join(sorted(set(x for x in s if x)))).reindex(out.index)
    out['tier_worst'] = out['tiers'].map(lambda s: max((TIER_RANK[x] for x in s.split(',') if x in TIER_RANK), default=0)
                                          if isinstance(s, str) else 0)
    return out


# ----------------------------------------------------------------------------------------------------------------------
def load_overrides(rdir: Path) -> pd.DataFrame:
    """``coord_overrides.tsv``: a register coordinate corrected by hand. v2.2: ``lat`` = ``-`` drops the register's coordinate (``ov_drop``): the register
    point was derived from a wrong authority id (Bullinger's Gärnitz, Longueville, Neuburg) and the place goes to the geocoder."""
    f = Path(rdir) / 'coord_overrides.tsv'
    if not f.exists():
        return pd.DataFrame(columns=['coll_slug', 'entity_key', 'ov_lat', 'ov_lon', 'ov_drop'])
    from .manual_overrides import read_tsv          # comment lines start with '#'; an entity key may contain one (Religionsfrieden's 'file#brueggen')
    o = read_tsv(f)
    o['ov_lat'] = pd.to_numeric(o['lat'], errors='coerce')
    o['ov_lon'] = pd.to_numeric(o['lon'], errors='coerce')
    o['ov_drop'] = o['lat'].str.strip().eq('-')
    o = o[valid_coord(o['ov_lat'], o['ov_lon']) | o['ov_drop']]
    return o[['coll_slug', 'entity_key', 'ov_lat', 'ov_lon', 'ov_drop']].drop_duplicates(['coll_slug', 'entity_key'])


def load_namecheck(rdir: Path) -> Optional[pd.DataFrame]:
    f = Path(rdir) / 'namecheck_local.tsv'
    if not f.exists():
        _t('no namecheck_local.tsv: register/authority conflicts are not arbitrated')
        return None
    n = pd.read_csv(f, sep='\t', dtype={'coll_slug': str, 'entity_key': str})
    return n.drop_duplicates(['coll_slug', 'entity_key']).set_index(['coll_slug', 'entity_key'])


def _auth_and_conflict(pl: pd.DataFrame) -> None:
    pl['auth_lat'] = np.where(pl['has_wd'], pl['wd_lat'], pl['gn_lat'])
    pl['auth_lon'] = np.where(pl['has_wd'], pl['wd_lon'], pl['gn_lon'])
    pl['auth_source'] = np.where(pl['has_wd'], 'wikidata', np.where(pl['has_gn'], 'geonames', None))
    both = pl['has_reg'] & (pl['has_wd'] | pl['has_gn'])
    pl['coord_conflict_km'] = np.nan
    pl.loc[both, 'coord_conflict_km'] = haversine_km(pl.loc[both, 'reg_lat'], pl.loc[both, 'reg_lon'],
                                                     pl.loc[both, 'auth_lat'], pl.loc[both, 'auth_lon'])
    pl['coord_conflict'] = pl['coord_conflict_km'] > CONFLICT_KM


def _score(d: pd.Series) -> np.ndarray:
    """Name check distance -> 3 (<=30 km), 2 (<=100 km), 1 (no same-name GeoNames feature), 0 (far), -1 (no coordinate)."""
    d = pd.to_numeric(d, errors='coerce').to_numpy(dtype='float64')
    return np.where(np.isnan(d), -1, np.where(d < 0, 1, np.where(d <= 30, 3, np.where(d <= 100, 2, 0))))


def arbitrate(pl: pd.DataFrame, nc: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Register-vs-authority conflicts: keep the coordinate with a GeoNames feature of the same name nearby."""
    pl['namecheck_reg_km'] = np.nan
    pl['namecheck_auth_km'] = np.nan
    pl['arbitration'] = None
    pl['auth_rejected'] = False
    pl['reg_rejected'] = False
    if nc is None or not len(nc):
        return pl
    key = pd.MultiIndex.from_arrays([pl['coll_slug'], pl['entity_key']])
    d = nc.reindex(key)
    pl['namecheck_reg_km'] = d['d_reg'].to_numpy()
    pl['namecheck_auth_km'] = d['d_auth'].to_numpy()
    sr, sa = _score(pl['namecheck_reg_km']), _score(pl['namecheck_auth_km'])
    conf = pl['coord_conflict'].to_numpy()
    rej_auth = conf & (sr > sa)
    rej_reg = conf & (sa > sr)
    pl.loc[rej_auth, 'arbitration'] = 'register'
    pl.loc[rej_reg, 'arbitration'] = 'authority'
    pl['auth_rejected'] = rej_auth
    pl['reg_rejected'] = rej_reg
    pl.loc[rej_auth, ['has_wd', 'has_gn']] = False
    pl.loc[rej_reg, 'has_reg'] = False
    # a rejected authority must not drag the place into the cluster of the wrong QID / id
    drop = ['qid', 'qid_via'] + [c for c in pl.columns if c.startswith('id_')]
    pl.loc[rej_auth, drop] = None
    _t(f'arbitration: {int(rej_auth.sum())} authority coordinates rejected, {int(rej_reg.sum())} register coordinates rejected, '
       f'{int((conf & ~rej_auth & ~rej_reg).sum())} conflicts kept')
    return pl


NAME_SOURCES = ('name_propagation', 'name_geonames', 'name_exonym', 'name_latin')      # v2.3: name_latin = a Latin name de-inflected and found in a lexicon
GC_COLS = ['gc_source', 'gc_conf', 'gc_lat', 'gc_lon', 'gc_radius_km', 'gc_cand', 'gc_cand_name', 'gc_key', 'gc_tier', 'gc_p',
           'gc_evidence', 'gc_pass', 'gc_via', 'gc_agree', 'gc_conflict_km', 'gc_reason', 'gc_tgn', 'gc_qid']
SRC_CODE = {'register': 'r', 'wikidata': 'w', 'geonames': 'g', 'name_propagation': 'p', 'name_geonames': 'n', 'name_exonym': 'e', 'name_latin': 'l', 'tgn': 't',
            'casebooks_directory': 'c', 'curated': 'k'}


def apply_geocode(pl: pd.DataFrame, path: Optional[Path], verify: Optional[Path] = None) -> pd.DataFrame:
    """Last-resort coordinates from name-based geocoding; never touches a place that already has coordinates.

    ``verify``: geocode_verify.tsv (``reconcile verify rules``): high / medium results that the audit takes back; they stay in
    the ``gc_*`` columns as low-confidence candidates (``gc_reason`` = ``verify_demoted``, ``verify_rule`` = the rule).
    """
    pl['coord_source_pre'] = pl['coord_source_geo']
    pl['verify_rule'] = None
    if path is None or not Path(path).exists():
        for c in GC_COLS:
            pl[c] = None
        pl['coord_conf'] = np.where(pl['coord_source_geo'] != 'none', 'authority', None)
        pl['coord_radius_km'] = np.nan
        return pl
    gc = pd.read_parquet(path).drop_duplicates(['coll_slug', 'entity_key'])
    cols = ['coll_slug', 'entity_key'] + [c for c in GC_COLS if c in gc.columns]
    out = pl.merge(gc[cols], how='left', on=['coll_slug', 'entity_key'], validate='m:1')
    for c in GC_COLS:
        if c not in out:
            out[c] = None
    if verify is not None and Path(verify).exists():
        v = pd.read_csv(verify, sep='\t', dtype=str, keep_default_na=False).drop_duplicates(['coll_slug', 'entity_key'])
        vr = dict(zip(zip(v['coll_slug'], v['entity_key']), v['rule']))
        rr = pd.Series([vr.get(k) for k in zip(out['coll_slug'], out['entity_key'])], index=out.index, dtype='object')
        hit = rr.notna() & (out['coord_source_geo'] == 'none') & out['gc_conf'].isin(['high', 'medium'])
        out.loc[hit, 'verify_rule'] = rr[hit]
        out.loc[hit, 'gc_conf'] = 'low'
        out.loc[hit, 'gc_reason'] = 'verify_demoted'
        _t(f'audit: {int(hit.sum())} high / medium name-based results taken back')
    free = (out['coord_source_geo'] == 'none') & out['gc_conf'].isin(['high', 'medium'])
    out.loc[free, 'coord_source_geo'] = out.loc[free, 'gc_source']
    out.loc[free, 'geo_lat'] = out.loc[free, 'gc_lat']
    out.loc[free, 'geo_lon'] = out.loc[free, 'gc_lon']
    # a result for a place that already had coordinates is ignored (and not reported)
    out.loc[out['coord_source_pre'] != 'none', GC_COLS] = None
    out['coord_conf'] = np.where(out['coord_source_pre'] != 'none', 'authority', np.where(free, out['gc_conf'], None))
    out['coord_radius_km'] = np.where(free, pd.to_numeric(out['gc_radius_km'], errors='coerce'), np.nan)
    _t(f'name-based geocoding: {int(free.sum())} places placed (high/medium), '
       f'{int(((out["coord_source_pre"] == "none") & (out["gc_conf"] == "low")).sum())} low-confidence candidates kept apart')
    return out


def _final_coords(pl: pd.DataFrame) -> None:
    pl['coord_source_geo'] = np.where(pl['has_reg'], 'register', np.where(pl['has_wd'], 'wikidata',
                                      np.where(pl['has_gn'], 'geonames', 'none')))
    pl['geo_lat'] = np.where(pl['has_reg'], pl['reg_lat'], np.where(pl['has_wd'], pl['wd_lat'], pl['gn_lat']))
    pl['geo_lon'] = np.where(pl['has_reg'], pl['reg_lon'], np.where(pl['has_wd'], pl['wd_lon'], pl['gn_lon']))
    pl.loc[pl['coord_source_geo'] == 'none', ['geo_lat', 'geo_lon']] = np.nan


def build(v1: Path, rdir: Path, dest: Path, geonames: Optional[Path] = None, geocode: Optional[Path] = None,
          verify: Optional[Path] = None) -> dict:
    t0 = time.time()
    dest.mkdir(parents=True, exist_ok=True)
    mapping, wdp, wdq = load_wd(rdir)
    gn = load_geonames(geonames)
    _t(f'wd map {len(mapping)} rows, wd places {len(wdp)}, wd persons {len(wdq)}, geonames {len(gn)}')
    votes = id_votes(v1)
    human_places = set(wdp.loc[wdp['classes'].map(is_non_place), 'qid'])      # human or otherwise not a place

    rep: dict = {'generated': time.strftime('%Y-%m-%d %H:%M:%S')}
    coll = pd.read_csv(v1 / 'collections.tsv', sep='\t', usecols=['slug', 'licence', 'tier', 'renascor_ids'])

    # ================================================================== places
    pl = pq.read_table(v1 / 'places_local.parquet').to_pandas()
    pl['scope'] = None
    dup_qids = set(wdp.loc[wdp['classes'].map(lambda c: any(x.split(':', 1)[0] in DUP_CLASSES for x in c.split('|') if x)), 'qid'])
    rq = resolve_qids(votes, mapping, human_places, 'place', dup_qids)
    dom = dominant_ids(votes, 'place', PLACE_FALLBACK + ('wikipedia',))
    pl = pl.merge(rq, how='left', on=['coll_slug', 'entity_key']).merge(dom, how='left', on=['coll_slug', 'entity_key'])
    for s in PLACE_FALLBACK:
        if 'id_' + s not in pl:
            pl['id_' + s] = None
    _t(f'places local {len(pl)}; with QID {pl["qid"].notna().sum()}')

    # --- coordinates
    reg_ok = pl['lat'].notna() & pl['lon'].notna() & valid_coord(pl['lat'], pl['lon'])
    pl['reg_lat'] = pl['lat'].where(reg_ok)
    pl['reg_lon'] = pl['lon'].where(reg_ok)
    pl['reg_source'] = pl['coord_source'].where(reg_ok)
    pl['reg_invalid'] = pl['lat'].notna() & ~reg_ok
    ov = load_overrides(rdir)
    if len(ov):
        ix = pl.merge(ov, how='left', on=['coll_slug', 'entity_key'])
        hit = ix['ov_lat'].notna().to_numpy()
        pl.loc[hit, 'reg_lat'] = ix.loc[hit, 'ov_lat'].to_numpy()
        pl.loc[hit, 'reg_lon'] = ix.loc[hit, 'ov_lon'].to_numpy()
        pl.loc[hit, 'reg_source'] = 'override'
        reg_ok = reg_ok | pd.Series(hit, index=pl.index)
        _t(f'coordinate overrides applied: {int(hit.sum())}')
    w = wdp.drop_duplicates('qid').set_index('qid')
    pl['wd_lat'] = pl['qid'].map(w['lat'])
    pl['wd_lon'] = pl['qid'].map(w['lon'])
    gn1 = gn.drop_duplicates('geonameid').set_index('geonameid') if len(gn) else gn
    pl['gn_lat'] = pl['id_geonames'].map(gn1['latitude']) if len(gn1) else np.nan
    pl['gn_lon'] = pl['id_geonames'].map(gn1['longitude']) if len(gn1) else np.nan
    pl['has_reg'] = pl['reg_lat'].notna()
    pl['has_wd'] = pl['wd_lat'].notna()
    pl['has_gn'] = pl['gn_lat'].notna()
    _auth_and_conflict(pl)
    pl = arbitrate(pl, load_namecheck(rdir))
    _final_coords(pl)
    pl = apply_geocode(pl, geocode, verify)
    pl['qid_conflict'] = (pl['qid_n_cand'].fillna(0) > 1) | pl['auth_rejected']

    # --- clusters
    pl = pl.reset_index(drop=True)
    cl = assign_clusters(pl, PLACE_FALLBACK)
    pl['global_id'] = np.where(cl['gid'].notna(), 'pl:' + cl['gid'].astype(str), None)
    pl['basis'] = cl['basis']
    pl['scope'] = np.where(cl['gid'].notna(), 'authority', 'edition')
    ed = pl['global_id'].isna()
    pl.loc[ed, 'global_id'] = [edition_id('pl', c, k) for c, k in zip(pl.loc[ed, 'coll_slug'], pl.loc[ed, 'entity_key'])]
    _t(f'place clusters: {pl["global_id"].nunique()} global ids for {len(pl)} locals')

    loc_cols = ['coll_slug', 'entity_key', 'renascor_id', 'name', 'norm_name', 'n_mentions', 'n_in_period', 'tier', 'licence',
                'lat', 'lon', 'coord_source', 'has_coords']
    geo = pl[loc_cols + ['qid', 'qid_via', 'qid_n_cand', 'qid_conflict', 'id_geonames', 'id_tgn', 'id_gnd',
                         'reg_invalid', 'coord_source_geo', 'geo_lat', 'geo_lon', 'auth_source', 'auth_lat', 'auth_lon',
                         'coord_conflict_km', 'coord_conflict', 'arbitration', 'namecheck_reg_km', 'namecheck_auth_km',
                         'auth_rejected', 'reg_rejected', 'global_id', 'basis', 'scope', 'coord_conf', 'coord_radius_km']
                    + GC_COLS + ['verify_rule']].rename(
        columns={'lat': 'register_lat', 'lon': 'register_lon', 'coord_source': 'register_coord_source',
                 'has_coords': 'register_has_coords', 'coord_source_geo': 'coord_source', 'geo_lat': 'lat', 'geo_lon': 'lon',
                 'id_geonames': 'geonames_id', 'id_tgn': 'tgn_id', 'id_gnd': 'gnd_id'})
    geo['has_coords'] = geo['lat'].notna()
    geo.to_parquet(dest / 'places_local_geo.parquet', index=False)

    # --- global places
    ag = aggregate_global(pl, 'pl')
    # QID / ids of the cluster
    ids_cols = {}
    for s in ('geonames', 'tgn', 'pleiades', 'gnd', 'hls', 'viaf', 'wikipedia'):
        ids_cols[s] = dominant(pl, 'id_' + s)
    for s, ser in ids_cols.items():
        ag[s] = ser.reindex(ag.index)
    ag['qid'] = dominant(pl, 'qid').reindex(ag.index)
    # coordinates: wikidata > geonames > register (most mentions)
    gp = pl.groupby('global_id')
    wd1 = pl[pl['has_wd']].drop_duplicates('global_id').set_index('global_id')
    gn1c = pl[pl['has_gn']].sort_values('n_mentions', ascending=False).drop_duplicates('global_id').set_index('global_id')
    rg = pl[pl['has_reg']].sort_values('n_mentions', ascending=False).drop_duplicates('global_id').set_index('global_id')
    lat = pd.Series(np.nan, index=ag.index)
    lon = pd.Series(np.nan, index=ag.index)
    src = pd.Series('none', index=ag.index, dtype=object)
    for name, tab, la, lo in (('register', rg, 'reg_lat', 'reg_lon'), ('geonames', gn1c, 'gn_lat', 'gn_lon'),
                              ('wikidata', wd1, 'wd_lat', 'wd_lon')):      # later overrides earlier
        ix = ag.index.intersection(tab.index)
        lat[ix] = tab.loc[ix, la]
        lon[ix] = tab.loc[ix, lo]
        src[ix] = name
    ag['lat'], ag['lon'], ag['coord_source'] = lat, lon, src
    ag['coord_conf'] = pd.Series(np.where(src != 'none', 'authority', None), index=ag.index, dtype=object)
    ag['coord_radius_km'] = np.nan
    ag['coord_cand'] = None
    ag['coord_evidence'] = None
    nmr = pl[pl['coord_source_geo'].isin(NAME_SOURCES) & pl['coord_source_pre'].eq('none')].copy()
    if len(nmr):
        nmr['_r'] = nmr['coord_conf'].map({'high': 2, 'medium': 1})
        nmr = nmr.sort_values(['_r', 'n_mentions', 'entity_key'], ascending=[False, False, True]).drop_duplicates('global_id')
        nmr = nmr.set_index('global_id')
        ix = [g for g in nmr.index if g in ag.index and ag.at[g, 'coord_source'] == 'none']
        for g in ix:
            r = nmr.loc[g]
            ag.at[g, 'lat'], ag.at[g, 'lon'], ag.at[g, 'coord_source'] = r['geo_lat'], r['geo_lon'], r['coord_source_geo']
            ag.at[g, 'coord_conf'], ag.at[g, 'coord_radius_km'] = r['coord_conf'], r['coord_radius_km']
            ag.at[g, 'coord_cand'], ag.at[g, 'coord_evidence'] = r['gc_cand'], r['gc_evidence']
        _t(f'global places: {len(ix)} clusters placed by name')
    ag['n_register_coords'] = pl.groupby('global_id')['has_reg'].sum().reindex(ag.index).astype('int64')
    ag['coord_conflict'] = pl.groupby('global_id')['coord_conflict'].any().reindex(ag.index).fillna(False)
    ag['qid_conflict'] = pl.groupby('global_id')['qid_conflict'].any().reindex(ag.index).fillna(False)
    # register vs chosen coordinates of the cluster (only if the chosen one is not the register)
    rr = ag.index.intersection(rg.index)
    dd = haversine_km(rg.loc[rr, 'reg_lat'], rg.loc[rr, 'reg_lon'], ag.loc[rr, 'lat'], ag.loc[rr, 'lon'])
    ag['coord_conflict_km'] = pd.Series(dd, index=rr).reindex(ag.index)
    ag.loc[ag['coord_source'] == 'register', 'coord_conflict_km'] = np.nan
    ag['coord_conflict'] = ag['coord_conflict'] | (ag['coord_conflict_km'] > CONFLICT_KM)
    # Wikidata attributes
    q = ag['qid']
    for l in ('en', 'fr', 'de', 'it'):
        ag['label_' + l] = _blank(q.map(w['label_' + l]))
    ag['label'] = ag['label_en'].fillna(ag['label_fr']).fillna(ag['label_de']).fillna(ag['label_it']).fillna(ag['name'])
    ag['classes'] = _blank(q.map(w['classes']))
    ag['country'] = _blank(q.map(w['country']))
    ag['country_iso'] = _blank(q.map(w['country_iso']))
    # GeoNames attributes when no QID
    if len(gn1):
        ag['gn_name'] = ag['geonames'].map(gn1['name'])
        ag['gn_country'] = ag['geonames'].map(gn1['country_code'])
        ag['gn_feature'] = ag['geonames'].map(gn1['feature_code'])
    ag = ag.reset_index().rename(columns={'global_id': 'id'})
    place_global = ag
    coll_sets = pl.groupby('global_id')['coll_slug'].agg(lambda x: sorted(set(x)))
    place_global['globe_id'] = merge_points(place_global)
    _write_global(place_global, dest / 'places_global.parquet')

    # ================================================================== persons
    pe = pq.read_table(v1 / 'persons_local.parquet').to_pandas()
    rqp = resolve_qids(votes, mapping, set(), 'person')
    domp = dominant_ids(votes, 'person', PERSON_FALLBACK + ('wikipedia',))
    pe = pe.merge(rqp, how='left', on=['coll_slug', 'entity_key']).merge(domp, how='left', on=['coll_slug', 'entity_key'])
    for s in PERSON_FALLBACK:
        if 'id_' + s not in pe:
            pe['id_' + s] = None
    pe = pe.reset_index(drop=True)
    pe['qid_conflict'] = pe['qid_n_cand'].fillna(0) > 1
    clp = assign_clusters(pe, PERSON_FALLBACK)
    pe['global_id'] = np.where(clp['gid'].notna(), 'pe:' + clp['gid'].astype(str), None)
    pe['basis'] = clp['basis']
    pe['scope'] = np.where(clp['gid'].notna(), 'authority', 'edition')
    ed = pe['global_id'].isna()
    pe.loc[ed, 'global_id'] = [edition_id('pe', c, k) for c, k in zip(pe.loc[ed, 'coll_slug'], pe.loc[ed, 'entity_key'])]
    _t(f'person clusters: {pe["global_id"].nunique()} global ids for {len(pe)} locals; with QID {pe["qid"].notna().sum()}')
    pe[['coll_slug', 'entity_key', 'name', 'norm_name', 'n_mentions', 'n_in_period', 'tier', 'licence', 'qid', 'qid_via',
        'qid_n_cand', 'qid_conflict', 'global_id', 'basis', 'scope']].to_parquet(dest / 'persons_local_global.parquet', index=False)
    ap = aggregate_global(pe, 'pe')
    for s in PERSON_FALLBACK + ('wikipedia',):
        ap[s] = dominant(pe, 'id_' + s).reindex(ap.index)
    ap['qid'] = dominant(pe, 'qid').reindex(ap.index)
    ap['qid_conflict'] = pe.groupby('global_id')['qid_conflict'].any().reindex(ap.index).fillna(False)
    ap['note_type'] = dominant(pe, 'note_type').reindex(ap.index)
    ap['reg_birth'] = dominant(pe, 'birth').reindex(ap.index)
    ap['reg_death'] = dominant(pe, 'death').reindex(ap.index)
    wq = wdq.drop_duplicates('qid').set_index('qid')
    q = ap['qid']
    for l in ('en', 'fr', 'de', 'it'):
        ap['label_' + l] = _blank(q.map(wq['label_' + l]))
    ap['label'] = ap['label_en'].fillna(ap['label_fr']).fillna(ap['label_de']).fillna(ap['label_it']).fillna(ap['name'])
    ap['birth'] = _blank(q.map(wq['birth']))
    ap['birth_precision'] = pd.to_numeric(q.map(wq['birth_prec']), errors='coerce').astype('Int64')
    ap['death'] = _blank(q.map(wq['death']))
    ap['death_precision'] = pd.to_numeric(q.map(wq['death_prec']), errors='coerce').astype('Int64')
    occ = q.map(wq['occupations']).fillna('')
    ap['occupations'] = occ.map(lambda s: [x.split(':', 1)[1] for x in s.split('|') if ':' in x])
    ap['occupation_qids'] = occ.map(lambda s: [x.split(':', 1)[0] for x in s.split('|') if ':' in x])
    ap['wd_human'] = q.map(wq['human'])
    ap = ap.reset_index().rename(columns={'global_id': 'id'})
    _write_global(ap, dest / 'persons_global.parquet')

    # ================================================================== reports and globe
    rep.update(report_numbers(pl, pe, place_global, ap, votes, mapping, wdp, wdq, gn, dest))
    globe = write_globe(place_global, coll, dest / 'globe_places.json', rep, coll_sets=coll_sets)
    rep['globe'] = globe
    (dest / 'reconcile_report.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=_js), 'utf-8')
    _t(f'done in {time.time() - t0:.0f}s')
    return rep


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _write_global(df: pd.DataFrame, path: Path):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().head(1)
            if len(sample) and isinstance(sample.iloc[0], list):
                continue
            df[c] = df[c].astype('string')
    df.to_parquet(path, index=False)
    _t(f'wrote {path.name}: {len(df)} rows')


# ----------------------------------------------------------------------------------------------------------------------
def report_numbers(pl, pe, pg, pgp, votes, mapping, wdp, wdq, gn, dest: Path) -> dict:
    r: dict = {}
    inp = pl['n_in_period'] > 0
    r['places_local'] = {
        'distinct': int(len(pl)), 'distinct_in_period': int(inp.sum()),
        'mentions_in_period': int(pl['n_in_period'].sum()),
        'register_coords_distinct_in_period': int((inp & pl['has_reg']).sum()),
        'register_coords_mentions_in_period': int(pl.loc[pl['has_reg'], 'n_in_period'].sum()),
        'register_invalid_coords': int(pl['reg_invalid'].sum()),
        'coords_now_distinct_in_period': int((inp & (pl['coord_source_geo'] != 'none')).sum()),
        'coords_now_mentions_in_period': int(pl.loc[pl['coord_source_geo'] != 'none', 'n_in_period'].sum()),
    }
    for s in ('register', 'wikidata', 'geonames', 'name_propagation', 'name_geonames', 'none'):
        m = pl['coord_source_geo'] == s
        r['places_local'][f'source_{s}_distinct_in_period'] = int((inp & m).sum())
        r['places_local'][f'source_{s}_mentions_in_period'] = int(pl.loc[m, 'n_in_period'].sum())
    r['places_local']['with_qid_distinct_in_period'] = int((inp & pl['qid'].notna()).sum())
    r['places_local']['with_qid_mentions_in_period'] = int(pl.loc[pl['qid'].notna(), 'n_in_period'].sum())
    r['places_local']['qid_conflict'] = int(pl['qid_conflict'].sum())
    r['places_local']['coord_conflicts_50km'] = int(pl['coord_conflict'].sum())
    r['places_local']['coord_conflicts_50km_mentions_in_period'] = int(pl.loc[pl['coord_conflict'], 'n_in_period'].sum())
    r['places_local']['qid_without_wd_coords_distinct_in_period'] = int((inp & pl['qid'].notna() & ~pl['has_wd']).sum())
    r['places_local']['geonames_id_but_no_coords_distinct_in_period'] = int(
        (inp & pl['id_geonames'].notna() & (pl['coord_source_geo'] == 'none')).sum())
    # name-based geocoding: before / after, by method and confidence
    placed_pre = pl['coord_source_pre'] != 'none'
    placed_now = pl['coord_source_geo'] != 'none'
    gcb: dict = {
        'before_distinct_in_period': int((inp & placed_pre).sum()),
        'before_mentions_in_period': int(pl.loc[placed_pre, 'n_in_period'].sum()),
        'after_distinct_in_period': int((inp & placed_now).sum()),
        'after_mentions_in_period': int(pl.loc[placed_now, 'n_in_period'].sum()),
        'mentions_in_period_total': int(pl['n_in_period'].sum()), 'distinct_in_period_total': int(inp.sum()),
        'by_method_confidence': {}, 'low_confidence_not_placed': {}}
    gcb['placed_share_mentions_before'] = round(gcb['before_mentions_in_period'] / max(1, gcb['mentions_in_period_total']), 4)
    gcb['placed_share_mentions_after'] = round(gcb['after_mentions_in_period'] / max(1, gcb['mentions_in_period_total']), 4)
    gcb['placed_share_distinct_before'] = round(gcb['before_distinct_in_period'] / max(1, gcb['distinct_in_period_total']), 4)
    gcb['placed_share_distinct_after'] = round(gcb['after_distinct_in_period'] / max(1, gcb['distinct_in_period_total']), 4)
    for src in NAME_SOURCES:
        for conf in ('high', 'medium'):
            m = (pl['coord_source_geo'] == src) & (pl['coord_conf'] == conf)
            gcb['by_method_confidence'][f'{src}:{conf}'] = {'distinct_in_period': int((inp & m).sum()),
                                                            'mentions_in_period': int(pl.loc[m, 'n_in_period'].sum())}
        m = (pl['coord_source_geo'] == 'none') & (pl['gc_source'] == src) & (pl['gc_conf'] == 'low')
        gcb['low_confidence_not_placed'][src] = {'distinct_in_period': int((inp & m).sum()),
                                                 'mentions_in_period': int(pl.loc[m, 'n_in_period'].sum())}
    if 'gc_reason' in pl:
        u = pl[inp & (pl['coord_source_geo'] == 'none')]
        gcb['unplaced_reasons'] = {k: {'distinct': int(len(g)), 'mentions': int(g['n_in_period'].sum())}
                                   for k, g in u.groupby(u['gc_reason'].fillna('not_attempted'))}
    if 'verify_rule' in pl:
        vb = pl[pl['verify_rule'].notna()]
        gcb['verify_taken_back'] = {k: {'distinct_in_period': int((g['n_in_period'] > 0).sum()), 'mentions_in_period': int(g['n_in_period'].sum())}
                                    for k, g in vb.groupby('verify_rule')}
    r['geocode'] = gcb
    # conflicts table
    cf = pl[pl['coord_conflict']].sort_values('n_in_period', ascending=False)
    cf[['coll_slug', 'entity_key', 'name', 'n_in_period', 'reg_lat', 'reg_lon', 'auth_source', 'auth_lat', 'auth_lon',
        'coord_conflict_km', 'qid']].head(2000).to_csv(dest / 'coord_conflicts.tsv', sep='\t', index=False)
    # global
    gi = pg['n_in_period'] > 0
    r['places_global'] = {
        'rows': int(len(pg)), 'rows_in_period': int(gi.sum()),
        'authority_clusters': int((pg['scope'] == 'authority').sum()),
        'authority_clusters_multi_local': int(((pg['scope'] == 'authority') & (pg['n_locals'] > 1)).sum()),
        'locals_in_authority_clusters': int(pl['scope'].eq('authority').sum()),
        'with_qid': int(pg['qid'].notna().sum()),
        'with_coords': int(pg['lat'].notna().sum()),
        'with_coords_in_period': int((pg['lat'].notna() & gi).sum()),
        'mentions_in_period_with_coords': int(pg.loc[pg['lat'].notna(), 'n_in_period'].sum()),
        'mentions_in_period_total': int(pg['n_in_period'].sum()),
        'coord_source_in_period': {k: int(v) for k, v in pg[gi]['coord_source'].value_counts().items()},
        'coord_conflict': int(pg['coord_conflict'].sum()),
        'max_collections_in_a_cluster': int(pg['n_collections'].max()),
        'clusters_in_2plus_collections': int((pg['n_collections'] >= 2).sum()),
    }
    ip = pgp['n_in_period'] > 0
    r['persons_global'] = {
        'rows': int(len(pgp)), 'rows_in_period': int(ip.sum()),
        'authority_clusters': int((pgp['scope'] == 'authority').sum()),
        'authority_clusters_multi_local': int(((pgp['scope'] == 'authority') & (pgp['n_locals'] > 1)).sum()),
        'locals': int(len(pe)), 'locals_in_authority_clusters': int(pe['scope'].eq('authority').sum()),
        'with_qid': int(pgp['qid'].notna().sum()),
        'with_qid_mentions_in_period': int(pgp.loc[pgp['qid'].notna(), 'n_in_period'].sum()),
        'mentions_in_period_total': int(pgp['n_in_period'].sum()),
        'with_birth_or_death': int((pgp['birth'].notna() | pgp['death'].notna()).sum()),
        'with_occupation': int(pgp['occupations'].map(len).gt(0).sum()),
        'non_human_qid': int((pgp['wd_human'] == False).sum()),   # noqa: E712
        'clusters_in_2plus_collections': int((pgp['n_collections'] >= 2).sum()),
    }
    r['wikidata'] = {
        'ids_mapped': int(mapping[['scheme', 'id']].drop_duplicates().shape[0]),
        'ids_mapped_by_scheme': {k: int(v) for k, v in mapping.drop_duplicates(['scheme', 'id'])['scheme'].value_counts().items()},
        'ids_ambiguous_2plus_qids': int((mapping.drop_duplicates(['scheme', 'id'])['k'] > 1).sum()),
        'places_described': int(len(wdp)), 'places_with_coords': int(wdp['lat'].notna().sum()),
        'persons_described': int(len(wdq)),
        'geonames_rows': int(len(gn)),
    }
    # distinct ids per scheme/kind and mapped share
    v = votes.drop_duplicates(['kind', 'scheme', 'id'])
    mk = set(zip(mapping['scheme'], mapping['id']))
    v = v.assign(mapped=[(s, i) in mk or s == 'wikidata' for s, i in zip(v['scheme'], v['id'])])
    r['ids_mapped_share'] = {f'{k}:{s}': [int(len(g)), round(float(g['mapped'].mean()), 3)]
                              for (k, s), g in v.groupby(['kind', 'scheme'])}
    # unplaced
    un = pl[inp & (pl['coord_source_geo'] == 'none')]
    byc = pl[inp].groupby('coll_slug').apply(lambda d: pd.Series({
        'places_in_period': len(d), 'mentions_in_period': int(d['n_in_period'].sum()),
        'placed_distinct': int((d['coord_source_geo'] != 'none').sum()),
        'placed_mentions': int(d.loc[d['coord_source_geo'] != 'none', 'n_in_period'].sum()),
        'unplaced_distinct': int((d['coord_source_geo'] == 'none').sum()),
        'unplaced_mentions': int(d.loc[d['coord_source_geo'] == 'none', 'n_in_period'].sum()),
        'name_placed_distinct': int(d['coord_source_geo'].isin(NAME_SOURCES).sum()),
        'name_placed_mentions': int(d.loc[d['coord_source_geo'].isin(NAME_SOURCES), 'n_in_period'].sum()),
        'placed_mentions_before': int(d.loc[d['coord_source_pre'] != 'none', 'n_in_period'].sum()),
        'lowconf_unplaced_mentions': int(d.loc[(d['coord_source_geo'] == 'none') & (d['gc_conf'] == 'low'), 'n_in_period'].sum()),
        'tier': d['tier'].iloc[0]}), include_groups=False)
    byc['placed_share'] = (byc['placed_mentions'] / byc['mentions_in_period']).round(3)
    byc['placed_share_before'] = (byc['placed_mentions_before'] / byc['mentions_in_period']).round(3)
    byc = byc.sort_values('unplaced_mentions', ascending=False)
    byc.reset_index().to_csv(dest / 'unplaced_by_collection.tsv', sep='\t', index=False)
    top = un.groupby('norm_name').agg(name=('name', 'first'), mentions=('n_in_period', 'sum'), n_colls=('coll_slug', 'nunique'),
                                      colls=('coll_slug', lambda s: ','.join(sorted(set(s))[:4])),
                                      lowconf=('gc_conf', lambda s: int((s == 'low').sum())),
                                      reason=('gc_reason', lambda s: s.dropna().mode().iat[0] if s.notna().any() else '')).reset_index()
    top = top.sort_values('mentions', ascending=False)
    top.head(3000).to_csv(dest / 'unplaced_top_names.tsv', sep='\t', index=False)
    # propagation potential of normalised names (what a name-based geocoder could do with data already in the atlas)
    pl_ok = pl[(pl['coord_source_geo'] != 'none') & (pl['norm_name'].fillna('') != '')]
    names = set(pl_ok['norm_name'])
    hit = un['norm_name'].isin(names)
    r['unplaced'] = {
        'distinct_in_period': int(len(un)), 'mentions_in_period': int(un['n_in_period'].sum()),
        'distinct_norm_names': int(un['norm_name'].nunique()),
        'distinct_with_name_matching_a_placed_entity': int(hit.sum()),
        'mentions_with_name_matching_a_placed_entity': int(un.loc[hit, 'n_in_period'].sum()),
        'top_collections_by_unplaced_mentions': {k: [int(x['unplaced_mentions']), int(x['mentions_in_period'])]
                                                  for k, x in byc.head(25).iterrows()},
    }
    # agreement of coordinates between placed entities that share a normalised name (precision of name propagation)
    pm = pl_ok.groupby('norm_name').filter(lambda d: d['coll_slug'].nunique() >= 2)
    if len(pm):
        refrow = pm.sort_values('n_mentions', ascending=False).drop_duplicates('norm_name')
        ref = refrow.set_index('norm_name')
        d = haversine_km(pm['geo_lat'], pm['geo_lon'], pm['norm_name'].map(ref['geo_lat']), pm['norm_name'].map(ref['geo_lon']))
        pm = pm.assign(_d=d)
        pm = pm[~pm.index.isin(refrow.index)]     # the reference entity of each name is not compared with itself
        if len(pm):
            r['name_agreement'] = {
                'entities_compared': int(len(pm)), 'names': int(pm['norm_name'].nunique()),
                'share_within_50km': round(float((pm['_d'] <= 50).mean()), 4),
                'share_within_50km_weighted': round(float(np.average(pm['_d'] <= 50, weights=pm['n_in_period'] + 1)), 4),
                'share_within_10km': round(float((pm['_d'] <= 10).mean()), 4)}
    return r


GENERIC_NAME = {'stadt', 'kloster', 'bistum', 'fluss', 'insel', 'kirche', 'schloss', 'burg', 'haus', 'hospital',
                'fürstentum', 'furstentum', 'königreich', 'konigreich', 'herzogtum', 'grafschaft', 'republik', 'amt',
                'church', 'castle', 'river', 'island', 'abbey', 'palace', 'street', 'bridge', 'gate', 'parish', 'hall',
                'manor', 'priory', 'hospital', 'college', 'school', 'inn', 'house', 'market', 'lane', 'ward', 'tower'}
MERGE_KM = 20.0


PREFIX_MIN = 8          # a first comma part shared by this many different names is a country / region, not a place


def _name_keys(r, generic_prefix: frozenset = frozenset()) -> set:
    """Normalised names that can identify the same point: the place name, its labels, the part before a comma
    ('Venedig, Republik' -> 'venedig') and the text without parentheses ('Halle (Saale)' -> 'halle'). Not the text
    inside parentheses (a disambiguator: 'Coswig (Anhalt)' is not 'Anhalt') and not a first comma part that is a country
    ('Portugal, Setúbal' is not 'Portugal')."""
    out: set = set()
    for n in (r.name, r.label, r.label_en, r.label_fr, r.label_de, r.label_it, getattr(r, 'gn_name', None)):
        if not isinstance(n, str) or not n:
            continue
        cands = [n, re.sub(r'\([^)]*\)', ' ', n)]
        first = _norm(n.split(',')[0])
        if ',' in n and first not in generic_prefix:
            cands.append(n.split(',')[0])
        for c in cands:
            k = _norm(c)
            if len(k) >= 3 and k not in GENERIC_NAME:
                out.add(k)
    return out


def merge_points(pg: pd.DataFrame, extra=None) -> pd.Series:
    """globe_id: places with coordinates and in-period mentions that share a normalised name and lie within MERGE_KM
    form one point (union-find); the member with most mentions is the representative. Others: None.
    ``extra`` (v2.3): pairs of global ids that are the same place for a reason other than the name (``reconcile_v2.point_unions``)."""
    d = pg[pg['lat'].notna() & (pg['n_in_period'] >= 1)]
    ids = d['id'].tolist()
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pref: dict = {}
    for n in set(d['name'].dropna()):
        if ',' in n:
            pref.setdefault(_norm(n.split(',')[0]), set()).add(n)
    generic = frozenset(k for k, v in pref.items() if len(v) >= PREFIX_MIN)
    buckets: dict = {}
    lat = d['lat'].to_numpy(dtype='float64')
    lon = d['lon'].to_numpy(dtype='float64')
    for k, r in enumerate(d.itertuples(index=False)):
        for nk in _name_keys(r, generic):
            buckets.setdefault(nk, []).append(k)
    for nk, mem in buckets.items():
        if len(mem) < 2:
            continue
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                i, j = mem[a], mem[b]
                if find(i) != find(j) and haversine_km(lat[i], lon[i], lat[j], lon[j]) <= MERGE_KM:
                    parent[find(i)] = find(j)
    pos = {g: k for k, g in enumerate(ids)}
    for a_, b_ in (extra or ()):
        if a_ in pos and b_ in pos and find(pos[a_]) != find(pos[b_]):
            parent[find(pos[a_])] = find(pos[b_])
    n = d['n_in_period'].to_numpy()
    auth = ((~d['coord_source'].isin(NAME_SOURCES)) if 'coord_source' in d else pd.Series(True, index=d.index)).to_numpy()   # edition / authority coordinates come first
    rep: dict = {}
    for k in range(len(ids)):
        root = find(k)
        if root not in rep or (auth[k], n[k], ids[k]) > (auth[rep[root]], n[rep[root]], ids[rep[root]]):
            rep[root] = k
    gid = {ids[k]: ids[rep[find(k)]] for k in range(len(ids))}
    return pg['id'].map(gid)


def write_globe(pg: pd.DataFrame, coll: pd.DataFrame, path: Path, rep: dict, max_bytes: int = 4_000_000,
                coll_sets: Optional[pd.Series] = None) -> dict:
    d = pg[pg['lat'].notna() & (pg['n_in_period'] >= 1)].copy()
    if 'globe_id' not in d:
        d['globe_id'] = d['id']
    d = d.sort_values(['n_in_period', 'id'], ascending=[False, True])
    qarr = d[QCOLS].to_numpy()
    d = d.reset_index(drop=True)
    by_gid: dict = {}
    for i, g in enumerate(d['globe_id'].tolist()):
        by_gid.setdefault(g, []).append(i)
    items = []
    for gid, mem in by_gid.items():
        lead = next((i for i in mem if d.at[i, 'coord_source'] not in NAME_SOURCES), mem[0])   # authority coordinates lead
        r = d.iloc[lead]                                      # members are sorted by mentions: the first one is the lead
        nm = r['name'] if isinstance(r['name'], str) and r['name'] else (r['label'] or '')
        q = qarr[mem].sum(axis=0)
        cnt: Counter = Counter()
        for i in mem:
            for c, k in zip(d.at[i, 'top_collections'], d.at[i, 'top_collection_counts']):
                cnt[c] += int(k)
        if coll_sets is not None:
            allc = set()
            for i in mem:
                allc |= set(coll_sets.get(d.at[i, 'id'], []))
            nc = len(allc)
        else:
            nc = int(max(d.at[i, 'n_collections'] for i in mem))
        qid = next((d.at[i, 'qid'] for i in mem if isinstance(d.at[i, 'qid'], str)), None)
        it = {'id': gid, 'name': nm[:60], 'lat': round(float(r['lat']), 4), 'lon': round(float(r['lon']), 4),
              'n': int(d.loc[mem, 'n_in_period'].sum()), 'q': [int(x) for x in q], 'c': nc,
              'cols': [c for c, _ in cnt.most_common(3)], 'qid': qid,
              't': int(d.loc[mem, 'tier_worst'].max()), 's': SRC_CODE.get(r['coord_source'], r['coord_source'][0])}
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
        'title': 'Renascor Atlas, places tagged by editors (v1.1, coordinates from edition registers, Wikidata, GeoNames, '
                 'and name-based geocoding of the rest)',
        'generated': time.strftime('%Y-%m-%d'),
        'period': [1450, 1700], 'quarters': [1450 + 25 * i for i in range(10)],
        'fields': {'n': 'in-period mentions', 'q': 'mentions per quarter-century by document year (sum can be below n: '
                    'documents dated across a boundary or before 1450)', 'c': 'collections', 'cols': 'top 3 collections',
                   't': 'worst licence tier among all contributing collections (1=T1, 2=T2, 3=T3: internal use only)',
                   's': 'coordinate source: r=register, w=wikidata, g=geonames id, p=name propagation (the same name is placed '
                        'elsewhere in the atlas), n=GeoNames name match; p and n are high / medium confidence only',
                   'm': 'number of authority / edition places merged into this point (same name within %d km); absent = 1'
                        % int(MERGE_KM)},
        'attribution': ['Wikidata (CC0)', 'GeoNames, https://www.geonames.org (CC BY 4.0)'],
        'coordinate_sources_note': 'p and n points are inferred from names, not from the editions; see places_local_geo.parquet '
                                   '(gc_* columns) for the evidence. When built with --geocode-verify, the strata that did not '
                                   'reach 85 % accuracy in a read-through of the TEI passages are not on the globe '
                                   '(verify_rule column)',
        'places': len(items), 'mentions': int(d['n_in_period'].sum()),
        'collections': [{'slug': r['slug'], 'licence': r['licence'], 'tier': r['tier'], 'renascor_ids': r['renascor_ids'],
                         'places_in_top3': int(used.get(r['slug'], 0))}
                        for _, r in cl.iterrows()],
        'trimmed': None,
    }

    def dump(its, hd):
        return json.dumps({'header': hd, 'places': its}, ensure_ascii=False, separators=(',', ':'))
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
            'places_eligible': int(len(d)), 'merged_points': int(sum(1 for x in items if x.get('m', 1) > 1))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile build')
    ap.add_argument('--v1', type=Path, required=True)
    ap.add_argument('--rdir', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--geonames', type=Path, default=None)
    ap.add_argument('--geocode', type=Path, default=None, help='geocode_places.parquet from `reconcile geocode run`')
    ap.add_argument('--geocode-verify', type=Path, default=None, dest='verify',
                    help='geocode_verify.tsv from `reconcile verify rules`: name-based results the audit takes back')
    a = ap.parse_args(argv)
    rep = build(a.v1, a.rdir, a.dest, a.geonames, a.geocode, a.verify)
    print(json.dumps(rep, indent=1, default=_js)[:6000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
