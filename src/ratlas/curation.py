"""Homonym audit (2026-10-23): places put on the wrong homonym, corrected by hand, and two guard rules run at every build.

The cause found by the audit: an edition's register (TEI ``<geo>``) or identifier can point at a modern homonym (Bullinger's Sodom is
Sodom, West Virginia; Emmaus is Emmaus, Pennsylvania; Troja is the Prague district), and the atlas trusted register and identifier
coordinates as the editors' own. Our name-based geocoder makes the same mistake with Latin and biblical names (Ladon in the Loiret).

Three pieces of data, all in ``data/curation/``:

* ``homonym_fixes.json`` -- one entry per edition place read in context: ``move`` (a new point), ``unplace`` (a real place, but which one is
  uncertain), ``not_a_place`` (a common noun, a person, an adjective) or ``keep`` (read and right: the rules leave it alone).
* ``ancient_names.tsv`` -- biblical, classical and mythical names with the region their place must lie in (rule ``ancient``).
* the model layer has its own file, ``data/model_layer_overrides.json`` (``scripts/ner2_layer_fix.py``).

Rules (applied when no entry of the fixes file covers the place):

* ``ancient`` -- a place whose name is in ``ancient_names.tsv`` and whose point lies outside the name's region is unplaced (Sodom outside the
  Levant, Troja outside the Aegean, Atlantis anywhere).
* ``far_homonym`` -- a point on another continent (the Americas, Africa south of the Sahara, Asia east of Persia, Oceania) is unplaced when
  the edition sends less than 1 % of its placed mentions there, the place is not an area, and the same name is placed in the Old World by
  other places of the atlas (2 mentions or more): Alcalá in Colombia, València in Venezuela.

* ``propagated`` -- a place whose point was copied (name propagation, Latin names) from a place that a fix or a rule changed loses that
  point too, unless an entry decides it (Sandrart's Bethulia copied Bullinger's Bangladesh).

Strict reading: a moved place stays strict only when an identifier of the edition itself confirms the new point (the same QID, or an
authority point within 30 km, 150 km for an area). Otherwise its source becomes ``curated`` (page code ``k``): our correction, not the
editors' location, so it counts in the Augmented reading only.

``check`` (also ``python -m ratlas curation check``) lists, at each build, every point the rules would touch and the per-edition outliers
(more than 1,500 km and three times the typical distance from the edition's centre) that have a same-name place much nearer: the list is
written next to the tables and read by a person; nothing in it is applied except the two rules.
"""
from __future__ import annotations

import io
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'curation'
FIXES = DATA / 'homonym_fixes.json'
ANCIENT = DATA / 'ancient_names.tsv'
ACTIONS = ('move', 'unplace', 'not_a_place', 'keep')
KINDS = ('c', 'a', 'n', 'b', 'u')
#: (lat min, lat max, lon min, lon max)
REGIONS = {
    'LEV': (10.0, 42.0, 24.0, 62.0),       # Holy Land, Syria, Egypt, Mesopotamia, Persia, Arabia, southern Anatolia
    'GR': (32.0, 47.0, 12.0, 45.0),        # Greece, the Aegean, Anatolia, the Black Sea
    'MED': (25.0, 48.0, -12.0, 45.0),      # the Roman Mediterranean
}
FAR_ZONES = ('americas', 'africa_s', 'asia', 'oceania')
SUPPORT_KM, SUPPORT_KM_AREA = 30.0, 150.0
FAR_SHARE = 0.01
FAR_OLD_MENTIONS = 2
OUTLIER_KM = 1500.0
CURATED = 'curated'
EDITOR_SOURCES = ('register', 'wikidata', 'geonames', 'tgn')


# ----------------------------------------------------------------------------------------------------------------------- helpers
def fold(s) -> str:
    """Lower case, no accents, punctuation as spaces (the key of ``ancient_names.tsv``)."""
    s = unicodedata.normalize('NFKD', str(s or '').casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[\W_]+', ' ', s).strip()


def zone(lat: float, lon: float) -> str:
    """old (Europe, the Mediterranean, the Near and Middle East), americas, africa_s, asia (east of Persia), oceania, other."""
    if lat != lat or lon != lon:
        return 'none'
    if lon < -30:
        return 'americas'
    if lat < -10 and lon > 100:
        return 'oceania'
    if lon > 66:
        return 'asia'
    if lat < 13 and -25 < lon < 55:
        return 'africa_s'
    if lat > 0:
        return 'old'
    return 'other'


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def in_regions(lat: float, lon: float, regions) -> bool:
    if 'NONE' in regions:
        return False
    return any(REGIONS[r][0] <= lat <= REGIONS[r][1] and REGIONS[r][2] <= lon <= REGIONS[r][3] for r in regions)


# ----------------------------------------------------------------------------------------------------------------------- data
def load_fixes(path: Optional[Path] = None) -> pd.DataFrame:
    """The validated entries of ``homonym_fixes.json`` (empty table when the file is missing). Raises ValueError on a bad entry."""
    path = Path(path or FIXES)
    cols = ['coll_slug', 'entity_key', 'name', 'action', 'lat', 'lon', 'kind', 'qid', 'reason_fr', 'origin', 'label']
    if not path.exists():
        return pd.DataFrame(columns=cols)
    ents = json.loads(path.read_text('utf-8'))['entries']
    f = pd.DataFrame(ents)
    for c in cols:
        if c not in f:
            f[c] = None
    f = f[cols].copy()
    bad = f[~f['action'].isin(ACTIONS)]
    if len(bad):
        raise ValueError(f'homonym_fixes: unknown action {sorted(set(bad["action"]))}')
    if (f['coll_slug'].fillna('').str.strip() == '').any() or (f['entity_key'].fillna('').str.strip() == '').any():
        raise ValueError('homonym_fixes: empty coll_slug or entity_key')
    mv = f['action'] == 'move'
    la, lo = pd.to_numeric(f['lat'], errors='coerce'), pd.to_numeric(f['lon'], errors='coerce')
    if (mv & ~(la.between(-90, 90) & lo.between(-180, 180))).any():
        raise ValueError(f'homonym_fixes: move without a valid point {list(f.loc[mv & la.isna(), "entity_key"][:3])}')
    if (mv & ~f['kind'].isin(KINDS)).any():
        raise ValueError('homonym_fixes: move without a kind (c a n b u)')
    q = f['qid'].dropna()
    if (~q.astype(str).str.match(r'^Q[1-9]\d*$')).any():
        raise ValueError('homonym_fixes: malformed qid')
    if f.duplicated(['coll_slug', 'entity_key']).any():
        raise ValueError(f'homonym_fixes: duplicate entries {list(f.loc[f.duplicated(["coll_slug", "entity_key"]), "entity_key"][:3])}')
    f['lat'], f['lon'] = la, lo
    return f.reset_index(drop=True)


def load_ancient(path: Optional[Path] = None) -> dict:
    """{folded name: [region, ...]} from ``ancient_names.tsv``."""
    path = Path(path or ANCIENT)
    if not path.exists():
        return {}
    text = '\n'.join(l for l in path.read_text('utf-8').splitlines() if not l.startswith('#'))
    t = pd.read_csv(io.StringIO(text), sep='\t', dtype=str, keep_default_na=False)
    out = {}
    for k, r in zip(t['key'], t['region']):
        regs = [x.strip() for x in r.split('|') if x.strip()]
        unknown = [x for x in regs if x != 'NONE' and x not in REGIONS]
        if not regs or unknown:
            raise ValueError(f'ancient_names: bad region {r!r} for {k!r}')
        out[fold(k)] = regs
    return out


# ----------------------------------------------------------------------------------------------------------------------- rules
def norm_frame(df: pd.DataFrame, lat='lat', lon='lon', source='coord_source', area=None) -> pd.DataFrame:
    """The columns the rules read, from a table of local places (``places_local_geo`` names by default)."""
    out = pd.DataFrame({
        'coll_slug': df['coll_slug'].to_numpy(), 'entity_key': df['entity_key'].to_numpy(), 'name': df['name'].to_numpy(),
        'lat': pd.to_numeric(df[lat], errors='coerce').to_numpy(), 'lon': pd.to_numeric(df[lon], errors='coerce').to_numpy(),
        'source': df[source].to_numpy(), 'n_mentions': pd.to_numeric(df['n_mentions'], errors='coerce').fillna(0).to_numpy(),
    })
    for c in ('qid', 'auth_lat', 'auth_lon', 'tgn_lat', 'tgn_lon', 'gc_cand'):
        out[c] = df[c].to_numpy() if c in df else None
    if area is None:
        a = np.zeros(len(df), dtype=bool)
        for c, v in (('coord_kind', 'area'), ('place_kind', 'area')):
            if c in df:
                a |= (df[c] == v).to_numpy()
        a |= (out['source'] == 'name_exonym').to_numpy()
        out['area'] = a
    else:
        out['area'] = np.asarray(area, dtype=bool)
    out['nk'] = [fold(n) for n in out['name']]
    out['zone'] = [zone(a, b) for a, b in zip(out['lat'], out['lon'])]
    return out


def rule_ancient(n: pd.DataFrame, ancient: dict) -> np.ndarray:
    """True for a placed place with an ancient name outside its region."""
    ok = n['lat'].notna().to_numpy()
    return np.array([bool(o) and k in ancient and not in_regions(a, b, ancient[k]) for o, k, a, b in zip(ok, n['nk'], n['lat'], n['lon'])])


def rule_far_homonym(n: pd.DataFrame) -> np.ndarray:
    """True for a point on another continent that the edition rarely visits, when the same name is placed in the Old World elsewhere."""
    placed = n['lat'].notna()
    p = n[placed]
    w = p.groupby(['coll_slug', 'zone'])['n_mentions'].sum().unstack(fill_value=0)
    share = w.div(w.sum(axis=1).clip(lower=1), axis=0)
    old = p[p['zone'] == 'old']
    old_by_name = old.groupby(['nk', 'coll_slug'])['n_mentions'].sum()
    tot = old_by_name.groupby(level=0).sum()
    out = np.zeros(len(n), dtype=bool)
    for i, (c, k, z, a, pl_) in enumerate(zip(n['coll_slug'], n['nk'], n['zone'], n['area'], placed)):
        if not pl_ or a or z not in FAR_ZONES:
            continue
        sh = share.at[c, z] if (c in share.index and z in share.columns) else 0.0
        if sh >= FAR_SHARE or k not in tot.index:
            continue
        other = tot[k] - (old_by_name.get((k, c), 0))            # the Old World mentions of other editions
        out[i] = other >= FAR_OLD_MENTIONS
    return out


def supported(fx, qid, auth_lat, auth_lon, tgn_lat, tgn_lon) -> Optional[str]:
    """Which identifier of the edition confirms the corrected point: 'qid' (the same QID), 'auth' (its Wikidata / GeoNames point is near),
    'tgn' (its TGN point is near), or None."""
    if isinstance(fx['qid'], str) and fx['qid'] and fx['qid'] == qid:
        return 'qid'
    tol = SUPPORT_KM_AREA if fx['kind'] == 'a' else SUPPORT_KM
    for via, la, lo in (('auth', auth_lat, auth_lon), ('tgn', tgn_lat, tgn_lon)):
        try:
            la, lo = float(la), float(lo)
        except (TypeError, ValueError):
            continue
        if la == la and lo == lo and float(haversine_km(fx['lat'], fx['lon'], la, lo)) <= tol:
            return via
    return None


def plan(n: pd.DataFrame, fixes: Optional[pd.DataFrame] = None, ancient: Optional[dict] = None) -> pd.DataFrame:
    """One row per place that changes: coll_slug, entity_key, name, action (move | unplace | not_a_place), lat, lon, kind, qid, strict,
    why (verdict | rule_checked | rule:ancient | rule:far_homonym), reason. ``n`` is a frame of ``norm_frame``."""
    fixes = load_fixes() if fixes is None else fixes
    ancient = load_ancient() if ancient is None else ancient
    key = list(zip(n['coll_slug'], n['entity_key']))
    fx = {(r['coll_slug'], r['entity_key']): r for r in fixes.to_dict('records')}
    r_anc = rule_ancient(n, ancient)
    r_far = rule_far_homonym(n)
    rows = []
    for i, k in enumerate(key):
        f = fx.get(k)
        if f is not None:
            if f['action'] == 'keep':
                continue
            via = None
            if f['action'] == 'move':
                via = supported(f, n.at[i, 'qid'], n.at[i, 'auth_lat'], n.at[i, 'auth_lon'], n.at[i, 'tgn_lat'], n.at[i, 'tgn_lon'])
            rows.append({'coll_slug': k[0], 'entity_key': k[1], 'name': n.at[i, 'name'], 'action': f['action'],
                         'lat': f['lat'] if f['action'] == 'move' else np.nan, 'lon': f['lon'] if f['action'] == 'move' else np.nan,
                         'kind': f['kind'] if f['action'] == 'move' else None, 'qid': f['qid'] if f['action'] == 'move' else None,
                         'strict': via is not None, 'strict_via': via, 'why': f['origin'] or 'verdict', 'reason': f['reason_fr'],
                         'old_lat': n.at[i, 'lat'], 'old_lon': n.at[i, 'lon'], 'old_source': n.at[i, 'source'],
                         'label': f.get('label') if isinstance(f.get('label'), str) else None})
        elif r_anc[i] or r_far[i]:
            why = 'rule:ancient' if r_anc[i] else 'rule:far_homonym'
            rows.append({'coll_slug': k[0], 'entity_key': k[1], 'name': n.at[i, 'name'], 'action': 'unplace', 'lat': np.nan, 'lon': np.nan,
                         'kind': None, 'qid': None, 'strict': False, 'strict_via': None, 'why': why,
                         'reason': 'nom antique hors de sa région' if r_anc[i] else 'homonyme lointain, sans contexte de ce continent dans l\'édition',
                         'old_lat': n.at[i, 'lat'], 'old_lon': n.at[i, 'lon'], 'old_source': n.at[i, 'source']})
    # rule 'propagated': a place that copied its point from a place corrected above (gc_cand = '<collection>:<entity key>' of the
    # name-propagation or Latin-name step) loses it too, unless an entry decides it: Bethulia of Sandrart copied Bullinger's Bangladesh
    done = {(r['coll_slug'], r['entity_key']) for r in rows} | set(fx)
    src_of = {}
    for i, c in enumerate(n['gc_cand']):
        if isinstance(c, str) and ':' in c and n.at[i, 'lat'] == n.at[i, 'lat']:
            a, b = c.split(':', 1)
            src_of[i] = (a, b)
    bad = {(r['coll_slug'], r['entity_key']) for r in rows}
    changed = True
    while changed:
        changed = False
        for i, s_ in src_of.items():
            k = key[i]
            if k in done or s_ not in bad:
                continue
            rows.append({'coll_slug': k[0], 'entity_key': k[1], 'name': n.at[i, 'name'], 'action': 'unplace', 'lat': np.nan, 'lon': np.nan,
                         'kind': None, 'qid': None, 'strict': False, 'strict_via': None, 'why': 'rule:propagated',
                         'reason': f'point copié de {s_[0]}:{s_[1]}, corrigé', 'old_lat': n.at[i, 'lat'], 'old_lon': n.at[i, 'lon'],
                         'old_source': n.at[i, 'source']})
            done.add(k)
            bad.add(k)
            changed = True
    cols = ['coll_slug', 'entity_key', 'name', 'action', 'lat', 'lon', 'kind', 'qid', 'strict', 'strict_via', 'why', 'reason', 'old_lat', 'old_lon',
            'old_source', 'label']
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------------------------------------------------- pipeline
def apply_reconcile(pl: pd.DataFrame, rep: Optional[dict] = None, fixes_path: Optional[Path] = None, log=lambda m: None) -> pd.DataFrame:
    """``reconcile build2`` (end of ``_places``): applies the plan to the local places before the clusters are made.

    unplace / not_a_place: no point and no identity (QID, ids: they led to the wrong place); not_a_place also sets place_class.
    move: the point of the entry; strict -> the source of the identifier that confirms it, else ``curated`` (a free source of the cluster,
    used when no register or authority point of another member gives one); the entry's QID becomes ``qid_cluster`` (the place joins the
    point of that item: Tunis joins Tunis), the edition's own wrong ids are dropped."""
    n = norm_frame(pl, lat='geo_lat', lon='geo_lon', source='coord_source_geo')
    n['auth_lat'], n['auth_lon'] = pl.get('auth_lat'), pl.get('auth_lon')
    n['tgn_lat'], n['tgn_lon'] = pl.get('tgn_lat'), pl.get('tgn_lon')
    p = plan(n, load_fixes(fixes_path))
    pl = pl.copy()
    pl['curation'] = None
    pl['curation_why'] = None
    if not len(p):
        return pl
    ix = pd.MultiIndex.from_arrays([pl['coll_slug'], pl['entity_key']])
    pos = {k: i for i, k in enumerate(ix)}
    ids = [c for c in pl.columns if c.startswith('id_')] + [c for c in ('qid', 'qid_via', 'qid_cluster', 'qid_exo') if c in pl]
    flags = [c for c in ('has_reg', 'has_wd', 'has_gn', 'has_tgn') if c in pl]
    for r in p.itertuples(index=False):
        i = pos.get((r.coll_slug, r.entity_key))
        if i is None:
            continue
        j = pl.index[i]
        pl.at[j, 'curation'] = r.action
        pl.at[j, 'curation_why'] = r.why
        for c in flags:
            pl.at[j, c] = False
        if r.action in ('unplace', 'not_a_place'):
            pl.loc[j, ['geo_lat', 'geo_lon']] = [np.nan, np.nan]
            pl.at[j, 'coord_source_geo'] = 'none'
            pl.at[j, 'coord_kind'] = None
            pl.at[j, 'coord_conf'] = None
            pl.at[j, 'coord_radius_km'] = np.nan
            for c in ids:
                pl.at[j, c] = None
            if 'identity_dropped' in pl:
                pl.at[j, 'identity_dropped'] = True
            if r.action == 'not_a_place':
                pl.at[j, 'place_class'] = 'not_a_place'
            continue
        # move
        pl.at[j, 'geo_lat'], pl.at[j, 'geo_lon'] = float(r.lat), float(r.lon)
        if 'gc_cand' in pl:                              # the global place shows why it is there, not the geocoder's wrong candidate
            pl.at[j, 'gc_cand'] = 'curation:' + str(r.name)
            pl.at[j, 'gc_evidence'] = str(r.reason)[:300]
        pl.at[j, 'coord_kind'] = 'area' if r.kind == 'a' else 'point'
        pl.at[j, 'coord_radius_km'] = 150.0 if r.kind == 'a' else np.nan
        if r.strict:
            src = {'qid': 'wikidata', 'tgn': 'tgn'}.get(r.strict_via) or (pl.at[j, 'auth_source'] if 'auth_source' in pl else None) or 'wikidata'
            pl.at[j, 'coord_source_geo'] = src
            pl.at[j, 'coord_conf'] = 'authority'
            col = {'wikidata': ('has_wd', 'wd_lat', 'wd_lon'), 'tgn': ('has_tgn', 'tgn_lat', 'tgn_lon'), 'geonames': ('has_gn', 'gn_lat', 'gn_lon')}[src]
            if col[0] in pl:
                pl.at[j, col[0]] = True
                pl.at[j, col[1]], pl.at[j, col[2]] = float(r.lat), float(r.lon)
        else:
            pl.at[j, 'coord_source_geo'] = CURATED
            pl.at[j, 'coord_conf'] = 'high'
            if 'coord_source_pre' in pl:
                pl.at[j, 'coord_source_pre'] = 'none'
            for c in ids:
                pl.at[j, c] = None
        if isinstance(r.qid, str) and 'qid_cluster' in pl:
            pl.at[j, 'qid_cluster'] = r.qid
        if 'kind5_override' in pl and r.kind in KINDS:
            pl.at[j, 'kind5_override'] = r.kind
    c = p.groupby(['why', 'action']).size()
    rep = rep if rep is not None else {}
    rep['curation'] = {f'{w}:{a}': int(v) for (w, a), v in c.items()}
    log(f'homonym curation: {rep["curation"]}')
    return pl


# ----------------------------------------------------------------------------------------------------------------------- check
def outliers(n: pd.DataFrame, samename: Optional[pd.DataFrame] = None, min_places: int = 15) -> pd.DataFrame:
    """Per edition: placed points more than 1,500 km and three root-mean-square distances from the edition's median centre. With
    ``samename`` (key, lat, lon, pop, fcode: GeoNames features by folded name), only those with a same-name feature at least 1,000 km
    nearer the centre (a homonym the edition more likely means)."""
    rows = []
    p = n[n['lat'].notna() & ~n['area']]
    for c, g in p.groupby('coll_slug'):
        if len(g) < min_places:
            continue
        clat, clon = float(g['lat'].median()), float(g['lon'].median())
        d = haversine_km(g['lat'], g['lon'], clat, clon)
        rms = float(np.sqrt((d ** 2).mean()))
        far = d > max(OUTLIER_KM, 3 * rms)
        for (_, r), dd in zip(g[far].iterrows(), d[far]):
            near = np.nan
            if samename is not None:
                s = samename[samename['key'] == r['nk']]
                if len(s):
                    sig = (s['pop'].fillna(0) >= 1000) | s['fcode'].astype(str).str.match(r'^(PPLA|PPLC|ADM|PCL|ISL|MT|STM|RGN)')
                    if sig.any():
                        near = float(haversine_km(s.loc[sig, 'lat'], s.loc[sig, 'lon'], clat, clon).min())
                if not (near == near and near < 0.5 * dd and dd - near > 1000):
                    continue
            rows.append({'coll_slug': c, 'entity_key': r['entity_key'], 'name': r['name'], 'lat': r['lat'], 'lon': r['lon'], 'source': r['source'],
                         'n_mentions': int(r['n_mentions']), 'km_from_centre': round(float(dd)), 'km_nearer_homonym': None if near != near else round(near)})
    return pd.DataFrame(rows, columns=['coll_slug', 'entity_key', 'name', 'lat', 'lon', 'source', 'n_mentions', 'km_from_centre', 'km_nearer_homonym'])


def apply_plan_frame(n: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """``n`` (norm_frame) with the plan applied: what the places look like after the fixes and the rules (idempotent)."""
    n = n.copy()
    pos = {k: i for i, k in enumerate(zip(n['coll_slug'], n['entity_key']))}
    for r in p.itertuples(index=False):
        i = pos.get((r.coll_slug, r.entity_key))
        if i is None:
            continue
        if r.action == 'move':
            n.loc[i, ['lat', 'lon']] = [float(r.lat), float(r.lon)]
            n.at[i, 'source'] = CURATED if not r.strict else 'register'
            n.at[i, 'area'] = r.kind == 'a'
        else:
            n.loc[i, ['lat', 'lon']] = [np.nan, np.nan]
            n.at[i, 'source'] = 'none'
        n.at[i, 'zone'] = zone(n.at[i, 'lat'], n.at[i, 'lon'])
    return n


def check(n: pd.DataFrame, model_points=None, samename: Optional[pd.DataFrame] = None, fixes: Optional[pd.DataFrame] = None,
          ancient: Optional[dict] = None, after: bool = True) -> pd.DataFrame:
    """The build-time report: every place a rule touches (with ``covered`` = an entry of the fixes file decides it), and the outliers.
    ``n`` = norm_frame of the local places; with ``after`` the fixes and rules are applied first, so that the report lists what is
    still on the map; ``model_points`` = the model layer's places (dicts)."""
    fixes = load_fixes() if fixes is None else fixes
    ancient = load_ancient() if ancient is None else ancient
    keyed = set(zip(fixes['coll_slug'], fixes['entity_key']))
    if after:
        n = apply_plan_frame(n, plan(n, fixes, ancient))
    out = []
    for name, hit in (('ancient', rule_ancient(n, ancient)), ('far_homonym', rule_far_homonym(n))):
        for _, r in n[hit].iterrows():
            out.append({'check': name, 'layer': 'editor', 'coll_slug': r['coll_slug'], 'entity_key': r['entity_key'], 'name': r['name'],
                        'lat': r['lat'], 'lon': r['lon'], 'source': r['source'], 'n_mentions': int(r['n_mentions']),
                        'covered': (r['coll_slug'], r['entity_key']) in keyed, 'detail': ''})
    for _, r in outliers(n, samename).iterrows():
        out.append({'check': 'edition_outlier', 'layer': 'editor', 'coll_slug': r['coll_slug'], 'entity_key': r['entity_key'], 'name': r['name'],
                    'lat': r['lat'], 'lon': r['lon'], 'source': r['source'], 'n_mentions': r['n_mentions'],
                    'covered': (r['coll_slug'], r['entity_key']) in keyed,
                    'detail': f"{r['km_from_centre']} km from the edition's centre" + (f"; a same-name place {r['km_nearer_homonym']} km from it" if r['km_nearer_homonym'] is not None else '')})
    for p in model_points or []:
        k = fold(p['name'])
        if k in ancient and not in_regions(p['lat'], p['lon'], ancient[k]):
            out.append({'check': 'ancient', 'layer': 'model', 'coll_slug': '', 'entity_key': p['id'], 'name': p['name'], 'lat': p['lat'], 'lon': p['lon'],
                        'source': p.get('gs', ''), 'n_mentions': int(p['n']), 'covered': False, 'detail': p.get('tier', '')})
    cols = ['check', 'layer', 'coll_slug', 'entity_key', 'name', 'lat', 'lon', 'source', 'n_mentions', 'covered', 'detail']
    return pd.DataFrame(out, columns=cols)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='ratlas curation')
    ap.add_argument('step', choices=['check', 'plan'])
    ap.add_argument('--atlas', type=Path, required=True, help='folder with places_local_geo.parquet (a build of the atlas)')
    ap.add_argument('--model', type=Path, default=None, help='model_layer_v2.json, for the ancient-name check of the model points')
    ap.add_argument('--samename', type=Path, default=None, help='gn_samename.parquet (reconcile dir), for the outlier check')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--before', action='store_true', help='check: the tables as they are, without applying the fixes and rules first')
    a = ap.parse_args(argv)
    geo = pd.read_parquet(a.atlas / 'places_local_geo.parquet')
    n = norm_frame(geo)
    if a.step == 'plan':
        t = plan(n)
    else:
        model = json.loads(a.model.read_text('utf-8'))['places'] if a.model else None
        sn = pd.read_parquet(a.samename) if a.samename else None
        t = check(n, model, sn, after=not a.before)
    out = a.out or (a.atlas / f'homonym_{a.step}.tsv')
    t.to_csv(out, sep='\t', index=False)
    print(f'{len(t)} rows -> {out}')
    if a.step == 'check' and len(t):
        print(t.groupby(['check', 'layer', 'covered']).size().to_string())
    return 0
