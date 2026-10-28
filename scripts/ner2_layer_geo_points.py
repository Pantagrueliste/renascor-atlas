"""NER layer v1, geocoding stage 3 (Hypatia, Slurm): precision filters, points, documents.

    python scripts/ner2_layer_geo_points.py --layer LAYER --geo LAYER/geo [--words words_en.txt]

Reads the geocoder's results (``geo/gc/gc_results.parquet``), keeps the forms placed with high or medium confidence and takes back, in this order:

* ``continent``          a continent pin (Europe, Asia, Africa, America ... or any area of 1500 km radius or more): a mention is never put on a continent;
* ``gn_uncorroborated``  a GeoNames-only match with a low certainty that is not a major feature (the v2.1 rule of ``verify_geocode.rules``);
* ``lowercase_geonames`` a GeoNames or Latin-lexicon match of a form written with a capital in fewer than half of its mentions that no editor-placed namesake corroborates;
* ``common_word``        an English form that is also a lowercase word of the English word list (Bath, Reading, Wells, Derby ...) placed by a GeoNames name
                         match that no editor-placed namesake of the atlas corroborates;
* ``far_namesake``       a GeoNames name match whose key also names a bigger place (a city of 100 000 or more, or a province / country) more than 150 km away,
                         unless an editor-placed namesake of the atlas corroborates it (Boston and Lincoln go, Durham stays: the atlas places it too).

The remaining forms are grouped in points (same Wikidata / GeoNames / atlas candidate, or the same modern name within 10 km), with mentions, documents,
quarter-centuries, written forms and top documents (title, date, collection, link).

Writes in ``geo/points/``: ``forms_placed.parquet`` (every geocoded form with its fate), ``points.json`` (the points, without editor matching), ``fates.json``
(counts of forms and mentions by fate), ``doc_titles.json``.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import multiprocessing as mp
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from ratlas import verify_geocode as VG
from ratlas import geotext as GT
from ratlas import latin_names as LN
from ratlas import placeclass as PCL

QUARTERS = list(range(1450, 1700, 25))
CONTINENTS = frozenset('europe asia africa america americas north america south america oceania australia antarctica'.split())
KM_MERGE = 10.0
GN_CAND = re.compile(r'^gn:(\d+)$')
EX_KIND = re.compile(r'\((\w+), radius')
GN_EV = re.compile(r'\((\w+), (\w*), pop (\d+)\)')


def _t(msg):
    print(f'[layer-pts {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def kind_of(row, majors=None) -> str:
    """c / a / n / b / u from the geocoder's own evidence: exonym table kind, GeoNames feature code, or the radius."""
    ev = row['gc_evidence'] if isinstance(row['gc_evidence'], str) else ''
    r = float(row['gc_radius_km'] or 0)
    if row['gc_source'] == 'name_exonym':
        m = EX_KIND.search(ev)
        k = m.group(1) if m else ''
        if k in ('city',):
            return 'c'
        if k in ('area',):
            return 'a'
        if k in ('sea', 'river', 'island', 'mountain'):
            return 'n'
        return 'a' if r >= 30 else 'c'
    m = GN_EV.search(ev)
    if m:
        f = m.group(1)
        if f.startswith('PPL') or f in ('P', 'LCTY'):
            return 'c'
        if f.startswith('PCL') or f.startswith('ADM') or f in ('RGN', 'AREA', 'TERR', 'PRSH'):
            return 'a'
        if f[:1] in 'HTV':
            return 'n'
        if f[:1] == 'S':
            return 'b'
    return 'a' if r >= 30 else ('c' if r > 0 else 'u')


def fold(s) -> str:
    s = unicodedata.normalize('NFKD', str(s or '').casefold())
    return ''.join(c for c in s if c.isalnum())


def hav1(lat1, lon1, lat2, lon2):
    return float(VG.hav(lat1, lon1, lat2, lon2))


# ---------------------------------------------------------------------------------------------------------------------
# documents: titles and links
# ---------------------------------------------------------------------------------------------------------------------
def load_ext(path):
    spec = importlib.util.spec_from_file_location('ner2_layer_extract_mod', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['ner2_layer_extract_mod'] = m
    spec.loader.exec_module(m)
    return m


EEBO_ID = re.compile(r'^[AB]\d{5}$')


def doc_link(doc_id: str, collection: str, lang: str):
    """A link built from the document's own identifier (not checked online): EEBO-TCP, Evans-TCP and the Wikisource pages."""
    stem = doc_id.split('/', 1)[-1]
    if EEBO_ID.match(stem) and collection.startswith('EEBO') or collection.startswith('LatinEEBO') and EEBO_ID.match(stem):
        return f'https://quod.lib.umich.edu/e/eebogroup/{stem}.0001.001'
    if collection.startswith('EvansTCP') and re.match(r'^N\d{5}$', stem):
        return f'https://quod.lib.umich.edu/e/evans/{stem}.0001.001'
    m = re.match(r'^WS_([a-z]{2})_(.+)$', stem)
    if m and 'Wikisource' in collection:
        return f'https://{m.group(1)}.wikisource.org/wiki/{quote(m.group(2).replace(" ", "_"))}'
    return None


def head_title(raw: bytes) -> str:
    """Best-effort title: <dc:title> of the Wikisource header, else the first line with letters, cut at 110 characters."""
    i = raw.find(b'"t": "')
    if i < 0:
        return ''
    s = raw[i + 6:i + 6 + 2500]
    s = s.rstrip(b'\\') if len(s) - len(s.rstrip(b'\\')) % 2 else s
    try:
        t = json.loads(b'"' + s + b'"')
    except Exception:
        for cut in range(1, 12):
            try:
                t = json.loads(b'"' + s[:-cut] + b'"')
                break
            except Exception:
                t = ''
    m = re.search(r'<dc:title>\s*(.+?)\s*</dc:title>', t)
    if m:
        return m.group(1)[:110]
    for ln in t.split('\n'):
        ln = ' '.join(ln.split())
        if sum(c.isalpha() for c in ln) >= 8:
            return ln[:110]
    return ''


def scan_shard(args):
    path, needed = args
    out = {}
    with gzip.open(path, 'rb') as f:
        for line in f:
            i = line.find(b'"id": "')
            if i < 0:
                continue
            j = line.index(b'"', i + 7)
            try:
                did = json.loads(b'"' + line[i + 7:j] + b'"')
            except Exception:
                continue
            if did in needed:
                out[did] = head_title(line[:j + 2800])
    return out


# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=Path, required=True)
    ap.add_argument('--geo', type=Path, required=True)
    ap.add_argument('--words', type=Path, default=None, help='English lowercase word list (one per line)')
    ap.add_argument('--db', type=Path, default=None)
    ap.add_argument('--extract-script', type=Path, default=Path.home() / 'renascor-atlas/scripts/ner2_layer_extract.py')
    ap.add_argument('--top-docs', type=int, default=5)
    ap.add_argument('--top-forms', type=int, default=12)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    out = a.geo / 'points'
    out.mkdir(parents=True, exist_ok=True)
    forms = pd.read_parquet(a.geo / 'forms.parquet')
    st = pd.read_parquet(a.geo / 'gc/forms_status.parquet')[['lang', 'key', 'status']].rename(columns={'status': 'status2'})
    gc = pd.read_parquet(a.geo / 'gc/gc_results.parquet')
    F = forms.merge(st, on=['lang', 'key'], how='left').merge(
        gc.drop(columns=['coll_slug', 'entity_key', 'name']), on=['lang', 'key'], how='left')
    F['fate'] = ''
    F.loc[(F['status'] == 'ok') & (F['status2'] == 'generic'), 'fate'] = 'dropped:generic_noun'
    geocoded = (F['fate'] == '')
    placed = geocoded & F['gc_conf'].isin(['high', 'medium'])
    F.loc[geocoded & ~placed, 'fate'] = 'unplaced:' + F.loc[geocoded & ~placed, 'gc_reason'].fillna('low_confidence').astype(str)
    # --- post filters
    lab = F['gc_cand_name'].fillna('').map(fold)
    names_cont = {fold(x) for x in CONTINENTS}
    cont = placed & (lab.isin(names_cont) | (F['gc_radius_km'].fillna(0) >= 1500))
    F.loc[cont, 'fate'] = 'dropped:continent'
    placed &= ~cont
    chk = F[placed].rename(columns={'gc_source': 'coord_source', 'gc_conf': 'coord_conf', 'gc_lat': 'lat', 'gc_lon': 'lon'}).copy()
    chk['coll_slug'] = 'nerlayer_' + chk['lang']
    chk['entity_key'] = chk['key']
    chk['n_in_period'] = chk['n']
    chk['name'] = chk['form']
    chk['coord_source'] = chk['coord_source'].where(chk['coord_source'] != 'name_latin', 'name_latin')
    rr = VG.rules(chk)
    rule_of = {(c, k): r for c, k, r in zip(rr['coll_slug'], rr['entity_key'], rr['rule'])}
    rl = pd.Series([rule_of.get((c, k), '') for c, k in zip(chk['coll_slug'], chk['entity_key'])], index=chk.index)
    F.loc[rl[rl != ''].index, 'fate'] = 'dropped:' + rl[rl != '']
    placed &= ~F['fate'].str.startswith('dropped:verify') & ~F['fate'].isin(['dropped:' + x for x in set(rl) if x])
    # the Channel Islands box (a Latin 'Caesarea' or a parish name taken for Jersey) and ethnonyms (Swede, Dane)
    ch = placed & pd.Series(VG.in_channel_box(F['gc_lat'].fillna(0), F['gc_lon'].fillna(0)), index=F.index) & ~F['form'].str.contains(r'(?i)jersey|guernsey|man\b|alderney', regex=True)
    F.loc[ch, 'fate'] = 'dropped:channel_islands'
    placed &= ~ch
    eth = placed & pd.Series([bool(PCL.is_ethnonym_surface(x)) for x in F['form']], index=F.index)
    F.loc[eth, 'fate'] = 'dropped:ethnonym'
    placed &= ~eth
    lc = placed & (F['cap_share'] < 0.5) & F['gc_source'].isin(['name_geonames', 'name_latin']) & ~F['gc_evidence'].fillna('').str.contains('corroborated by')
    F.loc[lc, 'fate'] = 'dropped:lowercase_geonames'
    placed &= ~lc
    if a.words and a.words.exists():
        W = {w.strip() for w in a.words.read_text('utf-8').split('\n') if w.strip()}
        corr = F['gc_evidence'].fillna('').str.contains('corroborated by')
        cw = placed & (F['lang'] == 'en') & (~F['key'].str.contains(' ')) & F['key'].isin(W) & (F['gc_source'] == 'name_geonames') & ~corr
        F.loc[cw, 'fate'] = 'dropped:common_word'
        placed &= ~cw
    # far_namesake: the key also names a bigger place far away (read from the GeoNames / TGN index of the run)
    ix = pd.read_parquet(a.geo / 'gc/gn_index_layer.parquet', columns=['key', 'gid', 'lat', 'lon', 'fclass', 'fcode', 'pop'])
    gk = placed & (F['gc_source'] == 'name_geonames') & ~F['gc_evidence'].fillna('').str.contains('corroborated by')
    need = set(F.loc[gk, 'gc_key'].dropna())
    ixs = ix[ix['key'].isin(need)]
    by_key = {k: d for k, d in ixs.groupby('key')}
    far = []
    for i, r in F[gk].iterrows():
        d = by_key.get(r['gc_key'])
        m = GN_EV.search(r['gc_evidence'] or '')
        pop0 = int(m.group(3)) if m else 0
        if d is None:
            continue
        dd = VG.hav(r['gc_lat'], r['gc_lon'], d['lat'].to_numpy(), d['lon'].to_numpy())
        big = ((d['pop'].to_numpy() >= max(100000, pop0)) & d['fclass'].isin(['P', 'A']).to_numpy()) | (d['fcode'].str.startswith(('PCL', 'ADM1')).to_numpy() & (d['pop'].to_numpy() >= max(1000000, pop0)))
        if bool((big & (dd > 150)).any()):
            far.append(i)
    F.loc[far, 'fate'] = 'dropped:far_namesake'
    placed &= ~F['fate'].eq('dropped:far_namesake')
    F.loc[placed, 'fate'] = 'placed'
    F['kind'] = ''
    F.loc[placed, 'kind'] = [kind_of(r) for _, r in F[placed].iterrows()]
    _t(f'fates: {F["fate"].value_counts().head(12).to_dict()}')

    # --- points
    P = F[placed].copy()
    P['cand'] = P['gc_cand'].fillna('')
    P['mname'] = P['gc_cand_name'].fillna(P['form'])
    P['pkey'] = P['gc_qid'].where(P['gc_qid'].notna(), P['cand'].where(P['cand'].str.startswith('gn:'), 'xy:' + P['gc_lat'].round(3).astype(str) + ',' + P['gc_lon'].round(3).astype(str)))
    P['mfold'] = P['mname'].map(fold)
    # greedy merge of points with the same modern name within KM_MERGE km
    P = P.sort_values('n', ascending=False)
    pid = {}
    reps = []                                   # (lat, lon, mfold, pid)
    by_name = defaultdict(list)
    counter = 0
    group = []
    for (k, mf), la, lo in zip(zip(P['pkey'], P['mfold']), P['gc_lat'], P['gc_lon']):
        if k in pid:
            group.append(pid[k]); continue
        hit = None
        for (rl, ro, rid) in by_name[mf]:
            if hav1(la, lo, rl, ro) <= KM_MERGE:
                hit = rid; break
        if hit is None:
            hit = counter; counter += 1
            by_name[mf].append((la, lo, hit))
        pid[k] = hit
        group.append(hit)
    P['pid'] = group
    # second pass: two points of the same place under related names (Milano / Milan, Paris / Parys, England / England, Königreich) or the same area with two centres
    lead = P.sort_values('n', ascending=False).drop_duplicates('pid')
    L_lat, L_lon = lead['gc_lat'].to_numpy(), lead['gc_lon'].to_numpy()
    L_r = lead['gc_radius_km'].fillna(0).to_numpy()
    L_n = lead['mfold'].to_numpy()
    L_sk = np.array([GT.skeleton(x) if x else '' for x in L_n], dtype=object)
    L_k = lead['kind'].to_numpy()
    L_id = lead['pid'].to_numpy()
    repmap = {}
    reps = []                                # indices into lead of the representatives
    for j in range(len(lead)):
        tgt = None
        if reps:
            ri = np.array(reps)
            d = VG.hav(L_lat[j], L_lon[j], L_lat[ri], L_lon[ri])
            rr_ = np.maximum(L_r[ri], L_r[j])
            thr = np.where(rr_ > 15, np.maximum(10.0, 0.3 * rr_), 10.0)
            for q in np.flatnonzero(d <= thr):
                i2 = ri[q]
                a_, b_ = str(L_n[i2]), str(L_n[j])
                if a_ == b_ or L_sk[i2] == L_sk[j] or (len(a_) >= 4 and len(b_) >= 4 and (a_.startswith(b_[:5]) or b_.startswith(a_[:5])) and a_[:4] == b_[:4]):
                    tgt = i2
                    break
        if tgt is None:
            reps.append(j)
            repmap[int(L_id[j])] = int(L_id[j])
        else:
            repmap[int(L_id[j])] = int(L_id[tgt])
    n_before = len(lead)
    P['pid'] = P['pid'].map(repmap)
    counter = P['pid'].nunique()
    _t(f'second merge: {n_before} -> {counter} points')
    F['pid'] = -1
    F.loc[P.index, 'pid'] = P['pid']
    _t(f'{len(P)} forms placed -> {counter} points')

    # --- documents per point
    df = pd.read_parquet(a.geo / 'doc_forms.parquet')
    mp_ = F.loc[P.index, ['lang', 'key', 'pid']]
    d = df.merge(mp_, on=['lang', 'key'])
    dp = d.groupby(['pid', 'doc_id'])['n'].sum().reset_index()
    nd = dp.groupby('pid')['doc_id'].nunique()
    top = dp.sort_values(['pid', 'n', 'doc_id'], ascending=[True, False, True]).groupby('pid').head(a.top_docs)
    docs = pd.read_parquet(a.layer / 'docs.parquet', columns=['doc_id', 'collection', 'member', 'lang', 'year', 'date_confidence', 'tier'])
    colls = d.merge(docs[['doc_id', 'collection']], on='doc_id').groupby(['pid', 'collection'])['n'].sum().reset_index()
    # --- titles and links
    needed = set(top['doc_id'])
    _t(f'{len(needed)} top documents, scanning the text shards for titles')
    shards = sorted((a.layer / 'extract/texts').glob('*/*.jsonl.gz'))
    titles = {}
    with mp.Pool(a.procs) as pool:
        for r in pool.imap_unordered(scan_shard, [(str(s), needed) for s in shards]):
            for k, v in r.items():
                titles.setdefault(k, v)
    _t(f'{len(titles)} titles ({time.time() - t0:.0f}s)')
    (out / 'doc_titles.json').write_text(json.dumps(titles, ensure_ascii=False), 'utf-8')
    dm = docs.set_index('doc_id')
    # --- assemble
    qcols = [f'q{i}' for i in range(10)]
    tot = F.loc[P.index].groupby(F.loc[P.index, 'pid'])
    pts = []
    rows = P.sort_values('n', ascending=False).groupby('pid')
    topd = {k: v for k, v in top.groupby('pid')}
    colls_by = {k: v.sort_values('n', ascending=False) for k, v in colls.groupby('pid')}
    for p, g in rows:
        lead = g.iloc[0]
        w = g['n'].to_numpy()
        lat = float(np.average(g['gc_lat'], weights=w)) if g['gc_lat'].nunique() > 1 else float(lead['gc_lat'])
        lon = float(np.average(g['gc_lon'], weights=w)) if g['gc_lon'].nunique() > 1 else float(lead['gc_lon'])
        lat, lon = float(lead['gc_lat']), float(lead['gc_lon'])          # the point of the most frequent form (all of them lie within 10 km)
        kinds = Counter()
        for k, n in zip(g['kind'], w):
            kinds[k] += int(n)
        q = [int(g[c].sum()) for c in qcols]
        forms_l = [[r['form'], int(r['n']), r['lang']] for _, r in g.sort_values('n', ascending=False).head(a.top_forms).iterrows()]
        srcs = Counter()
        for s_, n in zip(g['gc_source'], w):
            srcs[s_] += int(n)
        cb = colls_by.get(p)
        tdocs = []
        for _, r in topd.get(p, pd.DataFrame(columns=['doc_id', 'n'])).iterrows():
            did = r['doc_id']
            m = dm.loc[did] if did in dm.index else None
            tdocs.append({'id': did, 'title': titles.get(did) or None, 'year': (int(m['year']) if m is not None and pd.notna(m['year']) else None),
                          'collection': (m['collection'] if m is not None else did.split('/')[0]), 'url': doc_link(did, str(m['collection']) if m is not None else '', str(m['lang']) if m is not None else ''),
                          'n': int(r['n'])})
        conf = 'high' if (g['gc_conf'] == 'high').all() else 'medium'
        pt = {'id': f'ml:{p}', 'name': lead['mname'], 'lat': round(lat, 5), 'lon': round(lon, 5), 'k': kinds.most_common(1)[0][0],
              'n': int(w.sum()), 'nd': int(nd.get(p, 0)), 'q': q, 's': 'm', 'gs': srcs.most_common(1)[0][0], 'conf': conf,
              'qid': (lead['gc_qid'] if isinstance(lead['gc_qid'], str) else None), 'gid': (lead['cand'][3:] if lead['cand'].startswith('gn:') else None),
              'cand': lead['cand'] or None, 'forms': forms_l, 'nforms': int(len(g)),
              'langs': {k: int(v) for k, v in g.groupby('lang')['n'].sum().items()},
              'c': int(len(cb)) if cb is not None else 0, 'cols': (cb['collection'].head(3).tolist() if cb is not None else []), 'docs': tdocs}
        r_ = float(lead['gc_radius_km'] or 0)
        if pt['k'] == 'a' and r_ > 0:
            pt['r'] = round(r_)
        pts.append(pt)
    pts.sort(key=lambda x: -x['n'])
    (out / 'points.json').write_text(json.dumps(pts, ensure_ascii=False, separators=(',', ':')), 'utf-8')
    keep = ['lang', 'key', 'form', 'n', 'n_docs', 'status', 'fate', 'pid', 'kind', 'gc_lat', 'gc_lon', 'gc_source', 'gc_conf', 'gc_radius_km', 'gc_cand', 'gc_cand_name',
            'gc_qid', 'gc_p', 'gc_evidence', 'gc_reason']
    F[keep].to_parquet(out / 'forms_placed.parquet', index=False)
    fates = F.groupby('fate').agg(forms=('key', 'size'), mentions=('n', 'sum')).reset_index()
    bl = F.groupby(['lang', 'fate']).agg(forms=('key', 'size'), mentions=('n', 'sum')).reset_index()
    fates.to_csv(out / 'fates.tsv', sep='\t', index=False)
    bl.to_csv(out / 'fates_by_lang.tsv', sep='\t', index=False)
    summ = {'forms': int(len(F)), 'mentions': int(F['n'].sum()), 'points': len(pts), 'placed_forms': int(placed.sum()), 'placed_mentions': int(F.loc[placed, 'n'].sum()),
            'fates': {r['fate']: [int(r['forms']), int(r['mentions'])] for _, r in fates.iterrows()}}
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), 'utf-8')
    _t(json.dumps(summ, ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main()
