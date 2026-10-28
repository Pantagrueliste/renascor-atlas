"""NER layer v3 (Mac): the model layer v2 + the sets tei, extra and seen, all corrections applied, one layer.

    nice -n 19 python scripts/ner2_layer_v3_build.py --work FIX_DIR --v3 V3_DIR --globe globe_places.json --out-dir OUT [--no-v3]

``FIX_DIR`` = the v1 / es tables of the layer v2 (forms_placed, doc_forms, docs_v1, es/, fix/ctx, fix/form_pper, fix/titles). ``V3_DIR`` = layer/v3 (tei/docs, extra/docs,
seen/docs, geo_all/{points,layer,doc_forms}, fix_all/{ctx,form_pper}, dedup_all/{dup_flags,...}).

The units of the new sets (one per written form and language, run ``v3``) join the state of ner2_layer_fix.py after the Spanish step, so that every later step reaches
them: Nice / Ile-de-France / Jordan by the cue in the text, Antiochia, person-like forms, the blacklist, the table edits, the radius floor, the merge of duplicate points, then
the sweep verdicts (data/model_layer_overrides.json, 109 + 62 entries) and the merge again. Added here:

  * documents: Nakala out (decision 11), documents that are duplicates of the layer or of another new document out (dedup_all), no document id twice;
  * ``verdict forms``: a v3 form that the v2 layer holds in a point decided by the sweep joins that point before the verdicts are applied;
  * titles (v29, ``--titles``): the mentions inside a royal / noble title (and an ecclesiastical one with ``--drop-ecclesiastical`` or ratlas.titles.ECCLESIASTICAL_REMOVED) are taken out of the
    document counts of their form before anything else is computed (data/curation/title_excluded_model.tsv.gz, ratlas.titles);
  * guard rules of ratlas.curation on the model points: ``ancient`` (a name of ancient_names.tsv outside its region) and ``far_homonym`` (a point on another continent
    that the collection rarely visits, when the same name is placed in the Old World by other collections), applied to the units of the new sets (``--rules all`` also to v2).

``--no-v3`` rebuilds the v2 layer (regression test: it must give model_layer_v2.json of v28 exactly).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
import ner2_layer_fix as FX  # noqa: E402
from ratlas import curation as CUR  # noqa: E402
from ratlas import titles as TIT  # noqa: E402

EXCLUDED_COLLECTIONS = {'Nakala', 'LondonLives', 'London_Lives', 'MAP'}
DUP_COVER = 0.5
DUP_MIN_SHINGLES = 16


def _t(m):
    print(f'[v3-build {time.strftime("%H:%M:%S")}] {m}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# documents of the new sets: duplicates
# ---------------------------------------------------------------------------------------------------------------------
def decide_duplicates(flags: pd.DataFrame, intra: pd.DataFrame, docs: pd.DataFrame):
    """-> dict doc_id -> reason, for the new documents to leave out.

    layer: union containment >= 0.5 in the layer's documents (>= 16 sampled shingles), or an exact copy of a layer document;
    new: containment >= 0.5 in another new document: mutual -> the less tagged goes, then the shorter; contained in a larger one -> the contained goes, unless it carries editor tags
    and the larger one does not; exact copies inside the new sets: the less tagged / shorter goes.
    """
    drop = {}
    f = flags.set_index('doc_id')
    for d, r in f.iterrows():
        if r['exact_layer_doc']:
            drop[d] = f"copie exacte de {r['exact_layer_doc']}"
        elif r['layer_union_contain'] >= DUP_COVER and r['sampled_shingles'] >= DUP_MIN_SHINGLES:
            drop[d] = f"contenu à {100 * r['layer_union_contain']:.0f} % dans la couche (proche : {r['layer_best_doc']})"
    tags = docs['n_tags'].fillna(0).to_dict() if 'n_tags' in docs else {}
    words = f['words'].to_dict()

    def rank(d):
        return (tags.get(d, 0), words.get(d, 0), d)

    cand = intra[(intra['contain_cand'] >= DUP_COVER) & (intra['cand_shingles'] >= DUP_MIN_SHINGLES)].copy()
    cont = {(a, b): c for a, b, c in zip(cand['cand'], cand['other'], cand['contain_cand'])}
    allc = {(a, b): c for a, b, c in zip(intra['cand'], intra['other'], intra['contain_cand'])}
    for (a, b), c in sorted(cont.items(), key=lambda kv: (-kv[1], kv[0])):
        if a in drop or b in drop:
            continue
        back = allc.get((b, a), 0.0)
        if back >= DUP_COVER:
            lose = a if rank(a) < rank(b) else b
        else:
            if tags.get(a, 0) > 0 and tags.get(b, 0) == 0:
                continue
            lose = a
        keep = b if lose == a else a
        drop[lose] = f'contenu à {100 * cont[(a, b)] if lose == a else 100 * back:.0f} % dans un autre document nouveau ({keep})'
    for d, r in f.iterrows():
        ex = [x for x in str(r['exact_new_docs']).split('|') if x]
        if ex and d not in drop:
            grp = [d] + ex
            if all(g not in drop for g in grp):
                keep = max(grp, key=rank)
                if keep != d:
                    drop[d] = f'copie exacte de {keep}'
    return drop


# ---------------------------------------------------------------------------------------------------------------------
# state: add the new sets
# ---------------------------------------------------------------------------------------------------------------------
def load_v3(V3: Path, args):
    F3 = pd.read_parquet(V3 / 'geo_all/points/forms_placed.parquet')
    DF = pd.read_parquet(V3 / 'geo_all/doc_forms.parquet')
    dl = pd.read_parquet(V3 / 'geo_all/layer/docs.parquet')
    cols = [c for c in ('doc_id', 'collection', 'year', 'lang', 'set', 'n_tags', 'n_words') if c in dl.columns]
    docs3 = dl[cols].drop_duplicates('doc_id').set_index('doc_id')
    ctx3 = pd.read_parquet(V3 / 'fix_all/ctx.parquet')
    pp3 = pd.read_parquet(V3 / 'fix_all/form_pper.parquet')
    titles3 = json.loads((V3 / 'geo_all/points/doc_titles.json').read_text('utf-8'))
    return F3, DF, docs3, ctx3, pp3, titles3


def make_v3_units(F3: pd.DataFrame, first_uid: int):
    U3 = FX.make_units(F3, 'v3')
    U3.index = U3.index + first_uid
    U3['grp0'] = U3['grp']
    return U3


RUN_SRC = {'v1': 'ml', 'es': 'mles', 'v3': 'ml3'}


def subtract_titles(D: pd.DataFrame, U: pd.DataFrame, excl: pd.DataFrame):
    """Takes the mentions inside a title out of ``D`` (uid, doc_id, n): ``excl`` = ratlas.titles.load_model_exclusions. Returns (D without them, report)."""
    ex = {(r.src, r.doc_id, r.lang, r.key): int(r.n_excluded) for r in excl.itertuples()}
    key_of = {u: (RUN_SRC[r], lg, k) for u, r, lg, k in zip(U.index, U['run'], U['lang'], U['key'])}
    cat = [ex.get((key_of[u][0], d, key_of[u][1], key_of[u][2]), 0) if u in key_of else 0 for u, d in zip(D['uid'], D['doc_id'])]
    D = D.copy()
    cat = np.array(cat)
    removed = int(np.minimum(cat, D['n'].to_numpy()).sum())
    D['n'] = D['n'] - np.minimum(cat, D['n'].to_numpy())
    return D[D['n'] > 0].reset_index(drop=True), {'lines_matched': int((cat > 0).sum()), 'mentions_removed': removed}


def _pick(run, only_run, skip_run):
    return (only_run is None or run == only_run) and (skip_run is None or run != skip_run)


def rule_ancient_units(S, ancient, only_run=None, skip_run=None, apply=True):
    """Unplace the placed units whose folded form (or point name) is a name of ancient_names.tsv, outside its region."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0).astype(int)
    hit = []
    for uid in U.index[U['placed'] & (n > 0)]:
        if not _pick(U.at[uid, 'run'], only_run, skip_run):
            continue
        for k in {CUR.fold(U.at[uid, 'form']), CUR.fold(U.at[uid, 'cname'])}:
            if k in ancient and not CUR.in_regions(float(U.at[uid, 'lat']), float(U.at[uid, 'lon']), ancient[k]):
                hit.append((uid, k))
                break
    rows = []
    for uid, k in hit:
        rows.append((U.at[uid, 'form'], U.at[uid, 'lang'], U.at[uid, 'run'], int(n[uid]), round(float(U.at[uid, 'lat']), 2), round(float(U.at[uid, 'lon']), 2)))
        if apply:
            S.unplace(uid, 'curation:ancient')
    return rows


def rule_far_homonym(S, only_run=None, skip_run=None, apply=True):
    """A unit on another continent (zone americas, africa_s, asia, oceania), not an area, is unplaced in the documents of a collection that sends less than 1 % of its placed
    mentions to that zone, when the same folded name is placed in the Old World by the other collections for 2 mentions or more (ratlas.curation.rule_far_homonym, per unit)."""
    U, D, docs = S.U, S.D, S.docs
    pl = U[U['placed']].copy()
    pl['zone'] = [CUR.zone(float(a), float(b)) for a, b in zip(pl['lat'], pl['lon'])]
    pl['nk'] = pl['form'].map(CUR.fold)
    Dp = D[D['uid'].isin(pl.index)].copy()
    Dp['coll'] = Dp['doc_id'].map(docs['collection']).fillna(Dp['doc_id'].str.split('/').str[0])
    Dp['zone'] = Dp['uid'].map(pl['zone'])
    Dp['nk'] = Dp['uid'].map(pl['nk'])
    w = Dp.groupby(['coll', 'zone'])['n'].sum().unstack(fill_value=0)
    share = w.div(w.sum(axis=1).clip(lower=1), axis=0)
    old = Dp[Dp['zone'] == 'old']
    old_by = old.groupby(['nk', 'coll'])['n'].sum()
    tot = old_by.groupby(level=0).sum()
    rows = []
    moved = 0
    for uid in pl.index[(pl['zone'].isin(CUR.FAR_ZONES)) & (pl['kind'] != 'a')]:
        if not _pick(U.at[uid, 'run'], only_run, skip_run):
            continue
        k, z = pl.at[uid, 'nk'], pl.at[uid, 'zone']
        if k not in tot.index:
            continue
        have = S.doc_counts(uid)
        by_c = defaultdict(dict)
        for d, c in have.items():
            by_c[docs['collection'].get(d) or d.split('/')[0]][d] = c
        mv = {}
        for c, dd in by_c.items():
            sh = share.at[c, z] if (c in share.index and z in share.columns) else 0.0
            if sh >= CUR.FAR_SHARE:
                continue
            if tot[k] - old_by.get((k, c), 0) >= CUR.FAR_OLD_MENTIONS:
                mv.update(dd)
        if mv:
            if not apply:
                tot_n = sum(c for d, c in have.items() if d in mv)
            elif len(mv) == len(have):
                tot_n = sum(have.values())
                S.unplace(uid, 'curation:far_homonym')
            else:
                nu, tot_n = S.move_mentions(uid, mv, 'curation:far')
                if nu is not None:
                    S.unplace(nu, 'curation:far_homonym')
            rows.append((U.at[uid, 'form'], U.at[uid, 'lang'], U.at[uid, 'run'], int(tot_n), z, round(float(U.at[uid, 'lat']), 2), round(float(U.at[uid, 'lon']), 2)))
            moved += int(tot_n)
    return rows


def propagate_verdict_forms(S, entries):
    """A v3 form that the v2 layer holds in a point decided by the sweep joins that point (same lang + key), so that the verdict reaches it."""
    U = S.U
    n = S.unit_n().reindex(U.index).fillna(0).astype(int)
    groups = set()
    for e in entries:
        try:
            groups.add(U.at[FX._anchor_uid(S, e['anchor']), 'grp'])
        except RuntimeError:
            pass
    old = U[(U['run'] != 'v3') & U['placed'] & (U['variant'] == '') & U['grp'].isin(groups) & (n > 0)]
    ref = {}
    for uid, lg, k in zip(old.index, old['lang'], old['key']):
        ref.setdefault((lg, k), uid)
    rows = []
    for uid in U.index[(U['run'] == 'v3') & U['placed'] & (n > 0)]:
        r = ref.get((U.at[uid, 'lang'], U.at[uid, 'key']))
        if r is not None and U.at[uid, 'grp'] != U.at[r, 'grp']:
            rows.append((U.at[uid, 'form'], U.at[uid, 'lang'], int(n[uid]), str(U.at[uid, 'cname']), str(U.at[r, 'cname'])))
            FX.regroup(S, uid, r, 'v3:verdict_form')
    return rows


def sets_snapshot(S, setof):
    """Placed mentions per set (v2 for the documents of the v2 tables)."""
    n = S.unit_n()
    pl = S.U.index[S.U['placed']]
    Dp = S.D[S.D['uid'].isin(pl)]
    s = Dp['doc_id'].map(setof).fillna('v2')
    return {k: int(v) for k, v in Dp.groupby(s)['n'].sum().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--v3', required=True)
    ap.add_argument('--globe', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--overrides', default=None)
    ap.add_argument('--overrides-v3', default=str(HERE.parent / 'data' / 'model_layer_overrides_v3.json'), help='verdicts of the review of the new points (ignored with --no-v3, or when the file is missing)')
    ap.add_argument('--no-v3', action='store_true')
    ap.add_argument('--rules', default='all', choices=['new', 'all', 'none'])
    ap.add_argument('--tag', default='v3')
    ap.add_argument('--titles', action='store_true', help='remove the mentions inside a title (data/curation/title_excluded_model.tsv.gz)')
    ap.add_argument('--keep-ecclesiastical', action='store_true', help='with --titles: leave the ecclesiastical titles (bishop of ...) in, so that their places stay')
    ap.add_argument('--drop-ecclesiastical', action='store_true', help='with --titles: take the ecclesiastical titles out too (default: ratlas.titles.ECCLESIASTICAL_REMOVED)')
    a = ap.parse_args()
    W, V3, OUT = Path(a.work), Path(a.v3), Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    U, D, docs = FX.load_state(W)
    ctx = pd.read_parquet(W / 'fix/ctx.parquet')
    pper = pd.read_parquet(W / 'fix/form_pper.parquet')
    titles = json.loads((W / 'fix/titles.json').read_text('utf-8'))
    log_v3 = {}
    setof = {}
    D3 = None
    if not a.no_v3:
        F3, DF, docs3, ctx3, pp3, titles3 = load_v3(V3, a)
        U3 = make_v3_units(F3, int(U.index.max()) + 1)
        ix = {(lg, k): i for i, (lg, k) in zip(U3.index, zip(U3['lang'], U3['key']))}
        DF = DF.copy()
        DF['uid'] = [ix[(lg, k)] for lg, k in zip(DF['lang'], DF['key'])]
        old_ids = set(docs.index)
        clash = sorted(set(docs3.index) & old_ids)
        # documents to leave out
        reasons = {}
        for d in clash:
            reasons[d] = 'identifiant déjà dans la couche v2'
        for d, c in docs3['collection'].items():
            if c in EXCLUDED_COLLECTIONS:
                reasons.setdefault(d, f'collection {c} exclue')
        flags = pd.read_csv(V3 / 'dedup_all/dup_flags.tsv', sep='\t', keep_default_na=False)
        intra = pd.read_csv(V3 / 'dedup_all/intra.tsv', sep='\t', keep_default_na=False)
        dups = decide_duplicates(flags, intra, docs3)
        for d, r in dups.items():
            reasons.setdefault(d, 'doublon : ' + r)
        log_v3['reasons'] = reasons
        log_v3['dup_docs'] = dups
        D3 = DF[~DF['doc_id'].isin(reasons)][['uid', 'doc_id', 'n']].copy()
        docs3 = docs3[~docs3.index.isin(reasons)]
        log_v3['mentions_dropped_docs'] = int(DF[DF['doc_id'].isin(reasons)]['n'].sum())
        U3['placed_v3'] = (U3['fate'] == 'placed')
        U3['placed'] = False
        U = pd.concat([U, U3])
        U['grp0'] = U['grp0'].fillna(U['grp'])
        docs = pd.concat([docs, docs3[['collection', 'year', 'lang']]])
        setof = docs3['set'].to_dict()
        ctx = pd.concat([ctx, ctx3], ignore_index=True)
        pper = pd.concat([pper, pp3], ignore_index=True)
        titles = {**titles, **titles3}
    if a.titles:
        ecc = False if a.keep_ecclesiastical else True if a.drop_ecclesiastical else TIT.ECCLESIASTICAL_REMOVED
        excl = TIT.load_model_exclusions(ecclesiastical=ecc)
        D, r1 = subtract_titles(D, U, excl)
        if D3 is not None:
            D3, r3 = subtract_titles(D3, U, excl)
        else:
            r3 = {'lines_matched': 0, 'mentions_removed': 0}
        log_v3['titles'] = {'ecclesiastical_removed': ecc, 'lines_in_table': len(excl), 'v1_es': r1, 'v3': r3,
                            'mentions_removed': r1['mentions_removed'] + r3['mentions_removed']}
        _t(f"titles: {log_v3['titles']['mentions_removed']} mentions out ({r1['lines_matched']} + {r3['lines_matched']} lines of {len(excl)})")
    S = FX.State(U, D, docs)
    GM = FX.load_module(HERE / 'ner2_layer_geo_match.py', 'ner2_layer_geo_match_mod')
    track = []

    def snap(name):
        track.append((name, sets_snapshot(S, setof)))

    FX.stage_baseline(S)
    FX.stage_nakala(S)
    FX.stage_spanish(S)
    snap('v2 avant ajout')
    if D3 is not None:
        S.U.loc[S.U['run'] == 'v3', 'placed'] = S.U.loc[S.U['run'] == 'v3', 'placed_v3'].astype(bool)
        S.D = pd.concat([S.D, D3], ignore_index=True)
        S.step('v3 : tei + extra + seen ajoutés', documents=int(D3['doc_id'].nunique()), mentions_dropped_docs=log_v3['mentions_dropped_docs'])
        snap('ajout des jeux')
    for fn, nm in ((lambda: FX.stage_nice(S, ctx), 'Nice'), (lambda: FX.stage_antioch(S), 'Antioche'), (lambda: FX.stage_pper(S, pper), 'personnes'),
                   (lambda: FX.stage_idf(S, ctx), 'Île-de-France'), (lambda: FX.stage_blacklist(S), 'liste noire'), (lambda: FX.stage_table(S, ctx), 'tables'),
                   (lambda: FX.stage_floor(S), 'rayon')):
        fn()
        snap(nm)
    _, merged = FX.stage_merge(S, GM.keys_of)
    snap('fusion 1')
    entries = FX.load_overrides(a.overrides or FX.OVERRIDES)
    if D3 is not None and a.overrides_v3 and Path(a.overrides_v3).exists():
        entries = entries + FX.load_overrides(a.overrides_v3)
        log_v3['overrides_v3'] = len(entries)
    if D3 is not None:
        log_v3['verdict_forms'] = propagate_verdict_forms(S, entries)
        S.step('v3 : formes des points jugés', forms=len(log_v3['verdict_forms']), mentions=sum(r[2] for r in log_v3['verdict_forms']))
        snap('formes jugées')
    FX.stage_sweep(S, entries)
    snap('balayage')
    _, merged2 = FX.stage_merge(S, GM.keys_of, step_name='P2 merge again after the sweep')
    merged = merged + merged2
    snap('fusion 2')
    if a.rules != 'none' and D3 is not None:
        only = 'v3' if a.rules == 'new' else None
        anc = CUR.load_ancient()
        if a.rules == 'new':           # the v2 units are only listed, as the build check of the atlas does
            log_v3['ancient_v2_listed'] = rule_ancient_units(S, anc, skip_run='v3', apply=False)
            log_v3['far_v2_listed'] = rule_far_homonym(S, skip_run='v3', apply=False)
        r_anc = rule_ancient_units(S, anc, only)
        S.step('règle noms antiques', forms=len(r_anc), mentions=sum(r[3] for r in r_anc))
        snap('antiques')
        r_far = rule_far_homonym(S, only)
        S.step('règle homonymes lointains', forms=len(r_far), mentions=sum(r[3] for r in r_far))
        snap('lointains')
        log_v3['ancient'] = r_anc
        log_v3['far'] = r_far
    # assemble
    pts, docs_by = FX.assemble(S, titles)
    # the label of a point that the v2 layer already holds comes from its v2 forms only (labels stay as in v2); a point of the new sets alone is labelled from its forms
    import copy
    S2 = copy.copy(S)
    S2.D = S.D[~S.D['doc_id'].isin(setof.keys())]
    pts2, _ = FX.assemble(S2, titles)
    forms_v2 = {p['_g']: p['_forms_all'] for p in pts2}
    for p in pts:
        p['_forms_label'] = forms_v2.get(p['_g']) or p['_forms_all']
    pts.sort(key=lambda p: (-p['n'], p['id']))
    G = json.loads(Path(a.globe).read_text('utf-8'))['places']
    for i, p in enumerate(pts):
        p['id'] = f'ml2:{i}'
    em = FX.match_editors(pts, G, GM.keys_of)
    S.step('editor matching', **em)
    lookup = FX.exonym_lookup()
    for p in pts:
        if not p['name']:
            fl = p['_forms_label']
            p['name'] = FX.label_for(fl, FX.exonym_label_of(fl, lookup), (p.get('ed') or {}).get('name'))
        p['tier'] = FX.tier_of(p['n'])
        p['prev'] = [('ml:' + x[3:] if x.startswith('v1:') else 'mles:' + x[3:] if x.startswith('es:') else 'ml3:' + x[3:] if x.startswith('v3:') else x) for x in p['prev']]
    for p in pts:
        p.pop('_r_fixed', None)
        p.pop('_forms_all', None)
        p.pop('_forms_label', None)
    # per-set / per-collection accounting
    acc = account(S, pts, docs_by, setof, docs)
    ns = argparse.Namespace(out=str(OUT / 'model_layer_v2.json'), docs_out=str(OUT / 'model_layer_v2_docs.json'), report=None, stats_out=None)
    FX.write_outputs(ns, S, pts, docs_by, titles, {}, em, merged, {'old': {'places': []}})
    # header: v3
    lay = json.loads(Path(ns.out).read_text('utf-8'))
    h = lay['header']
    if D3 is not None:
        h['title'] = 'Renascor Atlas, places found by the NER model on untagged texts (layer m), version 3: v2 + TEI editions + repaired documents + training editions, corrected'
        h['sets'] = {'v2': 'couche v2 (22-23 oct.)', 'tei': 'éditions TEI du jeu principal', 'extra': 'documents réparés', 'seen': 'éditions TEI de l\'entraînement (texte non balisé)'}
        h['note'] = h['note'] + '; v3: only text outside editor tags'
    Path(ns.out).write_text(json.dumps(lay, ensure_ascii=False, separators=(',', ':')), 'utf-8')
    (OUT / 'build_log.json').write_text(json.dumps({'steps': h['steps'], 'track': track, 'accounting': acc, 'v3': {k: v for k, v in log_v3.items()}, 'seconds': round(time.time() - t0)},
                                                   ensure_ascii=False, default=str, indent=1), 'utf-8')
    _t(f'done: {len(pts)} points, {sum(p["n"] for p in pts)} mentions, main {sum(p["tier"] == "main" for p in pts)} in {round(time.time() - t0)} s')


def account(S, pts, docs_by, setof, docs):
    """Placed mentions, documents and points per set and per collection, from the final points (before the point ids are dropped)."""
    rows = []
    for p in pts:
        for d, k in docs_by.get(p['_g'], []):
            rows.append((p['id'], p['tier'], d, int(k)))
    R = pd.DataFrame(rows, columns=['pid', 'tier', 'doc_id', 'n'])
    R['set'] = R['doc_id'].map(setof).fillna('v2')
    R['coll'] = R['doc_id'].map(docs['collection']).fillna(R['doc_id'].str.split('/').str[0])
    out = {}
    g = R.groupby(['set', 'coll']).agg(mentions=('n', 'sum'), documents=('doc_id', 'nunique'), points=('pid', 'nunique'))
    out['by_set_coll'] = {f'{s}|{c}': {k: int(v) for k, v in r.items()} for (s, c), r in g.iterrows()}
    g = R.groupby('set').agg(mentions=('n', 'sum'), documents=('doc_id', 'nunique'), points=('pid', 'nunique'))
    out['by_set'] = {s: {k: int(v) for k, v in r.items()} for s, r in g.iterrows()}
    sets_of_pt = R.groupby('pid')['set'].agg(lambda x: tuple(sorted(set(x))))
    out['points_by_sets'] = {'+'.join(k): int(v) for k, v in sets_of_pt.value_counts().items()}
    out['points_new_only'] = int(sum(1 for t in sets_of_pt if 'v2' not in t))
    out['mentions_in_new_only_points'] = int(R[R['pid'].isin(sets_of_pt.index[[('v2' not in t) for t in sets_of_pt]])]['n'].sum())
    out['by_coll'] = {c: int(v) for c, v in R.groupby('coll')['n'].sum().items()}
    return out


if __name__ == '__main__':
    main()
