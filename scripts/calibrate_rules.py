#!/usr/bin/env python3
"""Recalibrate two rules of the reconcile stage on the labels the manual review and the audit produced (v2.1). Run on Hypatia (Slurm).

    python -I scripts/calibrate_rules.py homonym --v2 out_v2/all --rdir reconcile_v2 --labels manual_review.tsv --out homonym.json
    python -I scripts/calibrate_rules.py verify  --geocode geocode_places.parquet --verify geocode_verify.tsv --audit audit_samples.tsv \
        --casebooks casebooks_place_map.tsv --out verify.json
    python -I scripts/calibrate_rules.py merge homonym.json verify.json --out data/calibration_v2_1.json

Method (both rules). The labelled cases are split 70 / 30 by a hash of their key (seed in ``SEED``): the threshold is fitted on the 70 % (the one
that maximises the number of correct decisions, ties to the most conservative), and the confusion matrices of the old and the new rule are
reported on the other 30 % (held out, never used to choose anything), and on all labels. A label ``undecidable`` is not used to fit or to score; its
decisions are counted apart.

* ``homonym``: the 114 pairs (GeoNames id of a village against a larger namesake far away) that the v2 gate rejected and the manual review judged
  (``gate_homonym``: the id is right / wrong). Positive = the id is wrong = it should be rejected. Old rule: reject every candidate. New rule:
  reject a candidate whose feature lies more than ``A`` km from every other placed place of the edition (``namegate.homonym_check``).
* ``verify``: name-based GeoNames matches that no propagation corroborates, not major features, and that a person (the v1.1 audit of 765 places) or the
  Casebooks directory (match confidence high / medium, a result within 15 km of the directory's point is right) judged. Positive = the match is wrong =
  it should be taken back. Old rule: the v2 ``geocode_verify.tsv`` (v1.1 strata). New rule: take back when the geocoder's own certainty ``p`` is below ``P``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 'renascor-atlas-v2.1'
TRAIN_SHARE = 70
HOM_GRID = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
P_GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]
CB_OK_KM = 15.0


def split_of(key: str) -> str:
    h = int(hashlib.md5(f'{SEED}|{key}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < TRAIN_SHARE else 'test'


def confusion(pred_pos: np.ndarray, y_pos: np.ndarray) -> dict:
    tp, fp = int((pred_pos & y_pos).sum()), int((pred_pos & ~y_pos).sum())
    fn, tn = int((~pred_pos & y_pos).sum()), int((~pred_pos & ~y_pos).sum())
    n = tp + fp + fn + tn
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'n': n, 'accuracy': round((tp + tn) / n, 4) if n else None,
            'precision': round(tp / (tp + fp), 4) if tp + fp else None, 'recall': round(tp / (tp + fn), 4) if tp + fn else None}


def fit(grid, score_fn, conservative_last=True):
    """Best parameter by correct decisions on the training labels; ties go to the last (most conservative) value of ``grid``."""
    best, arg = -1, None
    for g in grid:
        s = score_fn(g)
        if s > best or (s == best and conservative_last):
            best, arg = s, g
    return arg, best


# ---------------------------------------------------------------------------------------------------------------------
def homonym(a) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from ratlas import namegate as NG
    from ratlas import reconcile_gates as RG
    from ratlas import reconcile_v2 as V2
    import pyarrow.parquet as pq
    auth = RG.AuthData(Path(a.rdir))
    votes = RG.id_votes(Path(a.v2))
    dc = V2.default_counts(Path(a.v2))
    pl = V2.apply_default_counts(pq.read_table(Path(a.v2) / 'places_local.parquet').to_pandas(), dc, 'place')
    ctx = RG.edition_context(votes[votes['kind'] == 'place'], pl, auth)
    lab = pd.read_csv(a.labels, sep='\t', dtype=str, keep_default_na=False)
    lab = lab[lab['group'] == 'gate_homonym']
    E = pl.drop_duplicates(['coll_slug', 'entity_key']).set_index(['coll_slug', 'entity_key'])
    rows = []
    for r in lab.itertuples(index=False):
        if (r.coll_slug, r.entity_key) not in E.index or r.id not in auth.gn:
            continue
        er = E.loc[(r.coll_slug, r.entity_key)]
        f = auth.gn[r.id]
        ekeys = NG.name_keys(er['name'], er['variants'])
        same = RG._same_name(ekeys, auth)
        fe = {'gid': r.id, 'lat': f['lat'], 'lon': f['lon'], 'pop': f['pop'], 'cc': f['cc'], 'fcode': f['fcode']}
        cand = NG.homonym_candidate(fe, same, er['name'])
        la, lo = RG.ctx_without(ctx, r.coll_slug, r.entity_key)
        iso = float(np.min(NG.hav(f['lat'], f['lon'], la, lo))) if len(la) else float('nan')
        rows.append(dict(case=r.case_id, coll=r.coll_slug, key=r.entity_key, name=er['name'], gid=r.id, verdict=r.verdict, n_context=int(len(la)),
                         iso_km=iso, candidate=cand is not None, town_km=(cand[1] if cand else float('nan')),
                         split=split_of(f'{r.coll_slug}|{er["name"]}|{r.id}')))
    d = pd.DataFrame(rows)
    out = {'rule': 'homonym gate (isolation of the candidate from the edition)', 'labels_total': int(len(lab)), 'labels_matched': int(len(d)),
           'verdicts': d['verdict'].value_counts().to_dict(), 'min_context': NG.HOMONYM_MIN_CONTEXT, 'grid_km': HOM_GRID}
    dec = d[d['verdict'].isin(['correct', 'wrong']) & d['candidate']].copy()
    dec['y'] = dec['verdict'] == 'wrong'
    ok_ctx = (dec['n_context'] >= NG.HOMONYM_MIN_CONTEXT).to_numpy()

    def predict(df, A):
        return ((df['n_context'] >= NG.HOMONYM_MIN_CONTEXT) & (df['iso_km'] > A)).to_numpy()
    tr, te = dec[dec['split'] == 'train'], dec[dec['split'] == 'test']
    A, best = fit(HOM_GRID, lambda A: int(((predict(tr, A)) == tr['y'].to_numpy()).sum()))
    out['fitted_km'] = A
    out['split'] = {'train': int(len(tr)), 'test': int(len(te)), 'train_wrong': int(tr['y'].sum()), 'test_wrong': int(te['y'].sum())}
    for name, df in (('train', tr), ('test_held_out', te), ('all_labelled', dec)):
        y = df['y'].to_numpy()
        out[name] = {'old_rule_reject_every_candidate': confusion(np.ones(len(df), bool), y), 'new_rule': confusion(predict(df, A), y)}
    # sensitivity: accuracy of every A on all labels
    out['accuracy_by_threshold_all'] = {str(g): confusion(predict(dec, g), dec['y'].to_numpy())['accuracy'] for g in HOM_GRID}
    und = d[(d['verdict'] == 'undecidable') & d['candidate']]
    out['undecidable'] = {'n': int(len(und)), 'old_rejects': int(len(und)), 'new_rejects': int(predict(und, A).sum()),
                          'new_suggested_no_context': int((und['n_context'] < NG.HOMONYM_MIN_CONTEXT).sum())}
    out['pairs_without_context'] = int((~ok_ctx).sum())
    d.to_csv(Path(a.out).with_suffix('.tsv'), sep='\t', index=False)
    return out


# ---------------------------------------------------------------------------------------------------------------------
_GN = re.compile(r"^(?:GeoNames|TGN) (\d+) .*? \((\w+), (\w*), pop (\d+)\)")
MAJOR = frozenset('PCLI PCLD PCLS PCLF PCL PCLH ADM1 ADM1H ADM2 ADMD PPLC PPLA PPLA2 PPLG'.split())


def _hav(a, b, c, d):
    a, b, c, d = (np.radians(np.asarray(x, dtype='float64')) for x in (a, b, c, d))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def verify(a) -> dict:
    g = pd.read_parquet(a.geocode)
    g = g[(g['gc_source'] == 'name_geonames') & g['gc_conf'].isin(['high', 'medium'])].copy()
    ev = g['gc_evidence'].fillna('')
    g['corr'] = ev.str.contains('corroborated by', regex=False)
    m = ev.map(lambda e: _GN.match(e))
    g['fcode'] = m.map(lambda x: x.group(2) if x else None)
    g['pop'] = m.map(lambda x: int(x.group(4)) if x else 0)
    g['cc'] = m.map(lambda x: x.group(3) if x else None)
    g['major'] = [(f in MAJOR) or ((p or 0) >= 50000 and str(f).startswith('PPL')) for f, p in zip(g['fcode'], g['pop'])]
    ver = pd.read_csv(a.verify, sep='\t', dtype=str, keep_default_na=False)
    old = dict(zip(zip(ver['coll_slug'], ver['entity_key']), ver['rule']))
    g['old_rule'] = [old.get(k, '') for k in zip(g['coll_slug'], g['entity_key'])]
    # labels 1: the v1.1 audit
    au = pd.read_csv(a.audit, sep='\t', dtype=str, keep_default_na=False)
    au['lab'] = np.where(au['verdict'] == 'correct', 'C', np.where(au['verdict'].str.startswith('wrong namesake'), 'W', 'x'))
    l1 = au[au['lab'].isin(['C', 'W'])][['coll_slug', 'entity_key', 'lab']].assign(source='audit_v1.1')
    # labels 2: the Casebooks directory (high / medium matches are 90-100 % right)
    cb = pd.read_csv(a.casebooks, sep='\t', dtype=str, keep_default_na=False)
    cb = cb[cb['confidence'].isin(['high', 'medium']) & (cb['lat'] != '')].copy()
    cb['entity_key'] = 'casebooks:' + cb['place_id']
    c = g[g['coll_slug'] == 'casebooks'].merge(cb[['entity_key', 'lat', 'lon']].rename(columns={'lat': 'dlat', 'lon': 'dlon'}), on='entity_key')
    c['km'] = _hav(c['gc_lat'].astype(float), c['gc_lon'].astype(float), c['dlat'].astype(float), c['dlon'].astype(float))
    l2 = c.assign(lab=np.where(c['km'] <= CB_OK_KM, 'C', 'W'), source='casebooks_directory')[['coll_slug', 'entity_key', 'lab', 'source']]
    L = pd.concat([l1, l2], ignore_index=True).drop_duplicates(['coll_slug', 'entity_key'], keep='first')
    d = g.merge(L, on=['coll_slug', 'entity_key'], how='inner')
    d = d[~d['corr'] & ~d['major']].copy()                               # the population the gn_uncorroborated rule is about
    d['y'] = d['lab'] == 'W'
    d['gc_p'] = pd.to_numeric(d['gc_p'], errors='coerce').fillna(0.0)
    d['old'] = d['old_rule'] != ''
    d['split'] = [split_of(f'{c_}|{k}') for c_, k in zip(d['coll_slug'], d['entity_key'])]
    tr, te = d[d['split'] == 'train'], d[d['split'] == 'test']
    P, best = fit(P_GRID, lambda P: int((((tr['gc_p'] < P).to_numpy()) == tr['y'].to_numpy()).sum()), conservative_last=False)
    out = {'rule': 'gn_uncorroborated (certainty p of an uncorroborated GeoNames-only match)', 'labelled': int(len(d)),
           'by_source': d['source'].value_counts().to_dict(), 'wrong': int(d['y'].sum()), 'grid_p': P_GRID, 'fitted_p': P,
           'split': {'train': int(len(tr)), 'test': int(len(te))}}
    for name, df in (('train', tr), ('test_held_out', te), ('all_labelled', d)):
        y = df['y'].to_numpy()
        out[name] = {'old_rule_v1.1_strata': confusion(df['old'].to_numpy(), y), 'new_rule_p_below_fitted': confusion((df['gc_p'] < P).to_numpy(), y),
                     'never_take_back': confusion(np.zeros(len(df), bool), y)}
    out['accuracy_by_threshold_all'] = {str(p): confusion((d['gc_p'] < p).to_numpy(), d['y'].to_numpy())['accuracy'] for p in P_GRID}
    # every old rule on the labels (precision of the take-backs)
    allg = g.merge(L, on=['coll_slug', 'entity_key'], how='inner')
    allg['y'] = allg['lab'] == 'W'
    out['old_rules_on_all_labelled_matches'] = {r: {'taken_back': int(len(x)), 'wrong': int(x['y'].sum()), 'right': int((~x['y']).sum())}
                                                for r, x in allg[allg['old_rule'] != ''].groupby('old_rule')}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('homonym')
    p.add_argument('--v2', required=True)
    p.add_argument('--rdir', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--geocode', required=True)
    p.add_argument('--verify', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--casebooks', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('merge')
    p.add_argument('files', nargs='+')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'merge':
        res = {Path(f).stem.replace('cal_', ''): json.loads(Path(f).read_text()) for f in a.files}
        res['seed'] = SEED
        res['train_share_percent'] = TRAIN_SHARE
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return 0
    res = homonym(a) if a.cmd == 'homonym' else verify(a)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
