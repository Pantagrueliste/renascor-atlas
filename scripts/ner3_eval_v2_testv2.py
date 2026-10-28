"""RenNER v2: LOC on test_v2 (reference version "v2" of ner2_score_testv2.py) for a few systems, per language, with paired differences.

    python scripts/ner3_eval_v2_testv2.py --gold test2/test_v2.jsonl --pred NAME=path ... --ref r3_mdeberta_HM --out scores.json [--boot 2000]

Same counting as ner2_score_testv2.py, version v2: repaired text, editor gold + 'add' - 'remove', ambiguous tokens masked in gold and
predictions, strict LOC spans, micro P/R/F1. Bootstrap by document (seed 1, 2000 draws, the same draws for every system), as there.
Subsets: all, no_koenigsfelden, each language. Paired: F1(system) - F1(ref) on the same draws. PER (secondary): editor PER where per_valid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
sys.path.insert(0, str(HERE))
from ner2_score_testv2 import KOE, NO_LOC, counts_of, drop, f1, prf  # noqa: E402
from ratlas.ner.ner2_cand import read_preds  # noqa: E402
from ratlas.ner.spans import tags_to_spans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', action='append', default=[])
    ap.add_argument('--ref', default='r3_mdeberta_HM')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    recs = []
    for line in open(a.gold, encoding='utf-8'):
        r = json.loads(line)
        if not r['loc_valid'] or r['edition'] == NO_LOC:
            continue
        recs.append({'id': r['id'], 'edition': r['edition'], 'doc': r['doc'], 'lang': r.get('lang') or 'und',
                     'gold': set(map(tuple, r['loc_gold_v2'])), 'mask': set(r['loc_mask']),
                     'per': set(tags_to_spans(r['ner_tags_per'])) if r.get('per_valid') else None})
    N = len(recs)
    docs = {}
    for r in recs:
        docs.setdefault((r['edition'], r['doc']), len(docs))
    doc_idx = np.array([docs[(r['edition'], r['doc'])] for r in recs])
    nd = len(docs)
    lang = np.array([r['lang'] for r in recs])
    edition = np.array([r['edition'] for r in recs])
    masks = {'all': np.ones(N, bool), 'no_koenigsfelden': edition != KOE}
    for lg in sorted(set(lang)):
        masks[f'lang={lg}'] = lang == lg
    pm = np.array([r['per'] is not None for r in recs])
    rng = np.random.default_rng(1)
    idx = rng.integers(0, nd, size=(a.boot, nd))
    W = np.zeros((a.boot, nd))
    for b in range(a.boot):
        W[b] = np.bincount(idx[b], minlength=nd)

    def summarize(c, m):
        dc = np.zeros((nd, 3))
        for j in range(3):
            dc[:, j] = np.bincount(doc_idx[m], weights=c[m, j], minlength=nd)
        p, r, f = prf(dc.sum(0))
        T = W @ dc
        fb = f1(T[:, 0], T[:, 1], T[:, 2])
        tot = dc.sum(0)
        return ({'tp': int(tot[0]), 'fp': int(tot[1]), 'fn': int(tot[2]), 'p': round(p, 4), 'r': round(r, 4), 'f1': round(f, 4),
                 'f1_ci95': [round(float(x), 4) for x in np.percentile(fb, [2.5, 97.5])]}, fb)

    out = {'gold': a.gold, 'n_passages': N, 'n_documents': nd, 'bootstrap': {'draws': a.boot, 'seed': 1, 'unit': 'document'},
           'passages_per_subset': {s: int(m.sum()) for s, m in masks.items()}, 'scores': {}, 'paired_vs_ref': {}}
    draws = {}
    for s in a.pred:
        name, path = s.split('=', 1)
        preds = read_preds(Path(path))
        c = np.zeros((N, 3), dtype=np.int64)
        cp = np.zeros((N, 3), dtype=np.int64)
        miss = 0
        for k, r in enumerate(recs):
            p = preds.get(r['id'])
            if p is None:
                miss += 1
                continue
            P = {(x['s'], x['e']) for x in p['LOC']}
            c[k] = counts_of(drop(r['gold'], r['mask']), drop(P, r['mask']))
            if r['per'] is not None:
                cp[k] = counts_of(r['per'], {(x['s'], x['e']) for x in p['PER']})
        sc = {'missing': miss}
        for sub, m in masks.items():
            sc[sub], draws[(name, sub)] = summarize(c, m)
        sc['per_editor'], _ = summarize(cp, pm)
        out['scores'][name] = sc
        print(name, 'missing', miss, ' '.join(f"{k.replace('lang=', '')}:{v['f1']:.3f}" for k, v in sc.items() if isinstance(v, dict)), flush=True)
    for name in out['scores']:
        if name == a.ref:
            continue
        e = {}
        for sub in masks:
            d = draws[(name, sub)] - draws[(a.ref, sub)]
            e[sub] = {'diff': round(out['scores'][name][sub]['f1'] - out['scores'][a.ref][sub]['f1'], 4),
                      'ci95': [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]}
        out['paired_vs_ref'][name] = e
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('written', a.out)


if __name__ == '__main__':
    main()
