"""Agreement of several models as a filter (audit): spans predicted by at least k of the n models, scored on the held-out test editions
(strict, relaxed, and the share of false positives that are strings tagged elsewhere in the edition), per language and in total."""
import argparse, gzip, json
from collections import Counter, defaultdict

def spans_of(tags):
    out, st = [], None
    for i, t in enumerate(list(tags) + [0]):
        if t in (0, 1):
            if st is not None: out.append((st, i)); st = None
            if t == 1: st = i
    return out
def norm(s): return s.lower().replace('ſ', 's')
def ov(a, b): return a[0] < b[1] and b[0] < a[1]

ap = argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--pred-dir'); ap.add_argument('--models'); ap.add_argument('--out'); ap.add_argument('--conf', type=float, default=0.0)
a = ap.parse_args()
models = a.models.split(',')
recs = {}
for line in open(a.data + '/test.jsonl', encoding='utf-8'):
    d = json.loads(line); recs[d['id']] = d
preds = {}
for m in models:
    preds[m] = {}
    for line in gzip.open(f'{a.pred_dir}/{m}__test.jsonl.gz', 'rt', encoding='utf-8'):
        d = json.loads(line); preds[m][d['id']] = {(x['s'], x['e']): x['conf'] for x in d['LOC']}
surf = defaultdict(set)
for i, r in recs.items():
    if r.get('loc_valid', True):
        for s, e in spans_of(r['ner_tags_loc']): surf[r['edition']].add(norm(' '.join(r['tokens'][s:e])))
res = {}
for k in range(1, len(models) + 1):
    c = defaultdict(Counter)
    for i, r in recs.items():
        if not r.get('loc_valid', True): continue
        g = set(spans_of(r['ner_tags_loc']))
        cnt = Counter()
        for m in models:
            for sp, cf in preds[m][i].items():
                if cf >= a.conf: cnt[sp] += 1
        ps = {sp for sp, n in cnt.items() if n >= k}
        for grp in ('all', 'lang=' + (r.get('lang') or 'unk'), 'edition=' + r['edition']):
            c[grp]['tp'] += len(g & ps); c[grp]['fp'] += len(ps - g); c[grp]['fn'] += len(g - ps)
            c[grp]['p_ov'] += sum(1 for x in ps if any(ov(x, y) for y in g)); c[grp]['n_pred'] += len(ps)
            c[grp]['g_ov'] += sum(1 for y in g if any(ov(x, y) for x in ps)); c[grp]['n_gold'] += len(g)
            c[grp]['fp_else'] += sum(1 for x in ps - g if not any(ov(x, y) for y in g) and norm(' '.join(r['tokens'][x[0]:x[1]])) in surf[r['edition']])
    out = {}
    for grp, v in c.items():
        tp, fp, fn = v['tp'], v['fp'], v['fn']
        p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        out[grp] = {'tp': tp, 'fp': fp, 'fn': fn, 'p': round(p, 4), 'r': round(rc, 4), 'f1': round(2 * p * rc / (p + rc), 4) if p + rc else 0,
                    'relaxed_p': round(v['p_ov'] / v['n_pred'], 4) if v['n_pred'] else 0, 'relaxed_r': round(v['g_ov'] / v['n_gold'], 4) if v['n_gold'] else 0,
                    'fp_tagged_elsewhere': v['fp_else']}
    res[f'k>={k}'] = out
json.dump({'models': models, 'min_conf': a.conf, 'results': res}, open(a.out, 'w'), indent=1)
for k, v in res.items(): print(k, v['all'])
