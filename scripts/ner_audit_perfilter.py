"""Does the PER head help to drop LOC false positives? For each LOC span predicted by a model: does it overlap a PER span predicted by the same model?
Counts true positives (strict), false positives of each kind, and the effect on strict P/R/F1 of dropping the LOC spans that overlap a PER span."""
import argparse, gzip, json
from collections import Counter, defaultdict
def spans_of(tags):
    out, st = [], None
    for i, t in enumerate(list(tags) + [0]):
        if t in (0, 1):
            if st is not None: out.append((st, i)); st = None
            if t == 1: st = i
    return out
def ov(a, b): return a[0] < b[1] and b[0] < a[1]
ap = argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--pred-dir'); ap.add_argument('--models'); ap.add_argument('--out'); a = ap.parse_args()
recs = {}
for line in open(a.data + '/test.jsonl', encoding='utf-8'):
    d = json.loads(line); recs[d['id']] = d
out = {}
for m in a.models.split(','):
    c = Counter(); by_ed = defaultdict(Counter)
    for line in gzip.open(f'{a.pred_dir}/{m}__test.jsonl.gz', 'rt', encoding='utf-8'):
        d = json.loads(line); r = recs[d['id']]
        if not r.get('loc_valid', True): continue
        g = set(spans_of(r['ner_tags_loc'])); per = [(x['s'], x['e']) for x in d['PER']]
        ps = {(x['s'], x['e']) for x in d['LOC']}
        for sp in ps:
            both = any(ov(sp, q) for q in per)
            kind = 'tp' if sp in g else ('bnd' if any(ov(sp, y) for y in g) else 'fp')
            c[(kind, both)] += 1; by_ed[r['edition']][(kind, both)] += 1
        c['gold'] += len(g)
    tp = c[('tp', False)] + c[('tp', True)]; fp = c[('fp', False)] + c[('fp', True)] + c[('bnd', False)] + c[('bnd', True)]
    tp2 = c[('tp', False)]; fp2 = c[('fp', False)] + c[('bnd', False)]
    f = lambda tp, fp, g: round(2 * tp / (2 * tp + fp + (g - tp)), 4)
    out[m] = {'loc_also_per': {f'{k[0]}|{k[1]}': v for k, v in c.items() if k != 'gold'}, 'f1_before': f(tp, fp, c['gold']), 'f1_after_dropping_loc_overlapping_per': f(tp2, fp2, c['gold']),
              'p_before': round(tp / (tp + fp), 4), 'p_after': round(tp2 / (tp2 + fp2), 4), 'r_before': round(tp / c['gold'], 4), 'r_after': round(tp2 / c['gold'], 4)}
    print(m, out[m])
json.dump(out, open(a.out, 'w'), indent=1)
