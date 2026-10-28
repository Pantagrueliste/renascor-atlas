"""Sample false positives of one model on the held-out test editions, with context, for manual judgement (audit).
Kinds (as in ner_audit_score.py): other / tagged_elsewhere_in_edition / boundary. Writes a TSV (id, kind, edition, lang, span text, context with the span in [[ ]])."""
import argparse, gzip, json, random, sys
from collections import defaultdict
from pathlib import Path

def spans_of(tags):
    out, st = [], None
    for i, t in enumerate(list(tags) + [0]):
        if t in (0, 1):
            if st is not None: out.append((st, i)); st = None
            if t == 1: st = i
    return out
def norm(s): return s.lower().replace('ſ', 's')
def ov(a, b): return a[0] < b[1] and b[0] < a[1]

ap = argparse.ArgumentParser()
ap.add_argument('--data'); ap.add_argument('--pred'); ap.add_argument('--out')
ap.add_argument('--n-other', type=int, default=18); ap.add_argument('--n-elsewhere', type=int, default=5); ap.add_argument('--n-boundary', type=int, default=4)
ap.add_argument('--n-fn', type=int, default=8); ap.add_argument('--seed', type=int, default=11); ap.add_argument('--min-conf', type=float, default=0.0)
a = ap.parse_args()
recs = {}
for line in open(a.data, encoding='utf-8'):
    d = json.loads(line); recs[d['id']] = d
pred = {}
for line in gzip.open(a.pred, 'rt', encoding='utf-8'):
    d = json.loads(line); pred[d['id']] = d
surf = defaultdict(set)
gold = {}
for i, r in recs.items():
    gold[i] = set(spans_of(r['ner_tags_loc'])) if r.get('loc_valid', True) else set()
    for s, e in gold[i]: surf[r['edition']].add(norm(' '.join(r['tokens'][s:e])))
pool = defaultdict(list)
for i, r in recs.items():
    if not r.get('loc_valid', True): continue
    g = gold[i]; ps = {(x['s'], x['e']) for x in pred[i]['LOC']}
    cf = {(x['s'], x['e']): x['conf'] for x in pred[i]['LOC']}
    for sp in ps - g:
        if cf[sp] < a.min_conf: continue
        t = norm(' '.join(r['tokens'][sp[0]:sp[1]]))
        kind = 'boundary' if any(ov(sp, y) for y in g) else ('tagged_elsewhere_in_edition' if t in surf[r['edition']] else 'other')
        pool[(r['edition'], 'fp_' + kind)].append((i, sp))
    for sp in (g - ps) if a.min_conf == 0 else ():
        pool[(r['edition'], 'fn')].append((i, sp))
rng = random.Random(a.seed)
rows = []
for (ed, kind), lst in sorted(pool.items()):
    n = {'fp_other': a.n_other, 'fp_tagged_elsewhere_in_edition': a.n_elsewhere, 'fp_boundary': a.n_boundary, 'fn': a.n_fn}[kind]
    for i, sp in rng.sample(lst, min(n, len(lst))):
        r = recs[i]; toks = r['tokens']
        ctx = ' '.join(toks[max(0, sp[0]-12):sp[0]]) + ' [[' + ' '.join(toks[sp[0]:sp[1]]) + ']] ' + ' '.join(toks[sp[1]:sp[1]+8])
        gl = ' | '.join(' '.join(toks[s:e]) for s, e in sorted(gold[i]) if ov((s, e), sp))
        rows.append((len(rows), i, kind, ed, r.get('lang') or 'unk', ' '.join(toks[sp[0]:sp[1]]), gl, ctx))
with open(a.out, 'w', encoding='utf-8') as f:
    f.write('n\tid\tkind\tedition\tlang\tspan\toverlapping_gold\tcontext\n')
    for r in rows: f.write('\t'.join(str(x) for x in r) + '\n')
print(len(rows), 'rows')
