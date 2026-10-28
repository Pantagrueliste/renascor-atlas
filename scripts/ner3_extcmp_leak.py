"""External comparison: overlap of the new public test sets (pub.jsonl) with everything the RenNER v2 students were trained on
(ours/train, hist_gold/train, modern_silver/train, silver balanced + reserve). Same 8-word shingles and normalisation as ner_audit_leak_ext.py.
Per test group (source/subset): mean share of shingles found, records with >= 50 % found, exact duplicates of the normalised text.

    python -I scripts/ner3_extcmp_leak.py --pub pub.jsonl --data DATA_V1 --agg AGG --silver SILVER_FINAL --out leak.json
"""
import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_audit_leak import norm_tokens, shingles, text_key, read  # noqa: E402


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument('--pub', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--agg', required=True)
ap.add_argument('--silver', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--per-record', help='also write id -> {group: share} for every test record (jsonl)')
a = ap.parse_args()
srcs = [('ours_train', [f'{a.data}/train.jsonl']), ('hist_gold_train', [f'{a.agg}/hist_gold/train.jsonl']),
        ('modern_silver_train', [f'{a.agg}/modern_silver/train.jsonl']),
        ('renner_silver', sorted(glob.glob(f'{a.silver}/silver_balanced_*.jsonl') + glob.glob(f'{a.silver}/silver_reserve_*.jsonl')))]
groups, exact = {}, {}
for name, paths in srcs:
    hs, ek = [], set()
    for p in paths:
        for r in read(p, ['tokens']):
            nt = norm_tokens(r['tokens'])
            hs.append(np.array(shingles(nt), dtype=np.int64))
            ek.add(text_key(nt))
    groups[name] = np.unique(np.concatenate(hs)) if hs else np.zeros(0, dtype=np.int64)
    exact[name] = ek
    log(name, len(paths), 'files', len(groups[name]), 'shingles')
acc = defaultdict(lambda: defaultdict(float))
ge50 = defaultdict(lambda: defaultdict(int))
ex = defaultdict(lambda: defaultdict(int))
n = defaultdict(int)
examples = defaultdict(list)
prf = open(a.per_record, 'w') if a.per_record else None
for r in read(a.pub, ['tokens', 'source', 'subset', 'id']):
    key = f"{r['source']}/{r['subset']}"
    nt = norm_tokens(r['tokens'])
    sh = np.array(shingles(nt), dtype=np.int64)
    n[key] += 1
    k = text_key(nt)
    rs = {}
    for g, h in groups.items():
        fr = 0.0
        if len(sh) and len(h):
            pos = np.searchsorted(h, sh)
            pos[pos >= len(h)] = 0
            fr = float((h[pos] == sh).mean())
        acc[key][g] += fr
        ge50[key][g] += fr >= 0.5
        ex[key][g] += k in exact[g]
        rs[g] = round(fr, 4)
        if fr >= 0.5 and len(examples[key]) < 10:
            examples[key].append({'id': r['id'], 'group': g, 'share': round(fr, 3), 'text': ' '.join(r['tokens'][:30])})
    if prf is not None:
        prf.write(json.dumps({'id': r['id'], **rs}) + '\n')
if prf is not None:
    prf.close()
res = {key: {'records': n[key], **{g: {'mean_share': round(acc[key][g] / n[key], 4), 'records_ge50': ge50[key][g], 'exact': ex[key][g]} for g in groups},
             'examples': examples[key]} for key in sorted(n)}
Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: {g: v[g] for g in groups} for k, v in res.items()}, indent=1))
