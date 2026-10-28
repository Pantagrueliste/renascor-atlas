"""Overlap of the EXTERNAL evaluation sets (hist_gold/test/*.jsonl: HIPE-2022, Europeana, charters, ...) with the training data of the R3 mixes
(ours/train, hist_gold/train, modern_silver/train): share of 8-word shingles of every test record found in each training group, per test file."""
import argparse, glob, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_audit_leak import norm_tokens, shingles, text_key, read  # noqa

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

ap = argparse.ArgumentParser(); ap.add_argument('--data'); ap.add_argument('--agg'); ap.add_argument('--out'); a = ap.parse_args()
groups, exact = {}, {}
for name, path in (('ours_train', f'{a.data}/train.jsonl'), ('hist_gold_train', f'{a.agg}/hist_gold/train.jsonl'), ('modern_silver_train', f'{a.agg}/modern_silver/train.jsonl')):
    hs, ek = [], set()
    for r in read(path, ['tokens']):
        nt = norm_tokens(r['tokens']); hs.append(np.array(shingles(nt), dtype=np.int64)); ek.add(text_key(nt))
    h = np.unique(np.concatenate(hs)); groups[name] = h; exact[name] = ek; log(name, len(h))
rows = []
for f in sorted(glob.glob(f'{a.agg}/hist_gold/test/*.jsonl')):
    n = 0; acc = defaultdict(float); ge50 = defaultdict(int); ex = defaultdict(int); words = 0
    for r in read(f, ['tokens']):
        nt = norm_tokens(r['tokens']); sh = np.array(shingles(nt), dtype=np.int64); n += 1; words += len(nt); k = text_key(nt)
        for g, h in groups.items():
            fr = 0.0
            if len(sh):
                pos = np.searchsorted(h, sh); pos[pos >= len(h)] = 0; fr = float((h[pos] == sh).mean())
            acc[g] += fr; ge50[g] += fr >= 0.5; ex[g] += k in exact[g]
    row = {'file': os.path.basename(f)[:-6], 'records': n, 'words': words}
    for g in groups: row[g] = f'{acc[g]/n:.3f}/{ge50[g]}/{ex[g]}'
    rows.append(row); log(row)
with open(a.out, 'w') as fo:
    keys = list(rows[0]); fo.write('\t'.join(keys) + '\n')
    for r in rows: fo.write('\t'.join(str(r[k]) for k in keys) + '\n')
