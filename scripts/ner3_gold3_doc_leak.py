"""gold v3: document-level 8-gram check of the candidate documents against the v1 data and the training mixes (Hypatia, CPU).

The passage check (ner2_leak_check.py) only sees the drawn passages; this one compares the WHOLE document, so another passage of a work that is in
the v1 training data is caught too. Self-contained (numpy only): 64-bit hashes of folded 8-word shingles, the same scheme as ner3_silver_pool.py.

    python scripts/ner3_gold3_doc_leak.py --docs docs.jsonl.gz --out doc_leak.json --stream FILE_OR_GLOB ...
"""
import argparse
import glob
import gzip
import hashlib
import json
import re
import time
import unicodedata

import numpy as np

PK = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x94D049BB133111EB,
               0xBF58476D1CE4E5B9], dtype=np.uint64) | np.uint64(1)
WORD = re.compile(r'\w+', re.U)
VOC = {}


def fold(s):
    s = s.lower().replace('ſ', 's')
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def wid(w):
    v = VOC.get(w)
    if v is None:
        if len(VOC) > 3_000_000:
            VOC.clear()
        v = int.from_bytes(hashlib.blake2b(fold(w).encode(), digest_size=8).digest(), 'little')
        VOC[w] = v
    return v


def grams(text, n=8):
    ws = WORD.findall(text)
    m = len(ws)
    if m < n:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((wid(w) for w in ws), dtype=np.uint64, count=m)
    h = np.zeros(m - n + 1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        for k in range(n):
            h += ids[k:m - n + 1 + k] * PK[k]
    return h


def in_sorted(h, E):
    if len(h) == 0 or len(E) == 0:
        return np.zeros(len(h), dtype=bool)
    i = np.searchsorted(E, h)
    i[i >= len(E)] = 0
    return E[i] == h


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stream', nargs='+', required=True)
    a = ap.parse_args()
    docs = [json.loads(l) for l in gzip.open(a.docs, 'rt', encoding='utf-8')]
    H = [np.unique(grams(d['text'])) for d in docs]
    log(len(docs), 'documents', sum(len(h) for h in H), '8-grams')
    allh = np.unique(np.concatenate(H))
    res = [{'doc_id': d['doc_id'], 'lang': d['lang'], 'n_8grams': int(len(h)), 'by_file': {}} for d, h in zip(docs, H)]
    files = []
    for s in a.stream:
        files += sorted(glob.glob(s)) or [s]
    for f in files:
        t0 = time.time()
        found = []
        n = 0
        for line in open(f, encoding='utf-8'):
            if not line.strip():
                continue
            d = json.loads(line)
            t = d.get('text') or ' '.join(d.get('tokens', []))
            h = grams(t)
            if len(h):
                found.append(h[in_sorted(h, allh)])
            n += 1
        F = np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.uint64)
        for r, h in zip(res, H):
            k = int(in_sorted(h, F).sum())
            if k:
                r['by_file'][f] = k
        log(f, n, 'records', len(F), 'shared 8-grams', f'{time.time() - t0:.0f}s')
    for r in res:
        tot = sum(r['by_file'].values())
        r['matched'] = tot
        r['frac'] = round(tot / max(1, r['n_8grams']), 5)
        r['flag'] = tot >= 25 or r['frac'] >= 0.01
    json.dump({'files': files, 'n_docs': len(res), 'flagged': [r for r in res if r['flag']], 'per_doc': res}, open(a.out, 'w'), indent=1)
    log('flagged', sum(r['flag'] for r in res))


if __name__ == '__main__':
    main()
