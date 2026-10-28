"""Leak check of the gold2 candidate passages against every text the v1 models were trained, tuned or tested on (run on Hypatia, CPU job).

For each passage: share of its 8-word shingles (folded: lower case, long s -> s, accents removed, punctuation dropped) that occur in the streamed files.
The streamed files are the v1 data (train, dev, test) and the external aggregates; a passage with a share >= --flag is listed (it is to be replaced).
Also reports the passages whose document name equals the file name of a v1 document.

    python scripts/ner2_leak_check.py --passages passages.jsonl --stream data_v1/train.jsonl data_v1/dev.jsonl ... --out leak_gold2.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.ner2_passages import fold  # noqa: E402

N = 8
W = re.compile(r'\w+', re.U)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def words(text):
    return [fold(w) for w in W.findall(text)]


def opener(p):
    return gzip.open(p, 'rt', encoding='utf-8') if p.endswith('.gz') else open(p, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passages', required=True)
    ap.add_argument('--stream', nargs='+', required=True, help='jsonl files or globs (record has text or tokens)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--flag', type=float, default=0.05)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.passages, encoding='utf-8') if l.strip()]
    grams = defaultdict(list)       # shingle -> passage indexes
    sizes = []
    for i, r in enumerate(recs):
        ws = words(r['text'])
        gs = {' '.join(ws[j:j + N]) for j in range(max(0, len(ws) - N + 1))}
        sizes.append(len(gs))
        for g in gs:
            grams[g].append(i)
    first = {g.split(' ', 1)[0] for g in grams}
    log(len(recs), 'passages,', len(grams), 'shingles')
    hits = [Counter() for _ in recs]        # per passage: file -> number of shingles found
    found = [set() for _ in recs]
    v1_doc_names = set()
    files = []
    for s in a.stream:
        files += sorted(glob.glob(s)) or [s]
    for f in files:
        t0 = time.time()
        n = 0
        for line in opener(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('doc'):
                v1_doc_names.add(re.sub(r'\.(xml|txt|json|jsonl)$', '', str(d['doc']).split('/')[-1], flags=re.I).lower())
            ws = words(d['text']) if d.get('text') else words(' '.join(d.get('tokens', [])))
            n += 1
            for j in range(len(ws) - N + 1):
                if ws[j] in first:
                    g = ' '.join(ws[j:j + N])
                    if g in grams:
                        for i in grams[g]:
                            found[i].add(g)
                            hits[i][f] += 1
        log(f, n, 'records', f'{time.time() - t0:.0f}s')
    out = []
    for i, r in enumerate(recs):
        share = len(found[i]) / max(1, sizes[i])
        out.append({'id': r['id'], 'share': round(share, 4), 'n_found': len(found[i]), 'files': dict(hits[i].most_common(3)),
                    'doc_name_in_v1': r['doc'].lower() in v1_doc_names or re.sub(r'\.(xml|txt)$', '', r['doc'], flags=re.I).lower() in v1_doc_names})
    flagged = [o for o in out if o['share'] >= a.flag or o['doc_name_in_v1']]
    rep = {'n_passages': len(recs), 'streamed_files': files, 'n_with_any_hit': sum(o['n_found'] > 0 for o in out), 'max_share': max(o['share'] for o in out),
           'flagged': flagged, 'per_passage': out}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    log('any hit:', rep['n_with_any_hit'], 'max share', rep['max_share'], 'flagged', len(flagged))


if __name__ == '__main__':
    main()
