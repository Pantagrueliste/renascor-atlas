"""Cavriana transfer test (RenNER v2): remove the Cavriana edition from every training source, by edition and by 8-gram.
Run on Hypatia (debug partition, CPU). Gold v2, gold v3 and dev3 are never read.

    python scripts/ner3_cav_filter.py --cav bench_cavriana.jsonl --file ours=data_v1/train.jsonl --file dev=data_v1/dev.jsonl ... \
        --out-dir OUT [--write]

A record is removed when its edition (``edition``, ``id`` or ``source``) names Cavriana, or when it shares at least --min-shared folded
8-word shingles with a Cavriana passage (fold of ner2_passages, as ner2_leak_check.py; text, or the tokens joined). Without --write:
report only (OUT/hits_<name>.tsv, OUT/report.json). With --write: OUT/<name>.jsonl for every file with at least one removal (the other
files are unchanged and keep their path), OUT/removed_<name>.tsv, OUT/report.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.ner2_passages import fold  # noqa: E402

N = 8
W = re.compile(r'\w+', re.U)
CAV = re.compile(r'cavriana', re.I)


def words(text):
    return [fold(w) for w in W.findall(text)]


def shingles(ws):
    return {' '.join(ws[j:j + N]) for j in range(max(0, len(ws) - N + 1))}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cav', required=True)
    ap.add_argument('--file', action='append', required=True, help='name=path')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-shared', type=int, default=1)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grams = set()
    for l in open(a.cav, encoding='utf-8'):
        if l.strip():
            grams |= shingles(words(json.loads(l)['text']))
    first = {g.split(' ', 1)[0] for g in grams}
    log('cavriana shingles', len(grams))
    rep = {'cav_shingles': len(grams), 'min_shared': a.min_shared, 'files': {}}
    for spec in a.file:
        name, path = spec.split('=', 1)
        t0 = time.time()
        st = Counter()
        hits = []
        fo = None
        if a.write:
            fo = open(out / f'{name}.jsonl.tmp', 'w', encoding='utf-8')
        for line in open(path, encoding='utf-8'):
            if not line.strip():
                continue
            d = json.loads(line)
            st['records'] += 1
            ed = ' '.join(str(d.get(k) or '') for k in ('edition', 'id', 'source'))
            by_ed = bool(CAV.search(ed))
            ws = words(d['text']) if d.get('text') else words(' '.join(d.get('tokens') or []))
            sh = set()
            for j in range(len(ws) - N + 1):
                if ws[j] in first:
                    g = ' '.join(ws[j:j + N])
                    if g in grams:
                        sh.add(g)
            n_all = max(1, len(ws) - N + 1)
            drop = by_ed or len(sh) >= a.min_shared
            if by_ed or sh:
                hits.append((d.get('id'), d.get('edition') or d.get('source'), d.get('lang'), int(by_ed), len(sh), round(len(sh) / n_all, 4),
                             len(ws), ' | '.join(sorted(sh)[:3])))
            if drop:
                st['removed'] += 1
                st['removed_by_edition' if by_ed else 'removed_by_8gram'] += 1
                st['removed_words'] += len(d.get('tokens') or ws)
            elif fo:
                fo.write(line if line.endswith('\n') else line + '\n')
        if fo:
            fo.close()
            if st['removed']:
                (out / f'{name}.jsonl.tmp').rename(out / f'{name}.jsonl')
            else:
                (out / f'{name}.jsonl.tmp').unlink()
        with open(out / (f'removed_{name}.tsv' if a.write else f'hits_{name}.tsv'), 'w', encoding='utf-8') as fh:
            fh.write('id\tedition\tlang\tby_edition\tshared_8grams\tshare\twords\texamples\n')
            for h in hits:
                fh.write('\t'.join(str(x) for x in h) + '\n')
        st['hits_8gram_not_edition'] = sum(1 for h in hits if not h[3])
        st['seconds'] = round(time.time() - t0)
        rep['files'][name] = {'path': path, **st, 'new_path': str(out / f'{name}.jsonl') if (a.write and st['removed']) else path}
        log(name, dict(st))
    (out / 'report.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
