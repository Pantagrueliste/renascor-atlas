"""RenNER variant without the T3 collections CELT and CIDTC: filtered copies of every training and early-stopping file.

    python scripts/ner3_noT3_filter.py --ref celt=data_v1/raw/passages/celt.jsonl.gz --ref cidtc=data_v1/raw/passages/cidtc.jsonl.gz \
        --ref-from data_v1/train.jsonl --ref-from data_v1/dev.jsonl --ref-from data_v1/test.jsonl \
        --file ours=IN:OUT --file dev=IN:OUT ... --aug aug_lc=IN:OUT --details shared.jsonl --out report.json

Reference: the text of every CELT and CIDTC passage of the v1 extraction (raw passages: tagged and untagged documents, every split and the
unused ones), plus the CELT / CIDTC records of the --ref-from files (by edition). 8-grams: ner2_passages.fingerprints (8 folded \\w+ words),
the same code as the silver leak checks.
A record is dropped (1) when its edition, id, doc or collection names CELT or CIDTC, or (2) when it shares at least one 8-gram with the
reference. --aug files (lowercase copies, id = origin id + "#lc") also drop every copy whose origin was dropped from any --file.
Kept lines are written unchanged, in the same order. OUT is written only when at least one record is dropped (otherwise the original file
is used as is); the report says which. Input files are never modified. Gold v2, gold v3, dev3, test_v2 and the editor bench are never read.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from ratlas.ner.ner2_passages import fingerprints  # noqa: E402

T3 = {'celt', 'cidtc'}
RX = re.compile(r'\bcelt\b|\bcidtc\b|(^|[/_:])celt[/_:]|(^|[/_:])cidtc[/_:]', re.I)
GRAMS: set = set()


def opener(p):
    return gzip.open(p, 'rt', encoding='utf-8') if str(p).endswith('.gz') else open(p, encoding='utf-8')


def text_of(d):
    t = d.get('text')
    if isinstance(t, str) and t:
        return t
    return ' '.join(d.get('tokens') or [])


def named_t3(d):
    if str(d.get('edition', '')).lower() in T3 or str(d.get('slug', '')).lower() in T3:
        return True
    return bool(RX.search(' '.join(str(d.get(k, '')) for k in ('id', 'edition', 'doc', 'doc_id', 'collection', 'source', 'subset'))))


def build_ref(refs, ref_from):
    texts = Counter()
    seen = set()
    for spec in refs:
        name, _, path = spec.partition('=')
        for line in opener(path):
            d = json.loads(line)
            t = d.get('text') or ''
            if t and t not in seen:
                seen.add(t); texts[name] += 1
                GRAMS.update(fingerprints(t))
    for path in ref_from:
        for line in opener(path):
            if '"celt' not in line and '"cidtc' not in line:
                continue
            d = json.loads(line)
            if str(d.get('edition', '')).lower() in T3:
                t = text_of(d)
                if t not in seen:
                    seen.add(t); texts['from_' + d['edition']] += 1
                    GRAMS.update(fingerprints(t))
    return dict(texts)


def filt(job):
    name, src, out, drop_origin = job
    tmp = out + '.part'
    c = Counter(); by_coll = Counter(); dropped = []
    with opener(src) as f, open(tmp, 'w', encoding='utf-8') as g:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            c['records'] += 1
            coll = d.get('edition') or d.get('collection') or d.get('source') or '?'
            reason = None
            if named_t3(d):
                reason = 'edition'
            elif drop_origin and d['id'].removesuffix('#lc') in drop_origin:
                reason = 'origin'
            else:
                fp = fingerprints(text_of(d))
                common = fp & GRAMS
                if common:
                    reason = '8gram'
                    dropped.append({'file': name, 'id': d['id'], 'coll': coll, 'shared': len(common), 'of': len(fp),
                                    'examples': sorted(common)[:3]})
            if reason:
                c['drop_' + reason] += 1
                by_coll[f'{reason}:{coll}'] += 1
                if reason != '8gram':
                    dropped.append({'file': name, 'id': d['id'], 'coll': coll, 'reason': reason})
                continue
            c['kept'] += 1
            g.write(line if line.endswith('\n') else line + '\n')
    n_drop = c['records'] - c['kept']
    if n_drop:
        os.replace(tmp, out); used = out
    else:
        os.remove(tmp); used = src
    return name, {'src': src, 'used': used, 'written': bool(n_drop), **dict(c), 'by_reason_coll': dict(by_coll.most_common())}, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', action='append', required=True, help='name=path of a CELT / CIDTC passage file')
    ap.add_argument('--ref-from', action='append', default=[], help='file whose CELT / CIDTC records (by edition) join the reference')
    ap.add_argument('--file', action='append', default=[], help='name=IN:OUT')
    ap.add_argument('--aug', action='append', default=[], help='name=IN:OUT (lowercase copies, filtered after the --file sets)')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--details', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rep = {'reference_texts': build_ref(a.ref, a.ref_from), 'reference_8grams': len(GRAMS)}
    assert GRAMS, 'empty reference'
    print('reference', rep, flush=True)

    def jobs(specs, origin=None):
        out = []
        for s in specs:
            name, _, io = s.partition('=')
            src, _, dst = io.partition(':')
            assert Path(src).resolve() != Path(dst).resolve(), 'OUT must differ from IN'
            out.append((name, src, dst, origin))
        return out

    det = open(a.details, 'w', encoding='utf-8')
    drop_ids = set()
    # fork: the workers inherit the reference 8-grams (GRAMS, filled above); spawn / forkserver would start them with an empty set
    with mp.get_context('fork').Pool(a.procs) as pool:
        for name, r, dropped in pool.imap_unordered(filt, jobs(a.file)):
            rep[name] = r; print(name, {k: v for k, v in r.items() if k != 'by_reason_coll'}, flush=True)
            for x in dropped:
                drop_ids.add(x['id']); det.write(json.dumps(x, ensure_ascii=False) + '\n')
    origin = frozenset(drop_ids)
    for job in jobs(a.aug, origin):
        name, r, dropped = filt(job)
        rep[name] = r; print(name, {k: v for k, v in r.items() if k != 'by_reason_coll'}, flush=True)
        for x in dropped:
            det.write(json.dumps(x, ensure_ascii=False) + '\n')
    det.close()
    json.dump(rep, open(a.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
