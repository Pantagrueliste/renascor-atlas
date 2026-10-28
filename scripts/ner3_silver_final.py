"""Training files of the RenNER silver, in the format ratlas/ner/train.py reads (nerdata.read_slim), from the outputs of ner3_silver_combine.py.

    python scripts/ner3_silver_final.py --silver silver_es.jsonl --subset balanced --out final/silver_balanced_es.jsonl [--report r.json]

Kept: passages with silver_ok true (a usable teacher answer). Passages with no entity are kept (negative examples).
--drop-ids FILE (repeatable) removes the passages whose id is listed (JSONL with an "id" field, e.g. the --details output of
ner3_silver_verify.py: passages sharing an 8-gram with gold v2, dev3, test_v2 or the v1 dev / test splits).
Each record: id, lang, edition ("silver/<collection>", the group of the train.py report), source "renner_silver", subset (balanced | reserve),
tokens, token_starts, text, ner_tags_loc / ner_tags_per (0 O, 1 B, 2 I), loc_valid / per_valid true, spans (as in the combined file),
collection, doc, year, genre, form, teacher. The tags are checked to give back exactly the spans (flat, no overlap, inside the passage).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from ratlas.ner.spans import spans_to_tags, tags_to_spans  # noqa: E402

KEEP = ('collection', 'doc', 'doc_id', 'year', 'period_bin_25y', 'genre', 'form', 'licence_note', 'v1_train_dev_edition_doc')


def to_train(rec: dict, subset: str) -> dict:
    toks = rec['tokens']
    n = len(toks)
    if len(rec['token_starts']) != n:
        raise ValueError(f"{rec['id']}: tokens and token_starts differ in length")
    by = {'LOC': [], 'PER': []}
    for s in rec['spans']:
        if not (0 <= s['start'] < s['end'] <= n):
            raise ValueError(f"{rec['id']}: span out of the passage {s}")
        by[s['label']].append((s['start'], s['end']))
    every = sorted(by['LOC'] + by['PER'])
    for (a, b), (c, d) in zip(every, every[1:]):
        if c < b:
            raise ValueError(f"{rec['id']}: overlapping spans {(a, b)} {(c, d)}")
    loc, per = spans_to_tags(n, by['LOC']), spans_to_tags(n, by['PER'])
    if tags_to_spans(loc) != sorted(by['LOC']) or tags_to_spans(per) != sorted(by['PER']):
        raise ValueError(f"{rec['id']}: tags do not give back the spans")
    out = {'id': rec['id'], 'lang': rec['lang'], 'edition': f"silver/{rec.get('collection', '?')}", 'source': 'renner_silver', 'subset': subset,
           'tokens': toks, 'token_starts': rec['token_starts'], 'text': rec['text'], 'ner_tags_loc': loc, 'ner_tags_per': per,
           'loc_valid': True, 'per_valid': True,
           'spans': [{k: s[k] for k in ('start', 'end', 'label', 'text', 'char_start', 'char_end')} for s in rec['spans']],
           'teacher': rec.get('teacher')}
    for k in KEEP:
        if k in rec:
            out[k] = rec[k]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--silver', required=True, help='output of ner3_silver_combine.py')
    ap.add_argument('--subset', required=True, choices=['balanced', 'reserve'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', help='JSON with the counts of the written file')
    ap.add_argument('--drop-ids', action='append', default=[], help='JSONL with an "id" field: passages to leave out (leak check)')
    a = ap.parse_args(argv)
    drop = {}
    for p in a.drop_ids:
        for line in open(p, encoding='utf8'):
            if line.strip():
                d = json.loads(line)
                drop.setdefault(d['id'], set()).add(d.get('set', '?'))
    seen = set()
    dropped_by = Counter()
    c = Counter()
    by_coll = Counter()
    with open(a.out, 'w', encoding='utf8') as f:
        for line in open(a.silver, encoding='utf8'):
            rec = json.loads(line)
            c['read'] += 1
            if rec['id'] in seen:
                c['duplicate_id_dropped'] += 1
                continue
            seen.add(rec['id'])
            if not rec.get('silver_ok'):
                c['not_silver_ok_dropped'] += 1
                continue
            if rec['id'] in drop:
                c['leak_dropped'] += 1
                dropped_by.update(drop[rec['id']])
                continue
            r = to_train(rec, a.subset)
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            c['passages'] += 1
            c['tokens'] += len(r['tokens'])
            nl, npr = sum(1 for t in r['ner_tags_loc'] if t == 1), sum(1 for t in r['ner_tags_per'] if t == 1)
            c['LOC'] += nl
            c['PER'] += npr
            c['passages_without_entity'] += nl + npr == 0
            by_coll[r.get('collection', '?')] += 1
    rep = {'file': a.out, 'subset': a.subset, **c, 'max_collection_share': round(max(by_coll.values()) / max(1, c['passages']), 4) if by_coll else 0,
           'leak_dropped_by_set': dict(dropped_by), 'by_collection': dict(by_coll.most_common())}
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_collection', 'leak_dropped_by_set')}))
    if a.report:
        json.dump(rep, open(a.report, 'w'), indent=1, ensure_ascii=False)
    return rep


if __name__ == '__main__':
    main()
