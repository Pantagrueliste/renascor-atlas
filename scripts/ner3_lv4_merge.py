"""Latin v4: put the prompt-v4 answers of the Latin verse and drama passages into the combined Latin silver (output of ner3_silver_combine.py).

    python scripts/ner3_lv4_merge.py --old combined/silver_la.jsonl --new combined/silver_la_v4vd.jsonl --out combined/silver_la_v4.jsonl \
        --report combined/merge_la_v4.json

A line of --old is replaced by the line of --new with the same id when the new line is usable (silver_ok); otherwise the old (v3) line stays
and is counted. Line order and every other passage are unchanged. The report counts the replaced passages by form and the span changes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def spans(d):
    return {(s['start'], s['end'], s['label']) for s in d.get('spans') or []}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    new = {}
    for line in open(a.new, encoding='utf8'):
        if line.strip():
            d = json.loads(line)
            new[d['id']] = d
    c = Counter()
    seen = set()
    with open(a.out, 'w', encoding='utf8') as f:
        for line in open(a.old, encoding='utf8'):
            if not line.strip():
                continue
            d = json.loads(line)
            n = new.get(d['id'])
            if n is None:
                f.write(line if line.endswith('\n') else line + '\n')
                continue
            seen.add(d['id'])
            if not n.get('silver_ok'):
                c['kept_v3_v4_not_ok'] += 1
                f.write(line if line.endswith('\n') else line + '\n')
                continue
            assert n['tokens'] == d['tokens'], d['id']
            c['replaced'] += 1
            c[f"replaced_{d.get('form')}"] += 1
            so, sn = spans(d), spans(n)
            for lab in ('LOC', 'PER'):
                c[f'{lab}_v3'] += sum(1 for x in so if x[2] == lab)
                c[f'{lab}_v4'] += sum(1 for x in sn if x[2] == lab)
                c[f'{lab}_removed'] += sum(1 for x in so - sn if x[2] == lab)
                c[f'{lab}_added'] += sum(1 for x in sn - so if x[2] == lab)
            c['passages_changed'] += so != sn
            c['empty_v3'] += not so
            c['empty_v4'] += not sn
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
    c['new_not_in_old'] = len(set(new) - seen)
    json.dump(dict(sorted(c.items())), open(a.report, 'w'), indent=1)
    print(json.dumps(dict(sorted(c.items()))))


if __name__ == '__main__':
    main()
