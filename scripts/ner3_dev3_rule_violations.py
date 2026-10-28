"""Owner-rule violations of a system on dev3, read from the v0.2 revision (gold_dev3_v02.jsonl, field v02 on each changed span).
A predicted span is a violation of rule X when it was right under v0.1 (core or long form) and is wrong under v0.2 because of a change
of rule X; an added v0.2 span (e.g. the Devil) that the system misses is an omission of that rule. Rate = violations / predicted spans.
    python scripts/ner3_dev3_rule_violations.py --v01 gold_dev3.jsonl --v02 gold_dev3_v02.jsonl --pred NAME=path ... --out out.json
Gold v2 and gold v3 are never read.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner3_teacher_score as S  # noqa: E402


def forms(r):
    out = {}
    for s in r['spans']:
        f = {(s['start'], s['end'], s['label'])}
        if s.get('alt_start') is not None:
            f.add((s['alt_start'], s['end'], s['label']))
        for x in f:
            out[x] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v01', required=True)
    ap.add_argument('--v02', required=True)
    ap.add_argument('--pred', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    g1 = {r['id']: r for r in S.load(a.v01)}
    g2 = {r['id']: r for r in S.load(a.v02)}
    res = {}
    for spec in a.pred:
        name, p = spec.split('=', 1)
        pred = {d['id']: S.to_pred(d) for d in S.load(p)}
        c = defaultdict(Counter)
        ex = defaultdict(list)
        for i, r2 in g2.items():
            lg = r2['lang']
            f1, f2 = forms(g1[i]), forms(r2)
            ps = {(x['s'], x['e'], lab) for lab in ('LOC', 'PER') for x in pred.get(i, {}).get(lab, [])}
            c[lg]['pred'] += len(ps)
            for x in ps:
                if x in f1 and x not in f2:
                    # which v0.2 rule made it wrong: the gold span of v0.2 covering the same core, else the v0.1 span's rule
                    rule = None
                    for s in r2['spans']:
                        if 'v02' in s and s['start'] < x[1] and x[0] < s['end']:
                            rule = s['v02'][0]['rule']
                            break
                    if rule is None:
                        s1 = f1[x]
                        rule = 'dropped:' + s1['label']
                        for s in r2.get('dropped_v02', []) if isinstance(r2.get('dropped_v02'), list) else []:
                            pass
                    c[lg]['viol:' + rule] += 1
                    ex[rule].append((i, ' '.join(r2['tokens'][x[0]:x[1]]), x[2]))
            for s in r2['spans']:
                if 'v02' in s and any(v['action'] in ('add', 'add_per', 'added', 'add_loc') or v.get('action', '').startswith('add') for v in s['v02']):
                    if (s['start'], s['end'], s['label']) not in ps:
                        rule = s['v02'][0]['rule']
                        c[lg]['miss:' + rule] += 1
                        ex['miss:' + rule].append((i, s['text'], s['label']))
        tot = Counter()
        for lg in c:
            tot.update(c[lg])
        c['all'] = tot
        res[name] = {lg: dict(v) for lg, v in c.items()}
        res[name]['examples'] = {k: v[:15] for k, v in ex.items()}
        print(name, {lg: {k: (v, round(100 * v / max(1, c[lg]['pred']), 2)) for k, v in c[lg].items() if k != 'pred'} | {'pred': c[lg]['pred']}
                     for lg in ('es', 'it', 'la', 'all')})
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
