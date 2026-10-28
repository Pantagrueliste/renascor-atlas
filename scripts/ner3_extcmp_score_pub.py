"""External comparison: score LOC / PER predictions on public test sets converted to our JSONL (ner3_extcmp_convert.py, external/aggregate/*/test).

    python -I scripts/ner3_extcmp_score_pub.py --gold a.jsonl --gold b.jsonl --pred NAME=pred.jsonl.gz ... --out scores.json --md scores.md

Groups: the record's 'source/subset' (and 'all' per source). Labels: LOC (records with loc_valid), PER (records with per_valid).
Three matchings, micro P / R / F1:
  strict   same first and last word (as ratlas.ner.train)
  trim     strict after removing punctuation-only words at both ends of every gold and predicted span (neutral for the open decision on the
           trailing abbreviation period: 'Aristotelis .' and 'Aristotelis' count as the same span)
  overlap  one-to-one greedy matching of spans that share at least one word
95 % interval of the strict F1: bootstrap (1000 draws, seed 20261024) over documents when a group has >= 20 documents, else over records.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.spans import tags_to_spans  # noqa: E402

MODES = ['strict', 'trim', 'overlap']


def is_punct(t: str) -> bool:
    return all(unicodedata.category(c)[0] in 'PS' for c in t)


def trim(sp, toks):
    s, e = sp
    while s < e and is_punct(toks[s]):
        s += 1
    while e > s and is_punct(toks[e - 1]):
        e -= 1
    return (s, e) if s < e else None


def match(g, p, mode, toks):
    if mode == 'trim':
        g = {x for x in (trim(y, toks) for y in g) if x}
        p = {x for x in (trim(y, toks) for y in p) if x}
        mode = 'strict'
    if mode == 'strict':
        g, p = set(g), set(p)
        tp = len(g & p)
        return tp, len(p) - tp, len(g) - tp
    used, tp = set(), 0
    for a in sorted(p):
        for b in sorted(g):
            if b not in used and a[0] < b[1] and b[0] < a[1]:
                used.add(b)
                tp += 1
                break
    return tp, len(p) - tp, len(g) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load_pred(path):
    op = gzip.open if str(path).endswith('.gz') else open
    out = {}
    with op(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d['id']] = {t: [(x['s'], x['e']) for x in d.get(t, [])] for t in ('LOC', 'PER')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', action='append', required=True)
    ap.add_argument('--pred', action='append', required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    ap.add_argument('--keep', help='file of record ids: score these records only (e.g. the records without 8-gram overlap with the training data)')
    a = ap.parse_args()
    recs = []
    for g in a.gold:
        for line in open(g, encoding='utf-8'):
            if line.strip():
                recs.append(json.loads(line))
    if a.keep:
        keep = {x.strip() for x in open(a.keep, encoding='utf-8') if x.strip()}
        recs = [r for r in recs if r['id'] in keep]
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[f"{r['source']}/{r['subset']}"].append(i)
        groups[f"{r['source']}/all"].append(i)
    gold = []
    for r in recs:
        gold.append({'LOC': tags_to_spans(r['ner_tags_loc']) if r.get('loc_valid', True) else None,
                     'PER': tags_to_spans(r['ner_tags_per']) if r.get('per_valid', True) else None})
    res = {}
    for spec in a.pred:
        name, path = spec.split('=', 1)
        pr = load_pred(path)
        miss = sum(1 for r in recs if r['id'] not in pr)
        # per record counts
        cnt = []
        for r, g in zip(recs, gold):
            p = pr.get(r['id'], {'LOC': [], 'PER': []})
            c = {}
            for lab in ('LOC', 'PER'):
                if g[lab] is None:
                    continue
                for m in MODES:
                    c[(lab, m)] = match(g[lab], p[lab], m, r['tokens'])
            cnt.append(c)
        out = {'missing_records': miss, 'groups': {}}
        for gname, idx in sorted(groups.items()):
            docs = defaultdict(list)
            for i in idx:
                docs[recs[i].get('doc') or recs[i]['id']].append(i)
            units = list(docs.values()) if len(docs) >= 20 else [[i] for i in idx]
            gd = {}
            for lab in ('LOC', 'PER'):
                tot = {m: [0, 0, 0] for m in MODES}
                ng = 0
                for i in idx:
                    if (lab, 'strict') in cnt[i]:
                        for m in MODES:
                            for k in range(3):
                                tot[m][k] += cnt[i][(lab, m)][k]
                        ng += 1
                if not ng or tot['strict'][0] + tot['strict'][2] == 0:
                    continue
                d = {'n_gold': tot['strict'][0] + tot['strict'][2], 'records': ng}
                for m in MODES:
                    p, r_, f = prf(*tot[m])
                    d[m] = {'p': round(p, 4), 'r': round(r_, 4), 'f1': round(f, 4), 'tp': tot[m][0], 'fp': tot[m][1], 'fn': tot[m][2]}
                # bootstrap of the strict F1
                ut = []
                for u in units:
                    t = [0, 0, 0]
                    for i in u:
                        if (lab, 'strict') in cnt[i]:
                            for k in range(3):
                                t[k] += cnt[i][(lab, 'strict')][k]
                    ut.append(t)
                rnd = random.Random(20261024)
                fs = []
                for _ in range(a.boot):
                    t = [0, 0, 0]
                    for _ in range(len(ut)):
                        x = ut[rnd.randrange(len(ut))]
                        t[0] += x[0]; t[1] += x[1]; t[2] += x[2]
                    fs.append(prf(*t)[2])
                fs.sort()
                d['strict']['ci'] = [round(fs[int(0.025 * len(fs))], 4), round(fs[int(0.975 * len(fs)) - 1], 4)]
                d['boot_unit'] = 'document' if len(docs) >= 20 else 'record'
                gd[lab] = d
            out['groups'][gname] = gd
        res[name] = out
        print(name, 'missing', miss, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if a.md:
        lines = []
        gnames = sorted({g for v in res.values() for g in v['groups']})
        for lab in ('LOC', 'PER'):
            lines.append(f'\n### {lab}: strict F1 [CI] / trim / overlap\n')
            lines.append('| system | ' + ' | '.join(gnames) + ' |')
            lines.append('|---|' + '---|' * len(gnames))
            for name, v in res.items():
                cells = []
                for g in gnames:
                    d = v['groups'].get(g, {}).get(lab)
                    cells.append(f"{d['strict']['f1']:.3f} [{d['strict']['ci'][0]:.2f}–{d['strict']['ci'][1]:.2f}] / {d['trim']['f1']:.3f} / {d['overlap']['f1']:.3f}" if d else '–')
                lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        Path(a.md).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
