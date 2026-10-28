"""Latin v4 prompt check: gemma-4-31B-it fewshot v3 against v4 (+pp), on dev3 and on the 40 Latin passages of the silver audit v1.

    python scripts/ner3_lv4_compare.py --dev3-gold dev3/gold_dev3.jsonl --dev3-passages dev3/passages.jsonl \
        --audit-sample silver/audit_silver_v1/sample_texts.jsonl --audit-ann silver/audit_silver_v1/ann_la.txt,.../ann_spares.txt \
        --pred v3=pred/gemma4_31b__fewshotv3.jsonl --pred v4=pred/gemma4_31b__fewshotv4.jsonl --json out.json

dev3: F1 strict, --accept-alt, all spans (ner3_teacher_score.py), per language and label; paired bootstrap of v4 - v3 per language
(passages resampled within the language, 1000 draws, seed 20261023) on the mean of the LOC and PER F1.
Audit: strict match (start, end, label) against the blind audit annotations (silver/audit_silver_v1/match.py); totals of false
positives and misses, paired bootstrap of the F1 difference, and the list of errors of each prompt. Gold v2 / gold v3 are never read.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sc = _load('sc', HERE / 'ner3_teacher_score.py')
bk = _load('bk', HERE / 'ner3_teacher_bakeoff.py')


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def pp(raw, toks):
    pred = {d['id']: sc.to_pred(d) for d in raw if d['id'] in toks}
    return bk.propagate(bk.strip_final_dot(pred, toks), toks)


def f1(t):
    return sc.prf(*t)[2]


def mean_f1(counts, ids):
    out = []
    for lab in ('LOC', 'PER'):
        t = [0, 0, 0]
        for i in ids:
            t = [a + b for a, b in zip(t, counts[i][lab])]
        out.append(f1(t))
    return sum(out) / 2, out


def paired(ca, cb, ids, n=1000, seed=20261023):
    rnd = random.Random(seed)
    d0 = mean_f1(cb, ids)[0] - mean_f1(ca, ids)[0]
    ds = []
    for _ in range(n):
        pick = [ids[rnd.randrange(len(ids))] for _ in ids]
        ds.append(mean_f1(cb, pick)[0] - mean_f1(ca, pick)[0])
    ds.sort()
    return d0, ds[int(0.025 * (n - 1))], ds[int(0.975 * (n - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev3-gold', required=True)
    ap.add_argument('--dev3-passages', required=True)
    ap.add_argument('--audit-sample', required=True)
    ap.add_argument('--audit-ann', required=True)
    ap.add_argument('--audit-match', default=None, help='path of audit_silver_v1/match.py (default: next to the sample)')
    ap.add_argument('--pred', action='append', required=True, help='TAG=raw teacher output (dev3 + audit ids)')
    ap.add_argument('--json')
    a = ap.parse_args()
    assert 'gold_v2' not in a.dev3_gold and 'gold3' not in a.dev3_gold and 'dev3' in a.dev3_gold
    gold = sc.load(a.dev3_gold)
    dtoks = {d['id']: d['tokens'] for d in jl(a.dev3_passages)}
    match = _load('match', a.audit_match or str(Path(a.audit_sample).with_name('match.py')))
    smp = match.load_sample(a.audit_sample)
    ann = match.parse(a.audit_ann.split(','))
    audit = {}
    for (st, rank), ents in ann.items():
        if st != 'la' or rank == 10:          # la 10 left the silver (leak with dev v1); la 40 replaces it, as in the audit
            continue
        d = smp[(st, rank)]
        spans, errs = match.to_spans(d, ents)
        assert not errs, errs
        audit[d['id']] = (d, set(spans))
    atoks = {i: d['tokens'] for i, (d, _) in audit.items()}
    tags = [s.split('=', 1)[0] for s in a.pred]
    raw = {t: jl(p) for t, p in (s.split('=', 1) for s in a.pred)}
    res = {'dev3': {}, 'audit': {}}
    dev_counts, aud_counts = {}, {}
    for t in tags:
        pr = pp(raw[t], dtoks)
        c = sc.per_passage_counts(gold, pr, accept_alt=True)
        dev_counts[t] = {i: {lab: v[(lab, 'strict')] for lab in ('LOC', 'PER')} for i, v in c.items()}
        sco = sc.score(gold, pr, accept_alt=True, counts=c)
        res['dev3'][t] = {f'{l}|{lab}': [round(x, 4) for x in sco[(l, lab, 'strict')]] for l in ('es', 'it', 'la', 'all') for lab in ('LOC', 'PER')}
        six = [sco[(l, lab, 'strict')][2] for l in ('es', 'it', 'la') for lab in ('LOC', 'PER')]
        res['dev3'][t]['choice_score'] = round(sum(six) / 6, 4)
        # audit
        pa = pp(raw[t], atoks)
        cnt, errs = {}, []
        for i, (d, g) in audit.items():
            p = {(x['s'], x['e'], lab) for lab in ('LOC', 'PER') for x in pa.get(i, {}).get(lab, [])}
            toks = d['tokens']
            cnt[i] = {}
            for lab in ('LOC', 'PER'):
                gg = {x for x in g if x[2] == lab}
                ppp = {x for x in p if x[2] == lab}
                cnt[i][lab] = (len(gg & ppp), len(ppp - gg), len(gg - ppp))
            for x in sorted(p - g):
                errs.append({'rank': d['rank'], 'form': d['form'], 'kind': 'FP', 'label': x[2], 'text': ' '.join(toks[x[0]:x[1]])})
            for x in sorted(g - p):
                errs.append({'rank': d['rank'], 'form': d['form'], 'kind': 'FN', 'label': x[2], 'text': ' '.join(toks[x[0]:x[1]])})
        aud_counts[t] = cnt
        tot = {lab: [sum(cnt[i][lab][k] for i in cnt) for k in range(3)] for lab in ('LOC', 'PER')}
        res['audit'][t] = {lab: {'tp': v[0], 'fp': v[1], 'fn': v[2], 'p_r_f1': [round(x, 4) for x in sc.prf(*v)]} for lab, v in tot.items()}
        res['audit'][t]['errors_total'] = sum(v[1] + v[2] for v in tot.values())
        res['audit'][t]['verse_drama_errors'] = sum(1 for e in errs if e['form'] != 'prose')
        res['audit'][t]['errors'] = errs
    if len(tags) == 2:
        A, B = tags
        res['paired_dev3'] = {}
        for l in ('es', 'it', 'la'):
            ids = [r['id'] for r in gold if r['lang'] == l]
            res['paired_dev3'][l] = [round(x, 4) for x in paired(dev_counts[A], dev_counts[B], ids)]
        ids = list(audit)
        res['paired_audit'] = [round(x, 4) for x in paired(aud_counts[A], aud_counts[B], ids, seed=20261024)]
        ids_v = [i for i, (d, _) in audit.items() if d['form'] != 'prose']
        res['audit_verse_drama_n'] = len(ids_v)
    for t in tags:
        d = res['dev3'][t]
        print(f"dev3 {t}: choice {d['choice_score']:.4f} | " + ' '.join(f"{k} {v[2]:.3f}" for k, v in d.items() if k != 'choice_score'))
        au = res['audit'][t]
        print(f"audit {t}: LOC P/R/F1 {au['LOC']['p_r_f1']} fp {au['LOC']['fp']} fn {au['LOC']['fn']} | PER {au['PER']['p_r_f1']} "
              f"fp {au['PER']['fp']} fn {au['PER']['fn']} | errors {au['errors_total']} (verse/drama {au['verse_drama_errors']})")
    if 'paired_dev3' in res:
        print('paired dev3 (B - A, mean LOC/PER F1):', res['paired_dev3'])
        print('paired audit:', res['paired_audit'])
    if a.json:
        json.dump(res, open(a.json, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
