"""Verse routing check: gemma-4-31B-it fewshot v3 against v4 (+pp) on the fresh Latin verse/drama dev set (80 passages, blind annotations).

    python scripts/ner3_verse_dev_score.py --passages verse_dev/passages.jsonl --ann verse_dev/ann_verse80.txt \
        --pred v3=pred/gemma4_31b__fewshotv3.jsonl --pred v4=pred/gemma4_31b__fewshotv4.jsonl --json out.json

Strict match (start, end, label) per label; score = mean of the LOC and PER F1. Two readings: all spans, and without the convention zones
(every reference span marked conv, and every predicted span overlapping a conv zone by one token, are dropped on both sides).
Paired bootstrap per passage of B - A, 2000 draws, seed 20261025 (rule: verse_routing_regle.md). Gold v2 / gold v3 are never read.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sc = _load('sc', HERE / 'ner3_teacher_score.py')
bk = _load('bk', HERE / 'ner3_teacher_bakeoff.py')
LABS = ('LOC', 'PER')


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def pp(raw, toks):
    pred = {d['id']: sc.to_pred(d) for d in raw if d['id'] in toks}
    return bk.propagate(bk.strip_final_dot(pred, toks), toks)


def score(counts, ids):
    f = []
    for lab in LABS:
        t = [sum(counts[i][lab][k] for i in ids) for k in range(3)]
        f.append(sc.prf(*t)[2])
    return sum(f) / 2


def paired(ca, cb, ids, n=2000, seed=20261025):
    rnd = random.Random(seed)
    d0 = score(cb, ids) - score(ca, ids)
    ds = []
    for _ in range(n):
        pick = [ids[rnd.randrange(len(ids))] for _ in ids]
        ds.append(score(cb, pick) - score(ca, pick))
    ds.sort()
    return round(d0, 4), round(ds[int(0.025 * (n - 1))], 4), round(ds[int(0.975 * (n - 1))], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passages', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--pred', action='append', required=True, help='TAG=raw teacher output')
    ap.add_argument('--json')
    ap.add_argument('--seed', type=int, default=20261025, help='bootstrap seed (default: the verse routing rule)')
    a = ap.parse_args()
    mv = _load('mv', Path(a.passages).with_name('match_vd.py'))
    P = jl(a.passages)
    ann = mv.parse(a.ann)
    ref = {}
    for r, d in enumerate(P):
        spans, errs = mv.to_spans(d, ann[r])
        assert not errs, errs
        ref[d['id']] = (d, spans)
    toks = {i: d['tokens'] for i, (d, _) in ref.items()}
    ids = [d['id'] for d in P]
    tags = [s.split('=', 1)[0] for s in a.pred]
    res = {'n_passages': len(ids), 'forms': {}, 'reading': {}, 'errors': {}}
    for d in P:
        res['forms'][d['form']] = res['forms'].get(d['form'], 0) + 1
    counts = {}
    for t, path in (s.split('=', 1) for s in a.pred):
        raw = jl(path)
        assert {d['id'] for d in raw} >= set(ids), 'missing passages in ' + path
        pr = pp(raw, toks)
        counts[t] = {'all': {}, 'noconv': {}}
        errs = []
        for i, (d, spans) in ref.items():
            g_all = {(s, e, lab) for s, e, lab, cv in spans if lab != 'O'}
            zones = [(s, e) for s, e, lab, cv in spans if cv]
            g_nc = {(s, e, lab) for s, e, lab, cv in spans if lab != 'O' and not cv}
            p_all = {(x['s'], x['e'], lab) for lab in LABS for x in pr.get(i, {}).get(lab, [])}
            p_nc = {x for x in p_all if not any(x[0] < ze and zs < x[1] for zs, ze in zones)}
            for key, g, p in (('all', g_all, p_all), ('noconv', g_nc, p_nc)):
                counts[t][key][i] = {lab: (len({x for x in g & p if x[2] == lab}), len({x for x in p - g if x[2] == lab}),
                                           len({x for x in g - p if x[2] == lab})) for lab in LABS}
            tk = d['tokens']
            conv_set = {(s, e, lab) for s, e, lab, cv in spans if cv and lab != 'O'}
            for x in sorted(p_all - g_all):
                errs.append({'id': i, 'form': d['form'], 'kind': 'FP', 'label': x[2], 'text': ' '.join(tk[x[0]:x[1]]),
                             'conv_zone': any(x[0] < ze and zs < x[1] for zs, ze in zones)})
            for x in sorted(g_all - p_all):
                errs.append({'id': i, 'form': d['form'], 'kind': 'FN', 'label': x[2], 'text': ' '.join(tk[x[0]:x[1]]), 'conv_zone': x in conv_set})
        res['errors'][t] = errs
        res['reading'][t] = {}
        for key in ('all', 'noconv'):
            c = counts[t][key]
            out = {}
            for lab in LABS:
                tot = [sum(c[i][lab][k] for i in ids) for k in range(3)]
                out[lab] = {'tp': tot[0], 'fp': tot[1], 'fn': tot[2], 'p_r_f1': [round(v, 4) for v in sc.prf(*tot)]}
            out['score'] = round(score(c, ids), 4)
            for form in ('verse', 'drama'):
                fid = [i for i in ids if ref[i][0]['form'] == form]
                out[f'score_{form}'] = round(score(c, fid), 4)
            out['errors'] = sum(out[lab]['fp'] + out[lab]['fn'] for lab in LABS)
            res['reading'][t][key] = out
    if len(tags) == 2:
        A, B = tags
        res['paired'] = {key: paired(counts[A][key], counts[B][key], ids, seed=a.seed) for key in ('all', 'noconv')}
        for form in ('verse', 'drama'):
            fid = [i for i in ids if ref[i][0]['form'] == form]
            res['paired'][f'all_{form}'] = paired(counts[A]['all'], counts[B]['all'], fid, seed=a.seed)
        lo = res['paired']['all'][1]
        res['routing_selected_by_rule'] = lo > 0
    for t in tags:
        for key in ('all', 'noconv'):
            o = res['reading'][t][key]
            print(f"{t} {key}: score {o['score']:.4f} | LOC {o['LOC']['p_r_f1']} fp {o['LOC']['fp']} fn {o['LOC']['fn']} | "
                  f"PER {o['PER']['p_r_f1']} fp {o['PER']['fp']} fn {o['PER']['fn']} | errors {o['errors']} | verse {o['score_verse']} drama {o['score_drama']}")
    if 'paired' in res:
        print('paired (B - A):', res['paired'], 'selected:', res['routing_selected_by_rule'])
    if a.json:
        json.dump(res, open(a.json, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
