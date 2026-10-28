"""Compare encoder setups per language on gold v2 (provisional reference: agreed + proposed spans).

    python scripts/ner2_es_setup_score.py --gold DIR --out FILE.md [--json FILE.json]

A setup returns a set of exact word spans (s, e) per passage. Scores are micro P / R / F1 per language, exact match.
Setups: single encoders; union of two; main model alone at confidence >= c; "recommended" = main model with confidence >= 0.98 and agreement of at least
2 of the 4 encoders (the main one counts), as in the layer (ner2_layer_merge.py).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ENC = ['r3_xlmr_large_H', 'r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_hmteams_HM']


def spans_of(rec, model, min_conf=0.0):
    return {(x['s'], x['e']): x['conf'] for x in rec['models'].get(model, []) if x['conf'] is None or x['conf'] >= min_conf}


def recommended(rec, main, min_conf=0.98, min_agree=2):
    """Spans of ``main`` with confidence >= min_conf and >= min_agree encoders (main included) predicting exactly the same span."""
    out = set()
    others = [spans_of(rec, m) for m in ENC if m != main]
    for k, c in spans_of(rec, main).items():
        if c >= min_conf and 1 + sum(k in o for o in others) >= min_agree:
            out.add(k)
    return out


def setup_spans(rec, name):
    if name in ENC:
        return set(spans_of(rec, name))
    if name == 'union xlmr+mdeberta':
        return set(spans_of(rec, ENC[0])) | set(spans_of(rec, ENC[1]))
    if name == 'union 4 encoders':
        return set().union(*[set(spans_of(rec, m)) for m in ENC])
    if name == 'mdeberta conf>=0.9':
        return set(spans_of(rec, 'r3_mdeberta_HM', 0.9))
    if name == 'mdeberta + agree>=2 (any conf)':
        return recommended(rec, 'r3_mdeberta_HM', 0.0, 2)
    if name == 'mdeberta + agree>=2, conf>=0.9':
        return recommended(rec, 'r3_mdeberta_HM', 0.9, 2)
    if name == 'recommended, xlmr main (layer v1)':
        return recommended(rec, 'r3_xlmr_large_H')
    if name == 'recommended, mdeberta main':
        return recommended(rec, 'r3_mdeberta_HM')
    if name == 'recommended, mdeberta main, agree>=1 (conf>=0.98)':
        return recommended(rec, 'r3_mdeberta_HM', 0.98, 1)
    if name == 'vote>=2 encoders (any main)':
        c = defaultdict(int)
        for m in ENC:
            for k in spans_of(rec, m):
                c[k] += 1
        return {k for k, v in c.items() if v >= 2}
    raise KeyError(name)


SETUPS = ENC + ['union xlmr+mdeberta', 'union 4 encoders', 'mdeberta conf>=0.9', 'mdeberta + agree>=2 (any conf)', 'mdeberta + agree>=2, conf>=0.9',
                'recommended, xlmr main (layer v1)', 'recommended, mdeberta main', 'recommended, mdeberta main, agree>=1 (conf>=0.98)', 'vote>=2 encoders (any main)']


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(preds, golds):
    """preds, golds: dict passage -> set of spans. Returns (tp, fp, fn)."""
    tp = fp = fn = 0
    for k, g in golds.items():
        p = preds.get(k, set())
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    gold = {}
    for l in open(a.gold / 'proposed_gold_v2.jsonl', encoding='utf-8'):
        d = json.loads(l)
        gold[d['id']] = (d['lang'], {(s['start'], s['end']) for s in d['spans'] if s['status'] in ('agreed', 'proposed')})
    preds = {}
    for l in open(a.gold / 'predictions.jsonl', encoding='utf-8'):
        d = json.loads(l)
        preds[d['id']] = d
    langs = ['en', 'la', 'it', 'es', 'fr', 'de']
    res = {}
    for st in SETUPS:
        res[st] = {}
        for lg in langs + ['all']:
            ids = [i for i, (l, _) in gold.items() if lg == 'all' or l == lg]
            gp = {i: gold[i][1] for i in ids}
            pp = {i: setup_spans(preds[i], st) for i in ids}
            tp, fp, fn = score(pp, gp)
            res[st][lg] = dict(zip(['tp', 'fp', 'fn'], [tp, fp, fn]), **dict(zip(['P', 'R', 'F1'], prf(tp, fp, fn))))
    ng = {lg: sum(len(g) for l, g in gold.values() if l == lg) for lg in langs}
    npass = {lg: sum(1 for l, g in gold.values() if l == lg) for lg in langs}
    lines = ['# Setups comparés sur gold v2 (référence provisoire : agreed + proposed)', '',
             'Provisoire : la référence est la mienne (510 spans), pas relue par un humain. Correspondance exacte. Cellule : F1 (P/R).', '',
             'Passages : ' + ', '.join(f'{l} {npass[l]}' for l in langs) + '. Spans de référence : ' + ', '.join(f'{l} {ng[l]}' for l in langs) + '.', '',
             '| setup | ' + ' | '.join(langs) + ' | toutes |', '|---|' + '---|' * (len(langs) + 1)]
    for st in SETUPS:
        lines.append(f'| {st} | ' + ' | '.join(f"{res[st][lg]['F1']:.2f} ({res[st][lg]['P']:.2f}/{res[st][lg]['R']:.2f})" for lg in langs + ['all']) + ' |')
    a.out.write_text('\n'.join(lines) + '\n', 'utf-8')
    if a.json:
        a.json.write_text(json.dumps({'n_ref': ng, 'n_pass': npass, 'scores': res}, indent=1), 'utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
