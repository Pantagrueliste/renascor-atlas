"""RenNER v2: score the students, r3_mdeberta_HM and the teacher on dev3 (never on gold v2), with paired bootstrap differences.

    python scripts/ner3_eval_v2_dev3.py --gold dev3/gold_dev3.jsonl --pred NAME=path ... --teacher es,it=v2.jsonl --teacher la=v3.jsonl \
        --pair A,B ... --out scores.json

Scorer: ner3_teacher_score.py with --accept-alt; two settings (all spans; sure only); strict and overlap; 95 % percentile intervals,
1000 draws, passages resampled within each language, seed 20261023 (S.bootstrap). Paired differences: the same draws for both systems
(one Random(20261023) per comparison, languages in the order es, it, la, all).
--teacher LANGS=path: the teacher's raw output (renner_llm_teacher format) for those languages; +pp is applied (final '.' dropped,
name propagated), exactly as for the silver (ner3_silver_combine.py).
Selection score (fixed before reading any dev3 score of the students): mean of the six strict F1 (es, it, la x LOC, PER), all spans.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ner3_teacher_score as S  # noqa: E402

_spec = importlib.util.spec_from_file_location('bk', HERE / 'ner3_teacher_bakeoff.py')
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)

LANGS = ['es', 'it', 'la']
LABS = ['LOC', 'PER']
MODES = ['strict', 'overlap']


def f1(t):
    return S.prf(*t)[2]


def paired(gold, ca, cb, n=1000, seed=20261023):
    rnd = random.Random(seed)
    ids = {lg: [r['id'] for r in gold if r['lang'] == lg] for lg in LANGS}
    ids['all'] = [r['id'] for r in gold]
    out = {}
    for lg in LANGS + ['all']:
        I = ids[lg]
        diffs = {(lab, m): [] for lab in LABS for m in MODES}
        diffs_sel = []
        for _ in range(n):
            pick = [I[rnd.randrange(len(I))] for _ in I]
            for lab in LABS:
                for m in MODES:
                    ta = [sum(ca[i][(lab, m)][k] for i in pick) for k in range(3)]
                    tb = [sum(cb[i][(lab, m)][k] for i in pick) for k in range(3)]
                    diffs[(lab, m)].append(f1(ta) - f1(tb))
        for (lab, m), xs in diffs.items():
            xs = sorted(xs)
            pa = [sum(ca[i][(lab, m)][k] for i in I) for k in range(3)]
            pb = [sum(cb[i][(lab, m)][k] for i in I) for k in range(3)]
            out[f'{lg}|{lab}|{m}'] = {'diff': f1(pa) - f1(pb), 'ci': [xs[int(0.025 * (n - 1))], xs[int(0.975 * (n - 1))]]}
    return out


def paired_selection(gold, ca, cb, n=1000, seed=20261023):
    """Paired interval of the difference of the selection score (mean of six strict F1); passages resampled within each language."""
    rnd = random.Random(seed)
    ids = {lg: [r['id'] for r in gold if r['lang'] == lg] for lg in LANGS}

    def sel(c, picks):
        v = []
        for lg in LANGS:
            for lab in LABS:
                v.append(f1([sum(c[i][(lab, 'strict')][k] for i in picks[lg]) for k in range(3)]))
        return sum(v) / len(v)
    d = []
    for _ in range(n):
        picks = {lg: [ids[lg][rnd.randrange(len(ids[lg]))] for _ in ids[lg]] for lg in LANGS}
        d.append(sel(ca, picks) - sel(cb, picks))
    d.sort()
    return {'diff': sel(ca, ids) - sel(cb, ids), 'ci': [d[int(0.025 * (n - 1))], d[int(0.975 * (n - 1))]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', action='append', default=[])
    ap.add_argument('--teacher', action='append', default=[], help='LANGS=path (raw teacher output); +pp applied')
    ap.add_argument('--teacher-name', default='teacher_gemma4_31b_pp')
    ap.add_argument('--pair', action='append', default=[])
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert 'final_gold_v2' not in a.gold and 'gold2' not in a.gold, 'gold v2 is the final test set'
    gold = S.load(a.gold)
    toks = {r['id']: r['tokens'] for r in gold}
    lang = {r['id']: r['lang'] for r in gold}
    preds = {}
    for s in a.pred:
        n, p = s.split('=', 1)
        preds[n] = {d['id']: S.to_pred(d) for d in S.load(p)}
    if a.teacher:
        t = {}
        for s in a.teacher:
            lg, p = s.split('=', 1)
            lgs = set(lg.split(','))
            raw = {d['id']: S.to_pred(d) for d in S.load(p) if lang.get(d['id']) in lgs}
            t.update(bk.propagate(bk.strip_final_dot(raw, toks), toks))
        preds[a.teacher_name] = t
    out = {'gold': a.gold, 'bootstrap': a.bootstrap, 'seed': 20261023, 'systems': {}, 'paired': {}, 'selection': {}}
    counts = {}
    for n, p in preds.items():
        miss = len(set(toks) - set(p))
        res = {'missing_ids': miss}
        for setting, sure in (('all', False), ('sure', True)):
            c = S.per_passage_counts(gold, p, True, sure)
            if setting == 'all':
                counts[n] = c
            sc = S.score(gold, p, counts=c)
            ci = S.bootstrap(gold, c, a.bootstrap)
            res[setting] = {'|'.join(k): {'p': v[0], 'r': v[1], 'f1': v[2], 'f1_ci': ci[k][2]} for k, v in sc.items()}
        res['selection_score'] = sum(res['all'][f'{lg}|{lab}|strict']['f1'] for lg in LANGS for lab in LABS) / 6
        out['systems'][n] = res
        print(f"{n:28s} sel {res['selection_score']:.4f} " + ' '.join(
            f"{lg}:{res['all'][f'{lg}|LOC|strict']['f1']:.3f}/{res['all'][f'{lg}|PER|strict']['f1']:.3f}" for lg in LANGS + ['all']), flush=True)
    for s in a.pair:
        x, y = s.split(',')
        out['paired'][f'{x} - {y}'] = paired(gold, counts[x], counts[y], a.bootstrap)
        out['paired'][f'{x} - {y}']['selection'] = paired_selection(gold, counts[x], counts[y], a.bootstrap)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('written', a.out)


if __name__ == '__main__':
    main()
