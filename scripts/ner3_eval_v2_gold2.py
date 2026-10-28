"""RenNER v2, FINAL TEST: LOC on gold v2 (final_gold_v2.jsonl) for the dev3-selected student(s) and r3_mdeberta_HM. Run once, at the end.

    python scripts/ner3_eval_v2_gold2.py --gold gold2/final_gold_v2.jsonl --pred NAME=path ... --ref r3_mdeberta_HM --out scores.json

Scoring logic of ner2_gold2_apply_review.py (score_passage): owner-accepted boundary variants form one entity (any variant = one TP),
doubt spans are neutral, strict (exact) and lenient (overlap) matching; micro P/R/F1 per language and overall.
Intervals: passages resampled with replacement within each language (1000 draws, seed 20261023); paired differences on the same draws.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner2_gold2_apply_review import LANGS, prf, score_passage  # noqa: E402


def load_pred(p):
    op = gzip.open if str(p).endswith('.gz') else open
    out = {}
    with op(p, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            out[d['id']] = {(x['s'], x['e']) for x in d.get('LOC', [])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', action='append', default=[])
    ap.add_argument('--ref', default='r3_mdeberta_HM')
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    gold = [json.loads(l) for l in open(a.gold, encoding='utf-8') if l.strip()]
    ids = {lg: [k for k, r in enumerate(gold) if r['lang'] == lg] for lg in LANGS}
    ids['all'] = list(range(len(gold)))
    counts = {}
    out = {'gold': a.gold, 'n_passages': len(gold), 'passages_per_lang': {k: len(v) for k, v in ids.items()},
           'bootstrap': {'draws': a.boot, 'seed': 20261023, 'unit': 'passage, within language'}, 'scores': {}, 'paired_vs_ref': {}}
    for s in a.pred:
        name, path = s.split('=', 1)
        P = load_pred(path)
        miss = sum(r['id'] not in P for r in gold)
        counts[name] = {mode: [score_passage(r, P.get(r['id'], set()), mode == 'lenient') for r in gold] for mode in ('strict', 'lenient')}
        out['scores'][name] = {'missing': miss}
    rnd = random.Random(20261023)
    draws = {lg: [[v[rnd.randrange(len(v))] for _ in v] for _ in range(a.boot)] for lg, v in ids.items()}

    def tot(c, sel):
        t = [0, 0, 0]
        for k in sel:
            for i in range(3):
                t[i] += c[k][i]
        return t

    def q(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]
    for name, cm in counts.items():
        for mode, c in cm.items():
            for lg, sel in ids.items():
                t = tot(c, sel)
                p, r, f = prf(*t)
                out['scores'][name][f'{mode}|{lg}'] = {'tp': t[0], 'fp': t[1], 'fn': t[2], 'p': round(p, 4), 'r': round(r, 4), 'f1': round(f, 4),
                                                        'f1_ci95': q([prf(*tot(c, d))[2] for d in draws[lg]])}
        print(name, ' '.join(f"{lg}:{out['scores'][name][f'strict|{lg}']['f1']:.3f}" for lg in ids), flush=True)
    if a.ref in counts:
        for name, cm in counts.items():
            if name == a.ref:
                continue
            e = {}
            for mode, c in cm.items():
                cr = counts[a.ref][mode]
                for lg, sel in ids.items():
                    e[f'{mode}|{lg}'] = {'diff': round(prf(*tot(c, sel))[2] - prf(*tot(cr, sel))[2], 4),
                                         'ci95': q([prf(*tot(c, d))[2] - prf(*tot(cr, d))[2] for d in draws[lg]])}
            out['paired_vs_ref'][name] = e
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('written', a.out)


if __name__ == '__main__':
    main()
