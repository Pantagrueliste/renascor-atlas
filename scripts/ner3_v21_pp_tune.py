"""RenNER v2.1 (b): tune the boundary post-processing rules (src/ratlas/ner/postproc.py) on dev3 ONLY.

    python scripts/ner3_v21_pp_tune.py --gold dev3/gold_dev3.jsonl --pred NAME=pred_dev3.jsonl.gz ... --out pp_tune.json [--md pp_tune.md]

For every system: the dev3 selection score (mean of six strict F1, es/it/la x LOC/PER, --accept-alt, all spans; ner3_eval_v2_dev3.py) with
no rule, with each rule alone, and greedy forward selection (add the rule with the best mean gain over the systems while the mean gain
is > 0 and no system loses). Paired bootstrap of the final set against no rule (1000 draws, seed 20261023). Number of spans changed.
Gold v2 / gold v3 / the editor bench are never read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
import ner3_teacher_score as S  # noqa: E402
import ner3_eval_v2_dev3 as E  # noqa: E402
from ratlas.ner import postproc as PP  # noqa: E402


def run(gold, raw, rules):
    out, changed = {}, 0
    for r in gold:
        d = raw.get(r['id'], {'id': r['id'], 'LOC': [], 'PER': []})
        new = PP.apply_pred_line(d, r['tokens'], r['lang'], rules)
        for lab in ('LOC', 'PER'):
            a = {(x['s'], x['e']) for x in d.get(lab) or []}
            b = {(x['s'], x['e']) for x in new[lab]}
            changed += len(a ^ b)
        out[r['id']] = new
    return out, changed


def sel(gold, pred):
    c = S.per_passage_counts(gold, pred, True, False)
    sc = S.score(gold, pred, counts=c)
    cells = {f'{lg}|{lab}': sc[(lg, lab, 'strict')][2] for lg in E.LANGS for lab in E.LABS}
    cells.update({f'all|{lab}': sc[('all', lab, 'strict')][2] for lab in E.LABS})
    return sum(cells[f'{lg}|{lab}'] for lg in E.LANGS for lab in E.LABS) / 6, cells, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args()
    assert 'gold2' not in a.gold and 'gold3' not in a.gold and 'dev3' in a.gold, 'dev3 only'
    gold = S.load(a.gold)
    raws = {}
    for s in a.pred:
        n, p = s.split('=', 1)
        raws[n] = {d['id']: S.to_pred(d) for d in S.load(p)}
    res = {'single': {}, 'greedy': [], 'final': {}}
    base = {n: sel(gold, run(gold, raws[n], ())[0]) for n in raws}
    for rule in PP.RULES:
        res['single'][rule] = {}
        for n in raws:
            pr, ch = run(gold, raws[n], (rule,))
            s, cells, _ = sel(gold, pr)
            res['single'][rule][n] = {'gain': round(s - base[n][0], 4), 'changed': ch,
                                      'cells': {k: round(v - base[n][1][k], 4) for k, v in cells.items()}}
    chosen = []
    while True:
        best = None
        for rule in PP.RULES:
            if rule in chosen:
                continue
            gains = []
            for n in raws:
                pr, _ = run(gold, raws[n], tuple(chosen + [rule]))
                cur = sel(gold, run(gold, raws[n], tuple(chosen))[0])[0]
                gains.append(sel(gold, pr)[0] - cur)
            mg = sum(gains) / len(gains)
            if mg > 1e-9 and min(gains) >= 0 and (best is None or mg > best[1]):
                best = (rule, mg, gains)
        if best is None:
            break
        chosen.append(best[0])
        res['greedy'].append({'rule': best[0], 'mean_gain': round(best[1], 4), 'gains': [round(g, 4) for g in best[2]]})
    res['chosen'] = chosen
    for n in raws:
        pr, ch = run(gold, raws[n], tuple(chosen))
        s, cells, c = sel(gold, pr)
        ps = E.paired_selection(gold, c, base[n][2])
        pc = E.paired(gold, c, base[n][2])
        res['final'][n] = {'base': round(base[n][0], 4), 'pp': round(s, 4), 'diff': round(ps['diff'], 4),
                           'ci': [round(x, 4) for x in ps['ci']], 'changed': ch,
                           'cells_base': {k: round(v, 4) for k, v in base[n][1].items()}, 'cells_pp': {k: round(v, 4) for k, v in cells.items()},
                           'paired_cells': {k: {'diff': round(v['diff'], 4), 'ci': [round(x, 4) for x in v['ci']]} for k, v in pc.items()
                                            if k.endswith('strict')}}
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
    names = list(raws)
    L = ['| règle | ' + ' | '.join(f'{n} gain (empans changés)' for n in names) + ' |', '|---|' + '---:|' * len(names)]
    for rule in PP.RULES:
        L.append(f'| {rule} | ' + ' | '.join(f"{res['single'][rule][n]['gain']:+.4f} ({res['single'][rule][n]['changed']})" for n in names) + ' |')
    L.append('')
    L.append('Choix glouton : ' + (', '.join(chosen) or 'aucune règle'))
    for n in names:
        f = res['final'][n]
        L.append(f"- {n} : {f['base']:.4f} -> {f['pp']:.4f} ({f['diff']:+.4f} [{f['ci'][0]:+.4f} ; {f['ci'][1]:+.4f}]), {f['changed']} empans changés")
    md = '\n'.join(L)
    if a.md:
        Path(a.md).write_text(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
