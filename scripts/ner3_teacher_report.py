"""dev3 summary for many teachers: F1 with bootstrap 95% CI per language (LOC, PER; strict and overlap), error types, as JSON.

    python scripts/ner3_teacher_report.py --gold dev3/gold_dev3.jsonl --pred name=path ... [--combine vote:a,b,c,d@3] --out report.json
Uses the dev3 scorer (ner3_teacher_score.py) with --accept-alt; two settings: all spans, and sure only. Never on gold v2.
Error types (strict, all spans, accept-alt): boundary (overlaps a gold span of the same label), label (exact span, other label),
spurious (touches no gold span), missed (gold span no prediction overlaps, any label).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner3_teacher_score as S  # noqa: E402


def errors(gold, pred):
    c = Counter()
    ex = {'missed': Counter(), 'spurious': Counter()}
    for r in gold:
        toks = r['tokens']
        p = pred.get(r['id'], {})
        for lab in ('LOC', 'PER'):
            other = 'PER' if lab == 'LOC' else 'LOC'
            g = S.gold_items(r, lab, True)
            go = S.gold_items(r, other, True)
            ps = S.pred_spans(p, lab)
            po = S.pred_spans(p, other)
            for x in ps:
                if any(x in f for f, _, _ in g):
                    continue
                if any(S._ov(x, rg) for _, rg, _ in g):
                    c[(r['lang'], lab, 'boundary')] += 1
                elif any(x in f for f, _, _ in go):
                    c[(r['lang'], lab, 'label')] += 1
                elif any(S._ov(x, rg) for _, rg, _ in go):
                    c[(r['lang'], lab, 'label+boundary')] += 1
                else:
                    c[(r['lang'], lab, 'spurious')] += 1
                    ex['spurious'][(r['lang'], lab, ' '.join(toks[x[0]:x[1]]))] += 1
            for f, rg, sure in g:
                if not any(S._ov(x, rg) for x in ps | po):
                    c[(r['lang'], lab, 'missed')] += 1
                    ex['missed'][(r['lang'], lab, ' '.join(toks[rg[0]:rg[1]]))] += 1
    return ({'|'.join(k): v for k, v in c.items()},
            {k: ['|'.join(x) + f' x{n}' for x, n in v.most_common(40)] for k, v in ex.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--pred', action='append', default=[])
    ap.add_argument('--combine', action='append', default=[])
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert 'final_gold_v2' not in a.gold and 'gold2' not in a.gold, 'gold v2 is the final test set'
    gold = S.load(a.gold)
    preds, meta = {}, {}
    for s in a.pred:
        n, p = s.split('=', 1)
        lines = S.load(p)
        preds[n] = {d['id']: S.to_pred(d) for d in lines}
        meta[n] = {'n_lines': len(lines), 'errors': sum(1 for d in lines if d.get('error')),
                   'unmatched': sum(len(d.get('unmatched') or []) for d in lines),
                   'missing_ids': len({r['id'] for r in gold} - set(preds[n]))}
    for s in a.combine:
        preds[s] = S.combine(s, preds)
        meta[s] = {}
    out = {'gold': a.gold, 'bootstrap': a.bootstrap, 'systems': {}}
    for n, p in preds.items():
        res = {'meta': meta[n]}
        for setting, sure in (('all', False), ('sure', True)):
            counts = S.per_passage_counts(gold, p, True, sure)
            sc = S.score(gold, p, counts=counts)
            ci = S.bootstrap(gold, counts, a.bootstrap) if a.bootstrap else {}
            res[setting] = {'|'.join(k): {'p': v[0], 'r': v[1], 'f1': v[2], 'f1_ci': ci.get(k, [None, None, None])[2]}
                            for k, v in sc.items()}
        res['errors'], res['examples'] = errors(gold, p)
        out['systems'][n] = res
        f = res['all']
        print(n, ' '.join(f"{lg}:{f[f'{lg}|LOC|strict']['f1']:.3f}/{f[f'{lg}|PER|strict']['f1']:.3f}" for lg in ('es', 'it', 'la', 'all')))
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
