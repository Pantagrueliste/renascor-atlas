"""Self-check of the test_v2 judgements: draw occurrences judged 'add' and 'error', show them without their label, re-judge, compare.

    python scripts/ner2_selfcheck_testv2.py sample --data test_v2_loc.jsonl --pred-dir pred --judged-dir judged --blind blind.json --key key.json [--n-add 40 --n-error 20 --seed 20261021]
    python scripts/ner2_selfcheck_testv2.py score --key key.json --rejudged mine.json --out selfcheck.json

``blind.json`` holds the contexts only (edition, language, text around the span, the span): no decision, no group, no statistics. The re-judge writes
``mine.json`` = {"<item>": {"label": "add" | "error" | "ambiguous", "note": "..."}} before ``key.json`` is opened. The draw is stratified by edition
(equal allocation) and reports the pooled rate re-weighted by the number of 'add' (or 'error') occurrences of each edition.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratlas.ner.ner2_cand import context, read_preds                                  # noqa: E402
from ratlas.ner.ner2_testgold import iter_decisions, load_judgements, occurrences     # noqa: E402

LABELS = ('add', 'error', 'ambiguous')


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def alloc(total, groups):
    base, extra = divmod(total, len(groups))
    return {g: base + (1 if i < extra else 0) for i, g in enumerate(groups)}


def sample(a):
    recs = {}
    with open(a.data, encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            recs[d['id']] = d
    preds = {p.name[:-len('.jsonl.gz')]: read_preds(p) for p in sorted(a.pred_dir.glob('*.jsonl.gz'))}
    cands, _ = occurrences(recs, preds)
    judged = load_judgements(a.judged_dir)
    pools = {'add': defaultdict(list), 'error': defaultdict(list)}
    for ed, surf, o, dec, fixes, explicit in iter_decisions(cands, judged):
        if dec in pools and not fixes and surf:
            pools[dec][ed].append((o, surf, explicit))
    eds = sorted(pools['add'])
    rng = random.Random(a.seed)
    blind, key = [], []
    for dec, total in (('add', a.n_add), ('error', a.n_error)):
        al = alloc(total, [e for e in eds if pools[dec][e]])
        for ed in sorted(al):
            for o, surf, explicit in rng.sample(pools[dec][ed], min(al[ed], len(pools[dec][ed]))):
                key.append({'decision': dec, 'edition': ed, 'id': o['id'], 's': o['s'], 'e': o['e'], 'group': surf, 'explicit': explicit,
                            'n_pool_edition': len(pools[dec][ed]), 'n_systems': len(o['systems'])})
    rng.shuffle(key)
    for i, k in enumerate(key):
        k['item'] = f'it{i:02d}'
        c = context(recs[k['id']], k['s'], k['e'], k=30)
        blind.append({'item': k['item'], 'edition': k['edition'], 'lang': c['lang'], 'left': c['left'], 'span': c['span'], 'right': c['right']})
    a.blind.write_text(json.dumps(blind, ensure_ascii=False, indent=1), encoding='utf-8')
    a.key.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    print(len(blind), 'items;', Counter(k['decision'] for k in key))


def score(a):
    key = json.loads(a.key.read_text(encoding='utf-8'))
    mine = json.loads(a.rejudged.read_text(encoding='utf-8'))
    rows = []
    for k in key:
        m = mine[k['item']]['label']
        assert m in LABELS, (k['item'], m)
        rows.append({**k, 'rejudged': m, 'note': mine[k['item']].get('note', '')})
    out = {'items': len(rows)}
    for dec in ('add', 'error'):
        rs = [r for r in rows if r['decision'] == dec]
        n = len(rs)
        same = sum(r['rejudged'] == dec for r in rs)
        conf = Counter(r['rejudged'] for r in rs)
        # not the opposite label: an 'add' re-judged 'ambiguous' is a doubt, not a contradiction
        not_opp = sum(r['rejudged'] != ('error' if dec == 'add' else 'add') for r in rs)
        per_ed = defaultdict(lambda: [0, 0])
        for r in rs:
            per_ed[r['edition']][1] += 1
            per_ed[r['edition']][0] += r['rejudged'] == dec
        pool_n = {r['edition']: r['n_pool_edition'] for r in rs}
        weighted = sum(pool_n[e] * (k / m) for e, (k, m) in per_ed.items()) / sum(pool_n[e] for e in per_ed)
        expl = [r for r in rs if r['explicit']]
        dflt = [r for r in rs if not r['explicit']]
        out[dec] = {'n': n, 'agree': same, 'agree_rate': round(same / n, 4), 'wilson95': wilson(same, n), 'rejudged_as': dict(conf),
                    'not_opposite': not_opp, 'not_opposite_rate': round(not_opp / n, 4), 'not_opposite_wilson95': wilson(not_opp, n),
                    'weighted_by_edition_agree_rate': round(weighted, 4),
                    'by_edition': {e: {'agree': k, 'n': m} for e, (k, m) in sorted(per_ed.items())},
                    'explicit_judged': {'n': len(expl), 'agree': sum(r['rejudged'] == dec for r in expl)},
                    'group_default_applied': {'n': len(dflt), 'agree': sum(r['rejudged'] == dec for r in dflt)}}
    out['all'] = {'agree': sum(r['rejudged'] == r['decision'] for r in rows), 'n': len(rows)}
    out['all']['agree_rate'] = round(out['all']['agree'] / out['all']['n'], 4)
    out['disagreements'] = [{k: r[k] for k in ('item', 'edition', 'group', 'decision', 'rejudged', 'explicit', 'note')} for r in rows if r['rejudged'] != r['decision']]
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'disagreements'}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--data', type=Path, required=True)
    s.add_argument('--pred-dir', type=Path, required=True)
    s.add_argument('--judged-dir', type=Path, required=True)
    s.add_argument('--blind', type=Path, required=True)
    s.add_argument('--key', type=Path, required=True)
    s.add_argument('--n-add', type=int, default=40)
    s.add_argument('--n-error', type=int, default=20)
    s.add_argument('--seed', type=int, default=20261021)
    c = sub.add_parser('score')
    c.add_argument('--key', type=Path, required=True)
    c.add_argument('--rejudged', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    sample(a) if a.cmd == 'sample' else score(a)


if __name__ == '__main__':
    main()
