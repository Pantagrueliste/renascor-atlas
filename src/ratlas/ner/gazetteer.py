"""Dictionary baseline: exact match (case-sensitive, token sequences, longest match first) of the gazetteer of the training spans.

The gazetteer of a label is the set of token sequences tagged with that label in the training passages that annotate it. For every sequence we also count how many
times it occurs in those passages, tagged or not; variants keep only the sequences whose share of tagged occurrences reaches ``min_prec`` (estimated precision).
Nothing from dev or test is used.

    python -m ratlas.ner.gazetteer --train ours=.../train.jsonl --set dev=.../dev.jsonl --set test=.../test.jsonl --out RUNS --name gazetteer_ours
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from . import nerdata as nd
from .spans import tags_to_spans
from .train import expand, log, parse_named
from .zeroshot import score_span_preds

MAXLEN = 8


def build(recs, task: str):
    tags_key, valid_key = ('ner_tags_loc', 'loc_valid') if task == 'LOC' else ('ner_tags_per', 'per_valid')
    tagged = Counter()
    for r in recs:
        if not r.get(valid_key, True):
            continue
        t = r['tokens']
        for s, e in tags_to_spans(r[tags_key]):
            if e - s <= MAXLEN:
                tagged[tuple(t[s:e])] += 1
    seen = Counter()
    for r in recs:
        if not r.get(valid_key, True):
            continue
        t = r['tokens']
        n = len(t)
        for i in range(n):
            for k in range(1, min(MAXLEN, n - i) + 1):
                g = tuple(t[i:i + k])
                if g in tagged:
                    seen[g] += 1
    return tagged, seen


def tag_record(tokens, gaz: set):
    n = len(tokens)
    out, i = set(), 0
    while i < n:
        hit = 0
        for k in range(min(MAXLEN, n - i), 0, -1):
            if tuple(tokens[i:i + k]) in gaz:
                hit = k
                break
        if hit:
            out.add((i, i + hit))
            i += hit
        else:
            i += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--train', action='append', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-prec', default='0,0.5,0.8')
    args = ap.parse_args(argv)
    t0 = time.time()
    recs = []
    for _, p, _ in parse_named(args.train):
        for f in expand(p):
            recs += nd.read_slim(f)
    log(f'{len(recs)} training records')
    stats = {}
    gaz = {}
    for task in ('LOC', 'PER'):
        tagged, seen = build(recs, task)
        stats[task] = {'types': len(tagged), 'tokens_tagged': sum(tagged.values())}
        for mp in (float(x) for x in args.min_prec.split(',')):
            gaz[(task, mp)] = {g for g, c in tagged.items() if c / max(seen[g], c) >= mp}
            log(task, 'min_prec', mp, 'entries', len(gaz[(task, mp)]), 'of', len(tagged))
    del recs
    sets = {n: expand(p) for n, p, _ in parse_named(args.set)}
    out = {}
    for name, files in sets.items():
        eval_recs = []
        for f in files:
            eval_recs += nd.read_slim(f)
        for mp in (float(x) for x in args.min_prec.split(',')):
            preds = {r['id']: {task: tag_record(r['tokens'], gaz[(task, mp)]) for task in ('LOC', 'PER')} for r in eval_recs}
            rep = score_span_preds(eval_recs, preds).report()
            rep['n_records'] = len(eval_recs)
            out[f'{name}@minprec{mp}'] = rep
            log(f'[{name} minprec {mp}] LOC {rep["LOC"]["overall"]["p"]}/{rep["LOC"]["overall"]["r"]}/{rep["LOC"]["overall"]["f1"]}  '
                f'PER {rep["PER"]["overall"]["p"]}/{rep["PER"]["overall"]["r"]}/{rep["PER"]["overall"]["f1"]}')
    for name in sets:
        out[name] = out[f'{name}@minprec0.5']
    d = Path(args.out) / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'metrics.json').write_text(json.dumps({'run': args.name, 'kind': 'dictionary', 'gazetteer': stats, 'selected': 'min_prec 0.5 (fixed, not tuned on test)',
                                                'minutes': round((time.time() - t0) / 60, 1), 'sets': out}, indent=1))
    (d / 'done.json').write_text('{}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
