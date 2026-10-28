"""RenNER v2.1: a DEVELOPMENT split of the editor bench, for the lowercase-name effect only. Never the main-score editions.

    python scripts/ner3_v21_bench_dev.py build --bench editor_bench/bench.jsonl --editions editor_bench/editions.json --out DIR
    python scripts/ner3_v21_bench_dev.py score --dir DIR --editions editor_bench/editions.json --pred NAME=pred.jsonl.gz ... \
        [--pred-lc NAME=pred_lc.jsonl.gz ...] --ref NAME --out scores.json [--md scores.md]

build: the documents of the editions outside main_score (corpus_corporum, ferdinand_i, rubens_online, corpus17_french) are split in two by
a hash of (edition, doc) (seed "v21-bench-dev"): about half go to bench-dev (DIR/bench_dev.jsonl, DIR/ids.txt, DIR/docs.tsv), the rest
stays untouched (bench-rest, ids in DIR/rest_ids.txt). DIR/bench_dev_lc.jsonl is a synthetic copy of bench-dev in which every word of an
editor span (LOC, PER) loses its capital initial (same transformation as ner3_v21_lc_augment.py; text and offsets keep their length).
score: recall of the editor spans (views LOC_editor and PER of ner_editor_bench_score.py, ``ignored`` masks applied), strict and
overlap, by casing class of the gold span: "lower" (no capital letter in the span) and "cap"; on bench_dev_lc every span is "lower".
"new/kw": predictions touching no editor span of their label (nor a masked token) per 1000 words, a rough precision guard (the editions
are not exhaustive: these are not all errors). Bootstrap over documents (1000 draws, seed 1), paired differences with --ref.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ebs', HERE / 'ner_editor_bench_score.py')
ebs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ebs)

SEED = 'v21-bench-dev'


def lower_word(w: str) -> str:
    if len(w) > 1 and w.isupper() and sum(c.isalpha() for c in w) >= 2:
        return w.lower()
    if w[:1].isupper():
        return w[:1].lower() + w[1:]
    return w


def build(a):
    eds = json.load(open(a.editions, encoding='utf-8'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = {}
    dev, rest = [], []
    for line in open(a.bench, encoding='utf-8'):
        r = json.loads(line)
        if eds[r['edition']].get('main_score'):
            continue
        h = hashlib.sha1(f"{SEED}|{r['edition']}|{r['doc']}".encode()).hexdigest()
        side = 'dev' if int(h[:8], 16) % 2 == 0 else 'rest'
        docs[(r['edition'], r['doc'])] = side
        (dev if side == 'dev' else rest).append(r)
    with open(out / 'bench_dev.jsonl', 'w', encoding='utf-8') as f, open(out / 'bench_dev_lc.jsonl', 'w', encoding='utf-8') as g:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            toks = list(r['tokens'])
            text = list(r['text'])
            for lab in ('loc', 'per'):
                for s, e in ebs.tags_to_spans(r[f'ner_tags_{lab}_editor']):
                    for k in range(s, e):
                        w = lower_word(toks[k])
                        if w != toks[k]:
                            assert len(w) == len(toks[k])
                            st = r['token_starts'][k]
                            assert r['text'][st:st + len(w)] == toks[k], (r['id'], k)
                            text[st:st + len(w)] = list(w)
                            toks[k] = w
            c = dict(r)
            c['tokens'] = toks
            c['text'] = ''.join(text)
            c['id'] = r['id']
            g.write(json.dumps(c, ensure_ascii=False) + '\n')
    (out / 'ids.txt').write_text(''.join(r['id'] + '\n' for r in dev))
    (out / 'rest_ids.txt').write_text(''.join(r['id'] + '\n' for r in rest))
    with open(out / 'docs.tsv', 'w', encoding='utf-8') as f:
        f.write('edition\tdoc\tside\n')
        for (e, d), s in sorted(docs.items()):
            f.write(f'{e}\t{d}\t{s}\n')
    st = Counter()
    for side, rs in (('dev', dev), ('rest', rest)):
        for r in rs:
            st[(side, r['edition'], 'passages')] += 1
            st[(side, r['edition'], 'words')] += len(r['tokens'])
            for lab in ('loc', 'per'):
                for s, e in ebs.tags_to_spans(r[f'ner_tags_{lab}_editor']):
                    low = not any(ch.isupper() for t in r['tokens'][s:e] for ch in t)
                    st[(side, r['edition'], f'{lab}_{"lower" if low else "cap"}')] += 1
        for e in sorted({k[1] for k in st if k[0] == side}):
            st[(side, e, 'docs')] = sum(1 for (ee, _), s in docs.items() if ee == e and s == side)
    rep = {f'{a}|{b}|{c}': v for (a, b, c), v in sorted(st.items())}
    json.dump(rep, open(out / 'build_stats.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))


def casing(r, sp):
    return 'lower' if not any(ch.isupper() for t in r['tokens'][sp[0]:sp[1]] for ch in t) else 'cap'


def doc_counts(recs, preds):
    """-> {key: {doc: [found, gold]}}, new-finds per doc, words per doc."""
    C = {}
    newf, words = Counter(), Counter()
    for r in recs:
        d = (r['edition'], r['doc'])
        words[d] += len(r['tokens'])
        p = preds.get(r['id'], {'LOC': set(), 'PER': set()})
        for view, lab in (('LOC_editor', 'LOC'), ('PER', 'PER')):
            v = r['views'].get(view)
            if v is None:
                continue
            gold = ebs.drop(v['gold'], v['mask'])
            pr = ebs.drop(p[lab], v['mask'])
            for g in gold:
                cl = casing(r, g)
                for mode in ('strict', 'overlap'):
                    ok = (g in pr) if mode == 'strict' else any(ebs._ov(x, g) for x in pr)
                    for key in (f'{lab}|{cl}|{mode}', f'ALL|{cl}|{mode}', f'{lab}|any|{mode}'):
                        C.setdefault(key, Counter())
                        C[key][(d, 'f')] += ok
                        C[key][(d, 'g')] += 1
        for lab in ('LOC', 'PER'):
            if not r['tagged'][lab]:
                continue
            for sp in p[lab]:
                if ebs.new_find_category(r, lab, sp) is not None:
                    newf[(d, lab)] += 1
    return C, newf, words


def score(a):
    eds = json.load(open(a.editions, encoding='utf-8'))
    D = Path(a.dir)
    sets = {'dev': ebs.load_bench(D / 'bench_dev.jsonl', eds)}
    if a.pred_lc:
        sets['dev_lc'] = ebs.load_bench(D / 'bench_dev_lc.jsonl', eds)
    docs = sorted({(r['edition'], r['doc']) for r in sets['dev']})
    rng = np.random.default_rng(1)
    W = np.stack([np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs)) for _ in range(1000)])
    res = {'docs': len(docs), 'systems': {}, 'paired': {}}
    draws = {}
    for setname, specs in (('dev', a.pred), ('dev_lc', a.pred_lc or [])):
        for spec in specs:
            n, path = spec.split('=', 1)
            preds, _ = ebs.read_preds(path)
            C, newf, words = doc_counts(sets[setname], preds)
            o = {}
            for key, c in sorted(C.items()):
                f = np.array([c[(d, 'f')] for d in docs], float)
                g = np.array([c[(d, 'g')] for d in docs], float)
                rb = (W @ f) / np.maximum(W @ g, 1)
                o[key] = {'gold': int(g.sum()), 'r': round(float(f.sum() / max(g.sum(), 1)), 4),
                          'r_ci95': [round(float(x), 4) for x in np.percentile(rb, [2.5, 97.5])]}
                draws[(setname, n, key)] = rb
            wd = np.array([words[d] for d in docs], float)
            for lab in ('LOC', 'PER'):
                nf = np.array([newf[(d, lab)] for d in docs], float)
                o[f'{lab}|new_per_kw'] = round(float(1000 * nf.sum() / wd.sum()), 2)
                draws[(setname, n, f'{lab}|new_per_kw')] = 1000 * (W @ nf) / np.maximum(W @ wd, 1)
            res['systems'][f'{setname}:{n}'] = o
    if a.ref:
        for (setname, n, key), x in draws.items():
            if n == a.ref or (setname, a.ref, key) not in draws:
                continue
            dlt = x - draws[(setname, a.ref, key)]
            res['paired'].setdefault(f'{setname}:{n} - {a.ref}', {})[key] = {
                'diff': round(float(np.median(dlt)), 4), 'ci95': [round(float(v), 4) for v in np.percentile(dlt, [2.5, 97.5])]}
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
    keys = ['LOC|lower|overlap', 'PER|lower|overlap', 'ALL|lower|overlap', 'ALL|lower|strict', 'LOC|cap|strict', 'PER|cap|strict',
            'LOC|new_per_kw', 'PER|new_per_kw']
    lines = ['| système | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
    for s, o in res['systems'].items():
        lines.append(f'| {s} | ' + ' | '.join(
            (f"{o[k]['r']:.3f}" if isinstance(o.get(k), dict) else str(o.get(k, '–'))) for k in keys) + ' |')
    md = '\n'.join(lines)
    if a.md:
        Path(a.md).write_text(md + '\n')
    print(md)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--bench', required=True)
    b.add_argument('--editions', required=True)
    b.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--dir', required=True)
    s.add_argument('--editions', required=True)
    s.add_argument('--pred', action='append', default=[])
    s.add_argument('--pred-lc', action='append', default=[])
    s.add_argument('--ref')
    s.add_argument('--out', required=True)
    s.add_argument('--md')
    a = ap.parse_args()
    if a.cmd == 'build':
        assert 'gold2' not in a.bench
        build(a)
    else:
        score(a)


if __name__ == '__main__':
    main()
