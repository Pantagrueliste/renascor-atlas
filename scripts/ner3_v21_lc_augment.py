"""RenNER v2.1 (a): lowercase-name augmentation. Training copies of a share of the S-full passages in which the entity names lose their
capital initial (and, for a part of the copies, every word of the passage: texts written almost all in lowercase, like Crenum).

    python scripts/ner3_v21_lc_augment.py --src ours=.../train.jsonl --src hist_gold=.../train.jsonl --src silver_bal_la=... \
        --sfull-passages 627600 --share 0.12 --p-all 0.3 --p-keep-span 0.2 --seed 21 --out aug_lc12.jsonl --report aug_lc12.json

Sources: the S-full sets read from disk (ours, hist_gold, the five silver files; modern_silver is left out: modern texts, sampled inside
train.py). Only passages with at least one entity of a valid label are eligible. Number of copies = share x number of S-full passages,
drawn uniformly among the eligible passages of all sources (seeded, without replacement).
Per copy, mode "names" (probability 1 - p_all): every word inside an entity span of a valid label loses its capital initial (an all-caps
word of 2+ letters is lowercased entirely), except that a span is kept unchanged with probability p_keep_span (mixed casing within a
passage); mode "all" (probability p_all): the same for every word of the passage. Tags are unchanged (the same words, the same spans).
A copy in which no word changed is not written. Ids get the suffix "#lc". Gold v2, gold v3, dev3 and the editor bench are never read.
Records whose id, edition or source names London Lives, MAP or a private collection are skipped (they are not in S-full anyway).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

FIELDS = ('id', 'tokens', 'ner_tags_loc', 'ner_tags_per', 'loc_valid', 'per_valid', 'lang', 'edition', 'source', 'subset')
EXCLUDE = re.compile(r'london.?lives|(^|[/_:])map([/_:]|$)|private', re.I)


def spans(tags):
    out, s = [], None
    for i, t in enumerate(tags):
        if t == 1:
            if s is not None:
                out.append((s, i))
            s = i
        elif t == 2 and s is not None:
            continue
        else:
            if s is not None:
                out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(tags)))
    return out


def lower_word(w: str) -> str:
    if len(w) > 1 and w.isupper() and sum(c.isalpha() for c in w) >= 2:
        return w.lower()
    if w[:1].isupper():
        return w[:1].lower() + w[1:]
    return w


def entity_spans(r):
    out = []
    if r.get('loc_valid', True) and r.get('ner_tags_loc'):
        out += spans(r['ner_tags_loc'])
    if r.get('per_valid', True) and r.get('ner_tags_per'):
        out += spans(r['ner_tags_per'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', action='append', required=True, help='name=path')
    ap.add_argument('--sfull-passages', type=int, required=True)
    ap.add_argument('--share', type=float, required=True)
    ap.add_argument('--p-all', type=float, default=0.3)
    ap.add_argument('--p-keep-span', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    for s in a.src:
        assert not re.search(r'gold2|gold3|gold_v2|dev3|editor_bench|test2|test_v2', s), s
    pool = []          # (source name, line offset in file, path)
    lines = {}
    skipped = Counter()
    for s in a.src:
        name, path = s.split('=', 1)
        L = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if EXCLUDE.search(' '.join(str(d.get(k, '')) for k in ('id', 'edition', 'source', 'subset'))):
                    skipped[name] += 1
                    continue
                d = {k: d[k] for k in FIELDS if k in d}
                if entity_spans(d):
                    L.append(d)
        lines[name] = L
        pool += [(name, i) for i in range(len(L))]
        print(f'{name}: {len(L)} eligible passages', flush=True)
    n = round(a.share * a.sfull_passages)
    rnd = random.Random(a.seed)
    order = list(range(len(pool)))
    rnd.shuffle(order)
    stats = Counter()
    by_src = Counter()
    by_lang = Counter()
    written = 0
    with open(a.out + '.tmp', 'w', encoding='utf-8') as out:
        for k in order:
            if written >= n:
                break
            name, i = pool[k]
            r = lines[name][i]
            mode = 'all' if rnd.random() < a.p_all else 'names'
            toks = list(r['tokens'])
            if mode == 'all':
                idx = range(len(toks))
            else:
                idx = set()
                for s, e in entity_spans(r):
                    if rnd.random() < a.p_keep_span:
                        continue
                    idx.update(range(s, e))
            changed = 0
            for j in idx:
                w = lower_word(toks[j])
                if w != toks[j]:
                    toks[j] = w
                    changed += 1
            if not changed:
                stats['unchanged_skipped'] += 1
                continue
            c = dict(r)
            c['id'] = f"{r['id']}#lc"
            c['tokens'] = toks
            c['source'] = f"{r.get('source', name)}+lc"
            out.write(json.dumps(c, ensure_ascii=False) + '\n')
            written += 1
            stats[f'mode_{mode}'] += 1
            stats['words_changed'] += changed
            stats['words'] += len(toks)
            by_src[name] += 1
            by_lang[r.get('lang') or '?'] += 1
    import os
    os.replace(a.out + '.tmp', a.out)
    rep = {'args': vars(a), 'target_copies': n, 'written': written, 'eligible': len(pool), 'stats': dict(stats),
           'by_source': dict(by_src), 'by_lang': dict(by_lang.most_common()), 'skipped_excluded': dict(skipped)}
    json.dump(rep, open(a.report, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == '__main__':
    main()
