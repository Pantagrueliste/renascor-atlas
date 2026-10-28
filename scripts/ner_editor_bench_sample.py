"""Stratified sample of the editor-tag bench for the LLM teacher (and the slim passage files the teacher reads).

    python scripts/ner_editor_bench_sample.py --bench editor_bench/bench.jsonl --outdir editor_bench/teacher_sample [--quota de=417,...]

Quotas per language (default: 2,000 passages; Italian taken whole); inside a language, passages are allotted to editions in proportion
to their size, editions outside main_score weighted by 0.25 (largest remainder, at least one per edition), then drawn by hash (seed
"editor-bench-teacher-2026-10-24").
Writes ids.txt (all sampled ids), passages_la.jsonl (Latin: guide v3) and passages_other.jsonl (other languages: guide v2), with the
fields the teacher needs (id, lang, text, tokens, token_starts), and sample_stats.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = 'editor-bench-teacher-2026-10-24'
QUOTA = {'de': 417, 'la': 416, 'fr': 350, 'en': 250, 'es': 250, 'it': 217, 'nl': 100}


def h(x: str) -> str:
    return hashlib.sha1(f'{SEED}|{x}'.encode()).hexdigest()


def allot(sizes: dict, q: int, weight: dict | None = None) -> dict:
    """Largest-remainder allotment of q among the keys of sizes (in proportion to size x weight), at least 1 each, never more than the size."""
    if q >= sum(sizes.values()):
        return dict(sizes)
    wt = {k: n * (weight or {}).get(k, 1.0) for k, n in sizes.items()}
    tot = sum(wt.values())
    base = {k: max(1, int(q * wt[k] / tot)) for k in sizes}
    base = {k: min(v, sizes[k]) for k, v in base.items()}
    rem = sorted(sizes, key=lambda k: (-(q * wt[k] / tot - int(q * wt[k] / tot)), k))
    i = 0
    while sum(base.values()) < q:
        k = rem[i % len(rem)]
        if base[k] < sizes[k]:
            base[k] += 1
        i += 1
    while sum(base.values()) > q:
        k = max(base, key=lambda k: (base[k], k))
        base[k] -= 1
    return base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--quota', help='lang=n,... (replaces the default quotas)')
    a = ap.parse_args(argv)
    quota = dict(QUOTA)
    if a.quota:
        quota = {k: int(v) for k, v in (x.split('=') for x in a.quota.split(','))}
    by = defaultdict(lambda: defaultdict(list))
    recs, secondary = {}, set()
    for line in open(a.bench, encoding='utf-8'):
        r = json.loads(line)
        if not r['edition_flags'].get('main_score'):
            secondary.add(r['edition'])
        if r.get('lang') in quota:
            by[r['lang']][r['edition']].append(r['id'])
            recs[r['id']] = {k: r[k] for k in ('id', 'lang', 'text', 'tokens', 'token_starts')}
    chosen, stats = [], {}
    for lang, q in quota.items():
        eds = by.get(lang, {})
        al = allot({e: len(v) for e, v in eds.items()}, q, {e: 0.25 for e in secondary})
        for e, n in sorted(al.items()):
            chosen += sorted(eds[e], key=h)[:n]
        stats[lang] = {'quota': q, 'available': sum(len(v) for v in eds.values()), 'by_edition': dict(sorted(al.items()))}
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    order = sorted(chosen, key=h)
    (out / 'ids.txt').write_text(''.join(i + '\n' for i in order), encoding='utf-8')
    n = Counter()
    with open(out / 'passages_la.jsonl', 'w', encoding='utf-8') as fl, open(out / 'passages_other.jsonl', 'w', encoding='utf-8') as fo:
        for i in order:
            r = recs[i]
            (fl if r['lang'] == 'la' else fo).write(json.dumps(r, ensure_ascii=False) + '\n')
            n['la' if r['lang'] == 'la' else 'other'] += 1
    json.dump({'seed': SEED, 'n': len(order), 'files': dict(n), 'languages': stats}, open(out / 'sample_stats.json', 'w'), indent=1, ensure_ascii=False)
    print(len(order), dict(n))


if __name__ == '__main__':
    main()
