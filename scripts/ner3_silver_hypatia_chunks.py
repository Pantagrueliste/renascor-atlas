"""Cut the silver pools into Hypatia job inputs (one GPU job per file, <= 3.5 h with the model load).

    python scripts/ner3_silver_hypatia_chunks.py --silver RQ/derived/atlas/ner/v2/renner/silver --per-job 25000 --out DIR \
        [--reserve] [--merge-es-it]

Order: the full_chunks files of the Ollama plan (c00 = pilot sample, then c01, c02 ...), concatenated per language, so the same
random order is kept and the pilot passages come first. With --reserve, pool_es_reserve and pool_it_reserve are cut as well
(separate files, *_reserve_jNN). --merge-es-it puts Spanish and Italian in the same files (esit_jNN, esit_reserve_jNN): they
share prompt v2, and the small pools then cost one model load. Writes <out>/<lang>_jNN.jsonl and <out>/manifest.tsv (file, lang, passages, first id, last id).
Checks: every pool id once, no gold v2 / dev3 file is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def lines(path):
    with open(path, encoding='utf8') as f:
        return [l if l.endswith('\n') else l + '\n' for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--silver', required=True)
    ap.add_argument('--per-job', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reserve', action='store_true')
    ap.add_argument('--merge-es-it', action='store_true')
    a = ap.parse_args()
    s, out = Path(a.silver), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = []
    for lang in ('es', 'it', 'la'):
        files = sorted((s / 'full_chunks').glob(f'{lang}_c*.jsonl'))
        rows = [l for f in files for l in lines(f)]
        pool = {json.loads(l)['id'] for l in lines(s / f'pool_{lang}.jsonl')}
        ids = [json.loads(l)['id'] for l in rows]
        assert len(ids) == len(set(ids)) and set(ids) == pool, f'{lang}: chunks differ from the pool'
        groups.append((lang, rows))
        if a.reserve and lang in ('es', 'it'):
            groups.append((f'{lang}_reserve', lines(s / f'pool_{lang}_reserve.jsonl')))
    if a.merge_es_it:
        g = dict(groups)
        merged = [('esit', g.pop('es') + g.pop('it'))]
        if a.reserve:
            merged.append(('esit_reserve', g.pop('es_reserve') + g.pop('it_reserve')))
        groups = merged + list(g.items())
    man = ['file\tlang\tpassages\tfirst_id\tlast_id']
    for name, rows in groups:
        n_jobs = -(-len(rows) // a.per_job)
        size = -(-len(rows) // n_jobs)            # even sizes
        for j in range(n_jobs):
            part = rows[j * size:(j + 1) * size]
            fn = out / f'{name}_j{j + 1:02d}.jsonl'
            fn.write_text(''.join(part), encoding='utf8')
            man.append(f'{fn.name}\t{"+".join(sorted({json.loads(x)["lang"] for x in part}))}\t{len(part)}\t{json.loads(part[0])["id"]}\t{json.loads(part[-1])["id"]}')
    (out / 'manifest.tsv').write_text('\n'.join(man) + '\n', encoding='utf8')
    print('\n'.join(man))


if __name__ == '__main__':
    main()
