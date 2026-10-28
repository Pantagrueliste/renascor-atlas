"""NER layer v1, stage 1b (CPU): order the English documents so that any prefix of the work list is balanced by decade, and write the work list.

English (EEBO-TCP is 85 % of the words) may not fit in the GPU budget. Documents of each decade are put in a seeded random order, and the global order is the
cumulative number of tokens inside the decade: reading the list from the top gives every decade the same number of tokens until the small decades run out.
The English texts are re-written, in that order, as ``texts/en_ranked/rNNN.jsonl.gz`` (about --shard-tokens tokens each). The other languages keep their shards.

    python scripts/ner2_layer_plan.py --extract EXTRACT [--shard-tokens 25000000] [--procs 20]

Writes ``plan/en_ranked.tsv`` (doc_id, decade, n_tok, key, shard) and ``plan/worklist.txt`` (shards in the order of the work: la, it, es, fr, de, en_ranked).
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

G = {}


def split_one(path):
    out = Path(G['extract'])
    rank = G['rank']                       # doc_id -> shard index
    bufs = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            i = line.index('"id": "') + 7
            j = line.index('"', i)
            did = json.loads('"' + line[i:j] + '"')
            bufs.setdefault(rank[did], []).append(line)
    d = out / 'texts' / 'en_ranked_parts'
    d.mkdir(parents=True, exist_ok=True)
    tag = Path(path).name.replace('.jsonl.gz', '')
    for k, lines in bufs.items():
        with gzip.open(d / f'r{k:03d}.{tag}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as g:
            g.writelines(lines)
    return len(bufs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', required=True)
    ap.add_argument('--shard-tokens', type=int, default=25_000_000)
    ap.add_argument('--procs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261021)
    a = ap.parse_args()
    X = Path(a.extract)
    dr = pd.read_parquet(X / 'docs_read.parquet')
    en = dr[(dr['status'] == 'ok') & (dr['lang'] == 'en')].copy()
    en['decade'] = (en['year'] // 10 * 10).astype(int)
    rng = np.random.default_rng(a.seed)
    en = en.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    en['key'] = en.groupby('decade')['n_tok'].cumsum() - en['n_tok']              # tokens of the decade ahead of the document
    en = en.sort_values(['key', 'decade']).reset_index(drop=True)
    en['cum'] = en['n_tok'].cumsum() - en['n_tok']
    en['shard'] = (en['cum'] // a.shard_tokens).astype(int)
    (X / 'plan').mkdir(exist_ok=True)
    en[['doc_id', 'decade', 'n_tok', 'n_words', 'key', 'shard', 'task']].to_csv(X / 'plan' / 'en_ranked.tsv', sep='\t', index=False)
    print('english docs', len(en), 'tokens', int(en['n_tok'].sum()), 'shards', int(en['shard'].max()) + 1)
    G['extract'] = str(X)
    G['rank'] = dict(zip(en['doc_id'], en['shard'].astype(int)))
    files = sorted(str(p) for p in (X / 'texts' / 'en').glob('*.jsonl.gz'))
    with mp.get_context('fork').Pool(a.procs) as pool:
        for _ in pool.imap_unordered(split_one, files):
            pass
    d = X / 'texts' / 'en_ranked'
    d.mkdir(parents=True, exist_ok=True)
    for k in range(int(en['shard'].max()) + 1):
        parts = sorted((X / 'texts' / 'en_ranked_parts').glob(f'r{k:03d}.*.jsonl.gz'))
        with open(d / f'r{k:03d}.jsonl.gz', 'wb') as out:                    # concatenated gzip members are one valid gzip stream
            for p in parts:
                out.write(p.read_bytes())
    for p in (X / 'texts' / 'en_ranked_parts').glob('*.gz'):
        p.unlink()
    (X / 'texts' / 'en_ranked_parts').rmdir()
    shards = pd.read_csv(X / 'shards.tsv', sep='\t')
    order = []
    for lang in ('la', 'it', 'es', 'fr', 'de'):
        sub = shards[shards['lang'] == lang].sort_values('file')
        order += list(sub['file'])
    order += [f'texts/en_ranked/r{k:03d}.jsonl.gz' for k in range(int(en['shard'].max()) + 1)]
    (X / 'plan' / 'worklist.txt').write_text('\n'.join(order) + '\n')
    print(len(order), 'shards in the work list')
    t = en.groupby('decade').agg(docs=('doc_id', 'size'), tokens=('n_tok', 'sum'))
    print(t.to_string())


if __name__ == '__main__':
    main()
