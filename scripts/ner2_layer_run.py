"""NER layer v1, stage 2 (GPU): run the best model on untagged texts, then the other three encoders on the passages where it found a confident span.

    python scripts/ner2_layer_run.py --out O --shards texts/la/raw__x.00.jsonl.gz ... [--limit-docs N] [--smoke]

Per chunk of documents (about --chunk-tokens tokens; a chunk is the unit of resumption): r3_xlmr_large_H on every passage of 200 words (stride 180), keeping
every span with its confidence; then r3_mdeberta_HM, r3_mmbert_base_HM and r3_hmteams_HM on the passages where the main model has a span of confidence >= 0.9;
n_agree = number of the four models (the main one included) that predict exactly the same word span (NULL when the other models were not run on the passage).
Output per chunk: ``spans/<shard>__cNNN.parquet``. Chunks whose output exists are skipped, so a job that is killed or reaches its time limit is re-submitted as is.

Time limit: with --stop-after SECONDS the job stops cleanly after the chunk in progress once that time has elapsed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODELS = ['r3_xlmr_large_H', 'r3_mdeberta_HM', 'r3_mmbert_base_HM', 'r3_hmteams_HM']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_shard(path: Path):
    ids, texts = [], []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ids.append(d['id'])
            texts.append(d['t'])
    return ids, texts


def chunk_docs(texts, chunk_tokens: int):
    """Consecutive documents grouped to about chunk_tokens tokens (estimated as 1.25 x the number of whitespace words)."""
    chunks, cur, tot = [], [], 0
    for i, t in enumerate(texts):
        n = int(len(t.split()) * 1.25)
        cur.append(i)
        tot += n
        if tot >= chunk_tokens:
            chunks.append(cur)
            cur, tot = [], 0
    if cur:
        chunks.append(cur)
    return chunks


class PackSet:
    """Dataset of packs for a DataLoader: each item is prepare_pack() of a group of documents / selected passages."""

    def __init__(self, ti, texts, groups):
        self.ti, self.texts, self.groups = ti, texts, groups      # groups: list of (doc indices, plan)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        from ratlas.ner import layerpred as LP
        docs, plan = self.groups[i]
        return LP.prepare_pack(self.ti, docs, [self.texts[d] for d in docs], plan)


def plan_groups(texts, docs, pack_chars: int):
    groups, cur, tot = [], [], 0
    for d in docs:
        cur.append(d)
        tot += len(texts[d])
        if tot >= pack_chars:
            groups.append((cur, [None] * len(cur)))
            cur, tot = [], 0
    if cur:
        groups.append((cur, [None] * len(cur)))
    return groups


def plan_selected(sel_doc, sel_pass, pack_passages: int):
    """sel_doc / sel_pass: sorted unique (document, passage) pairs -> groups of whole documents with about pack_passages passages."""
    groups, cur_d, cur_p, n = [], [], [], 0
    if len(sel_doc) == 0:
        return groups
    brk = np.flatnonzero(np.r_[True, sel_doc[1:] != sel_doc[:-1]])
    brk = np.append(brk, len(sel_doc))
    for a, b in zip(brk[:-1], brk[1:]):
        cur_d.append(int(sel_doc[a]))
        cur_p.append(sel_pass[a:b])
        n += b - a
        if n >= pack_passages:
            groups.append((cur_d, cur_p))
            cur_d, cur_p, n = [], [], 0
    if cur_d:
        groups.append((cur_d, cur_p))
    return groups


def run_model(LP, bundle, texts, groups, workers: int):
    import torch
    ds = PackSet(bundle.ti, texts, groups)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=3 if workers else None, collate_fn=lambda x: x,
                                     persistent_workers=False)
    parts = []
    n_tok = 0
    for pack in dl:
        lab, pm, pp = LP.run_batches(bundle, pack)
        n_tok += int(pack['pass_off'][-1])
        parts.append(LP.pack_spans(pack, lab, pm, pp))
    keys = parts[0].keys() if parts else ['doc', 'passage', 'w_start', 'w_end', 'conf', 'p_per', 'char_start', 'char_end']
    out = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)) for k in keys}
    return out, n_tok


def year_array(y: np.ndarray):
    import pyarrow as pa
    bad = np.isnan(y)
    return pa.array(np.where(bad, 0, y).astype(np.int16), mask=bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--texts-root', required=True)
    ap.add_argument('--shards', nargs='*', default=[], help='paths relative to --texts-root')
    ap.add_argument('--shards-file', default='', help='one shard path per line, in the order of the work')
    ap.add_argument('--docs', required=True, help='docs table (parquet) with doc_id, collection, lang, year')
    ap.add_argument('--out', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--chunk-tokens', type=int, default=12_000_000)
    ap.add_argument('--pack-chars', type=int, default=2_000_000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--select-conf', type=float, default=0.9)
    ap.add_argument('--stop-after', type=float, default=0, help='seconds; stop cleanly after the chunk in progress')
    ap.add_argument('--limit-docs', type=int, default=0)
    ap.add_argument('--only-main', action='store_true')
    ap.add_argument('--tag', default='', help='prefix of the chunk files (use one tag per document filter)')
    ap.add_argument('--doc-filter', default='', help='text file: one doc_id per line; only these documents are run')
    a = ap.parse_args()

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from ratlas.ner import layerpred as LP

    t_start = time.time()
    if a.shards_file:
        a.shards = a.shards + [x for x in Path(a.shards_file).read_text().split() if x]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    out = Path(a.out)
    (out / 'spans').mkdir(parents=True, exist_ok=True)
    (out / 'chunks').mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(a.docs).set_index('doc_id')
    keep = None
    if a.doc_filter:
        keep = set(Path(a.doc_filter).read_text().split())
    bundles = {}
    names = MODELS[:1] if a.only_main else MODELS
    for nm in names:
        t0 = time.time()
        bundles[nm] = LP.load_bundle(str(Path(a.models_dir) / nm), dev, nm)
        log('loaded', nm, f'{time.time() - t0:.0f}s', 'prefix', bundles[nm].ti.prefix, 'suffix', bundles[nm].ti.suffix, 'L', bundles[nm].ti.L)
    main_b = bundles[MODELS[0]]
    tot_tok = 0
    tot_t = 0.0
    for sh in a.shards:
        lang = Path(sh).parent.name.split('_')[0]
        # with a tag the language is part of the chunk name: shards of different languages share task names (raw__x.00)
        sname = (a.tag + '__' + lang + '__' if a.tag else '') + Path(sh).name.replace('.jsonl.gz', '')
        todo = []
        ids = texts = None
        # decide the chunks without reading the text twice: chunk numbers depend only on the shard content, so read once
        ids, texts = read_shard(Path(a.texts_root) / sh)
        if keep is not None:
            sel = [i for i, d in enumerate(ids) if d in keep]
            ids = [ids[i] for i in sel]
            texts = [texts[i] for i in sel]
        if a.limit_docs:
            ids, texts = ids[:a.limit_docs], texts[:a.limit_docs]
        chunks = chunk_docs(texts, a.chunk_tokens)
        log(f'shard {sh}: {len(ids)} docs, {len(chunks)} chunks')
        for ci, docs in enumerate(chunks):
            fn = out / 'spans' / f'{sname}__c{ci:03d}.parquet'
            if fn.exists():
                continue
            if a.stop_after and time.time() - t_start > a.stop_after:
                log('time budget reached: stop before', fn.name)
                sys.exit(3)
            t0 = time.time()
            groups = plan_groups(texts, docs, a.pack_chars)
            h, n_tok = run_model(LP, main_b, texts, groups, a.workers)
            t1 = time.time()
            n_sp = len(h['doc'])
            n_agree = np.full(n_sp, -1, dtype=np.int8)
            n_ovl = np.full(n_sp, -1, dtype=np.int8)
            n_sel = 0
            if not a.only_main and n_sp:
                hi = h['conf'] >= a.select_conf
                pair = np.unique(np.stack((h['doc'][hi], h['passage'][hi]), 1), axis=0) if hi.any() else np.zeros((0, 2), dtype=np.int64)
                n_sel = len(pair)
                if n_sel:
                    sd, sp = pair[:, 0], pair[:, 1]
                    sel_mask = np.isin(h['doc'] * 10_000_000 + h['passage'], sd * 10_000_000 + sp)       # H spans of the passages that were run again
                    n_agree[sel_mask] = 1
                    n_ovl[sel_mask] = 1
                    groups2 = plan_selected(sd, sp, 1500)
                    for nm in MODELS[1:]:
                        o, _ = run_model(LP, bundles[nm], texts, groups2, a.workers)
                        ex, ov = LP.agreement(h, o)
                        n_agree[sel_mask] += ex[sel_mask]
                        n_ovl[sel_mask] += ov[sel_mask]
            t2 = time.time()
            doc_id = np.array([ids[d] for d in h['doc']], dtype=object) if n_sp else np.array([], dtype=object)
            surf = [texts[d][s:e] for d, s, e in zip(h['doc'], h['char_start'], h['char_end'])] if n_sp else []
            m = meta.reindex(doc_id) if n_sp else None
            tbl = pa.table({
                'doc_id': pa.array(doc_id.tolist(), pa.string()),
                'passage': pa.array(h['passage'].astype(np.int32)),
                'w_start': pa.array(h['w_start'].astype(np.int32)), 'w_end': pa.array(h['w_end'].astype(np.int32)),
                'char_start': pa.array(h['char_start'].astype(np.int32)), 'char_end': pa.array(h['char_end'].astype(np.int32)),
                'surface': pa.array(surf, pa.string()),
                'conf': pa.array(h['conf'].astype(np.float32)), 'p_per': pa.array(h['p_per'].astype(np.float32)),
                'n_agree': pa.array(np.where(n_agree < 0, 0, n_agree).astype(np.int8), mask=(n_agree < 0)),
                'n_overlap': pa.array(np.where(n_ovl < 0, 0, n_ovl).astype(np.int8), mask=(n_ovl < 0)),
                'lang': pa.array([lang] * n_sp, pa.string()),
                'year': year_array(m['year'].to_numpy(dtype='float64')) if n_sp else pa.array([], pa.int16()),
                'collection': pa.array(m['collection'].tolist() if n_sp else [], pa.string()),
            })
            tmp = fn.with_suffix('.tmp')
            pq.write_table(tbl, tmp, compression='zstd')
            os.replace(tmp, fn)
            (out / 'chunks').mkdir(exist_ok=True)
            (out / 'chunks' / f'{sname}__c{ci:03d}.docs').write_text('\n'.join(ids[d] for d in docs) + '\n')
            dt = time.time() - t0
            tot_tok += n_tok
            tot_t += t1 - t0
            log(f'{fn.name}: {len(docs)} docs, {n_tok} tokens, {n_sp} spans ({int((h["conf"] >= 0.98).sum()) if n_sp else 0} >= 0.98), {n_sel} passages re-run; '
                f'main {t1 - t0:.0f}s ({n_tok / max(1e-9, t1 - t0):.0f} tok/s), others {t2 - t1:.0f}s, total {dt:.0f}s')
    log(f'done: {tot_tok} tokens in {tot_t:.0f}s of main-model time')


if __name__ == '__main__':
    main()
