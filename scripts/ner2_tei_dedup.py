"""NER layer, TEI editions, step 2 (CPU, Hypatia): duplicate control at document level, and the share of each document already seen by the v1 models.

Method: the one of the leak checks (folded words, 8-word shingles, hashed, one in four kept; ``ner2_tei_text.shingle_hashes``), plus an exact fingerprint of the
folded word sequence for texts too short to have sampled shingles. The small side is indexed (the TEI documents read in step 1), the big side is streamed:

    index   --work W                      -> W/dedup/index.npz
    scan    --work W --name layer --glob 'layer_v1/extract/texts/[a-z][a-z]/*.jsonl.gz' --procs 22
    scan    --work W --name train --glob 'data_v1/*.jsonl' 'external/aggregate/hist_gold/*.jsonl' ... [--kind text]
    within  --work W                      -> W/dedup/within.parquet   (pairs of TEI documents)
    decide  --work W                      -> W/dedup/decisions.parquet, pairs_layer.tsv, pairs_set.tsv

Rules (``decide``):
    dup_layer       >= DUP_COVER of the sampled shingles of the document are found in documents of the model layer (union), or the exact fingerprint equals one
    dup_set         the same against another TEI document that is kept: mutual (both >= DUP_COVER) -> keep the one with more editor tags, then more words, then the
                    smaller id; one-way (the document is inside a bigger one) -> the contained one goes, unless it carries editor tags and the bigger one carries none
    overlap_layer   OVERLAP_MIN <= cover < DUP_COVER: kept, flagged
    seen_share      share of the sampled shingles found in the v1 train / dev / test passages, the external aggregates and the gold seed
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner2_tei_text as T  # noqa: E402

DUP_COVER = 0.5
OVERLAP_MIN = 0.1
MIN_PAIR = 3            # sampled shingles in common for a pair to be kept
MIN_SIZE = 16           # sampled shingles below which only the exact fingerprint counts
MAX_OWNERS = 40         # a shingle shared by more TEI documents than this is boilerplate
G: dict = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def read_jsonl(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------------------------------------------------
def _hash_shard(path):
    out = []
    for d in read_jsonl(path):
        ids = T.word_ids(d['t'])
        out.append((d['id'], T.shingles_of_ids(ids), T.fingerprint(ids), len(ids)))
    return out


def cmd_index(a):
    work = Path(a.work)
    files = sorted(glob.glob(str(work / 'texts' / '*.jsonl.gz')))
    ids, sizes, fps, nw, hs = [], [], [], [], []
    with mp.get_context('fork').Pool(a.procs) as pool:
        for res in pool.imap(_hash_shard, files):
            for did, h, fp, n in res:
                ids.append(did)
                sizes.append(len(h))
                fps.append(fp)
                nw.append(n)
                hs.append(h)
    H = np.concatenate(hs) if hs else np.zeros(0, dtype=np.uint64)
    owner = np.repeat(np.arange(len(ids), dtype=np.int32), sizes)
    order = np.argsort(H, kind='stable')
    H, owner = H[order], owner[order]
    UH, starts, counts = np.unique(H, return_index=True, return_counts=True)
    (work / 'dedup').mkdir(exist_ok=True)
    np.savez(work / 'dedup' / 'index.npz', UH=UH, starts=starts.astype(np.int64), counts=counts.astype(np.int32), owner=owner, sizes=np.array(sizes, dtype=np.int32),
             fps=np.array(fps, dtype=np.uint64), nwords=np.array(nw, dtype=np.int64))
    json.dump(ids, open(work / 'dedup' / 'ids.json', 'w'))
    log(len(ids), 'documents,', len(H), 'sampled shingles,', len(UH), 'distinct,', int((counts > 1).sum()), 'shared between documents')


def load_index(work):
    z = np.load(Path(work) / 'dedup' / 'index.npz')
    G['UH'], G['starts'], G['counts'], G['owner'] = z['UH'], z['starts'], z['counts'], z['owner']
    G['sizes'], G['fps'] = z['sizes'], z['fps']
    G['ids'] = json.load(open(Path(work) / 'dedup' / 'ids.json'))
    G['fp_map'] = {}
    for i, f in enumerate(G['fps'].tolist()):
        G['fp_map'].setdefault(f, []).append(i)


def match_ids(ids, hit_union):
    """For a word-id array: (hit positions in UH, owners array, exact owners)."""
    h = T.shingles_of_ids(ids)
    UH = G['UH']
    if len(h) == 0 or len(UH) == 0:
        return len(h), None, None
    idx = np.searchsorted(UH, h)
    idx[idx >= len(UH)] = len(UH) - 1
    m = UH[idx] == h
    if not m.any():
        return len(h), None, None
    hi = idx[m]
    hit_union[hi] = True
    cnt = G['counts'][hi].astype(np.int64)
    st = G['starts'][hi]
    keep = cnt <= MAX_OWNERS
    cnt, st = cnt[keep], st[keep]
    if len(cnt) == 0:
        return len(h), None, None
    tot = int(cnt.sum())
    pos = np.repeat(st - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
    owners = G['owner'][pos]
    u, c = np.unique(owners, return_counts=True)
    return len(h), u, c


def _scan_file(task):
    path, kind = task
    n_idx = len(G['UH'])
    union = np.zeros(n_idx, dtype=bool)
    pairs, exact = [], []
    nd = 0
    for d in read_jsonl(path):
        text = d['t'] if 't' in d else (d['text'] if d.get('text') else ' '.join(d.get('tokens', [])))
        did = d.get('id', '')
        ids = T.word_ids(text)
        nd += 1
        fp = T.fingerprint(ids)
        if fp in G['fp_map'] and len(ids) >= 20:
            for o in G['fp_map'][fp]:
                exact.append((o, did))
        ns, u, c = match_ids(ids, union)
        if u is not None:
            for o, k in zip(u.tolist(), c.tolist()):
                if k >= MIN_PAIR:
                    pairs.append((o, did, k, ns))
    return path, nd, pairs, exact, np.flatnonzero(union)


def cmd_scan(a):
    work = Path(a.work)
    load_index(work)
    files = []
    for g in a.glob:
        files += sorted(glob.glob(g))
    files = [f for f in files if 'en_ranked' not in f]
    log(len(files), 'files to stream')
    pairs, exact, union = [], [], np.zeros(len(G['UH']), dtype=bool)
    nd = 0
    with mp.get_context('fork').Pool(a.procs) as pool:
        for path, n, p, e, u in pool.imap_unordered(_scan_file, [(f, a.kind) for f in files], chunksize=1):
            pairs += p
            exact += e
            union[u] = True
            nd += n
            log(Path(path).name, n, 'docs', len(p), 'pairs')
    import pandas as pd
    out = work / 'dedup'
    pd.DataFrame(pairs, columns=['owner', 'other', 'shared', 'other_size']).to_parquet(out / f'scan_{a.name}_pairs.parquet', index=False)
    pd.DataFrame(exact, columns=['owner', 'other']).to_parquet(out / f'scan_{a.name}_exact.parquet', index=False)
    # per document: sampled shingles found anywhere in the streamed set
    cover = np.zeros(len(G['sizes']), dtype=np.int64)
    sel = np.flatnonzero(union)
    if len(sel):
        cnt = G['counts'][sel].astype(np.int64)
        st = G['starts'][sel]
        tot = int(cnt.sum())
        pos = np.repeat(st - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
        own = G['owner'][pos]
        cover = np.bincount(own, minlength=len(G['sizes']))
    np.save(out / f'scan_{a.name}_cover.npy', cover)
    log(nd, 'documents streamed;', len(pairs), 'pairs;', len(exact), 'exact')


def cmd_within(a):
    work = Path(a.work)
    load_index(work)
    UH, starts, counts, owner = G['UH'], G['starts'], G['counts'], G['owner']
    c = Counter()
    m2 = counts == 2
    s2 = starts[m2]
    x, y = owner[s2].astype(np.int64), owner[s2 + 1].astype(np.int64)
    lo, hi = np.minimum(x, y), np.maximum(x, y)
    keys = lo * len(G['ids']) + hi
    u, k = np.unique(keys, return_counts=True)
    for kk, vv in zip(u.tolist(), k.tolist()):
        c[kk] += vv
    for s, n in zip(starts[(counts > 2) & (counts <= MAX_OWNERS)].tolist(), counts[(counts > 2) & (counts <= MAX_OWNERS)].tolist()):
        o = sorted(set(owner[s:s + n].tolist()))
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                c[o[i] * len(G['ids']) + o[j]] += 1
    import pandas as pd
    N = len(G['ids'])
    rows = [(kk // N, kk % N, v) for kk, v in c.items() if v >= MIN_PAIR]
    pd.DataFrame(rows, columns=['a', 'b', 'shared']).to_parquet(work / 'dedup' / 'within.parquet', index=False)
    # exact fingerprints inside the set
    ex = [(v[i], v[j]) for v in G['fp_map'].values() if len(v) > 1 for i in range(len(v)) for j in range(i + 1, len(v))]
    pd.DataFrame(ex, columns=['a', 'b']).to_parquet(work / 'dedup' / 'within_exact.parquet', index=False)
    log(len(rows), 'pairs of TEI documents sharing >=', MIN_PAIR, 'sampled shingles;', len(ex), 'exact')


def cmd_decide(a):
    import pandas as pd
    work = Path(a.work)
    d = work / 'dedup'
    z = np.load(d / 'index.npz')
    ids = json.load(open(d / 'ids.json'))
    sizes = z['sizes'].astype(np.int64)
    N = len(ids)
    docs = pd.read_parquet(work / 'docs_read.parquet').set_index('doc_id')
    ntags = docs['n_tags'].reindex(ids).fillna(0).astype(int).values
    nwords = z['nwords']
    # layer
    lp = pd.read_parquet(d / 'scan_layer_pairs.parquet')
    le = pd.read_parquet(d / 'scan_layer_exact.parquet')
    lcov = np.load(d / 'scan_layer_cover.npy')
    cover_layer = np.where(sizes > 0, lcov / np.maximum(sizes, 1), 0.0)
    lp['c_doc_in_other'] = lp['shared'] / np.maximum(sizes[lp['owner'].values], 1)
    lp['c_other_in_doc'] = lp['shared'] / np.maximum(lp['other_size'], 1)
    best = lp.sort_values('shared', ascending=False).drop_duplicates('owner').set_index('owner')
    exact_layer = le.groupby('owner')['other'].first()
    # training sets
    seen = np.zeros(N)
    for p in sorted(d.glob('scan_train*_cover.npy')):
        seen = np.maximum(seen, np.load(p) / np.maximum(sizes, 1))
    # decisions
    status = np.array(['keep'] * N, dtype=object)
    reason = np.array([''] * N, dtype=object)
    other = np.array([''] * N, dtype=object)
    for i in range(N):
        if i in exact_layer.index:
            status[i], reason[i], other[i] = 'drop', 'dup_layer_exact', exact_layer[i]
        elif sizes[i] >= MIN_SIZE and cover_layer[i] >= DUP_COVER:
            status[i], reason[i] = 'drop', 'dup_layer'
            other[i] = best.loc[i, 'other'] if i in best.index else ''
    # within the set
    w = pd.read_parquet(d / 'within.parquet')
    we = pd.read_parquet(d / 'within_exact.parquet')
    rows = []
    for r in w.itertuples():
        ca, cb = r.shared / max(sizes[r.a], 1), r.shared / max(sizes[r.b], 1)
        rows.append((r.a, r.b, r.shared, ca, cb))
    for r in we.itertuples():
        rows.append((r.a, r.b, -1, 1.0, 1.0))
    pairs = pd.DataFrame(rows, columns=['a', 'b', 'shared', 'c_a_in_b', 'c_b_in_a'])
    pairs['doc_a'] = [ids[i] for i in pairs['a']]
    pairs['doc_b'] = [ids[i] for i in pairs['b']]

    def pref(i):          # smaller = kept first
        return (-ntags[i], -nwords[i], ids[i])

    excl = (docs['v1_doc'].reindex(ids).fillna(False).astype(bool) | docs['gold'].reindex(ids).fillna(False).astype(bool)).values     # kept out of the main set: do not compete
    cover_set = np.zeros(N)
    for r in pairs.sort_values(['shared'], ascending=False).itertuples():
        a_, b_ = r.a, r.b
        cover_set[a_] = max(cover_set[a_], r.c_a_in_b)
        cover_set[b_] = max(cover_set[b_], r.c_b_in_a)
        if status[a_] == 'drop' or status[b_] == 'drop' or excl[a_] or excl[b_]:
            continue
        in_a, in_b = r.c_a_in_b >= DUP_COVER, r.c_b_in_a >= DUP_COVER
        if not (in_a or in_b):
            continue
        if in_a and in_b:
            lose = a_ if pref(a_) > pref(b_) else b_
        else:
            lose = a_ if in_a else b_                      # the contained one
            win = b_ if in_a else a_
            if ntags[lose] > 0 and ntags[win] == 0:
                continue
        win = b_ if lose == a_ else a_
        status[lose], reason[lose], other[lose] = 'drop', 'dup_set', ids[win]
    out = pd.DataFrame({'doc_id': ids, 'n_shingles': sizes, 'n_tags': ntags, 'cover_layer': cover_layer, 'cover_set': cover_set, 'seen_share': seen,
                        'dedup': status, 'dedup_reason': reason, 'dup_of': other})
    out['overlap_layer'] = (out['cover_layer'] >= OVERLAP_MIN) & (out['dedup'] == 'keep')
    out.to_parquet(d / 'decisions.parquet', index=False)
    lp.sort_values(['owner', 'shared'], ascending=[True, False]).assign(doc=lambda x: [ids[i] for i in x['owner']]).query('c_doc_in_other >= 0.1 or c_other_in_doc >= 0.1')\
        .to_csv(d / 'pairs_layer.tsv', sep='\t', index=False)
    pairs[(pairs.c_a_in_b >= 0.1) | (pairs.c_b_in_a >= 0.1)].to_csv(d / 'pairs_set.tsv', sep='\t', index=False)
    log(out['dedup'].value_counts().to_dict(), out['dedup_reason'].value_counts().to_dict(), 'overlap flagged', int(out['overlap_layer'].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['index', 'scan', 'within', 'decide'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--name', default='layer')
    ap.add_argument('--glob', nargs='*', default=[])
    ap.add_argument('--kind', default='text')
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    {'index': cmd_index, 'scan': cmd_scan, 'within': cmd_within, 'decide': cmd_decide}[a.cmd](a)


if __name__ == '__main__':
    main()
