"""NER layer v3 (CPU, Hypatia, debug partition): duplicate control of the new sets and the per-form facts of the corrections.

    python scripts/ner2_layer_v3_check.py --data ~/renascor-atlas-data/ner --out ~/renascor-atlas-data/ner/layer_v3 [--procs 16] [--part dedup|fix|both]

Inputs: the texts of the three new sets (tei = layer_tei/chunks, seen = layer_tei/seen/chunks, extra = layer_extra/extract/texts), the layer's texts
(layer_v1/extract/texts), the spans sent to the geocoder (layer_v3/geo_all/layer/spans.parquet).

Part 1, duplicates (``dedup_all/``). Same unit as ner2_layer_extra_dedup.py (8-word folded shingles, one in four) plus an exact hash of the folded word sequence.
  * new set against new set (tei, seen, extra): ``intra.tsv``, ``exact_new.tsv``
  * new set against the layer (67 842 documents): ``layer_matches.tsv``, ``layer_union.tsv``, ``exact_layer.tsv``
  * ``dup_flags.tsv``: one row per new document (set, words, sampled shingles, union containment in the layer, best containment in another new document, exact duplicates)
Part 2, corrections (``fix_all/``). ``form_pper.parquet`` (run v3, lang, key, n, p_per) and ``ctx.parquet`` (the council / Ile-de-France / Jordan cues of the mentions),
the same files as ner2_layer_fix_ctx.py gives for the v1 and es runs.
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner2_layer_extra_dedup as DD  # noqa: E402
import ner2_layer_fix as FX  # noqa: E402
import ner2_layer_fix_ctx as FC  # noqa: E402

G: dict = {}


def _t(m):
    print(f'[v3-check {time.strftime("%H:%M:%S")}] {m}', file=sys.stderr, flush=True)


def shingles_and_hash(text: str):
    """(sampled shingle hashes, exact hash of the folded word sequence, number of words)."""
    w = DD.words(text)
    n = len(w)
    ex = hashlib.sha1(' '.join(w).encode('utf-8')).hexdigest() if n else ''
    if n < DD.N:
        return np.zeros(0, dtype=np.uint64), ex, n
    tok = np.fromiter((zlib.crc32(x.encode('utf-8')) for x in w), dtype=np.uint64, count=n)
    m = n - DD.N + 1
    with np.errstate(over='ignore'):
        h = np.zeros(m, dtype=np.uint64)
        for k in range(DD.N):
            h += tok[k:k + m] * DD._M[k]
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return np.unique(h[(h & DD.KEEP_MASK) == 0]), ex, n


def work_new(job):
    path, keep = job
    out = []
    for did, t in DD.iter_jsonl_gz(path):
        if did in keep:
            h, ex, n = shingles_and_hash(t)
            out.append((did, h, ex, n))
    return out


def work_layer(path):
    idx: DD.CandIndex = G['idx']
    exset = G['exact']
    rows, mpos, exm = [], [], []
    for rid, t in DD.iter_jsonl_gz(path):
        h, ex, n = shingles_and_hash(t)
        c, m = idx.scan(h)
        if c:
            rows.append((rid, len(h), c))
            mpos.append(m)
        if ex and ex in exset and n >= 30:
            exm.append((rid, ex))
    return path, rows, (np.unique(np.concatenate(mpos)) if mpos else np.zeros(0, dtype=np.int64)), exm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--part', default='both', choices=['dedup', 'fix', 'both'])
    a = ap.parse_args()
    D = Path(a.data).expanduser()
    V3 = Path(a.out).expanduser()
    if a.part in ('fix', 'both'):
        part_fix(D, V3, a.procs)
    if a.part in ('dedup', 'both'):
        part_dedup(D, V3, a.procs)
    print('done')


def part_dedup(D: Path, V3: Path, procs: int):
    out = V3 / 'dedup_all'
    out.mkdir(parents=True, exist_ok=True)
    roots = {'tei': D / 'layer_tei/chunks', 'seen': D / 'layer_tei/seen/chunks', 'extra': D / 'layer_extra/extract/texts'}
    ids, setof = {}, {}
    for s in roots:
        d = pd.read_parquet(V3 / s / 'docs.parquet', columns=['doc_id'])
        ids[s] = set(d['doc_id'])
        for x in ids[s]:
            setof[x] = s
    _t({s: len(v) for s, v in ids.items()})
    names, hashes, exacts, nwords = [], [], {}, {}
    for s, root in roots.items():
        files = sorted(str(p) for p in root.glob('*/*.jsonl.gz'))
        with mp.get_context('fork').Pool(procs) as pool:
            for res in pool.imap_unordered(work_new, [(f, ids[s]) for f in files], chunksize=1):
                for did, h, ex, n in res:
                    if did in exacts:
                        continue
                    names.append(did)
                    hashes.append(h)
                    exacts[did] = ex
                    nwords[did] = n
        _t(f'{s}: {sum(1 for x in names if setof[x] == s)} documents read')
    missing = [x for x in setof if x not in exacts]
    _t(f'documents without text: {len(missing)}')
    (out / 'no_text.txt').write_text('\n'.join(missing), 'utf-8')
    idx = DD.CandIndex(names, hashes)
    _t(f'{len(names)} documents, {int(idx.nsh.sum())} sampled shingles, {len(idx.U)} distinct')
    by_hash = {}
    for d, ex in exacts.items():
        if ex and nwords[d] >= 30:
            by_hash.setdefault(ex, []).append(d)
    with open(out / 'exact_new.tsv', 'w', encoding='utf-8') as f:
        f.write('exact\tdoc_id\tset\twords\n')
        for ex, ds in by_hash.items():
            if len(ds) > 1:
                for d in ds:
                    f.write(f'{ex}\t{d}\t{setof[d]}\t{nwords[d]}\n')
    from collections import Counter
    from itertools import combinations
    multi = np.flatnonzero(np.diff(idx.start) > 1)
    pairs = Counter()
    for u in multi:
        ds = sorted(set(int(x) for x in idx.D[idx.start[u]:idx.start[u + 1]]))
        if len(ds) > 40:
            continue
        for x, y in combinations(ds, 2):
            pairs[(x, y)] += 1
    with open(out / 'intra.tsv', 'w', encoding='utf-8') as f:
        f.write('cand\tcand_set\tother\tother_set\tmatched\tcand_shingles\tother_shingles\tcontain_cand\n')
        for (x, y), n in sorted(pairs.items()):
            if n < 8:
                continue
            for p_, q_ in ((x, y), (y, x)):
                f.write(f'{idx.names[p_]}\t{setof[idx.names[p_]]}\t{idx.names[q_]}\t{setof[idx.names[q_]]}\t{n}\t{int(idx.nsh[p_])}\t{int(idx.nsh[q_])}\t{n / max(1, idx.nsh[p_]):.4f}\n')
    _t(f'{len(pairs)} pairs of new documents share shingles')
    G['idx'] = idx
    G['exact'] = {ex for ex in exacts.values() if ex}
    root = D / 'layer_v1/extract/texts'
    files = []
    for lang in ('en', 'la', 'it', 'es', 'fr', 'de'):
        files += sorted(str(p) for p in (root / lang).glob('*.jsonl.gz'))
    _t(f'{len(files)} layer shards')
    rows, mpos, exm = [], [], []
    with mp.get_context('fork').Pool(procs) as pool:
        for k, (p, r, m, e) in enumerate(pool.imap_unordered(work_layer, files, chunksize=1)):
            rows += r
            mpos.append(m)
            exm += e
            if k % 25 == 0:
                _t(f'{k + 1} / {len(files)} {Path(p).name} {len(rows)} matching layer documents')
    DD.write_matches(out / 'layer_matches.tsv', rows, idx)
    mp_all = np.concatenate(mpos) if mpos else np.zeros(0, dtype=np.int64)
    DD.write_union(out / 'layer_union.tsv', idx, mp_all)
    ex_new = {}
    for d, ex in exacts.items():
        ex_new.setdefault(ex, []).append(d)
    with open(out / 'exact_layer.tsv', 'w', encoding='utf-8') as f:
        f.write('layer_doc\tnew_doc\tset\n')
        for rid, ex in exm:
            for d in ex_new.get(ex, []):
                f.write(f'{rid}\t{d}\t{setof[d]}\n')
    u = idx.union_per_doc(mp_all)
    best = {}
    for (x, y), n in pairs.items():
        for p_, q_ in ((x, y), (y, x)):
            c = n / max(1, idx.nsh[p_])
            if c > best.get(p_, (0, None))[0]:
                best[p_] = (c, idx.names[q_])
    single = {}
    for rid, nref, c in rows:
        for ci, n in c.items():
            v = n / max(1, idx.nsh[ci])
            if v > single.get(ci, (0, None))[0]:
                single[ci] = (v, rid)
    ex_layer = {}
    for rid, ex in exm:
        for d in ex_new.get(ex, []):
            ex_layer.setdefault(d, rid)
    ex_in = {d: [x for x in ds if x != d] for ex, ds in by_hash.items() for d in ds if len(ds) > 1}
    recs = []
    for i, d in enumerate(idx.names):
        recs.append({'doc_id': d, 'set': setof[d], 'words': nwords[d], 'sampled_shingles': int(idx.nsh[i]),
                     'layer_union_contain': round(float(u[i] / max(1, idx.nsh[i])), 4), 'layer_best_contain': round(single.get(i, (0, None))[0], 4),
                     'layer_best_doc': single.get(i, (0, None))[1] or '',
                     'new_best_contain': round(best.get(i, (0, None))[0], 4), 'new_best_doc': best.get(i, (0, None))[1] or '',
                     'exact_layer_doc': ex_layer.get(d, ''), 'exact_new_docs': '|'.join(ex_in.get(d, []))})
    pd.DataFrame(recs).to_csv(out / 'dup_flags.tsv', sep='\t', index=False)
    _t('dedup done')


def part_fix(D: Path, V3: Path, procs: int):
    out = V3 / 'fix_all'
    out.mkdir(parents=True, exist_ok=True)
    s = FC.load_spans(V3 / 'geo_all/layer/spans.parquet', 0.0)
    _t(f'{len(s)} spans after cleaning')
    g = s.groupby(['lang', 'key']).agg(n=('p_per', 'size'), p_per=('p_per', 'mean')).reset_index()
    g.insert(0, 'run', 'v3')
    g.to_parquet(out / 'form_pper.parquet', index=False)
    tg = s[s['key'].map(FX.context_rule_of).notna()].copy()
    tg['rule'] = tg['key'].map(FX.context_rule_of)
    need = set(tg['doc_id'])
    _t(f'{len(tg)} spans of the context rules {tg["rule"].value_counts().to_dict()}; {len(need)} documents to read')
    shards = sorted(Path(p) for root in (D / 'layer_tei/chunks', D / 'layer_tei/seen/chunks', D / 'layer_extra/extract/texts') for p in root.glob('*/*.jsonl.gz'))
    texts = {}
    with mp.get_context('fork').Pool(procs) as pool:
        for r in pool.imap_unordered(FC.scan_texts, [(str(p), need) for p in shards]):
            for k, v in r.items():
                texts.setdefault(k, v)
    _t(f'{len(texts)} texts read')
    rows, bad = [], 0
    for r in tg.itertuples():
        t = texts.get(r.doc_id)
        if t is None:
            bad += 1
            rows.append(('v3', r.doc_id, r.lang, r.key, r.rule, 1, -1))
            continue
        cs, ce = int(r.char_start), int(r.char_end)
        ok = FX.apply_context_rule(r.rule, t[max(0, cs - 120):cs], t[ce:ce + 120])
        rows.append(('v3', r.doc_id, r.lang, r.key, r.rule, 1, int(ok)))
    _t(f'mentions without text: {bad}')
    c = pd.DataFrame(rows, columns=['run', 'doc_id', 'lang', 'key', 'rule', 'n_total', 'n_match'])
    c = c.groupby(['run', 'doc_id', 'lang', 'key', 'rule'], as_index=False).agg(n_total=('n_total', 'sum'), n_match=('n_match', lambda x: int((x == 1).sum())),
                                                                              n_notext=('n_match', lambda x: int((x == -1).sum())))
    c.to_parquet(out / 'ctx.parquet', index=False)
    _t(f'ctx rows {len(c)}')


if __name__ == '__main__':
    main()
