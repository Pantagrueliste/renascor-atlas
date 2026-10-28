"""NER layer, extra documents: duplicate control by 8-word shingles (same unit as the leak check of the gold set, ner2_leak_check.py).

A shingle is 8 consecutive words, folded (lower case, long s -> s, accents removed, punctuation dropped). Both sides keep only one shingle in four (the
shingle's mixed 64-bit hash has its two low bits at zero), so that the counts stay comparable and the cost falls by four. For a candidate document A and a
reference document R:

    containment(A in R) = (sampled shingles of A that occur in R) / (sampled shingles of A)

The union over all the reference documents of one reference set (the layer, the tagged editions) is kept too: a text that the layer holds cut into pieces has a
small containment in each piece and a large one in the union.

    python scripts/ner2_layer_extra_dedup.py cands   --texts DIR --out cands.npz
    python scripts/ner2_layer_extra_dedup.py layer   --cands cands.npz --layer LAYER_EXTRACT_TEXTS --out layer_matches --procs 22
    python scripts/ner2_layer_extra_dedup.py tagged  --cands cands.npz --mentions mentions.parquet --raw RQ/data/raw --out tagged_matches --procs 2
    python scripts/ner2_layer_extra_dedup.py intra   --cands cands.npz --out intra_matches
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import re
import sys
import time
import unicodedata
import zipfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner2_layer_extra_readers import decode_any, ltag, parse_xml_loose  # noqa: E402

N = 8
KEEP_MASK = np.uint64(3)           # one shingle in four
_M = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x27D4EB2F165667C5,
               0x94D049BB133111EB], dtype=np.uint64)
_COMBINING = re.compile('[̀-ͯ]')
_WORD = re.compile(r'\w+', re.U)


def words(text: str) -> list[str]:
    t = text.lower().replace('ſ', 's')
    t = _COMBINING.sub('', unicodedata.normalize('NFD', t))
    return _WORD.findall(t)


def shingle_hashes(text: str) -> np.ndarray:
    """Unique sampled shingle hashes (uint64) of a text."""
    w = words(text)
    n = len(w)
    if n < N:
        return np.zeros(0, dtype=np.uint64)
    tok = np.fromiter((zlib.crc32(x.encode('utf-8')) for x in w), dtype=np.uint64, count=n)
    m = n - N + 1
    with np.errstate(over='ignore'):
        h = np.zeros(m, dtype=np.uint64)
        for k in range(N):
            h += tok[k:k + m] * _M[k]
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return np.unique(h[(h & KEEP_MASK) == 0])


class CandIndex:
    """Candidate shingles: CSR layout, hash -> documents that have it."""

    def __init__(self, names: list[str], hashes: list[np.ndarray]):
        self.names = list(names)
        self.nsh = np.array([len(h) for h in hashes], dtype=np.int64)
        if hashes and self.nsh.sum():
            H = np.concatenate(hashes)
            D = np.repeat(np.arange(len(hashes), dtype=np.int32), self.nsh)
            o = np.argsort(H, kind='stable')
            H, D = H[o], D[o]
            self.U, first = np.unique(H, return_index=True)
            self.start = np.r_[first, len(H)].astype(np.int64)
            self.D = D
        else:
            self.U = np.zeros(0, dtype=np.uint64)
            self.start = np.zeros(1, dtype=np.int64)
            self.D = np.zeros(0, dtype=np.int32)

    def scan(self, ref_hashes: np.ndarray):
        """-> (counts per candidate document (dict idx -> n), array of matched positions in U)."""
        if len(self.U) == 0 or len(ref_hashes) == 0:
            return {}, np.zeros(0, dtype=np.int64)
        pos = np.searchsorted(self.U, ref_hashes)
        pos[pos == len(self.U)] = 0
        hit = self.U[pos] == ref_hashes
        mp_ = pos[hit]
        if len(mp_) == 0:
            return {}, mp_
        lens = self.start[mp_ + 1] - self.start[mp_]
        tot = int(lens.sum())
        idx = np.repeat(self.start[mp_], lens) + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
        cnt = np.bincount(self.D[idx], minlength=len(self.names))
        nz = np.flatnonzero(cnt)
        return {int(i): int(cnt[i]) for i in nz}, mp_

    def union_per_doc(self, matched_positions: np.ndarray) -> np.ndarray:
        """Number of sampled shingles of each candidate found in the union of the references (positions in U matched at least once)."""
        if len(matched_positions) == 0:
            return np.zeros(len(self.names), dtype=np.int64)
        mpos = np.unique(matched_positions)
        lens = self.start[mpos + 1] - self.start[mpos]
        tot = int(lens.sum())
        idx = np.repeat(self.start[mpos], lens) + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
        return np.bincount(self.D[idx], minlength=len(self.names)).astype(np.int64)

    def save(self, path):
        np.savez_compressed(path, names=np.array(self.names, dtype=object), nsh=self.nsh, U=self.U, start=self.start, D=self.D)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        o = cls.__new__(cls)
        o.names, o.nsh, o.U, o.start, o.D = list(z['names']), z['nsh'], z['U'], z['start'], z['D']
        return o


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


G: dict = {}


# ---------------------------------------------------------------------------------------------------------------------
def iter_jsonl_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            yield d['id'], d['t']


def work_layer(path):
    """Scan one shard of the layer's texts. -> (path, [(ref_id, n_ref_shingles, {cand: n})], matched positions)."""
    idx: CandIndex = G['idx']
    out, mpos = [], []
    for rid, t in iter_jsonl_gz(path):
        h = shingle_hashes(t)
        c, m = idx.scan(h)
        if c:
            out.append((rid, len(h), c))
            mpos.append(m)
    return path, out, (np.unique(np.concatenate(mpos)) if mpos else np.zeros(0, dtype=np.int64))


def tei_plain_text(data: bytes) -> str:
    """Body text of a TEI file for the shingles (notes and header left out; no diplomatic care: only words matter)."""
    root = parse_xml_loose(data)
    body = None
    for el in root.iter():
        if ltag(el) == 'body':
            body = el
            break
    if body is None:
        body = root
    drop = {'note', 'fw', 'pb', 'figure', 'teiHeader'}
    buf = []

    def walk(e):
        t = ltag(e)
        if t in drop:
            return
        if e.text:
            buf.append(e.text)
        for c in e:
            walk(c)
            if c.tail:
                buf.append(c.tail)
        if t in ('p', 'l', 'div', 'head', 'item', 'sp', 'lg', 'lb'):
            buf.append('\n')

    walk(body)
    return ''.join(buf)


def text_of_bytes(data: bytes) -> str:
    """XML / TEI -> body text; anything else (treebank .psd, plain text) -> decoded text."""
    if data.lstrip()[:1] == b'<':
        return tei_plain_text(data)
    return decode_any(data)


def scan_text(idx: CandIndex, text: str):
    h = shingle_hashes(text)
    c, m = idx.scan(h)
    return len(h), c, m


def work_tagged(job):
    """job: ('file', rid, path) or ('container', path, [(rid, member)]). -> list of (rid, n_ref_shingles, {cand: n}, matched positions, error)."""
    idx: CandIndex = G['idx']
    out = []
    try:
        if job[0] == 'file':
            _, rid, path = job
            try:
                n, c, m = scan_text(idx, text_of_bytes(Path(path).read_bytes()))
                out.append((rid, n, c, m, ''))
            except Exception as e:                  # unreadable reference: reported, never fatal
                out.append((rid, -1, {}, np.zeros(0, dtype=np.int64), repr(e)[:80]))
            return out
        _, path, members = job
        want = {mem.lstrip('./'): rid for rid, mem in members}
        done = set()
        if path.endswith(('.tar.gz', '.tgz')):
            import tarfile
            with tarfile.open(path, 'r|gz') as tf:
                for t in tf:
                    nm = t.name.lstrip('./')
                    if t.isfile() and nm in want:
                        rid = want[nm]
                        done.add(nm)
                        try:
                            n, c, m = scan_text(idx, text_of_bytes(tf.extractfile(t).read()))
                            out.append((rid, n, c, m, ''))
                        except Exception as e:
                            out.append((rid, -1, {}, np.zeros(0, dtype=np.int64), repr(e)[:80]))
        else:
            with zipfile.ZipFile(path) as z:
                for nm, rid in want.items():
                    try:
                        n, c, m = scan_text(idx, text_of_bytes(z.read(nm)))
                        out.append((rid, n, c, m, ''))
                    except Exception as e:
                        out.append((rid, -1, {}, np.zeros(0, dtype=np.int64), repr(e)[:80]))
                    done.add(nm)
        for nm, rid in want.items():
            if nm not in done:
                out.append((rid, -1, {}, np.zeros(0, dtype=np.int64), 'member not found'))
    except Exception as e:
        for rid, _ in (job[2] if job[0] == 'container' else []):
            out.append((rid, -1, {}, np.zeros(0, dtype=np.int64), repr(e)[:80]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
def cmd_cands(a):
    texts = Path(a.texts)
    names, hashes = [], []
    for p in sorted(texts.rglob('*.jsonl.gz')):
        for did, t in iter_jsonl_gz(p):
            names.append(did)
            hashes.append(shingle_hashes(t))
    idx = CandIndex(names, hashes)
    idx.save(a.out)
    log(len(names), 'candidate documents,', int(idx.nsh.sum()), 'sampled shingles,', len(idx.U), 'distinct')


def write_matches(path, rows, idx: CandIndex):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('cand\tref\tmatched\tref_shingles\tcand_shingles\tcontain_cand\tcontain_ref\n')
        for rid, nref, c in rows:
            for ci, n in c.items():
                f.write(f'{idx.names[ci]}\t{rid}\t{n}\t{nref}\t{int(idx.nsh[ci])}\t{n / max(1, idx.nsh[ci]):.4f}\t{n / max(1, nref):.4f}\n')


def write_union(path, idx: CandIndex, mpos: np.ndarray):
    u = idx.union_per_doc(mpos)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('cand\tcand_shingles\tunion_matched\tunion_contain\n')
        for i, nm in enumerate(idx.names):
            f.write(f'{nm}\t{int(idx.nsh[i])}\t{int(u[i])}\t{u[i] / max(1, idx.nsh[i]):.4f}\n')


def cmd_layer(a):
    idx = CandIndex.load(a.cands)
    G['idx'] = idx
    root = Path(a.layer)
    files = []
    for lang in ('en', 'la', 'it', 'es', 'fr', 'de'):            # not en_ranked: the same English documents a second time
        files += sorted(str(p) for p in (root / lang).glob('*.jsonl.gz'))
    log(len(files), 'layer shards')
    rows, mpos = [], []
    with mp.get_context('fork').Pool(a.procs) as pool:
        for k, (p, out, m) in enumerate(pool.imap_unordered(work_layer, files, chunksize=1)):
            rows += out
            mpos.append(m)
            if k % 20 == 0:
                log(k + 1, '/', len(files), Path(p).name, len(rows), 'matching reference docs so far')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_matches(Path(a.out) / 'matches.tsv', rows, idx)
    write_union(Path(a.out) / 'union.tsv', idx, np.concatenate(mpos) if mpos else np.zeros(0, dtype=np.int64))
    log('done')


def tagged_jobs(mentions_path, raw_root, kind='', zones=()):
    import pandas as pd
    m = pd.read_parquet(mentions_path, columns=['coll_slug', 'source_path', 'kind', 'zone'])
    if kind:
        m = m[m['kind'] == kind]
    if zones:
        m = m[m['zone'].isin(list(zones))]
    m = m.drop_duplicates(['coll_slug', 'source_path'])
    jobs, groups = [], {}
    for c, p in zip(m['coll_slug'], m['source_path']):
        if '!' in p:
            z, mem = p.split('!', 1)
            groups.setdefault(str(Path(raw_root) / z), []).append((f'{c}:{p}', mem))
        else:
            jobs.append(('file', f'{c}:{p}', str(Path(raw_root) / p)))
    jobs += [('container', z, mems) for z, mems in groups.items()]
    return jobs


def cmd_tagged(a):
    idx = CandIndex.load(a.cands)
    G['idx'] = idx
    jobs = tagged_jobs(a.mentions, a.raw, a.kind, a.zones)
    log(len(jobs), 'jobs (files and containers)')
    rows, mpos, errs = [], [], []
    k = 0
    with mp.get_context('fork').Pool(a.procs) as pool:
        for res in pool.imap_unordered(work_tagged, jobs, chunksize=8):
            for rid, nref, c, m, err in res:
                k += 1
                if err:
                    errs.append((rid, err))
                if c:
                    rows.append((rid, nref, c))
                    mpos.append(m)
            if k % 5000 < len(res):
                log(k, 'reference files,', len(rows), 'matching')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_matches(Path(a.out) / 'matches.tsv', rows, idx)
    write_union(Path(a.out) / 'union.tsv', idx, np.concatenate(mpos) if mpos else np.zeros(0, dtype=np.int64))
    (Path(a.out) / 'errors.tsv').write_text(''.join(f'{r}\t{e}\n' for r, e in errs), encoding='utf-8')
    log('done', k, 'reference files,', len(errs), 'unreadable')


def cmd_intra(a):
    """Candidate against candidate: the shingles held by two or more candidates, counted by pair."""
    from collections import Counter
    from itertools import combinations
    idx = CandIndex.load(a.cands)
    multi = np.flatnonzero(np.diff(idx.start) > 1)
    pairs = Counter()
    for u in multi:
        ds = sorted(set(int(d) for d in idx.D[idx.start[u]:idx.start[u + 1]]))
        for x, y in combinations(ds, 2):
            pairs[(x, y)] += 1
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / 'matches.tsv', 'w', encoding='utf-8') as f:
        f.write('cand\tother\tmatched\tcand_shingles\tother_shingles\tcontain_cand\n')
        for (x, y), n in sorted(pairs.items()):
            for p_, q_ in ((x, y), (y, x)):
                f.write(f'{idx.names[p_]}\t{idx.names[q_]}\t{n}\t{int(idx.nsh[p_])}\t{int(idx.nsh[q_])}\t{n / max(1, idx.nsh[p_]):.4f}\n')
    log('done', len(pairs), 'pairs of candidates that share shingles')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cands')
    c.add_argument('--texts', required=True)
    c.add_argument('--out', required=True)
    for nm in ('layer', 'tagged', 'intra'):
        s = sub.add_parser(nm)
        s.add_argument('--cands', required=True)
        s.add_argument('--out', required=True)
        s.add_argument('--procs', type=int, default=4)
        if nm == 'layer':
            s.add_argument('--layer', required=True, help='directory with en/ la/ it/ es/ fr/ de/ shards (layer_v1/extract/texts)')
        if nm == 'tagged':
            s.add_argument('--mentions', required=True)
            s.add_argument('--raw', required=True)
            s.add_argument('--zones', nargs='*', default=[], help='zones of the mentions to keep (text footnote apparatus editorial_commentary = the text itself)')
            s.add_argument('--kind', default='', help="'place': only the files in which the editors tagged places (the references that can double count the model's places)")
    a = ap.parse_args()
    {'cands': cmd_cands, 'layer': cmd_layer, 'tagged': cmd_tagged, 'intra': cmd_intra}[a.cmd](a)


if __name__ == '__main__':
    main()
