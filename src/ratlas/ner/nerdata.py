"""Loading, subword alignment and windowing of NER records for fine-tuning and evaluation (numpy only; the tokenizer is passed in).

Record fields used: ``tokens`` (words), ``ner_tags_loc`` / ``ner_tags_per`` (0 O, 1 B, 2 I), ``loc_valid`` / ``per_valid``, ``id``, ``lang``,
``edition`` or ``source`` + ``subset``.

Alignment: the words are joined with one space and tokenised in one go with character offsets, so that the tokenizer sees natural text; each word is
labelled on its first subword only. Long passages are cut at word boundaries into windows of at most ``max_len`` subwords including the special tokens
(training: no overlap; evaluation: overlap, each word is predicted by the window in which it is farthest from the window edge).
"""
from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

IGNORE = -100


# ---------------------------------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------------------------------
def open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else open(path, encoding='utf-8')


def read_jsonl(path, limit: Optional[int] = None) -> list[dict]:
    out = []
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


SLIM = ('id', 'tokens', 'ner_tags_loc', 'ner_tags_per', 'loc_valid', 'per_valid', 'lang', 'edition', 'source', 'subset', 'text', 'token_starts', 'licence_flag')


def read_slim(path, limit: Optional[int] = None, keep_text: bool = False) -> list[dict]:
    """Read a JSONL and keep only the fields needed for NER (the text and offsets only when asked: they are large)."""
    keep = set(SLIM) if keep_text else set(SLIM) - {'text', 'token_starts'}
    out = []
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append({k: d[k] for k in keep if k in d})
            if limit and len(out) >= limit:
                break
    return out


def sample_records(recs: list[dict], max_tokens: Optional[int], seed: int, max_share_source: Optional[float] = None) -> list[dict]:
    """Random subset (without replacement) with at most ``max_tokens`` words; all records when the cap is None or large enough."""
    if not max_tokens:
        return recs
    total = sum(len(r['tokens']) for r in recs)
    if total <= max_tokens:
        return recs
    idx = list(range(len(recs)))
    random.Random(seed).shuffle(idx)
    out, n = [], 0
    for i in idx:
        out.append(recs[i])
        n += len(recs[i]['tokens'])
        if n >= max_tokens:
            break
    out.sort(key=lambda r: r['id'])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------------------------------------------------
def word_of_offsets(starts: list[int], ends: list[int], offsets: list[tuple[int, int]]) -> list[int]:
    """Word index of each subword from its character offsets (-1 for a zero-width or out-of-range subword). A subword whose offset starts on the space
    before a word (sentencepiece '▁' pieces) belongs to the following word."""
    out = []
    for s, e in offsets:
        if e <= s:
            out.append(-1)
            continue
        i = bisect.bisect_right(starts, s) - 1
        if i < 0 or s >= ends[i]:
            i += 1
        out.append(i if 0 <= i < len(starts) and starts[i] < e else -1)
    return out


def special_tokens(tok) -> tuple[list[int], list[int]]:
    """(prefix ids, suffix ids) that ``tok`` adds around a sequence."""
    a = tok('hello', add_special_tokens=False)['input_ids']
    b = tok('hello', add_special_tokens=True)['input_ids']
    for i in range(len(b) - len(a) + 1):
        if b[i:i + len(a)] == a:
            return list(b[:i]), list(b[i + len(a):])
    return list(b[:1]), []


class Encoded:
    """Tokenised records as flat arrays. ``ids[sub_off[i]:sub_off[i+1]]`` are the subwords of record i, ``first[...]`` the word index of the subword when it is
    the first subword of its word (else -1), word-level labels ``lab_loc / lab_per[word_off[i]:word_off[i+1]]``."""

    def __init__(self, ids, first, sub_off, lab_loc, lab_per, word_off, valid_loc, valid_per, meta, n_unaligned):
        self.ids, self.first, self.sub_off = ids, first, sub_off
        self.lab_loc, self.lab_per, self.word_off = lab_loc, lab_per, word_off
        self.valid_loc, self.valid_per = valid_loc, valid_per
        self.meta = meta
        self.n_unaligned = n_unaligned

    def __len__(self):
        return len(self.meta)

    def wstart(self, i: int) -> np.ndarray:
        """Subword start of each word of record i (length n_words + 1, non-decreasing; a word without subword gets the start of the next word)."""
        s, e = int(self.sub_off[i]), int(self.sub_off[i + 1])
        nw = int(self.word_off[i + 1] - self.word_off[i])
        n_sub = e - s
        pos = np.full(nw + 1, n_sub + 1, dtype=np.int64)
        f = self.first[s:e]
        k = np.nonzero(f >= 0)[0]
        pos[f[k]] = k
        pos[nw] = n_sub
        return np.minimum.accumulate(pos[::-1])[::-1]


def encode_records(tok, recs: list[dict], tag: str = '') -> Encoded:
    """Tokenise all records (fast tokenizer with offsets, in batches)."""
    ids_l, first_l, sub_off = [], [], [0]
    loc_l, per_l, word_off = [], [], [0]
    vloc, vper, meta = [], [], []
    n_un = 0
    B = 512
    for b in range(0, len(recs), B):
        batch = recs[b:b + B]
        texts, spans = [], []
        for r in batch:
            ws = r['tokens']
            st, en, p = [], [], 0
            for w in ws:
                st.append(p)
                p += len(w)
                en.append(p)
                p += 1
            texts.append(' '.join(ws))
            spans.append((st, en))
        enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        for r, (st, en), ids, offs in zip(batch, spans, enc['input_ids'], enc['offset_mapping']):
            nw = len(r['tokens'])
            wo = word_of_offsets(st, en, offs)
            first = np.full(len(ids), -1, dtype=np.int32)
            seen = set()
            for j, w in enumerate(wo):
                if w >= 0 and w not in seen:
                    seen.add(w)
                    first[j] = w
            n_un += nw - len(seen)
            ids_l.append(np.asarray(ids, dtype=np.int32))
            first_l.append(first)
            sub_off.append(sub_off[-1] + len(ids))
            lo = np.asarray(r.get('ner_tags_loc') or [0] * nw, dtype=np.int8)
            pe = np.asarray(r.get('ner_tags_per') or [0] * nw, dtype=np.int8)
            loc_l.append(lo)
            per_l.append(pe)
            word_off.append(word_off[-1] + nw)
            vloc.append(bool(r.get('loc_valid', True)))
            vper.append(bool(r.get('per_valid', True)))
            meta.append({k: r.get(k) for k in ('id', 'lang', 'edition', 'source', 'subset')})
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
    return Encoded(cat(ids_l, np.int32), cat(first_l, np.int32), np.asarray(sub_off, dtype=np.int64), cat(loc_l, np.int8), cat(per_l, np.int8),
                   np.asarray(word_off, dtype=np.int64), np.asarray(vloc), np.asarray(vper), meta, n_un)


# ---------------------------------------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------------------------------------
def make_windows(wstart: np.ndarray, limit: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Word ranges [a, b) with at most ``limit`` subwords each (one word at least). ``overlap`` subwords are shared by consecutive windows."""
    n = len(wstart) - 1
    ws = wstart.tolist()
    wins, a = [], 0
    while a < n:
        b = bisect.bisect_right(ws, ws[a] + limit, a + 1, n + 1) - 1
        b = min(max(b, a + 1), n)
        wins.append((a, b))
        if b >= n:
            break
        if overlap <= 0:
            a = b
        else:
            na = bisect.bisect_left(ws, ws[a] + limit - overlap, a + 1, b + 1)
            a = min(max(na, a + 1), b)
    return wins


def build_windows(enc: Encoded, limit: int, overlap: int = 0, only_valid: bool = False) -> np.ndarray:
    """Array of rows (record, word_a, word_b, sub_a, sub_b) for every window of every record."""
    rows = []
    for i in range(len(enc)):
        if only_valid and not (enc.valid_loc[i] or enc.valid_per[i]):
            continue
        ws = enc.wstart(i)
        for a, b in make_windows(ws, limit, overlap):
            sa, sb = int(ws[a]), int(ws[b])
            rows.append((i, a, b, sa, min(sb, sa + limit)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 5)


def batch_windows(enc: Encoded, rows: np.ndarray, prefix: list[int], suffix: list[int], pad_id: int, with_labels: bool):
    """Padded numpy batch for a set of window rows: input_ids, attention_mask, (lab_loc, lab_per), and for evaluation the (row, position) of each first subword."""
    L = max(int(r[4] - r[3]) for r in rows) + len(prefix) + len(suffix)
    n = len(rows)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    mask = np.zeros((n, L), dtype=np.int64)
    lab_loc = np.full((n, L), IGNORE, dtype=np.int64) if with_labels else None
    lab_per = np.full((n, L), IGNORE, dtype=np.int64) if with_labels else None
    maps = []
    for k, (i, a, b, sa, sb) in enumerate(rows):
        base = int(enc.sub_off[i])
        seg = enc.ids[base + sa: base + sb]
        m = sb - sa
        ids[k, :len(prefix)] = prefix
        ids[k, len(prefix):len(prefix) + m] = seg
        ids[k, len(prefix) + m:len(prefix) + m + len(suffix)] = suffix
        mask[k, :len(prefix) + m + len(suffix)] = 1
        fw = enc.first[base + sa: base + sb]
        pos = np.nonzero(fw >= 0)[0]
        words = fw[pos]                       # absolute word index within the record
        if with_labels:
            wo = int(enc.word_off[i])
            if enc.valid_loc[i]:
                lab_loc[k, len(prefix) + pos] = enc.lab_loc[wo + words]
            if enc.valid_per[i]:
                lab_per[k, len(prefix) + pos] = enc.lab_per[wo + words]
        maps.append((len(prefix) + pos, words))
    out = {'input_ids': ids, 'attention_mask': mask}
    if with_labels:
        out['lab_loc'], out['lab_per'] = lab_loc, lab_per
    out['maps'] = maps
    return out


def length_batches(rows: np.ndarray, max_tokens: int, max_seqs: int, rng: Optional[np.random.Generator], bucket: int = 64) -> list[np.ndarray]:
    """Batches of window indices of similar length. With ``rng``: shuffled (training), without: sorted by length (evaluation)."""
    n = len(rows)
    lens = (rows[:, 4] - rows[:, 3]) if n else np.zeros(0)
    if rng is not None:
        order = rng.permutation(n)
        chunks = [order[i:i + bucket * max_seqs] for i in range(0, n, bucket * max_seqs)]
        order = np.concatenate([c[np.argsort(lens[c], kind='stable')] for c in chunks]) if chunks else order
    else:
        order = np.argsort(lens, kind='stable')
    batches, cur, cur_max = [], [], 0
    for idx in order:
        l = int(lens[idx]) + 4
        m = max(cur_max, l)
        if cur and (m * (len(cur) + 1) > max_tokens or len(cur) >= max_seqs):
            batches.append(np.asarray(cur))
            cur, m = [], l
        cur.append(idx)
        cur_max = m
    if cur:
        batches.append(np.asarray(cur))
    if rng is not None:
        rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------------------------------------------------
# cache of tokenised sets
# ---------------------------------------------------------------------------------------------------------------------
def vocab_hash(tok) -> str:
    """Identity of a tokenizer: hash of its vocabulary and special-token wrapping (a saved copy has the same hash as the original)."""
    h = hashlib.sha1()
    h.update(json.dumps(sorted(tok.get_vocab().items())).encode())
    h.update(json.dumps(special_tokens(tok)).encode())
    return h.hexdigest()[:16]


def cache_key(tok_name: str, vocab: int, files: Iterable[str], extra: str = '') -> str:
    h = hashlib.sha1()
    h.update(f'{tok_name}|{vocab}|{extra}'.encode())
    for f in files:
        st = os.stat(f)
        h.update(f'{f}|{st.st_size}|{int(st.st_mtime)}'.encode())
    return h.hexdigest()[:20]


def save_encoded(enc: Encoded, path: Path) -> None:
    tmp = Path(str(path) + f'.tmp{os.getpid()}')
    with open(tmp, 'wb') as f:
        np.savez(f, ids=enc.ids, first=enc.first, sub_off=enc.sub_off, lab_loc=enc.lab_loc, lab_per=enc.lab_per, word_off=enc.word_off,
                 valid_loc=enc.valid_loc, valid_per=enc.valid_per, meta=np.asarray([json.dumps(m) for m in enc.meta], dtype=object),
                 n_unaligned=np.asarray(enc.n_unaligned))
    os.replace(tmp, path)


def load_encoded(path: Path) -> Encoded:
    z = np.load(path, allow_pickle=True)
    return Encoded(z['ids'], z['first'], z['sub_off'], z['lab_loc'], z['lab_per'], z['word_off'], z['valid_loc'], z['valid_per'],
                   [json.loads(m) for m in z['meta']], int(z['n_unaligned']))
