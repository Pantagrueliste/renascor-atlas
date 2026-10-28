"""Batch inference of the NER v1 encoders on long untagged documents, and span decoding (the "found by the model" layer).

The model, its windows and its decoding are the ones of ``scripts/ner_audit_predict.py`` (the independent v1 inference), rewritten with numpy so that it
keeps up with a GPU on billions of words:

* words are ``\\w+|[^\\w\\s]`` tokens of the document text; a *passage* is a window of ``size`` words (200) that starts every ``stride`` words (180), so that
  consecutive passages share ``size - stride`` words; a passage *owns* the words [start + overlap/2, end - overlap/2) (the first passage owns its start, the
  last one its end) and a span belongs to the passage that owns its first word;
* a passage is fed as its words joined by one space; subword windows of ``max_len`` tokens (with the special tokens of the tokenizer) overlap by 64 tokens; a
  word takes the prediction of its first subword in the window where that subword is farthest from the window edge (an edge that is the start or the end of
  the passage does not count);
* strict IOB2 decoding of the arg-max of the 3-way softmax of the LOC head: a span starts at B and goes on with I; a stray I is ignored; the confidence of a
  span is the mean over its words of the probability of the predicted class; ``p_per`` is the mean over its words of 1 - P(O) of the PER head.

Everything outside the model call is numpy (no per-token python loops), so that the CPU side can run in DataLoader workers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

WORD = re.compile(r'\w+|[^\w\s]', re.U)
BIG = 10 ** 6


# ---------------------------------------------------------------------------------------------------------------------
# passages
# ---------------------------------------------------------------------------------------------------------------------
def passage_plan(n_words: int, size: int = 200, stride: int = 180):
    """Passages of a document of ``n_words`` words: arrays (w0, w1, own0, own1), all in document word indices."""
    if n_words <= 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    if n_words <= size:
        n = 1
    else:
        n = (n_words - size + stride - 1) // stride + 1
    w0 = np.arange(n, dtype=np.int64) * stride
    w1 = np.minimum(w0 + size, n_words)
    half = (size - stride) // 2
    own0 = np.where(np.arange(n) == 0, 0, w0 + half)
    own1 = np.where(np.arange(n) == n - 1, n_words, w1 - (size - stride - half))
    return w0, w1, own0, own1


# ---------------------------------------------------------------------------------------------------------------------
# tokenisation, windows (CPU side)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class TokInfo:
    tok: object
    prefix: list
    suffix: list
    pad: int
    max_len: int = 256
    overlap: int = 64

    @property
    def L(self) -> int:
        return self.max_len - len(self.prefix) - len(self.suffix)


def tok_info(tok, max_len: int = 256, overlap: int = 64, force_bos: bool = False) -> TokInfo:
    """Special tokens the tokenizer puts around a sequence (a tokenizer rebuilt from tokenizer.json has them only in its post-processor)."""
    bare = tok('probe', add_special_tokens=False)['input_ids']
    full = tok('probe', add_special_tokens=True)['input_ids']
    k = next(i for i in range(len(full) - len(bare) + 1) if full[i:i + len(bare)] == bare)
    prefix, suffix = list(full[:k]), list(full[k + len(bare):])
    if force_bos and tok.bos_token_id is not None and (not prefix or prefix[0] != tok.bos_token_id):
        prefix = [int(tok.bos_token_id)] + prefix
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    return TokInfo(tok, prefix, suffix, int(pad), max_len, overlap)


def token_windows(n: int, L: int, step: int):
    """The windows of the v1 inference over a sequence of n tokens: [(s, e)]."""
    wins = []
    s = 0
    while True:
        e = min(n, s + L)
        wins.append((s, e))
        if e >= n:
            break
        s += step
        if s + L > n:
            s = max(0, n - L)
            wins.append((s, n))
            break
    return wins


def first_subwords(offsets: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the first subword of each word (-1 if a word has none). ``offsets`` (n, 2) are the character offsets of the subwords in the passage text
    (words joined by one space), ``starts`` / ``ends`` those of the words."""
    nw = len(starts)
    out = np.full(nw, -1, dtype=np.int64)
    if len(offsets) == 0 or nw == 0:
        return out
    ts, te = offsets[:, 0], offsets[:, 1]
    i = np.searchsorted(starts, ts, side='right') - 1
    ic = np.clip(i, 0, nw - 1)
    bad = (i < 0) | (ts >= ends[ic])
    i = np.where(bad, i + 1, i)
    ic = np.clip(i, 0, nw - 1)
    valid = (te > ts) & (i >= 0) & (i < nw) & (starts[ic] < te)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    w = i[idx]
    uniq, pos = np.unique(w, return_index=True)       # np.unique returns the first occurrence
    out[uniq] = idx[pos]
    return out


def prepare_pack(ti: TokInfo, doc_idx: Sequence[int], texts: Sequence[str], plan: Sequence[Optional[np.ndarray]], size: int = 200, stride: int = 180,
                 batch_tokens: int = 32000, max_seqs: int = 256) -> dict:
    """One pack = some documents (or the selected passages of some documents) -> everything the GPU loop and the decoder need.

    ``plan[i]`` is None (all passages of document i) or an int array of passage indices. Returns a dict of numpy arrays:
      pass_doc, pass_idx, pass_w0, pass_own0, pass_own1, pass_off (n_pass + 1; offsets into the pack word arrays), cs, ce (document character offsets of the words)
      batches: list of dict(ids (B, T) int32, lens (B,), gword (m,) int32, flat (m,) int64 (row * T + position), margin (m,) int32)
    """
    pd_, pi_, pw0, po0, po1, noff = [], [], [], [], [], [0]
    cs_l, ce_l = [], []
    words_all: list[list[str]] = []
    for di, text, pl in zip(doc_idx, texts, plan):
        ms = list(WORD.finditer(text))
        n = len(ms)
        if n == 0:
            continue
        words = [m.group() for m in ms]
        cs = np.fromiter((m.start() for m in ms), dtype=np.int64, count=n)
        ce = np.fromiter((m.end() for m in ms), dtype=np.int64, count=n)
        w0, w1, o0, o1 = passage_plan(n, size, stride)
        sel = np.arange(len(w0)) if pl is None else np.asarray(pl, dtype=np.int64)
        for p in sel:
            a, b = int(w0[p]), int(w1[p])
            words_all.append(words[a:b])
            cs_l.append(cs[a:b])
            ce_l.append(ce[a:b])
            pd_.append(di)
            pi_.append(int(p))
            pw0.append(a)
            po0.append(int(o0[p]))
            po1.append(int(o1[p]))
            noff.append(noff[-1] + (b - a))
    out = {'pass_doc': np.asarray(pd_, dtype=np.int64), 'pass_idx': np.asarray(pi_, dtype=np.int64), 'pass_w0': np.asarray(pw0, dtype=np.int64),
           'pass_own0': np.asarray(po0, dtype=np.int64), 'pass_own1': np.asarray(po1, dtype=np.int64), 'pass_off': np.asarray(noff, dtype=np.int64),
           'cs': np.concatenate(cs_l) if cs_l else np.zeros(0, dtype=np.int64), 'ce': np.concatenate(ce_l) if ce_l else np.zeros(0, dtype=np.int64), 'batches': []}
    if not words_all:
        return out
    joined = [' '.join(w) for w in words_all]
    enc = ti.tok(joined, add_special_tokens=False, return_offsets_mapping=True)
    ids_all, off_all = enc['input_ids'], enc['offset_mapping']
    L = ti.L
    step = max(1, L - ti.overlap)
    pre, suf = np.asarray(ti.prefix, dtype=np.int32), np.asarray(ti.suffix, dtype=np.int32)
    win_ids, win_len, win_g, win_pos, win_mar = [], [], [], [], []
    for pi, (words, ids, offs) in enumerate(zip(words_all, ids_all, off_all)):
        nw = len(words)
        n = len(ids)
        if n == 0:
            continue
        lens = np.fromiter(map(len, words), dtype=np.int64, count=nw)
        starts = np.cumsum(lens + 1) - (lens + 1)
        ends = starts + lens
        off = np.asarray(offs, dtype=np.int64).reshape(-1, 2)
        ft = first_subwords(off, starts, ends)
        has = np.flatnonzero(ft >= 0)
        if len(has) == 0:
            continue
        fw = ft[has]
        ida = np.asarray(ids, dtype=np.int32)
        base = noff[pi]
        for s, e in token_windows(n, L, step):
            m = (fw >= s) & (fw < e)
            if not m.any():
                continue
            f = fw[m]
            left = (f - s) if s > 0 else np.full(len(f), BIG)
            right = (e - 1 - f) if e < n else np.full(len(f), BIG)
            win_ids.append(np.concatenate((pre, ida[s:e], suf)))
            win_g.append(base + has[m])
            win_pos.append(len(pre) + f - s)
            win_mar.append(np.minimum(left, right))
    if not win_ids:
        return out
    wl = np.fromiter((len(x) for x in win_ids), dtype=np.int64, count=len(win_ids))
    order = np.argsort(wl, kind='stable')
    i = 0
    while i < len(order):
        j, mx = i, 0
        while j < len(order):
            ln = int(wl[order[j]])
            if j > i and (max(mx, ln) * (j - i + 1) > batch_tokens or (j - i) >= max_seqs):
                break
            mx = max(mx, ln)
            j += 1
        sel = order[i:j]
        B = len(sel)
        ids_b = np.full((B, mx), ti.pad, dtype=np.int32)
        for r, w in enumerate(sel):
            ids_b[r, :wl[w]] = win_ids[w]
        gword = np.concatenate([win_g[w] for w in sel]).astype(np.int32)
        flat = np.concatenate([r * mx + win_pos[w] for r, w in enumerate(sel)]).astype(np.int64)
        margin = np.concatenate([win_mar[w] for w in sel]).astype(np.int32)
        out['batches'].append({'ids': ids_b, 'lens': wl[sel].astype(np.int32), 'gword': gword, 'flat': flat, 'margin': margin})
        i = j
    return out


# ---------------------------------------------------------------------------------------------------------------------
# merge of the windows and decoding (CPU side, after the model)
# ---------------------------------------------------------------------------------------------------------------------
def merge_windows(n_words: int, gword, margin, lab, pmax, pper):
    """A word keeps the output of the window in which it is farthest from the window edge. Inputs are the concatenated per-window arrays;
    returns (lab int8 (n_words), pmax float32, pper float32); words without any output are O."""
    labs = np.zeros(n_words, dtype=np.int8)
    pm = np.ones(n_words, dtype=np.float32)
    pp = np.zeros(n_words, dtype=np.float32)
    if len(gword) == 0:
        return labs, pm, pp
    o = np.lexsort((np.arange(len(gword)), -margin.astype(np.int64), gword))     # per word: largest margin first, then the earliest window
    g = gword[o]
    first = np.r_[True, g[1:] != g[:-1]]
    take = o[first]
    labs[gword[take]] = lab[take]
    pm[gword[take]] = pmax[take]
    pp[gword[take]] = pper[take]
    return labs, pm, pp


def decode_spans(lab: np.ndarray, pmax: np.ndarray, pper: np.ndarray, pass_off: np.ndarray):
    """Strict IOB2 spans over the concatenated words of a pack. Returns (pass index, start (pack word index), end (exclusive), conf, p_per)."""
    n = len(lab)
    starts = np.flatnonzero(lab == 1)
    if len(starts) == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, np.zeros(0), np.zeros(0)
    stop = np.union1d(np.flatnonzero(lab != 2), pass_off[:-1])
    stop = np.append(stop, n)
    ends = stop[np.searchsorted(stop, starts + 1, side='left')]
    # a span cannot cross a passage boundary: the boundary offsets are in ``stop``
    c1 = np.concatenate(([0.0], np.cumsum(pmax.astype(np.float64))))
    c2 = np.concatenate(([0.0], np.cumsum(pper.astype(np.float64))))
    ln = (ends - starts).astype(np.float64)
    conf = (c1[ends] - c1[starts]) / ln
    pp = (c2[ends] - c2[starts]) / ln
    pid = np.searchsorted(pass_off, starts, side='right') - 1
    return pid, starts, ends, conf, pp


def owned_spans(pack: dict, pid, s, e, conf, pp):
    """Keep the spans whose first word is owned by their passage; returns a dict of arrays in document coordinates."""
    off = pack['pass_off']
    sw = s - off[pid]                                   # start word within the passage
    w0 = pack['pass_w0'][pid]
    ds = w0 + sw                                        # start word within the document
    keep = (ds >= pack['pass_own0'][pid]) & (ds < pack['pass_own1'][pid])
    pid, s, e, conf, pp, ds = pid[keep], s[keep], e[keep], conf[keep], pp[keep], ds[keep]
    de = ds + (e - s)
    return {'doc': pack['pass_doc'][pid], 'passage': pack['pass_idx'][pid], 'w_start': ds, 'w_end': de, 'conf': conf.astype(np.float32), 'p_per': pp.astype(np.float32),
            'char_start': pack['cs'][s], 'char_end': pack['ce'][e - 1]}


def span_keys(doc, w_start, w_end):
    """int64 key of an exact span: document (< 2^20), first word (< 2^31 / 2^11 words), length (< 2^12)."""
    return (doc.astype(np.int64) << 40) | (w_start.astype(np.int64) << 12) | np.minimum(w_end - w_start, 4095).astype(np.int64)


def agreement(h: dict, other: dict):
    """For the spans of the main model ``h`` (document coordinates): (exact, overlap) booleans against the spans of another model ``other``."""
    kh = span_keys(h['doc'], h['w_start'], h['w_end'])
    ko = span_keys(other['doc'], other['w_start'], other['w_end'])
    exact = np.isin(kh, ko)
    if len(ko) == 0 or len(kh) == 0:
        return exact, np.zeros(len(kh), dtype=bool)
    # overlap: spans of one model never overlap each other; sort the other spans by (doc, start)
    BIGW = 1 << 31
    oo = np.lexsort((other['w_start'], other['doc']))
    od, os_, oe = other['doc'][oo].astype(np.int64), other['w_start'][oo].astype(np.int64), other['w_end'][oo].astype(np.int64)
    okey = od * BIGW + os_
    # candidate: the last other span starting before h.end in the same document
    hq = h['doc'].astype(np.int64) * BIGW + h['w_end'].astype(np.int64)
    j = np.searchsorted(okey, hq, side='left') - 1
    jc = np.clip(j, 0, len(okey) - 1)
    ov = (j >= 0) & (od[jc] == h['doc']) & (oe[jc] > h['w_start'])
    return exact, ov


# ---------------------------------------------------------------------------------------------------------------------
# the model (GPU side)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Bundle:
    name: str
    model: object
    ti: TokInfo
    device: object


def load_bundle(model_dir: str, device, name: Optional[str] = None) -> Bundle:
    import torch
    from .train import build_model, load_tokenizer
    md = Path(model_dir)
    cfg = json.loads((md / 'config.json').read_text())
    tok = load_tokenizer(str(md / 'tokenizer'), cfg.get('trust_remote_code', False))
    model = build_model(cfg['encoder_path'], cfg.get('trust_remote_code', False), head_norm=cfg.get('head_norm', False)).to(device)
    model.load_state_dict(torch.load(md / 'model.pt', map_location=device))
    model.eval()
    ti = tok_info(tok, cfg.get('max_len', 256), 64, bool(cfg.get('force_bos')))
    return Bundle(name or cfg.get('run', md.name), model, ti, device)


def run_batches(bundle: Bundle, pack: dict):
    """Run the model on the batches of a pack; returns the merged word arrays (lab, pmax, pper)."""
    import torch
    n_words = int(pack['pass_off'][-1])
    gw, mg, lb, pm, pp = [], [], [], [], []
    dev = bundle.device
    with torch.inference_mode():
        for b in pack['batches']:
            ids = torch.from_numpy(b['ids']).to(dev, dtype=torch.long, non_blocking=True)
            lens = torch.from_numpy(b['lens']).to(dev)
            mask = (torch.arange(ids.shape[1], device=dev)[None, :] < lens[:, None]).long()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                hl, hp = bundle.model(ids, mask)
            pl = torch.softmax(hl.float(), -1)
            pmax, lab = pl.max(-1)
            pper = 1.0 - torch.softmax(hp.float(), -1)[..., 0]
            flat = torch.from_numpy(b['flat']).to(dev)
            lab = lab.reshape(-1)[flat].to(torch.int8).cpu().numpy()
            pmax = pmax.reshape(-1)[flat].float().cpu().numpy()
            pper = pper.reshape(-1)[flat].float().cpu().numpy()
            gw.append(b['gword'])
            mg.append(b['margin'])
            lb.append(lab)
            pm.append(pmax)
            pp.append(pper)
    if not gw:
        return merge_windows(n_words, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int8), np.zeros(0), np.zeros(0))
    return merge_windows(n_words, np.concatenate(gw), np.concatenate(mg), np.concatenate(lb), np.concatenate(pm), np.concatenate(pp))


def pack_spans(pack: dict, lab, pmax, pper) -> dict:
    pid, s, e, conf, pp = decode_spans(lab, pmax, pper, pack['pass_off'])
    return owned_spans(pack, pid, s, e, conf, pp)
