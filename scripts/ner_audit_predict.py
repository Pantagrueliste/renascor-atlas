"""Independent inference for the audit of the NER models (GPU, slurm/ner_audit_gpu.sbatch).

Written from scratch (own tokenisation check, own windowing, own decoding) and not through ratlas.ner.train.predict_words: it only reuses the model class
(build_model) and the tokenizer loader, which define the model itself. The words fed to the model are the record's words joined by one space (the
training convention). Output per record: predicted LOC and PER spans as word spans, with character offsets in the record's own ``text`` and the mean
softmax probability of the span (used for confidence thresholds).

    python scripts/ner_audit_predict.py --model-dir MODELS/r3_mdeberta_HM --data data_v1/test.jsonl --out audit/pred/r3_mdeberta_HM__test.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

WORD = re.compile(r'\w+|[^\w\s]', re.U)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def word_chars(text):
    return [(m.start(), m.end()) for m in WORD.finditer(text)]


def decode(pred, prob):
    """Strict IOB2 decoding of a 0/1/2 sequence: a span starts at B, continues with I; a stray I is ignored. -> [(start, end, mean prob)]"""
    spans, start = [], None
    for i, t in enumerate(list(pred) + [0]):
        if t == 1 or t == 0:
            if start is not None:
                spans.append((start, i, float(np.mean(prob[start:i]))))
                start = None
            if t == 1:
                start = i
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data', required=True, help='jsonl with text and/or tokens')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--batch-tokens', type=int, default=20000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--words-from', choices=['tokens', 'text'], default='tokens')
    a = ap.parse_args()

    import torch
    from ratlas.ner.train import build_model, load_tokenizer
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    md = Path(a.model_dir)
    cfg = json.loads((md / 'config.json').read_text())
    tok = load_tokenizer(str(md / 'tokenizer'), cfg.get('trust_remote_code', False))
    model = build_model(cfg['encoder_path'], cfg.get('trust_remote_code', False), head_norm=cfg.get('head_norm', False)).to(dev)
    model.load_state_dict(torch.load(md / 'model.pt', map_location=dev))
    model.eval()
    max_len = cfg.get('max_len', a.max_len)
    # special tokens added by the tokenizer itself (a tokenizer rebuilt from tokenizer.json has them only in its post-processor, not in
    # build_inputs_with_special_tokens): compare the encoding of a probe word with and without them
    bare = tok('probe', add_special_tokens=False)['input_ids']
    full = tok('probe', add_special_tokens=True)['input_ids']
    k = next(i for i in range(len(full) - len(bare) + 1) if full[i:i + len(bare)] == bare)
    prefix, suffix = list(full[:k]), list(full[k + len(bare):])
    if cfg.get('force_bos') and tok.bos_token_id is not None and (not prefix or prefix[0] != tok.bos_token_id):
        prefix = [int(tok.bos_token_id)] + prefix
    L = max_len - len(prefix) - len(suffix)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    log('model', cfg['run'], 'prefix', prefix, 'suffix', suffix, 'window', L)

    recs = []
    with open(a.data, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
                if a.limit and len(recs) >= a.limit:
                    break
    n_tok_mismatch = 0
    items = []     # per record: dict
    for r in recs:
        text = r.get('text', '')
        if a.words_from == 'tokens' and r.get('tokens'):
            words = r['tokens']
            if text:
                mine = [text[s:e] for s, e in word_chars(text)]
                n_tok_mismatch += mine != words
        else:
            wc = word_chars(text)
            words = [text[s:e] for s, e in wc]
        joined = ' '.join(words)
        ws, p = [], 0
        for w in words:
            ws.append((p, p + len(w)))
            p += len(w) + 1
        enc = tok(joined, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc['input_ids']
        word_of = []
        starts = np.array([s for s, _ in ws])
        for (s, e) in enc['offset_mapping']:
            if e <= s:
                word_of.append(-1)
                continue
            i = int(np.searchsorted(starts, s, side='right')) - 1
            if i < 0 or s >= ws[i][1]:
                i += 1
            word_of.append(i if 0 <= i < len(words) and ws[i][0] < e else -1)
        first = {}
        for j, w in enumerate(word_of):
            if w >= 0 and w not in first:
                first[w] = j
        items.append({'words': words, 'ids': ids, 'first': first})
    log(f'{len(recs)} records, tokenisation mismatches with the stored tokens: {n_tok_mismatch}')

    # windows
    wins = []   # (item, a, b)
    for ii, it in enumerate(items):
        n = len(it['ids'])
        step = max(1, L - a.overlap)
        s = 0
        while True:
            e = min(n, s + L)
            wins.append((ii, s, e))
            if e >= n:
                break
            s += step
            if s + L > n:
                s = max(0, n - L)
                wins.append((ii, s, n))
                break
    order = sorted(range(len(wins)), key=lambda w: wins[w][2] - wins[w][1])
    best = [np.full(len(it['ids']), -1.0) for it in items]
    probs = [np.zeros((len(it['ids']), 2, 3), dtype=np.float32) for it in items]
    t0 = time.time()
    bi = 0
    with torch.no_grad():
        while bi < len(order):
            batch, mx = [], 0
            while bi < len(order):
                w = wins[order[bi]]
                ln = w[2] - w[1] + len(prefix) + len(suffix)
                if batch and (max(mx, ln) * (len(batch) + 1) > a.batch_tokens or len(batch) >= 128):
                    break
                batch.append(w)
                mx = max(mx, ln)
                bi += 1
            ids = np.full((len(batch), mx), pad, dtype=np.int64)
            mask = np.zeros((len(batch), mx), dtype=np.int64)
            for k_, (ii, s, e) in enumerate(batch):
                seq = prefix + items[ii]['ids'][s:e] + suffix
                ids[k_, :len(seq)] = seq
                mask[k_, :len(seq)] = 1
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                hl, hp = model(torch.from_numpy(ids).to(dev), torch.from_numpy(mask).to(dev))
            pl = torch.softmax(hl.float(), -1).cpu().numpy()
            pp = torch.softmax(hp.float(), -1).cpu().numpy()
            for k_, (ii, s, e) in enumerate(batch):
                n = len(items[ii]['ids'])
                for j in range(s, e):
                    left = (j - s) if s > 0 else 10 ** 6
                    right = (e - 1 - j) if e < n else 10 ** 6
                    m = min(left, right)
                    if m > best[ii][j]:
                        best[ii][j] = m
                        probs[ii][j, 0] = pl[k_, len(prefix) + j - s]
                        probs[ii][j, 1] = pp[k_, len(prefix) + j - s]
    log(f'inference {time.time() - t0:.0f}s, {len(wins)} windows')

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, 'wt', encoding='utf-8') as out:
        for ri, (r, it) in enumerate(zip(recs, items)):
            nw = len(it['words'])
            wl = np.zeros((nw, 3), dtype=np.float32)
            wp = np.zeros((nw, 3), dtype=np.float32)
            wl[:, 0] = 1
            wp[:, 0] = 1
            for w, j in it['first'].items():
                wl[w] = probs[ri][j, 0]
                wp[w] = probs[ri][j, 1]
            text = r.get('text', '')
            wc = None
            if a.words_from == 'text' or not r.get('token_starts'):
                wc = word_chars(text) if text else None
            elif r.get('token_starts'):
                wc = [(s, s + len(w)) for s, w in zip(r['token_starts'], it['words'])]
            rec_out = {'id': r['id']}
            for task, w_ in (('LOC', wl), ('PER', wp)):
                pred = w_.argmax(1)
                conf = w_.max(1)
                spans = []
                for s, e, c in decode(pred, conf):
                    d = {'s': s, 'e': e, 'conf': round(c, 4)}
                    if wc and e <= len(wc):
                        d['cs'], d['ce'] = wc[s][0], wc[e - 1][1]
                        d['text'] = text[wc[s][0]:wc[e - 1][1]] if text else ' '.join(it['words'][s:e])
                    else:
                        d['text'] = ' '.join(it['words'][s:e])
                    spans.append(d)
                rec_out[task] = spans
            out.write(json.dumps(rec_out, ensure_ascii=False) + '\n')
    log('written', a.out)


if __name__ == '__main__':
    main()
