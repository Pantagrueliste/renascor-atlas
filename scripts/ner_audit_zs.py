"""Off-the-shelf models with their predicted spans saved (audit): same label mapping, windows and thresholds as the evaluation of the previous task
(ratlas.ner.zeroshot), but the spans are written out in the format of ner_audit_predict.py so that the same scorer can compare every system
(strict, relaxed, seen/unseen, false-positive types).

    python scripts/ner_audit_zs.py --backend flair --model flair/ner-english-large --map loc=LOC,per=PER --data test.jsonl --out audit/pred/zs_flair_ner_en_large__test.jsonl.gz
    python scripts/ner_audit_zs.py --backend gliner --model LOCAL --labels LOC=place,PER=person --threshold 0.7 ...
    python scripts/ner_audit_zs.py --backend hf --model Davlan/xlm-roberta-large-ner-hrl --map loc=LOC,per=PER ...
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['hf', 'gliner', 'flair'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--map')
    ap.add_argument('--labels', default='LOC=place,PER=person')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--chunk-words', type=int, default=100)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--gl-batch', type=int, default=16)
    ap.add_argument('--flair-link')
    ap.add_argument('--words-from', choices=['tokens', 'text'], default='tokens')
    a = ap.parse_args()

    import re
    from ratlas.ner import nerdata as nd
    from ratlas.ner import zeroshot as zs
    WORD = re.compile(r'\w+|[^\w\s]', re.U)
    recs = []
    with open(a.data, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if a.words_from == 'text' or not d.get('tokens'):
                    ms = list(WORD.finditer(d['text']))
                    d['tokens'] = [m.group() for m in ms]
                    d['token_starts'] = [m.start() for m in ms]
                recs.append(d)
    log(len(recs), 'records')
    preds = {r['id']: {'LOC': {}, 'PER': {}} for r in recs}     # span -> score

    if a.backend == 'hf':
        import torch
        from torch import nn
        from transformers import AutoModelForTokenClassification
        from ratlas.ner.train import load_tokenizer, pad_id_of, predict_words
        path = zs.snapshot(a.model)
        tok = load_tokenizer(path, False)
        model = AutoModelForTokenClassification.from_pretrained(path)
        mp = zs.parse_map(a.map)
        id2 = model.config.id2label
        lut = {'LOC': np.zeros(len(id2), dtype=np.int64), 'PER': np.zeros(len(id2), dtype=np.int64)}
        used = {}
        for i, name in id2.items():
            i = int(i)
            if name.upper() in ('O', 'OUT'):
                continue
            pre, _, ty = name.partition('-') if '-' in name else ('I', '', name)
            pre = 'B' if pre.upper() in ('B', 'S', 'U') else 'I'
            task = mp.get(ty.lower())
            if task:
                lut[task][i] = 1 if pre == 'B' else 2
                used[name] = task
        log('label map', used)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model
                self.register_buffer('ll', torch.from_numpy(lut['LOC']))
                self.register_buffer('lp', torch.from_numpy(lut['PER']))

            def forward(self, ids, mask):
                pred = self.m(input_ids=ids, attention_mask=mask).logits.argmax(-1)
                return nn.functional.one_hot(self.ll[pred], 3).float(), nn.functional.one_hot(self.lp[pred], 3).float()

        ad = Adapter().to(dev).eval()
        prefix, suffix = nd.special_tokens(tok)
        enc = nd.encode_records(tok, [{k: r[k] for k in ('id', 'tokens', 'ner_tags_loc', 'ner_tags_per') if k in r} | {'loc_valid': True, 'per_valid': True, 'edition': 'x'} for r in recs])
        pl, pp = predict_words(ad, enc, prefix, suffix, pad_id_of(tok), a.max_len - len(prefix) - len(suffix), a.overlap, dev)
        from ratlas.ner.spans import tags_to_spans
        for i, m in enumerate(enc.meta):
            s0, s1 = int(enc.word_off[i]), int(enc.word_off[i + 1])
            for task, arr in (('LOC', pl), ('PER', pp)):
                for sp in tags_to_spans(arr[s0:s1].tolist()):
                    preds[m['id']][task][sp] = 1.0
    elif a.backend == 'gliner':
        import torch
        from gliner import GLiNER
        path = zs.snapshot(a.model)
        model = GLiNER.from_pretrained(path, map_location='cpu').to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        labels = zs.parse_labels(a.labels)
        raw = zs.gliner_predict(model, recs, labels, a.chunk_words, a.threshold, a.gl_batch)
        for rid, v in raw.items():
            for task, d in v.items():
                for sp, sc in d.items():
                    if sc >= a.threshold:
                        preds[rid][task][sp] = sc
    else:
        import flair
        import torch
        from flair.data import Sentence
        from flair.models import SequenceTagger
        flair.device = torch.device('cuda')
        path = zs.snapshot(a.model)
        f = next(p for p in Path(path).iterdir() if p.suffix == '.pt' or p.name.endswith('.bin'))
        tagger = SequenceTagger.load(str(f))
        mp = zs.parse_map(a.map)
        items = [(r['id'], off, ws) for r in recs for off, ws in zs.chunks_of(r, a.chunk_words) if ws]
        B = 4096
        for b in range(0, len(items), B):
            part = items[b:b + B]
            sents = [Sentence(ws) for _, _, ws in part]
            tagger.predict(sents, mini_batch_size=a.gl_batch, verbose=False)
            for (rid, off, ws), s in zip(part, sents):
                for sp in s.get_spans('ner'):
                    task = mp.get(sp.get_label('ner').value.lower())
                    if task:
                        preds[rid][task][(sp.tokens[0].idx - 1 + off, sp.tokens[-1].idx + off)] = float(sp.get_label('ner').score)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, 'wt', encoding='utf-8') as out:
        for r in recs:
            text = r.get('text', '')
            ts = r.get('token_starts')
            o = {'id': r['id']}
            for task in ('LOC', 'PER'):
                lst = []
                for (s, e), sc in sorted(preds[r['id']][task].items()):
                    d = {'s': s, 'e': e, 'conf': round(sc, 4)}
                    if ts and e <= len(ts):
                        d['cs'], d['ce'] = ts[s], ts[e - 1] + len(r['tokens'][e - 1])
                        d['text'] = text[d['cs']:d['ce']] if text else ' '.join(r['tokens'][s:e])
                    else:
                        d['text'] = ' '.join(r['tokens'][s:e])
                    lst.append(d)
                o[task] = lst
            out.write(json.dumps(o, ensure_ascii=False) + '\n')
    log('written', a.out)


if __name__ == '__main__':
    main()
