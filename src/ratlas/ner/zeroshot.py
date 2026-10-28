"""Off-the-shelf NER models used as they are (no fine-tuning), scored with the same strict span metrics as the fine-tuned models.

Backends
  hf      a token-classification model of the Hub (WikiNEuRal, Davlan XLM-R, impresso, hmBERT HIPE models, ...). Our word alignment and windows are used, the
          labels of the model are mapped to LOC / PER (``--map``), everything else is O.
  gliner  GLiNER family: the label names are prompts (``--labels "LOC=place|location,PER=person"``); the score threshold is chosen on the dev set among
          ``--thresholds`` (one pass at the lowest threshold, the others filter by score).
  flair   flair sequence taggers (ner-multi, hmBench HIPE models): run in the flair environment.

    python -m ratlas.ner.zeroshot --backend hf --name zs_wikineural --model Babelscape/wikineural-multilingual-ner --map LOC=LOC,PER=PER \
        --set dev=.../dev.jsonl --set test=.../test.jsonl --out RUNS
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import nerdata as nd
from .spans import Counts, char_to_token_spans, record_groups, tags_to_spans, spans_to_tags
from .train import expand, load_tokenizer, log, pad_id_of, parse_named, predict_words, score_words

DEFAULT_MAP = {'loc': 'LOC', 'location': 'LOC', 'gpe': 'LOC', 'geo': 'LOC', 'place': 'LOC', 'locality': 'LOC', 'ort': 'LOC', 'lieu': 'LOC',
               'per': 'PER', 'person': 'PER', 'pers': 'PER', 'people': 'PER', 'human': 'PER', 'persona': 'PER'}


def snapshot(model_id: str) -> str:
    """Local snapshot directory of a cached Hub model (or the path itself)."""
    import os
    if Path(model_id).is_dir():
        return model_id
    cache = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache/huggingface/hub'))
    d = cache / ('models--' + model_id.replace('/', '--')) / 'snapshots'
    snaps = sorted(d.iterdir()) if d.is_dir() else []
    if not snaps:
        raise SystemExit(f'{model_id}: not in {cache}')
    return str(snaps[-1])


def parse_map(s: str | None) -> dict:
    if not s:
        return dict(DEFAULT_MAP)
    m = {}
    for kv in s.split(','):
        k, _, v = kv.partition('=')
        m[k.strip().lower()] = v.strip().upper()
    return m


def read_set(files, keep_text=False):
    recs = []
    for f in files:
        recs += nd.read_slim(f, keep_text=keep_text)
    return recs


def gold_spans(rec, task):
    tags = rec['ner_tags_loc'] if task == 'LOC' else rec['ner_tags_per']
    return set(tags_to_spans(tags))


def score_span_preds(recs, preds):
    """preds: id -> {'LOC': set of token spans, 'PER': set}. Records are scored only for the labels they annotate."""
    c = Counts()
    for r in recs:
        g = record_groups(r)
        p = preds.get(r['id'], {})
        for task, valid in (('LOC', r.get('loc_valid', True)), ('PER', r.get('per_valid', True))):
            if valid:
                c.add(task, g, gold_spans(r, task), set(p.get(task, ())))
    return c


def chunks_of(rec, n_words):
    toks = rec['tokens']
    return [(a, toks[a:a + n_words]) for a in range(0, max(1, len(toks)), n_words)]


# ---------------------------------------------------------------------------------------------------------------------
# hf token classification
# ---------------------------------------------------------------------------------------------------------------------
def run_hf(args, sets):
    import torch
    from torch import nn
    from transformers import AutoModelForTokenClassification
    path = snapshot(args.model)
    kw = {'trust_remote_code': True} if args.trust_remote_code else {}
    tok = load_tokenizer(path, args.trust_remote_code)
    model = AutoModelForTokenClassification.from_pretrained(path, **kw)
    mp = parse_map(args.map)
    id2 = model.config.id2label
    lut = {'LOC': np.zeros(len(id2), dtype=np.int64), 'PER': np.zeros(len(id2), dtype=np.int64)}
    used = {}
    for i, name in id2.items():
        i = int(i)
        pre, _, ty = name.partition('-') if '-' in name else ('I', '', name)
        if pre.upper() in ('S', 'U'):
            pre = 'B'
        elif pre.upper() in ('E', 'L'):
            pre = 'I'
        task = mp.get(ty.lower())
        if name.upper() in ('O', 'OUT'):
            continue
        if task:
            lut[task][i] = 1 if pre.upper() == 'B' else 2
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
    pad = pad_id_of(tok)
    limit = args.max_len - len(prefix) - len(suffix)
    out = {}
    for name, files in sets.items():
        t0 = time.time()
        recs = read_set(files)
        enc = nd.encode_records(tok, recs)
        pl, pp = predict_words(ad, enc, prefix, suffix, pad, limit, args.overlap, dev)
        c, _ = score_words(enc, pl, pp)
        out[name] = c.report()
        out[name]['n_records'] = len(recs)
        log_set(name, out[name], t0)
    return out, {'label_map': used}


def log_set(name, rep, t0):
    loc, per = rep.get('LOC', {}).get('overall', {}), rep.get('PER', {}).get('overall', {})
    log(f'[zs {name}] {time.time() - t0:.0f}s LOC P/R/F {loc.get("p")}/{loc.get("r")}/{loc.get("f1")}  PER {per.get("p")}/{per.get("r")}/{per.get("f1")}')


# ---------------------------------------------------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------------------------------------------------
def parse_labels(s: str) -> dict:
    d = {}
    for part in s.split(','):
        k, _, v = part.partition('=')
        d[k.strip().upper()] = [x.strip() for x in v.split('|') if x.strip()]
    return d


def gliner_predict(model, recs, labels: dict, chunk_words, threshold, batch_size):
    """id -> {'LOC': {(s, e): score}, 'PER': ...} at the given minimum threshold (best score kept when several labels give the same span)."""
    items = []
    for ri, r in enumerate(recs):
        for off, ws in chunks_of(r, chunk_words):
            if ws:
                items.append((ri, off, ws))
    names = [(t, l) for t, ls in labels.items() for l in ls]
    all_labels = [l for _, l in names]
    to_task = {l: t for t, l in names}
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    texts = [' '.join(items[i][2]) for i in order]
    res = model.batch_predict_entities(texts, all_labels, threshold=threshold, flat_ner=True, batch_size=batch_size)
    preds = {r['id']: {'LOC': {}, 'PER': {}} for r in recs}
    for k, ents in zip(order, res):
        ri, off, ws = items[k]
        starts, p = [], 0
        for w in ws:
            starts.append(p)
            p += len(w) + 1
        lens = [len(w) for w in ws]
        for e in ents:
            task = to_task.get(e['label'])
            if not task:
                continue
            for (s, t) in char_to_token_spans(starts, lens, [(e['start'], e['end'])]):
                key = (s + off, t + off)
                d = preds[recs[ri]['id']][task]
                d[key] = max(d.get(key, 0), float(e.get('score', 1)))
    return preds


def filter_threshold(preds, t):
    return {i: {task: {s for s, sc in d.items() if sc >= t} for task, d in v.items()} for i, v in preds.items()}


class SpaceSplitter:
    """One word per whitespace-separated piece: exactly our own tokens (GLiNER's default also cuts words at combining marks and inner punctuation)."""

    def __call__(self, text):
        import re
        for m in re.finditer(r'\S+', text):
            yield m.group(), m.start(), m.end()


def run_gliner(args, sets):
    import torch
    from gliner import GLiNER
    path = snapshot(args.model)
    model = GLiNER.from_pretrained(path, map_location='cpu', **({'variant': args.variant} if args.variant else {}))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    if getattr(args, 'split_ws', False):
        model.data_processor.words_splitter = SpaceSplitter()
    labels = parse_labels(args.labels)
    ths = sorted(float(x) for x in args.thresholds.split(','))
    raw = {}
    out = {}
    for name, files in sets.items():
        t0 = time.time()
        recs = read_set(files)
        preds = gliner_predict(model, recs, labels, args.chunk_words, ths[0], args.gl_batch)
        raw[name] = (recs, preds)
        for t in ths:
            rep = score_span_preds(recs, filter_threshold(preds, t)).report()
            rep['n_records'] = len(recs)
            out[f'{name}@{t}'] = rep
        log(f'[gliner {name}] {time.time() - t0:.0f}s thresholds ' + ' '.join(f'{t}:{out[f"{name}@{t}"]["LOC"]["overall"]["f1"]}' for t in ths))
    # threshold chosen on the dev set (LOC F1)
    sel = ths[0]
    if 'dev' in sets:
        sel = max(ths, key=lambda t: out[f'dev@{t}']['LOC']['overall']['f1'])
    for name in sets:
        out[name] = out[f'{name}@{sel}']
    log('selected threshold', sel)
    return out, {'labels': labels, 'thresholds': ths, 'selected_threshold': sel}


# ---------------------------------------------------------------------------------------------------------------------
# GLiNER2 (fastino): own library, run in the gliner2 environment
# ---------------------------------------------------------------------------------------------------------------------
def gliner2_predict(model, recs, labels: dict, chunk_words, threshold, batch_size):
    items = []
    for ri, r in enumerate(recs):
        for off, ws in chunks_of(r, chunk_words):
            if ws:
                items.append((ri, off, ws))
    names = [(t, l) for t, ls in labels.items() for l in ls]
    all_labels = [l for _, l in names]
    to_task = {l: t for t, l in names}
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    texts = [' '.join(items[i][2]) for i in order]
    res = model.batch_extract_entities(texts, all_labels, batch_size=batch_size, threshold=threshold, format_results=True, include_confidence=True,
                                       include_spans=True)
    preds = {r['id']: {'LOC': {}, 'PER': {}} for r in recs}
    for k, out in zip(order, res):
        ri, off, ws = items[k]
        starts, p = [], 0
        for w in ws:
            starts.append(p)
            p += len(w) + 1
        lens = [len(w) for w in ws]
        for label, ents in (out.get('entities') or {}).items():
            task = to_task.get(label)
            if not task:
                continue
            for e in ents:
                if not isinstance(e, dict) or 'start' not in e:
                    continue
                for (s_, t_) in char_to_token_spans(starts, lens, [(e['start'], e['end'])]):
                    key = (s_ + off, t_ + off)
                    d = preds[recs[ri]['id']][task]
                    d[key] = max(d.get(key, 0), float(e.get('confidence', 1)))
    return preds


def run_gliner2(args, sets):
    import torch
    from gliner2 import GLiNER2
    path = snapshot(args.model)
    cfg_text = (Path(path) / 'config.json').read_text().lower()
    if 'boundary' in cfg_text:              # GLiNER2.5 checkpoints use the boundary head
        from gliner2 import BoundaryExtractor
        model = BoundaryExtractor.from_pretrained(path)
    else:
        model = GLiNER2.from_pretrained(path)
    model = model.to('cuda').eval()
    labels = parse_labels(args.labels)
    ths = sorted(float(x) for x in args.thresholds.split(','))
    out = {}
    for name, files in sets.items():
        t0 = time.time()
        recs = read_set(files)
        preds = gliner2_predict(model, recs, labels, args.chunk_words, ths[0], args.gl_batch)
        for t in ths:
            rep = score_span_preds(recs, filter_threshold(preds, t)).report()
            rep['n_records'] = len(recs)
            out[f'{name}@{t}'] = rep
        log(f'[gliner2 {name}] {time.time() - t0:.0f}s thresholds ' + ' '.join(f'{t}:{out[f"{name}@{t}"]["LOC"]["overall"]["f1"]}' for t in ths))
    sel = ths[0]
    if 'dev' in sets:
        sel = max(ths, key=lambda t: out[f'dev@{t}']['LOC']['overall']['f1'])
    for name in sets:
        out[name] = out[f'{name}@{sel}']
    log('selected threshold', sel)
    return out, {'labels': labels, 'thresholds': ths, 'selected_threshold': sel}


# ---------------------------------------------------------------------------------------------------------------------
# flair
# ---------------------------------------------------------------------------------------------------------------------
def run_flair(args, sets):
    import flair
    import torch
    from flair.data import Sentence
    from flair.models import SequenceTagger
    flair.device = torch.device('cuda')
    path = snapshot(args.model)
    if args.flair_link:           # the model was saved with a relative path as embedding name (e.g. embeddings/dalembert): give it a directory with that name
        import os
        rel, _, hub_id = args.flair_link.partition('=')
        cwd = Path(args.out) / args.name / '_cwd'
        (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (cwd / rel).exists():
            (cwd / rel).symlink_to(snapshot(hub_id))
        os.chdir(cwd)
    f = next(p for p in Path(path).iterdir() if p.suffix == '.pt' or p.name.endswith('.bin'))
    tagger = SequenceTagger.load(str(f))
    mp = parse_map(args.map)
    out = {}
    for name, files in sets.items():
        t0 = time.time()
        recs = read_set(files)
        items = [(r['id'], off, ws) for r in recs for off, ws in chunks_of(r, args.chunk_words) if ws]
        preds = {r['id']: {'LOC': set(), 'PER': set()} for r in recs}
        B = 4096
        for b in range(0, len(items), B):
            part = items[b:b + B]
            sents = [Sentence(ws) for _, _, ws in part]
            tagger.predict(sents, mini_batch_size=args.gl_batch, verbose=False)
            for (rid, off, ws), s in zip(part, sents):
                for sp in s.get_spans('ner'):
                    task = mp.get(sp.get_label('ner').value.lower())
                    if task:
                        preds[rid][task].add((sp.tokens[0].idx - 1 + off, sp.tokens[-1].idx + off))
        out[name] = score_span_preds(recs, preds).report()
        out[name]['n_records'] = len(recs)
        log_set(name, out[name], t0)
    return out, {'label_map': mp}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['hf', 'gliner', 'gliner2', 'flair'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--set', action='append', default=[], help='name=path (file, directory or glob)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--map', help='hf/flair: label=TASK pairs, e.g. loc=LOC,pers=PER')
    ap.add_argument('--labels', default='LOC=place,PER=person', help='gliner: TASK=prompt1|prompt2,...')
    ap.add_argument('--thresholds', default='0.3,0.5,0.7')
    ap.add_argument('--chunk-words', type=int, default=100)
    ap.add_argument('--flair-link', help='flair: relpath=hub_id, e.g. embeddings/dalembert=pjox/dalembert')
    ap.add_argument('--variant', help='gliner: weight variant (bf16, fp16) when the repo has no plain model file')
    ap.add_argument('--split-ws', action='store_true', help='gliner: split the text at whitespace only (our own tokens) instead of the GLiNER word splitter')
    ap.add_argument('--gl-batch', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=64)
    args = ap.parse_args(argv)
    sets = {n: expand(p) for n, p, _ in parse_named(args.set)}
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fn = {'hf': run_hf, 'gliner': run_gliner, 'gliner2': run_gliner2, 'flair': run_flair}[args.backend]
    res, extra = fn(args, sets)
    (out_dir / 'metrics.json').write_text(json.dumps({'run': args.name, 'model': args.model, 'kind': 'zeroshot', 'backend': args.backend, **extra,
                                                      'minutes': round((time.time() - t0) / 60, 1), 'sets': res}, indent=1))
    (out_dir / 'done.json').write_text(json.dumps({'done': time.strftime('%F %T')}))
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
