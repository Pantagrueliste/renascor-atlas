"""Fine-tune an encoder for LOC and PER tagging (two token-classification heads, one per label, each trained only on the records where that label is annotated)
and evaluate it with strict span metrics.

    python -m ratlas.ner.train --name xlmr_base --model FacebookAI/xlm-roberta-base --train ours=.../train.jsonl --dev .../dev.jsonl --test ours=.../test.jsonl --out RUNS
    python -m ratlas.ner.train --eval_only --name xlmr_base --load MODELS/xlmr_base --eval hipe=.../hipe2022__newseye__fr.jsonl

Training sets: ``--train name=path[:max_words]`` (repeatable; max_words caps the words taken from that set, sampled with the seed). A path that is a directory
takes every ``*.jsonl`` in it. Early stopping on the LOC F1 of the dev set. Checkpoint every ~25 min and at every evaluation; ``--time_limit_min`` stops cleanly
(exit code 3) so that a job can be resubmitted and resume.

Large encoders (NER v2: XLM-R XL, EuroBERT 2.1B): ``--grad-accum N`` splits every batch of the v1 recipe into N micro-batches (same batches, same
optimizer steps, same loss normalisation as v1: the counts of labelled words are taken over the whole batch); ``--grad-ckpt`` turns on activation
checkpointing in the encoder; bf16 autocast with fp32 master weights and fp32 AdamW states as in v1. ``--encode-only`` tokenises the training and dev
sets into the cache and exits (run it on CPU before the GPU job). Smoke tests: ``--max-steps`` (the LR schedule is laid out over these steps),
``--stop-at-step`` (save the checkpoint and exit 3 at that step: tests the resume), ``--limit-dev``. Throughput and peak GPU memory go to perf.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import nerdata as nd
from .spans import Counts, record_groups, tags_to_spans

LABEL_SETS = {'LOC': 'lab_loc', 'PER': 'lab_per'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def expand(path: str) -> list[str]:
    """A file, a directory (every *.jsonl in it), a glob pattern, or several of them separated by ';'."""
    if ';' in path:
        return sorted(set(f for part in path.split(';') for f in expand(part)))
    if any(c in path for c in '*?['):
        import glob
        return sorted(glob.glob(path))
    p = Path(path)
    if p.is_dir():
        return sorted(str(x) for x in p.glob('*.jsonl'))
    return [str(p)]


def parse_named(specs: list[str]) -> list[tuple[str, str, int]]:
    out = []
    for s in specs or []:
        name, _, rest = s.partition('=')
        path, _, cap = rest.partition(':')
        out.append((name, path, int(cap) if cap else 0))
    return out


def load_set(specs: list[tuple[str, str, int]], seed: int, tag: str):
    """Records of the named sets (sampled to their caps). Returns (records, mix table)."""
    recs, mix = [], []
    for name, path, cap in specs:
        r = []
        for f in expand(path):
            r += nd.read_slim(f)
        total = sum(len(x['tokens']) for x in r)
        r = nd.sample_records(r, cap, seed)
        taken = sum(len(x['tokens']) for x in r)
        for x in r:
            x['_set'] = name
        mix.append({'set': name, 'path': path, 'records': len(r), 'words': taken, 'words_available': total,
                    'loc_spans': sum(sum(1 for t in x['ner_tags_loc'] if t == 1) for x in r if x.get('loc_valid', True)),
                    'per_spans': sum(sum(1 for t in x['ner_tags_per'] if t == 1) for x in r if x.get('per_valid', True))})
        recs += r
    tot = sum(m['words'] for m in mix) or 1
    for m in mix:
        m['share'] = round(m['words'] / tot, 4)
    log(f'[{tag}] mix', json.dumps(mix))
    return recs, mix


def encode_cached(tok, tok_name, recs, files, extra, cache_dir):
    key = nd.cache_key(nd.vocab_hash(tok), tok.vocab_size, files, extra)
    path = Path(cache_dir) / f'enc_{key}.npz' if cache_dir else None
    if path and path.exists():
        try:
            return nd.load_encoded(path)
        except Exception as e:                      # corrupt cache: rebuild
            log('cache unreadable', path, e)
    enc = nd.encode_records(tok, recs)
    if path:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        nd.save_encoded(enc, path)
    return enc


# ---------------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------------
def build_model(path: str, trust_remote_code: bool, dropout: float = 0.1, grad_ckpt: bool = False, attn: str = '', head_norm: bool = False):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class Tagger(nn.Module):
        def __init__(self):
            super().__init__()
            kw = {'trust_remote_code': True} if trust_remote_code else {}
            if attn:
                kw['attn_implementation'] = attn
            self.enc = AutoModel.from_pretrained(path, **kw)
            if grad_ckpt and getattr(self.enc, 'supports_gradient_checkpointing', False):
                self.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            elif grad_ckpt:
                # e.g. XLMRobertaXLModel in transformers 4.53: wrap every layer of the encoder in a non-reentrant checkpoint by hand
                import torch.utils.checkpoint as cp
                layers = None
                for dotted in ('encoder.layer', 'encoder.layers', 'layers'):
                    m = self.enc
                    try:
                        for part in dotted.split('.'):
                            m = getattr(m, part)
                    except AttributeError:
                        continue
                    layers = m
                    break
                if layers is None:
                    raise SystemExit(f'{path}: no gradient checkpointing and no layer list found')
                for layer in layers:
                    def fwd(*a, _f=layer.forward, _l=layer, **k):
                        if _l.training and torch.is_grad_enabled():
                            return cp.checkpoint(_f, *a, use_reentrant=False, **k)
                        return _f(*a, **k)
                    layer.forward = fwd
                if hasattr(self.enc.config, 'use_cache'):
                    self.enc.config.use_cache = False
            h = self.enc.config.hidden_size
            # XLM-R XL: the last hidden states have a std of ~0.04 (XLM-R large ~1), so heads initialised at 0.02 barely learn; a LayerNorm restores the scale
            self.head_norm = nn.LayerNorm(h) if head_norm else None
            self.drop = nn.Dropout(dropout)
            self.head_loc = nn.Linear(h, 3)
            self.head_per = nn.Linear(h, 3)
            for l in (self.head_loc, self.head_per):
                nn.init.normal_(l.weight, std=0.02)
                nn.init.zeros_(l.bias)

        def forward(self, input_ids, attention_mask):
            hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if self.head_norm is not None:
                hs = self.head_norm(hs)
            hs = self.drop(hs)
            return self.head_loc(hs), self.head_per(hs)

    return Tagger()


def load_tokenizer(path: str, trust_remote_code: bool):
    from transformers import AutoTokenizer
    kw = {'trust_remote_code': True} if trust_remote_code else {}
    try:
        tok = AutoTokenizer.from_pretrained(path, use_fast=True, **kw)
    except ValueError:
        # saved by transformers 5 ('TokenizersBackend'): the plain tokenizer.json is enough
        from transformers import PreTrainedTokenizerFast
        d = Path(path)
        cfg = json.loads((d / 'tokenizer_config.json').read_text()) if (d / 'tokenizer_config.json').exists() else {}
        special = {k: cfg[k] for k in ('bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token') if isinstance(cfg.get(k), str)}
        tok = PreTrainedTokenizerFast(tokenizer_file=str(d / 'tokenizer.json'), **special)
    if not getattr(tok, 'is_fast', False):
        raise SystemExit(f'{path}: no fast tokenizer (offsets are needed)')
    return tok


def wrapping(tok, force_bos: bool):
    prefix, suffix = nd.special_tokens(tok)
    if force_bos and tok.bos_token_id is not None and (not prefix or prefix[0] != tok.bos_token_id):
        prefix = [int(tok.bos_token_id)] + prefix
    return prefix, suffix


def pad_id_of(tok) -> int:
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    return int(tok.eos_token_id if tok.eos_token_id is not None else 0)


# ---------------------------------------------------------------------------------------------------------------------
# prediction and scoring
# ---------------------------------------------------------------------------------------------------------------------
def merge_window(enc, row, content, words, tag_loc, tag_per, pl, pp, best, nwords):
    """Write the tags predicted in one window into the word-level arrays; a word keeps the tag of the window in which it is farthest from the window edge
    (an edge that is the start or the end of the passage does not count)."""
    i, a, bb, sa, sb = row
    wo = int(enc.word_off[i])
    big = 10 ** 9
    left = content if a > 0 else np.full_like(content, big)
    right = (sb - sa - 1 - content) if bb < nwords[i] else np.full_like(content, big)
    margin = np.minimum(left, right)
    gi = wo + words
    upd = margin > best[gi]
    gi = gi[upd]
    best[gi] = margin[upd]
    pl[gi] = tag_loc[upd]
    pp[gi] = tag_per[upd]


def predict_words(model, enc: nd.Encoded, prefix, suffix, pad_id, limit, overlap, device, max_tokens=24000, max_seqs=64):
    """Word-level predicted tags (flat int8 arrays like enc.lab_loc / lab_per). Words with no subword stay O."""
    import torch
    rows = nd.build_windows(enc, limit, overlap)
    nW = int(enc.word_off[-1])
    pl = np.zeros(nW, dtype=np.int8)
    pp = np.zeros(nW, dtype=np.int8)
    best = np.full(nW, -1, dtype=np.int64)
    nwords = (enc.word_off[1:] - enc.word_off[:-1])
    model.eval()
    batches = nd.length_batches(rows, max_tokens, max_seqs, None)
    with torch.no_grad():
        for idx in batches:
            r = rows[idx]
            b = nd.batch_windows(enc, r, prefix, suffix, pad_id, False)
            ids = torch.from_numpy(b['input_ids']).to(device)
            mask = torch.from_numpy(b['attention_mask']).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                hl, hp = model(ids, mask)
            al = hl.argmax(-1).cpu().numpy()
            ap = hp.argmax(-1).cpu().numpy()
            for k, (pos, words) in enumerate(b['maps']):
                merge_window(enc, [int(x) for x in r[k]], pos - len(prefix), words, al[k, pos], ap[k, pos], pl, pp, best, nwords)
    return pl, pp


def score_words(enc: nd.Encoded, pl, pp, want_preds=False):
    c = Counts()
    preds = {}
    for i, m in enumerate(enc.meta):
        a, b = int(enc.word_off[i]), int(enc.word_off[i + 1])
        groups = record_groups(m)
        out = {}
        for task, lab, pred, valid in (('LOC', enc.lab_loc, pl, enc.valid_loc[i]), ('PER', enc.lab_per, pp, enc.valid_per[i])):
            ps = set(tags_to_spans(pred[a:b].tolist()))
            if want_preds:
                out[task] = sorted(ps)
            if valid:
                c.add(task, groups, set(tags_to_spans(lab[a:b].tolist())), ps)
        if want_preds:
            preds[m['id']] = out
    return c, preds


def evaluate_sets(model, tok, sets: dict, args, device, out_dir: Path, save_preds: set):
    """sets: name -> list of jsonl files. Returns {name: report}."""
    prefix, suffix = wrapping(tok, args.force_bos)
    pad = pad_id_of(tok)
    limit = args.max_len - len(prefix) - len(suffix)
    res = {}
    for name, files in sets.items():
        t0 = time.time()
        recs = []
        for f in files:
            recs += nd.read_slim(f)
        enc = encode_cached(tok, args.model if not args.load else args.load, recs, files, f'eval|{args.max_len}', args.cache)
        pl, pp = predict_words(model, enc, prefix, suffix, pad, limit, args.overlap, device)
        c, preds = score_words(enc, pl, pp, want_preds=name in save_preds)
        rep = c.report()
        rep['n_records'] = len(recs)
        rep['n_unaligned_words'] = enc.n_unaligned
        res[name] = rep
        if name in save_preds:
            import gzip
            with gzip.open(out_dir / f'pred_{name}.jsonl.gz', 'wt') as f:
                for k, v in preds.items():
                    f.write(json.dumps({'id': k, **{t: v.get(t, []) for t in ('LOC', 'PER')}}) + '\n')
        loc, per = rep.get('LOC', {}).get('overall', {}), rep.get('PER', {}).get('overall', {})
        log(f'[eval {name}] {len(recs)} records {time.time() - t0:.0f}s LOC P/R/F {loc.get("p")}/{loc.get("r")}/{loc.get("f1")}  PER {per.get("p")}/{per.get("r")}/{per.get("f1")}')
    return res


# ---------------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', help='HF id or local path of the pretrained encoder')
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--force-bos', action='store_true', help='put the BOS token in front of every window (EuroBERT: its tokenizer only adds EOS)')
    ap.add_argument('--train', action='append', default=[], help='name=path[:max_words]')
    ap.add_argument('--dev', help='dev jsonl (early stopping on LOC F1)')
    ap.add_argument('--test', action='append', default=[], help='name=path (file or directory) evaluated at the end')
    ap.add_argument('--eval', action='append', default=[], help='extra evaluation sets name=path')
    ap.add_argument('--save-preds', default='test', help='comma list of eval set names whose predicted spans are saved')
    ap.add_argument('--out', required=True, help='run directory root')
    ap.add_argument('--models', help='directory where the best model is saved (default: <out>/../models)')
    ap.add_argument('--cache', help='cache directory of tokenised sets')
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max-tokens', type=int, default=12000)
    ap.add_argument('--warmup', type=float, default=0.06)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--eval-every', type=float, default=0.5, help='epoch fraction between dev evaluations')
    ap.add_argument('--time-limit-min', type=float, default=0, help='stop cleanly after this many minutes (exit 3)')
    ap.add_argument('--init', help='model directory to start from (two-stage training)')
    ap.add_argument('--load', help='model directory to evaluate (with --eval-only)')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--limit-train', type=int, default=0, help='debug: first N training records')
    ap.add_argument('--tag', default='', help='suffix of the metrics file in eval-only mode')
    ap.add_argument('--grad-accum', type=int, default=1, help='micro-batches per batch (the batch of the v1 recipe is split; same optimizer steps)')
    ap.add_argument('--head-norm', action='store_true', help='LayerNorm on the encoder output before the two heads (needed by XLM-R XL)')
    ap.add_argument('--head-lr', type=float, default=0, help='learning rate of the heads (default: --lr)')
    ap.add_argument('--grad-ckpt', action='store_true', help='activation (gradient) checkpointing in the encoder')
    ap.add_argument('--attn', default='', help='attn_implementation passed to from_pretrained (sdpa, eager, flash_attention_2); default: the model default')
    ap.add_argument('--encode-only', action='store_true', help='tokenise the training and dev sets into the cache, then exit (CPU)')
    ap.add_argument('--max-steps', type=int, default=0, help='smoke test: stop after this many optimizer steps (the LR schedule is laid out over them)')
    ap.add_argument('--stop-at-step', type=int, default=0, help='smoke test: save the checkpoint and exit 3 at this step (resume test)')
    ap.add_argument('--limit-dev', type=int, default=0, help='smoke test: first N dev records only')
    ap.add_argument('--skip-final-eval', action='store_true', help='smoke probe: no final evaluation (perf.json only)')
    ap.add_argument('--ckpt-every-min', type=float, default=25, help='minutes between checkpoints outside the dev evaluations')
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        log('WARNING: no GPU')
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models) if args.models else Path(args.out).parent / 'models'
    model_dir = models_dir / args.name
    save_preds = set(x for x in args.save_preds.split(',') if x)
    eval_sets = {n: expand(p) for n, p, _ in parse_named(args.test + args.eval)}

    # ------------------------------------------------------------------ evaluation only
    if args.eval_only:
        src = Path(args.load or model_dir)
        cfg = json.loads((src / 'config.json').read_text())
        args.model = cfg['model']
        tok = load_tokenizer(str(src / 'tokenizer'), cfg.get('trust_remote_code', False))
        model = build_model(cfg['encoder_path'], cfg.get('trust_remote_code', False), args.dropout, head_norm=cfg.get('head_norm', False)).to(device)
        model.load_state_dict(torch.load(src / 'model.pt', map_location=device))
        args.max_len = cfg.get('max_len', args.max_len)
        args.force_bos = cfg.get('force_bos', False)
        res = evaluate_sets(model, tok, eval_sets, args, device, out_dir, save_preds)
        (out_dir / f'metrics_eval{args.tag}.json').write_text(json.dumps({'run': args.name, 'model_dir': str(src), 'sets': res}, indent=1))
        return 0

    # ------------------------------------------------------------------ data
    done_file = out_dir / 'done.json'
    if done_file.exists():
        log('already done', done_file)
        return 0
    cfg_path = out_dir / 'config.json'
    cfg_path.write_text(json.dumps({k: v for k, v in vars(args).items()}, indent=1))
    tok = load_tokenizer(args.model, args.trust_remote_code)
    prefix, suffix = wrapping(tok, args.force_bos)
    pad = pad_id_of(tok)
    limit = args.max_len - len(prefix) - len(suffix)
    log('model', args.model, 'prefix', prefix, 'suffix', suffix, 'pad', pad, 'device', device)

    tspecs = parse_named(args.train)
    recs, mix = load_set(tspecs, args.seed, 'train')
    if args.limit_train:
        recs = recs[:args.limit_train]
    files = [f for _, p, _ in tspecs for f in expand(p)]
    enc = encode_cached(tok, args.model, recs, files, f'train|{args.max_len}|{args.seed}|{[(n, c) for n, _, c in tspecs]}|{args.limit_train}', args.cache)
    del recs
    log(f'train encoded: {len(enc)} records, {len(enc.ids)} subwords, {enc.n_unaligned} words without subword')
    rows = nd.build_windows(enc, limit, 0, only_valid=True)
    log(f'train windows: {len(rows)}')
    dev_recs = nd.read_slim(args.dev, limit=args.limit_dev or None)
    dev_enc = encode_cached(tok, args.model, dev_recs, [args.dev], f'dev|{args.max_len}' + (f'|{args.limit_dev}' if args.limit_dev else ''), args.cache)
    del dev_recs
    if args.encode_only:
        log(f'encode only: train {len(enc)} records, dev {len(dev_enc)} records, cached in {args.cache}')
        cfg_path.unlink(missing_ok=True)
        return 0

    model = build_model(args.model, args.trust_remote_code, args.dropout, args.grad_ckpt, args.attn, args.head_norm)
    if args.init:
        sd = torch.load(Path(args.init) / 'model.pt', map_location='cpu')
        model.load_state_dict(sd)
        log('initialised from', args.init)
    model.to(device)
    no_decay = ('bias', 'LayerNorm.weight', 'layer_norm', 'norm.weight', 'norm')
    is_head = lambda n: not n.startswith('enc.')
    params = []
    for head in (False, True):
        lr = (args.head_lr or args.lr) if head else args.lr
        for dec in (True, False):
            ps = [p for n, p in model.named_parameters() if is_head(n) == head and (not any(x in n for x in no_decay)) == dec]
            if ps:
                params.append({'params': ps, 'weight_decay': args.wd if dec else 0.0, 'lr': lr})
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=device.type == 'cuda')
    probe = nd.length_batches(rows, args.max_tokens, args.batch, np.random.default_rng(args.seed))
    steps_per_epoch = len(probe)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warm = int(total_steps * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    eval_every = max(1, int(steps_per_epoch * args.eval_every))
    n_params = sum(p.numel() for p in model.parameters())
    log(f'steps/epoch {steps_per_epoch}, total {total_steps}, warmup {warm}, dev every {eval_every}, params {n_params / 1e6:.0f} M, '
        f'grad_accum {args.grad_accum}, grad_ckpt {args.grad_ckpt}')

    state = {'step': 0, 'best_f1': -1.0, 'bad': 0, 'history': [], 'train_loss_ema': None, 'elapsed': 0.0}
    ckpt = out_dir / 'ckpt.pt'
    if ckpt.exists():
        c = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(c['model'])
        opt.load_state_dict(c['opt'])
        sched.load_state_dict(c['sched'])
        state = c['state']
        log(f'resumed from step {state["step"]}')
        del c
    prev_elapsed = state.get('elapsed', 0.0)

    def save_ckpt():
        state['elapsed'] = prev_elapsed + time.time() - t_start
        tmp = out_dir / 'ckpt.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'state': state}, tmp)
        os.replace(tmp, ckpt)

    def save_best():
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = model_dir / 'model.tmp'
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, model_dir / 'model.pt')
        tok.save_pretrained(model_dir / 'tokenizer')
        (model_dir / 'config.json').write_text(json.dumps({'model': args.model, 'encoder_path': args.model, 'trust_remote_code': args.trust_remote_code,
                                                           'max_len': args.max_len, 'force_bos': args.force_bos, 'head_norm': args.head_norm, 'labels': {'LOC': ['O', 'B', 'I'], 'PER': ['O', 'B', 'I']},
                                                           'run': args.name, 'step': state['step'], 'dev_loc_f1': state['best_f1']}, indent=1))

    perf = {'model': args.model, 'params': n_params, 'grad_accum': args.grad_accum, 'grad_ckpt': args.grad_ckpt, 'batch': args.batch,
            'max_tokens': args.max_tokens, 'steps_per_epoch': steps_per_epoch, 'windows_per_epoch': int(len(rows)),
            'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', 'segments': []}
    seg = {'steps': 0, 'windows': 0, 'tokens': 0, 'padded_tokens': 0, 'seconds': 0.0, 'start_step': state['step']}
    perf_path = out_dir / 'perf.json'
    if perf_path.exists():
        try:
            perf['segments'] = json.loads(perf_path.read_text()).get('segments', [])
        except Exception:                           # noqa: BLE001
            pass

    def write_perf(final=False):
        s = dict(seg)
        if s['seconds'] > 0:
            s['steps_per_s'] = round(s['steps'] / s['seconds'], 3)
            s['windows_per_s'] = round(s['windows'] / s['seconds'], 2)
            s['tokens_per_s'] = round(s['tokens'] / s['seconds'], 1)
        if device.type == 'cuda':
            s['peak_alloc_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            s['peak_reserved_gb'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        s['final'] = final
        out = dict(perf)
        out['segments'] = perf['segments'] + [s]
        perf_path.write_text(json.dumps(out, indent=1))

    def dev_eval():
        t0 = time.time()
        pl, pp = predict_words(model, dev_enc, prefix, suffix, pad, limit, args.overlap, device)
        c, _ = score_words(dev_enc, pl, pp)
        rep = c.report()
        perf['dev_eval_seconds'] = round(time.time() - t0, 1)
        perf['dev_records'] = len(dev_enc)
        perf['dev_words'] = int(dev_enc.word_off[-1])
        model.train()
        return rep

    total_batches = total_steps
    last_ckpt = time.time()
    t_loop = time.time()
    model.train()
    stop_reason = None
    epoch0 = state['step'] // steps_per_epoch
    for epoch in range(epoch0, math.ceil(args.epochs)):
        batches = nd.length_batches(rows, args.max_tokens, args.batch, np.random.default_rng(args.seed * 1000 + epoch))
        skip = state['step'] - epoch * steps_per_epoch if epoch == epoch0 else 0
        for bi, idx in enumerate(batches):
            if bi < max(0, skip):
                continue
            if state['step'] >= total_steps:
                break
            t_step = time.time()
            b = nd.batch_windows(enc, rows[idx], prefix, suffix, pad, True)
            # counts of labelled words over the WHOLE batch (v1 normalisation), whatever the number of micro-batches
            n_loc = int((b['lab_loc'] != nd.IGNORE).sum())
            n_per = int((b['lab_per'] != nd.IGNORE).sum())
            if n_loc == 0 and n_per == 0:
                state['step'] += 1
                sched.step()
                continue
            loss_v = 0.0
            for part in np.array_split(np.arange(len(idx)), max(1, min(args.grad_accum, len(idx)))):
                ids = torch.from_numpy(b['input_ids'][part]).to(device, non_blocking=True)
                mask = torch.from_numpy(b['attention_mask'][part]).to(device, non_blocking=True)
                ll = torch.from_numpy(b['lab_loc'][part]).to(device)
                lp = torch.from_numpy(b['lab_per'][part]).to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    hl, hp = model(ids, mask)
                loss = 0.0
                for h, lab, n in ((hl, ll, n_loc), (hp, lp, n_per)):
                    if n > 0:
                        loss = loss + F.cross_entropy(h.float().view(-1, 3), lab.view(-1), ignore_index=nd.IGNORE, reduction='sum') / n
                if torch.is_tensor(loss):
                    loss.backward()
                    loss_v += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state['step'] += 1
            lv = loss_v
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seg['steps'] += 1
            seg['windows'] += len(idx)
            seg['tokens'] += int(b['attention_mask'].sum())
            seg['padded_tokens'] += int(b['attention_mask'].size)
            seg['seconds'] += time.time() - t_step
            state.setdefault('loss_log', []).append([state['step'], round(lv, 5)]) if state['step'] <= 2000 else None
            state['train_loss_ema'] = lv if state['train_loss_ema'] is None else 0.98 * state['train_loss_ema'] + 0.02 * lv
            if not math.isfinite(lv):
                raise SystemExit('loss is not finite')
            if state['step'] % 100 == 0 or state['step'] in (10, 50):
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0
                log(f'step {state["step"]}/{total_steps} epoch {epoch} loss {state["train_loss_ema"]:.4f} lr {sched.get_last_lr()[0]:.2e} '
                    f'{(time.time() - t_loop) / 60:.1f} min  {seg["windows"] / max(seg["seconds"], 1e-9):.1f} windows/s  peak {mem:.1f} GiB')
                write_perf()
            if state['step'] % eval_every == 0 or state['step'] == total_steps:
                rep = dev_eval()
                f1 = rep['LOC']['overall']['f1']
                pf1 = rep.get('PER', {}).get('overall', {}).get('f1')
                state['history'].append({'step': state['step'], 'epoch': round(state['step'] / steps_per_epoch, 2), 'dev_loc_f1': f1, 'dev_per_f1': pf1,
                                         'loss': round(state['train_loss_ema'], 4)})
                log(f'[dev] step {state["step"]} LOC F1 {f1} PER F1 {pf1}')
                if f1 > state['best_f1']:
                    state['best_f1'], state['bad'] = f1, 0
                    save_best()
                else:
                    state['bad'] += 1
                save_ckpt()
                last_ckpt = time.time()
                if state['bad'] >= args.patience:
                    stop_reason = 'early_stopping'
                    break
            elif time.time() - last_ckpt > args.ckpt_every_min * 60:
                save_ckpt()
                last_ckpt = time.time()
            if args.stop_at_step and state['step'] == args.stop_at_step:
                t0 = time.time()
                save_ckpt()
                perf['ckpt_save_seconds'] = round(time.time() - t0, 1)
                perf['ckpt_gb'] = round(ckpt.stat().st_size / 1e9, 2)
                write_perf()
                log(f'stop at step {state["step"]} (resume test), checkpoint saved in {time.time() - t0:.0f}s')
                return 3
            if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
                save_ckpt()
                write_perf()
                log('time limit reached, checkpoint saved')
                return 3
        if stop_reason or state['step'] >= total_steps:
            break
    if not (model_dir / 'model.pt').exists():
        save_best()
    write_perf(final=True)
    if args.skip_final_eval:
        (out_dir / 'metrics.json').write_text(json.dumps({'run': args.name, 'model': args.model, 'history': state['history'], 'steps': state['step'],
                                                          'loss_log': state.get('loss_log', []), 'probe': True}, indent=1))
        if ckpt.exists():
            ckpt.unlink()
        log('done (no final evaluation)')
        return 0

    # ------------------------------------------------------------------ final evaluation with the best model
    model.load_state_dict(torch.load(model_dir / 'model.pt', map_location=device))
    sets = dict(eval_sets)
    sets = {'dev': [args.dev], **sets}
    res = evaluate_sets(model, tok, sets, args, device, out_dir, save_preds)
    metrics = {'run': args.name, 'model': args.model, 'seed': args.seed, 'lr': args.lr, 'epochs': args.epochs, 'mix': mix, 'history': state['history'],
               'best_dev_loc_f1': state['best_f1'], 'steps': state['step'], 'stop': stop_reason or 'epochs_done',
               'loss_log': state.get('loss_log', [])[:2000], 'resumed': len(perf['segments']) > 0,
               'train_minutes': round((prev_elapsed + time.time() - t_start) / 60, 1), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
               'sets': res}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    done_file.write_text(json.dumps({'done': time.strftime('%F %T')}))
    if ckpt.exists():
        ckpt.unlink()
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
