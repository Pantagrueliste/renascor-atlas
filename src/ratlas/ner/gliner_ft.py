"""Fine-tune GLiNER on our data (label prompts 'place' / 'person'; each passage lists only the labels it annotates) and evaluate it like the zero-shot models.

    python -m ratlas.ner.gliner_ft --name gliner_ft_ours --model urchade/gliner_multi-v2.1 --train ours=.../train.jsonl --set dev=... --set test=... --out RUNS --models MODELS

Passages are cut into chunks of ``--chunk-words`` words (never inside a span), at most ``--max-samples`` chunks are drawn at random with the seed, trained for
``--epochs`` epochs, one checkpoint per epoch; the checkpoint with the best dev LOC F1 (best threshold) is kept.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

from . import nerdata as nd
from .spans import tags_to_spans
from .train import expand, load_set, log, parse_named
from .zeroshot import filter_threshold, gliner_predict, parse_labels, read_set, run_gliner, score_span_preds, snapshot

PROMPTS = {'LOC': 'place', 'PER': 'person'}


def chunk_bounds(n, tags_loc, tags_per, size):
    """Chunk [a, b) bounds of about ``size`` words that never cut a span."""
    out, a = [], 0
    while a < n:
        b = min(n, a + size)
        while b < n and (tags_loc[b] == 2 or tags_per[b] == 2):
            b += 1
        out.append((a, b))
        a = b
    return out


def to_samples(recs, size):
    samples = []
    for r in recs:
        toks = r['tokens']
        lv, pv = r.get('loc_valid', True), r.get('per_valid', True)
        if not (lv or pv):
            continue
        tl, tp = r['ner_tags_loc'], r['ner_tags_per']
        for a, b in chunk_bounds(len(toks), tl, tp, size):
            ner = []
            if lv:
                ner += [[s - a, e - a - 1, PROMPTS['LOC']] for s, e in tags_to_spans(tl[a:b])]
            if pv:
                ner += [[s - a, e - a - 1, PROMPTS['PER']] for s, e in tags_to_spans(tp[a:b])]
            labels = ([PROMPTS['LOC']] if lv else []) + ([PROMPTS['PER']] if pv else [])
            samples.append({'tokenized_text': toks[a:b], 'ner': ner, 'ner_labels': labels})
    return samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--train', action='append', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--chunk-words', type=int, default=100)
    ap.add_argument('--max-samples', type=int, default=100000)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--others-lr', type=float, default=5e-5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--thresholds', default='0.1,0.2,0.3,0.5,0.7')
    ap.add_argument('--gl-batch', type=int, default=16)
    ap.add_argument('--chunk-eval', dest='chunk_words_eval', type=int, default=100)
    args = ap.parse_args(argv)
    import torch
    from gliner import GLiNER
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / 'trainer'
    recs, mix = load_set(parse_named(args.train), args.seed, 'train')
    samples = to_samples(recs, args.chunk_words)
    del recs
    random.Random(args.seed).shuffle(samples)
    samples = samples[:args.max_samples]
    log(f'{len(samples)} training chunks')
    sets = {n: expand(p) for n, p, _ in parse_named(args.set)}
    dev_recs = read_set(sets['dev'])
    dev_chunks = to_samples(dev_recs[:1500], args.chunk_words)[:3000]

    model = GLiNER.from_pretrained(snapshot(args.model), map_location='cpu').to('cuda')
    steps = max(1, len(samples) // args.batch) * args.epochs
    targs = model.create_training_args(output_dir=str(work), learning_rate=args.lr, others_lr=args.others_lr, weight_decay=0.01, others_weight_decay=0.01,
                                       warmup_ratio=0.1, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch, bf16=True,
                                       max_steps=-1, num_train_epochs=args.epochs, save_strategy='epoch', save_total_limit=args.epochs, logging_steps=100,
                                       dataloader_num_workers=2, seed=args.seed, eval_strategy='no')
    log(f'training {args.epochs} epochs, about {steps} steps')
    model.train_model(samples, dev_chunks, training_args=targs)
    log('training done', round((time.time() - t0) / 60, 1), 'min')
    ckpts = sorted(work.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    log('checkpoints', [c.name for c in ckpts])

    # best checkpoint on the dev set
    ths = sorted(float(x) for x in args.thresholds.split(','))
    labels = {t: [p] for t, p in PROMPTS.items()}
    best, hist = None, []
    del model
    torch.cuda.empty_cache()
    for c in ckpts:
        m = GLiNER.from_pretrained(str(c), map_location='cpu').to('cuda').eval()
        preds = gliner_predict(m, dev_recs, labels, args.chunk_words_eval, ths[0], args.gl_batch)
        scores = {t: score_span_preds(dev_recs, filter_threshold(preds, t)).report()['LOC']['overall']['f1'] for t in ths}
        f = max(scores.values())
        hist.append({'checkpoint': c.name, 'dev_loc_f1_by_threshold': scores})
        log(c.name, scores)
        if best is None or f > best[0]:
            best = (f, c)
        del m
        torch.cuda.empty_cache()
    dest = Path(args.models) / args.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(best[1], dest)
    log('best checkpoint', best[1].name, best[0])
    ns = argparse.Namespace(model=str(dest), labels='LOC=place,PER=person', thresholds=args.thresholds, chunk_words=args.chunk_words_eval, gl_batch=args.gl_batch, variant=None, split_ws=True)
    res, extra = run_gliner(ns, sets)
    (out_dir / 'metrics.json').write_text(json.dumps({'run': args.name, 'model': args.model, 'kind': 'gliner_finetuned', 'mix': mix, 'train_chunks': len(samples),
                                                      'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed, 'checkpoints': hist, 'best_checkpoint': best[1].name,
                                                      **extra, 'minutes': round((time.time() - t0) / 60, 1), 'sets': res}, indent=1))
    (out_dir / 'done.json').write_text('{}')
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
