"""Open instruction LLM, zero-shot, on a few hundred passages of the test set (local GPU only, nothing leaves the machine).

  python -m ratlas.ner.llm sample --test .../test.jsonl --out .../llm_sample.jsonl --per-edition 60 --words 150 --seed 1
  python -m ratlas.ner.llm run --name zs_llm_qwen25_7b --model Qwen/Qwen2.5-7B-Instruct --sample .../llm_sample.jsonl --out RUNS

The sample is a JSONL in our record format (so that every other model can be scored on exactly the same passages with ``--set llm=...``): the first
``--words`` words of randomly chosen passages of each held-out edition, cut so as not to split a span. The model must return JSON with the place and the
person names copied from the text; each name is then marked wherever it occurs in the passage (exact match on the words, ignoring spaces).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import nerdata as nd
from .gliner_ft import chunk_bounds
from .spans import tags_to_spans
from .train import log
from .zeroshot import score_span_preds, snapshot

PROMPT = ('You are an expert in historical documents. Below is a passage from a transcribed historical text (16th-18th century or earlier; it may be in Latin, '
          'French, German, Italian, Dutch, Spanish or English, and may contain transcription noise and odd spacing). List every place name (towns, regions, '
          'countries, rivers, mountains, buildings that are named places, ...) and every person name (including names with titles or by-names) that occurs in it. '
          'Copy each name exactly as it is written in the passage. Do not translate, normalise or add names that are not in the passage. '
          'Answer with JSON only, in the form {{"places": [...], "persons": [...]}}.\n\nPassage:\n{text}')


def make_sample(args):
    recs = nd.read_slim(args.test)
    by_ed = defaultdict(list)
    for r in recs:
        if r.get('loc_valid') and len(r['tokens']) >= 40:
            by_ed[r['edition']].append(r)
    out = []
    rnd = random.Random(args.seed)
    for ed, rs in sorted(by_ed.items()):
        rs.sort(key=lambda r: r['id'])
        rnd.shuffle(rs)
        n = 0
        for r in rs:
            a, b = chunk_bounds(len(r['tokens']), r['ner_tags_loc'], r['ner_tags_per'], args.words)[0]
            c = dict(r)
            c['id'] = r['id'] + '#0'
            c['tokens'] = r['tokens'][a:b]
            c['ner_tags_loc'] = r['ner_tags_loc'][a:b]
            c['ner_tags_per'] = r['ner_tags_per'][a:b]
            if not tags_to_spans(c['ner_tags_loc']):        # keep passages with at least one place, as most passages of an edition have
                continue
            out.append(c)
            n += 1
            if n >= args.per_edition:
                break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    log(f'{len(out)} passages in {len(by_ed)} editions -> {args.out}')


def parse_answer(text: str):
    m = re.search(r'\{.*\}', text, re.S) or re.search(r'\{.*', text, re.S)      # the second form tolerates an answer cut off before the closing brace
    if not m:
        return [], [], False
    try:
        d = json.loads(m.group(0))
    except Exception:
        # tolerate a truncated or slightly broken answer: pull the quoted strings of each list
        d = {}
        for key in ('places', 'persons'):
            mm = re.search(r'"%s"\s*:\s*\[(.*?)(\]|$)' % key, m.group(0), re.S)
            d[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', mm.group(1)) if mm else []
        return d['places'], d['persons'], False
    clean = lambda xs: [x for x in (xs if isinstance(xs, list) else []) if isinstance(x, str) and x.strip()]
    return clean(d.get('places')), clean(d.get('persons')), True


def mark(tokens, names, maxlen=10):
    """Token spans of every occurrence of each name (ignoring spaces); longer matches win over the ones they contain."""
    keys = {re.sub(r'\s+', '', n) for n in names}
    keys.discard('')
    found = set()
    n = len(tokens)
    for i in range(n):
        acc = ''
        for j in range(i, min(n, i + maxlen)):
            acc += tokens[j]
            if acc in keys:
                found.add((i, j + 1))
    keep = set()
    for s, e in sorted(found, key=lambda x: (x[0], -(x[1] - x[0]))):
        if not any(k[0] <= s and e <= k[1] and (k[0], k[1]) != (s, e) for k in found):
            keep.add((s, e))
    return keep


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda').eval()
    recs = nd.read_slim(args.sample)
    prompts = []
    for r in recs:
        msgs = [{'role': 'user', 'content': PROMPT.format(text=' '.join(r['tokens']))}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    order = sorted(range(len(recs)), key=lambda i: len(prompts[i]))
    answers = {}
    t0 = time.time()
    for b in range(0, len(order), args.batch):
        idx = order[b:b + args.batch]
        enc = tok([prompts[i] for i in idx], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, o in zip(idx, out):
            answers[recs[i]['id']] = tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True)
        log(f'{b + len(idx)}/{len(recs)} {time.time() - t0:.0f}s')
    preds, ok = {}, 0
    for r in recs:
        places, persons, valid = parse_answer(answers[r['id']])
        ok += valid
        preds[r['id']] = {'LOC': mark(r['tokens'], places), 'PER': mark(r['tokens'], persons)}
    rep = score_span_preds(recs, preds).report()
    rep['n_records'] = len(recs)
    d = Path(args.out) / args.name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'answers.jsonl', 'w') as f:        # the model outputs contain passage text: kept on Hypatia only
        for r in recs:
            f.write(json.dumps({'id': r['id'], 'answer': answers[r['id']]}, ensure_ascii=False) + '\n')
    (d / 'metrics.json').write_text(json.dumps({'run': args.name, 'model': args.model, 'kind': 'llm_zeroshot', 'prompt': PROMPT, 'valid_json': ok, 'n': len(recs),
                                                'minutes': round((time.time() - t0) / 60, 1), 'sets': {'llm_sample': rep}}, indent=1))
    (d / 'done.json').write_text('{}')
    l, p = rep['LOC']['overall'], rep['PER']['overall']
    log(f'LOC P/R/F {l["p"]}/{l["r"]}/{l["f1"]}  PER {p["p"]}/{p["r"]}/{p["f1"]}  valid json {ok}/{len(recs)}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('sample')
    a.add_argument('--test', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--per-edition', type=int, default=60)
    a.add_argument('--words', type=int, default=150)
    a.add_argument('--seed', type=int, default=1)
    b = sub.add_parser('run')
    b.add_argument('--name', required=True)
    b.add_argument('--model', required=True)
    b.add_argument('--sample', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--batch', type=int, default=16)
    b.add_argument('--max-new', type=int, default=500)
    args = ap.parse_args(argv)
    if args.cmd == 'sample':
        make_sample(args)
    else:
        run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
