"""Zero-shot place finder with a local instruction LLM (Qwen3.5-9B, as in the v1 evaluation), output in the format of ner_audit_predict.py.

Same prompt, same decoding (greedy, thinking off) and same marking rule as ``ratlas.ner.llm`` (every occurrence of an answered name is marked,
longest match wins). Added here: a confidence for each marked span, the mean probability of the tokens that spell the name inside the model's own
answer (a generation-time probability, not a calibrated one). The passages stay on the cluster; nothing leaves the machine.

    python scripts/ner2_predict_llm.py --model Qwen/Qwen3.5-9B --data passages.jsonl --out pred/llm_qwen35_9b.jsonl.gz --answers answers.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def name_confidences(answer: str, probs: list[float], offsets: list[tuple[int, int]], key: str = 'places') -> dict[str, float]:
    """Mean token probability of every quoted string inside the list ``key`` of the JSON-like answer. offsets: character span of each generated token."""
    m = re.search(r'"%s"\s*:\s*\[(.*?)(\]|$)' % key, answer, re.S)
    if not m:
        return {}
    out: dict[str, float] = {}
    base = m.start(1)
    for q in re.finditer(r'"((?:[^"\\]|\\.)*)"', m.group(1)):
        a, b = base + q.start(1), base + q.end(1)
        ps = [p for p, (s, e) in zip(probs, offsets) if e > a and s < b and e > s]
        if ps:
            name = q.group(1)
            c = sum(ps) / len(ps)
            out[name] = max(out.get(name, 0.0), c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--answers', help='jsonl with the raw answers (kept on the cluster)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=500)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ratlas.ner.llm import PROMPT, mark, parse_answer
    from ratlas.ner.zeroshot import snapshot

    path = snapshot(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda').eval()
    recs = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]
    prompts = []
    for r in recs:
        msgs = [{'role': 'user', 'content': PROMPT.format(text=' '.join(r['tokens']))}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    order = sorted(range(len(recs)), key=lambda i: len(prompts[i]))
    answers, confs = {}, {}
    t0 = time.time()
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        enc = tok([prompts[i] for i in idx], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, enc['input_ids'].shape[1]:]
        lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu()
        for k, i in enumerate(idx):
            ids = gen[k].tolist()
            n = len(ids)
            while n and ids[n - 1] == tok.pad_token_id:
                n -= 1
            ids = ids[:n]
            text = tok.decode(ids, skip_special_tokens=True)
            offs, prev = [], 0
            for j in range(len(ids)):
                cur = min(len(tok.decode(ids[:j + 1], skip_special_tokens=True).rstrip('\ufffd')), len(text))
                offs.append((prev, cur))
                prev = max(prev, cur)
            answers[recs[i]['id']] = text
            probs = [math.exp(float(x)) for x in lp[k, :n]]
            try:
                confs[recs[i]['id']] = name_confidences(text, probs, offs)
            except Exception:
                confs[recs[i]['id']] = {}
        log(f'{b + len(idx)}/{len(recs)} {time.time() - t0:.0f}s')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    with gzip.open(a.out, 'wt', encoding='utf-8') as f:
        for r in recs:
            places, persons, valid = parse_answer(answers[r['id']])
            n_valid += valid
            ck = {re.sub(r'\s+', '', k): v for k, v in confs[r['id']].items()}
            spans = []
            for s, e in sorted(mark(r['tokens'], places)):
                d = {'s': s, 'e': e}
                key = ''.join(r['tokens'][s:e])
                d['conf'] = round(ck[key], 4) if key in ck else None
                if r.get('token_starts'):
                    d['cs'], d['ce'] = r['token_starts'][s], r['token_starts'][e - 1] + len(r['tokens'][e - 1])
                    d['text'] = r['text'][d['cs']:d['ce']]
                else:
                    d['text'] = ' '.join(r['tokens'][s:e])
                spans.append(d)
            f.write(json.dumps({'id': r['id'], 'LOC': spans, 'PER': [], 'valid_json': valid}, ensure_ascii=False) + '\n')
    if a.answers:
        with open(a.answers, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps({'id': r['id'], 'answer': answers[r['id']]}, ensure_ascii=False) + '\n')
    log(f'written {a.out}; valid json {n_valid}/{len(recs)}; {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
