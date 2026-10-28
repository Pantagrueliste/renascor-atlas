"""NER v2: Qwen3.5-9B (zero-shot, local GPU) on a sample of the repaired test passages, in the prediction format of ner_audit_zs.py.

Extends the v1 sample (200 passages per edition, ``samples/llm_sample_1200.jsonl``, first 150 words) to ``--per-edition`` passages per edition: the passages of the
v1 sample come first (matched by their v1 id), the others are drawn at random (seed). The whole passage (<= 200 words) is sent, with the v1 prompt and the v1
decoding (greedy, thinking off, JSON answer). Names returned by the model are marked wherever they occur in the passage (ratlas.ner.llm.mark).
Answers are appended to ``<out>.answers.jsonl`` while the job runs (resumable); the model outputs contain passage text and stay on Hypatia.

    python scripts/ner2_llm.py --data test_v2_loc.jsonl --v1-sample samples/llm_sample_1200.jsonl --per-edition 600 --out pred/zs_llm_qwen35_9b.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def choose(recs, v1_ids, per_edition, seed, min_tokens=40):
    """Passages to send: the v1 sample first, then random ones up to ``per_edition`` per edition."""
    by_ed = defaultdict(list)
    for r in recs:
        if len(r['tokens']) >= min_tokens:
            by_ed[r['edition']].append(r)
    out = []
    rnd = random.Random(seed)
    for ed, rs in sorted(by_ed.items()):
        rs.sort(key=lambda r: r['id'])
        first = [r for r in rs if r.get('v1_id', r['id']) in v1_ids]
        rest = [r for r in rs if r.get('v1_id', r['id']) not in v1_ids]
        rnd.shuffle(rest)
        out += (first + rest)[:max(per_edition, len(first))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--v1-sample', default=None)
    ap.add_argument('--per-edition', type=int, default=600)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-new', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ratlas.ner.llm import PROMPT, mark, parse_answer
    from ratlas.ner.zeroshot import snapshot

    recs = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]
    v1_ids = set()
    if a.v1_sample and Path(a.v1_sample).exists():
        v1_ids = {json.loads(l)['id'].split('#')[0] for l in open(a.v1_sample, encoding='utf-8') if l.strip()}
    sel = choose(recs, v1_ids, a.per_edition, a.seed)
    if a.limit:
        sel = sel[:a.limit]
    log(f'{len(sel)} passages selected ({sum(1 for r in sel if r.get("v1_id", r["id"]) in v1_ids)} of the v1 sample)')
    ans_path = Path(a.out + '.answers.jsonl')
    done = {}
    if ans_path.exists():
        for l in open(ans_path, encoding='utf-8'):
            try:
                d = json.loads(l)
                done[d['id']] = d['answer']
            except Exception:       # noqa: BLE001  a half-written last line
                pass
    log(f'{len(done)} answers already there')
    todo = [r for r in sel if r['id'] not in done]
    if todo:
        path = snapshot(a.model)
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = 'left'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda').eval()
        prompts = {}
        for r in todo:
            msgs = [{'role': 'user', 'content': PROMPT.format(text=' '.join(r['tokens']))}]
            prompts[r['id']] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        order = sorted(todo, key=lambda r: len(prompts[r['id']]))
        t0 = time.time()
        with open(ans_path, 'a', encoding='utf-8') as fa:
            for b in range(0, len(order), a.batch):
                part = order[b:b + a.batch]
                enc = tok([prompts[r['id']] for r in part], return_tensors='pt', padding=True).to('cuda')
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                for r, o in zip(part, out):
                    ans = tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True)
                    done[r['id']] = ans
                    fa.write(json.dumps({'id': r['id'], 'answer': ans}, ensure_ascii=False) + '\n')
                fa.flush()
                if (b // a.batch) % 10 == 0:
                    log(f'{b + len(part)}/{len(order)} {time.time() - t0:.0f}s')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with gzip.open(a.out, 'wt', encoding='utf-8') as fo:
        for r in sel:
            places, persons, valid = parse_answer(done[r['id']])
            ok += valid
            toks, ts, text = r['tokens'], r['token_starts'], r['text']
            o = {'id': r['id']}
            for task, names in (('LOC', places), ('PER', persons)):
                lst = []
                for s, e in sorted(mark(toks, names)):
                    cs, ce = ts[s], ts[e - 1] + len(toks[e - 1])
                    lst.append({'s': s, 'e': e, 'conf': 1.0, 'cs': cs, 'ce': ce, 'text': text[cs:ce]})
                o[task] = lst
            fo.write(json.dumps(o, ensure_ascii=False) + '\n')
    log(f'written {a.out}: {len(sel)} passages, valid JSON {ok}')


if __name__ == '__main__':
    main()
