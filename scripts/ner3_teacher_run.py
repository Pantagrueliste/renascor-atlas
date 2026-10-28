"""Teacher run on dev3 (or a silver pool): an open instruction LLM, JSON answer, resumable, throughput logged. Passages and answers stay on Hypatia.

    python scripts/ner3_teacher_run.py --model Qwen/Qwen3.8-27B --data dev3/passages.jsonl --prompt guided --out $DATA/ner/v3/teacher/qwen38_27b_guided
Prompts: zero = the v1 prompt (ratlas.ner.llm.PROMPT); guided = owner rules of the guidelines + few-shot (ner3_teacher_fewshot.json, invented examples, not from dev3 or gold v2).
Never run with gold v2 as --data.
"""
from __future__ import annotations

import argparse, gzip, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
HERE = Path(__file__).resolve().parent

RULES = """You annotate named entities in Renaissance texts (1450-1700; Latin, Italian, Spanish, French, German, English; spelling is old, OCR noise possible).
PLACES: towns, regions, countries, rivers, mountains, islands, named buildings used as places. Rules:
- A place inside a person's name (e.g. 'Maria of Burgundy') is not a place; list the person only.
- Adjectives in -ensis / ethnic adjectives are places only if they name the place ('Parisiensis' -> yes).
- Metonymy such as 'the king of France' or 'the Court' (la Corte, Palacio) is not a place.
- A name merely quoted as a word, an image or a description is not a place.
PERSONS: individuals, with titles or by-names ('Messer Andrea Doria', 'Fra Girolamo Savonarola'); not gods, groups, or generic roles.
Copy every name exactly as written in the passage (same spelling, same spaces). Do not translate or normalise. Do not add names that are not in the passage.
Answer with JSON only: {{"places": [...], "persons": [...]}}."""


def build_prompt(kind, text, lang, shots):
    from ratlas.ner.llm import PROMPT
    if kind == 'zero':
        return [{'role': 'user', 'content': PROMPT.format(text=text)}]
    msgs = [{'role': 'system', 'content': RULES}]
    for s in shots:
        msgs.append({'role': 'user', 'content': 'Passage:\n' + s['text']})
        msgs.append({'role': 'assistant', 'content': json.dumps({'places': s['places'], 'persons': s['persons']}, ensure_ascii=False)})
    msgs.append({'role': 'user', 'content': 'Passage:\n' + text})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt', choices=['zero', 'guided'], default='zero')
    ap.add_argument('--engine', choices=['vllm', 'hf'], default='vllm')
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--max-new', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    assert 'gold' not in Path(a.data).name.lower() or 'dev3' in str(a.data), 'gold v2 is the final test set'
    from ratlas.ner.llm import mark, parse_answer
    from ratlas.ner.zeroshot import snapshot
    recs = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    shots = json.load(open(HERE / 'ner3_teacher_fewshot.json'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ans_path = out / 'answers.jsonl'
    done = {}
    if ans_path.exists():
        for l in open(ans_path, encoding='utf-8'):
            try:
                d = json.loads(l); done[d['id']] = d['answer']
            except Exception:
                pass
    todo = [r for r in recs if r['id'] not in done]
    t0 = time.time()
    path = snapshot(a.model)
    if todo:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        prompts = [tok.apply_chat_template(build_prompt(a.prompt, ' '.join(r['tokens']), r['lang'], shots), tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in todo]
        if a.engine == 'vllm':
            from vllm import LLM, SamplingParams
            llm = LLM(path, tensor_parallel_size=a.tp, dtype='bfloat16', max_model_len=8192, gpu_memory_utilization=0.92)
            t1 = time.time()
            res = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=a.max_new))
            answers = [x.outputs[0].text for x in res]
        else:
            import torch
            from transformers import AutoModelForCausalLM
            tok.padding_side = 'left'
            tok.pad_token = tok.pad_token or tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to('cuda').eval()
            t1 = time.time(); answers = []
            order = sorted(range(len(todo)), key=lambda i: len(prompts[i])); tmp = {}
            for b in range(0, len(order), 8):
                idx = order[b:b + 8]
                enc = tok([prompts[i] for i in idx], return_tensors='pt', padding=True).to('cuda')
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                for i, x in zip(idx, o):
                    tmp[i] = tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True)
            answers = [tmp[i] for i in range(len(todo))]
        gen_s = time.time() - t1
        with open(ans_path, 'a', encoding='utf-8') as f:
            for r, an in zip(todo, answers):
                done[r['id']] = an
                f.write(json.dumps({'id': r['id'], 'answer': an}, ensure_ascii=False) + '\n')
        (out / 'throughput.json').write_text(json.dumps({'model': a.model, 'prompt': a.prompt, 'engine': a.engine, 'tp': a.tp, 'n': len(todo),
                                                         'generation_s': round(gen_s, 1), 'passages_per_hour': round(len(todo) / gen_s * 3600), 'total_s': round(time.time() - t0, 1)}))
    ok = 0
    with gzip.open(out / 'pred.jsonl.gz', 'wt', encoding='utf-8') as fo:
        for r in recs:
            pl, pe, valid = parse_answer(done[r['id']]); ok += valid
            fo.write(json.dumps({'id': r['id'], 'valid_json': valid, 'LOC': [{'s': s, 'e': e} for s, e in sorted(mark(r['tokens'], pl))],
                                 'PER': [{'s': s, 'e': e} for s, e in sorted(mark(r['tokens'], pe))]}) + '\n')
    print(f'valid json {ok}/{len(recs)}')


if __name__ == '__main__':
    main()
