"""Open teachers on Hypatia with vLLM (offline batch), with the SAME prompts as the Ollama runner (renner_llm_teacher.py).

    python scripts/renner_vllm_teacher.py --model google/gemma-4-31B-it --name gemma4_31b --passages dev3/passages.jsonl \
        --prompts rules,fewshot --outdir $D/v3/teacher_vllm [--tp 2] [--mistral] [--max-model-len 12288]

The messages (system prompt + condensed rules, and the 8 v1-train worked examples for "fewshot"), the answer parser and the
location of the strings in the text are imported from renner_llm_teacher.py, so a vLLM run and an Ollama run are comparable.
Decoding: temperature 0, no reasoning (enable_thinking false / reasoning_effort none), the answer constrained to the JSON object
{"entities": [{"text", "label": LOC|PER}]} (Ollama uses format=json, also a constrained decoder).
Output: <outdir>/<name>__<prompt><tag>.jsonl (same records as the Ollama runner) and <outdir>/<name>__<prompt><tag>.throughput.json
(passages/s over the whole batch, model load excluded). The model is loaded once for all prompts and guides. Never run on gold v2.

Guide and examples, as in the Ollama runner: --guide PATH (default scripts/renner_prompt_rules_en.md) and --fewshot PATH (default
scripts/renner_fewshot_v1train.jsonl). --guide may be repeated as TAG=PATH to run several prompt versions with one model load:
    --prompts fewshot --guide v2=scripts/renner_prompt_rules_en_v2.md --guide v3=scripts/renner_prompt_rules_en_v3.md
writes <name>__fewshotv2.jsonl and <name>__fewshotv3.jsonl (the names of the Ollama dev3 checks).
--guide-fewshot TAG=PATH gives one guide its own worked examples (v4: --guide-fewshot v4=scripts/renner_fewshot_v4.jsonl); the
other guides keep --fewshot. With --out and a plain PATH guide (no tag), use --fewshot.

Plain baseline (reviewer question: a general-purpose LLM with a plain prompt): --prompts plain sends one user message, PLAIN_PROMPT +
the passage, with no system prompt, no RenNER guidelines and no examples (--guide is ignored); same decoding and parser as above.
Output <name>__plain.jsonl.

Silver runs: --out FILE (one prompt, one guide) writes to FILE instead of <outdir>; ids already in FILE are skipped (resume), and
--batch N sends the passages N at a time, appending after each batch, so a job stopped by its time limit keeps its finished batches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import renner_llm_teacher as T  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "label": {"type": "string", "enum": ["LOC", "PER"]}},
        "required": ["text", "label"]}}},
    "required": ["entities"],
}

PLAIN_PROMPT = ("Extract the names of places and persons from the text below. Answer in JSON only: "
                '{"entities": [{"text": "...", "label": "LOC" or "PER"}]} (LOC = place, PER = person), '
                "with each name copied exactly as it is written in the text.\n\nText:\n")


def plain_messages(p):
    return [{"role": "user", "content": PLAIN_PROMPT + p["text"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--prompts", default="rules,fewshot")
    ap.add_argument("--outdir")
    ap.add_argument("--out", help="output file (one prompt and one guide only); resumable")
    ap.add_argument("--guide", action="append", help="PATH or TAG=PATH (repeatable); default the v1 condensed rules")
    ap.add_argument("--fewshot", default=str(T.DEFAULT_FEWSHOT), help="worked examples for the fewshot prompt")
    ap.add_argument("--guide-fewshot", action="append", default=[],
                    help="TAG=PATH (repeatable): worked examples for the guide TAG instead of --fewshot (e.g. v4=scripts/renner_fewshot_v4.jsonl)")
    ap.add_argument("--batch", type=int, default=0, help="passages per llm.chat call (0 = all at once)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--mistral", action="store_true", help="Mistral consolidated checkpoint: tokenizer/config/load format mistral")
    ap.add_argument("--no-schema", action="store_true", help="JSON object mode instead of the entity schema")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--quantization", help="e.g. fp8 (on-the-fly weight quantization, Marlin kernels on A100)")
    ap.add_argument("--text-only", action="store_true", help="multimodal checkpoint (Gemma 4, Qwen 3.5/3.6, Mistral 3): language model only")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if "gold" in Path(a.passages).name.lower() and "dev3" not in a.passages:
        ap.error("gold v2 is the final test set")
    guides = []
    for g in a.guide or [str(T.DEFAULT_GUIDE)]:
        tag, _, path = g.rpartition("=") if "=" in g else ("", "", g)
        guides.append((tag, path))
    prompts = a.prompts.split(",")
    if a.out and (len(prompts) != 1 or len(guides) != 1):
        ap.error("--out takes exactly one prompt and one guide")
    if not a.out and not a.outdir:
        ap.error("--outdir or --out is required")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    passages = T.load_jsonl(a.passages)[: a.limit]
    shots_all = T.load_jsonl(a.fewshot)
    fewshot_by_tag = dict(x.split("=", 1) for x in a.guide_fewshot)
    shots_by_tag = {t: T.load_jsonl(fp) for t, fp in fewshot_by_tag.items()}
    out = Path(a.outdir) if a.outdir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    kw = dict(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              dtype="auto", seed=0, enable_prefix_caching=True, trust_remote_code=True, enforce_eager=a.enforce_eager)
    if a.mistral:
        kw.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.text_only:
        kw["language_model_only"] = True
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    print(f"[{a.name}] model loaded in {load_s:.0f} s", file=sys.stderr, flush=True)

    so = StructuredOutputsParams(json_object=True) if a.no_schema else StructuredOutputsParams(json=SCHEMA)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, structured_outputs=so, seed=0)
    ctk = {"enable_thinking": False, "reasoning_effort": "none"}
    use_ctk = [True]

    def chat(msgs):
        if use_ctk[0]:
            try:
                return llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs=ctk)
            except Exception as e:  # a template that rejects an argument: retry without the reasoning switches
                print(f"[{a.name}] chat with template kwargs failed ({type(e).__name__}: {e}); retrying without", file=sys.stderr)
                use_ctk[0] = False
        return llm.chat(msgs, sp, use_tqdm=False)

    for prompt, (tag, gpath) in [(pr, g) for pr in prompts for g in guides]:
        plain = prompt == "plain"
        if plain:
            tag, gpath = "", "none"
        guide = "" if plain else open(gpath, encoding="utf8").read()
        shots = shots_by_tag.get(tag, shots_all) if prompt == "fewshot" else []
        path = Path(a.out) if a.out else out / f"{a.name}__{prompt}{tag}.jsonl"
        done = set()
        if a.out and path.exists():
            for l in open(path, encoding="utf8"):
                try:
                    done.add(json.loads(l)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        todo = [p for p in passages if p["id"] not in done]
        print(f"[{a.name}] {prompt}{tag}: {len(todo)} to annotate ({len(done)} already in {path.name}), guide {gpath}",
              file=sys.stderr, flush=True)
        step = a.batch if a.batch > 0 else max(len(todo), 1)
        n_err = n_trunc = ptok = gtok = 0
        gen_s = 0.0
        with open(path, "a" if a.out else "w", encoding="utf8") as f:
            for b0 in range(0, len(todo), step):
                chunk = todo[b0:b0 + step]
                msgs = [plain_messages(p) if plain else T.build_messages(p, guide, shots) for p in chunk]
                t1 = time.time()
                res = chat(msgs)
                gen_s += time.time() - t1
                for p, r in zip(chunk, res):
                    o = r.outputs[0]
                    rec = {"id": p["id"], "model": a.model, "prompt": prompt, "guide": Path(gpath).name, "engine": "vllm",
                   **({"fewshot": Path(fewshot_by_tag[tag]).name} if tag in fewshot_by_tag else {}),
                           "spans": [], "unmatched": [], "raw": o.text, "error": None, "finish_reason": o.finish_reason,
                           "prompt_tokens": len(r.prompt_token_ids or []), "eval_tokens": len(o.token_ids or [])}
                    ptok += rec["prompt_tokens"]
                    gtok += rec["eval_tokens"]
                    n_trunc += o.finish_reason == "length"
                    ents = T.parse(o.text)
                    if ents is None:
                        rec["error"] = "invalid_json"
                        n_err += 1
                    else:
                        rec["spans"], rec["unmatched"] = T.locate(p, ents)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if a.batch > 0:
                    print(f"[{a.name}] {prompt}{tag}: {b0 + len(chunk)}/{len(todo)} done, {gen_s:.0f} s", file=sys.stderr, flush=True)
        n_done = len(todo)
        thr = {"name": a.name, "model": a.model, "prompt": prompt + tag, "guide": gpath, "n_shots": len(shots), "n_passages": n_done, "tp": a.tp,
               "gpu": os.environ.get("R3_GPU", ""), "load_seconds": round(load_s, 1), "gen_seconds": round(gen_s, 1),
               "passages_per_second": round(n_done / gen_s, 3) if gen_s else None, "batch": a.batch, "prompt_tokens": ptok, "gen_tokens": gtok,
               "invalid_json": n_err, "truncated": n_trunc, "max_model_len": a.max_model_len, "schema": not a.no_schema,
               "quantization": a.quantization or "none"}
        tpath = path.with_name(path.name[: -len(".jsonl")] + ".throughput.json") if path.name.endswith(".jsonl") else Path(str(path) + ".throughput.json")
        if n_done or not tpath.exists():
            json.dump(thr, open(tpath, "w"), indent=1)
        print(f"[{a.name}] {prompt}{tag}: {json.dumps(thr)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
