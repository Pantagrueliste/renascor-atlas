"""Annotate passages with LOC and PER using an LLM served by Ollama (local or :cloud models).

Usage:
  python3 scripts/renner_llm_teacher.py --model mistral-large-4:cloud \
      --passages passages.jsonl [--prompt rules|fewshot] [--guide rules.md] [--fewshot fewshot.jsonl] \
      --out preds.jsonl [--workers 4] [--think false|true|low|medium|high] [--limit N]

Prompts:
  rules   = system prompt + the condensed guidelines (--guide, default scripts/renner_prompt_rules_en.md), zero-shot.
  fewshot = the same + the worked examples of --fewshot (default scripts/renner_fewshot_v1train.jsonl: 8 passages of the v1
            TRAINING split, la/es/it, re-annotated to the guideline conventions) as user/assistant turns.
  Without --prompt, the old behaviour holds: --guide is required and --fewshot is used only if given.

Input passages: one JSON per line with id, lang, text, tokens, token_starts (as in gold v2 and dev3).
Output: one JSON per line {id, model, prompt, spans: [{start, end, label, text, char_start, char_end}], unmatched, raw, error,
        attempts, seconds, prompt_tokens, eval_tokens}. Token spans are end-exclusive token indices, like the gold files.

The model is asked for the exact surface strings in reading order; each string is located in the text after the previous match
(exact match first, then a match that tolerates different whitespace, then a case-insensitive one), then snapped to token
boundaries. Strings that cannot be located are dropped and listed in "unmatched". The run is resumable: ids already in --out are
skipped, except those that ended with a transport error (rerun with --retry-errors).
"""
import argparse, concurrent.futures as cf, json, os, re, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GUIDE = HERE / "renner_prompt_rules_en.md"
DEFAULT_FEWSHOT = HERE / "renner_fewshot_v1train.jsonl"
LABELS = {"LOC", "PER"}
SYSTEM = """You annotate named entities in early modern texts (1450-1700) for a Renaissance NER corpus.
Follow the annotation guidelines below exactly. Return ONLY JSON: {"entities": [{"text": "...", "label": "LOC"|"PER"}, ...]}
- "text" must be copied exactly as it appears in the passage (same spelling, same characters), in reading order;
  repeat an entity each time it occurs.
- Labels: LOC (places) and PER (persons). Nothing else.
- If there is no entity, return {"entities": []}.

GUIDELINES
"""
_cooldown = {"until": 0.0}          # shared back-off after a rate limit or a server error
_cool_lock = threading.Lock()
_stop = threading.Event()          # set when the account quota is exhausted: the remaining passages are left for a later resume


class QuotaError(RuntimeError):
    """HTTP 429 that names a usage limit (not a short rate limit): retrying now is useless."""


def load_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def user_msg(p):
    return f"Language: {p.get('lang', '?')}\nPassage:\n<<<\n{p['text']}\n>>>"


def build_messages(p, guide, shots):
    msgs = [{"role": "system", "content": SYSTEM + guide}]
    for s in shots:
        ents = [{"text": x["text"], "label": x["label"]} for x in s["entities"]]
        msgs.append({"role": "user", "content": user_msg(s)})
        msgs.append({"role": "assistant", "content": json.dumps({"entities": ents}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_msg(p)})
    return msgs


def think_value(s):
    s = str(s).lower()
    return {"false": False, "true": True}.get(s, s)


def call_ollama(host, model, messages, timeout=600, retries=5, think=False, stats=None):
    """One chat call; retries with exponential back-off (shared by all workers). Returns the answer text.

    ``stats`` (a dict) receives attempts, prompt_tokens, eval_tokens and the errors met on the way."""
    body = {"model": model, "messages": messages, "stream": False, "think": think, "options": {"temperature": 0}}
    if think is False:
        body["format"] = "json"
    data = json.dumps(body).encode()
    stats = stats if stats is not None else {}
    stats.setdefault("errors", [])
    last = None
    for k in range(retries):
        wait = _cooldown["until"] - time.time()
        if wait > 0:
            time.sleep(wait)
        stats["attempts"] = k + 1
        try:
            req = urllib.request.Request(f"{host}/api/chat", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            if d.get("error"):
                raise RuntimeError(d["error"])
            stats["prompt_tokens"], stats["eval_tokens"] = d.get("prompt_eval_count"), d.get("eval_count")
            return d["message"]["content"]
        except Exception as e:  # network errors, 429, 5xx
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode("utf8", "replace")[:200]
                except Exception:
                    detail = ""
                last = f"HTTP {e.code} {detail}".strip()
            else:
                last = f"{type(e).__name__}: {e}"[:300]
            stats["errors"].append(last)
            if isinstance(e, urllib.error.HTTPError) and e.code == 429 and "limit" in last.lower() and "reached" in last.lower():
                raise QuotaError(last)
            pause = min(120, 5 * 2 ** k)
            with _cool_lock:
                _cooldown["until"] = max(_cooldown["until"], time.time() + pause)
            time.sleep(pause)
    raise RuntimeError(f"ollama call failed: {last}")


def _candidates(raw):
    """JSON values found in an answer: the whole text, then every object that parses from a '{'."""
    dec = json.JSONDecoder()
    out = []
    try:
        out.append(json.loads(raw))
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r"[\[{]", raw):
        try:
            v, _ = dec.raw_decode(raw[m.start():])
            out.append(v)
        except json.JSONDecodeError:
            continue
    return out


def parse(raw):
    """Entities of an answer, or None if no JSON answer is found. Reasoning before '</think>' and code fences are ignored;
    when several JSON objects are present, the last one holding "entities" wins."""
    if raw is None:
        return None
    raw = raw.split("</think>")[-1]
    raw = re.sub(r"```(?:json)?", "", raw)
    found = None
    for d in _candidates(raw):
        if isinstance(d, dict) and isinstance(d.get("entities"), list):
            found = d["entities"]
        elif found is None and isinstance(d, list) and all(isinstance(x, dict) for x in d):
            found = d
    if found is None:
        return None
    return [e for e in found if isinstance(e, dict) and e.get("label") in LABELS and str(e.get("text", "")).strip()]


def _find(text, s, pos):
    """(start, end) of s in text at or after pos, or None. Tried in order: exact; any whitespace between the words; the same
    ignoring case; the letters alone, allowing spaces, hyphens and soft hyphens between them (a word split in the source)."""
    i = text.find(s, pos)
    if i >= 0:
        return i, i + len(s)
    words = s.split()
    if not words:
        return None
    loose = r"\s+".join(re.escape(w) for w in words)
    chars = r"[\s\-\u00ad]*".join(re.escape(c) for c in "".join(words))
    for pat, flags in ((loose, 0), (loose, re.I), (chars, re.I)):
        m = re.compile(pat, flags).search(text, pos)
        if m:
            return m.start(), m.end()
    return None


def locate(p, ents):
    """Map surface strings to token spans, scanning left to right; returns (spans, unmatched)."""
    text, starts, toks = p["text"], p["token_starts"], p["tokens"]
    ends = [s + len(t) for s, t in zip(starts, toks)]
    spans, unmatched, pos = [], [], 0
    for e in ents:
        s = str(e["text"]).strip()
        hit = _find(text, s, pos)
        if hit is None:
            hit = _find(text, s, 0)  # out of order: fall back to first occurrence
        if hit is None:
            unmatched.append(s)
            continue
        i, j = hit
        a = next((k for k, (ts, te) in enumerate(zip(starts, ends)) if te > i), None)
        b = next((k for k in range(len(toks) - 1, -1, -1) if starts[k] < j), None)
        if a is None or b is None or b < a:
            unmatched.append(s)
            continue
        span = {"start": a, "end": b + 1, "label": e["label"], "text": " ".join(toks[a:b + 1]),
                "char_start": starts[a], "char_end": ends[b]}
        if all((x["start"], x["end"]) != (span["start"], span["end"]) for x in spans):
            spans.append(span)
        pos = j
    return sorted(spans, key=lambda x: (x["start"], x["end"])), unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--prompt", choices=["rules", "fewshot"], help="rules = zero-shot guidelines; fewshot = guidelines + examples")
    ap.add_argument("--guide")
    ap.add_argument("--fewshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--think", default="false", help="false (default), true, or a level (low, medium, high) for models that reason")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retry-errors", action="store_true", help="rerun the ids whose previous attempt ended with a transport error")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.prompt:
        a.guide = a.guide or str(DEFAULT_GUIDE)
        if a.prompt == "fewshot":
            a.fewshot = a.fewshot or str(DEFAULT_FEWSHOT)
        else:
            a.fewshot = None
    if not a.guide:
        ap.error("--guide is required without --prompt")
    if "gold" in Path(a.passages).name.lower() and "dev3" not in a.passages:
        ap.error("gold v2 is the final test set: never annotate it for teacher selection")
    guide = open(a.guide, encoding="utf8").read()
    shots = load_jsonl(a.fewshot) if a.fewshot else []
    think = think_value(a.think)
    passages = load_jsonl(a.passages)[: a.limit]
    done = set()
    if os.path.exists(a.out):
        keep = []
        for l in open(a.out, encoding="utf8"):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if a.retry_errors and d.get("error") and d["error"] != "invalid_json":
                continue
            done.add(d["id"])
            keep.append(l if l.endswith("\n") else l + "\n")
        if a.retry_errors:
            with open(a.out, "w", encoding="utf8") as f:
                f.writelines(keep)
    todo = [p for p in passages if p["id"] not in done]
    lock = threading.Lock()
    prompt_name = a.prompt or ("fewshot" if shots else "guide")
    print(f"{len(todo)} to annotate ({len(done)} already done) with {a.model}, prompt {prompt_name}, {len(shots)} shots",
          file=sys.stderr)

    def work(p):
        if _stop.is_set():
            return None
        t0 = time.time()
        rec = {"id": p["id"], "model": a.model, "prompt": prompt_name, "spans": [], "unmatched": [], "raw": None, "error": None}
        st = {}
        try:
            try:
                rec["raw"] = call_ollama(a.host, a.model, build_messages(p, guide, shots), timeout=a.timeout, think=think, stats=st)
            except QuotaError as q:
                _stop.set()
                print(f"quota exhausted, stopping (passage left for a resume): {q}", file=sys.stderr)
                return None
            ents = parse(rec["raw"])
            if ents is None:
                rec["error"] = "invalid_json"
            else:
                rec["spans"], rec["unmatched"] = locate(p, ents)
        except Exception as e:
            rec["error"] = str(e)
        rec["seconds"] = round(time.time() - t0, 2)
        rec["t_end"] = round(time.time(), 1)
        rec["attempts"] = st.get("attempts")
        rec["retry_errors"] = st.get("errors", [])
        rec["prompt_tokens"], rec["eval_tokens"] = st.get("prompt_tokens"), st.get("eval_tokens")
        with lock, open(a.out, "a", encoding="utf8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n = err = 0
    t_start = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(work, todo):
            if rec is None:
                continue
            n += 1
            err += rec["error"] is not None
            if n % 20 == 0:
                print(f"{n}/{len(todo)} done, {err} errors, {time.time() - t_start:.0f} s", file=sys.stderr)
    print(f"finished: {n} passages, {err} errors, {time.time() - t_start:.0f} s wall", file=sys.stderr)
    if _stop.is_set():
        print("STOPPED_ON_QUOTA", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
