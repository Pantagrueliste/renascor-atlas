"""renner_vllm_teacher.py with a fake vLLM: guide and few-shot options, resumable --out, and output readable by the silver tools."""
import importlib.util, json, pathlib, sys, types

SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Out:
    def __init__(self, text):
        self.text, self.finish_reason, self.token_ids = text, "stop", [0, 1]


class _Res:
    def __init__(self, text):
        self.outputs, self.prompt_token_ids = [_Out(text)], [0] * 5


def fake_vllm(calls):
    class LLM:
        def __init__(self, **kw):
            pass

        def chat(self, msgs, sp, use_tqdm=False, chat_template_kwargs=None):
            calls.append(msgs)
            res = []
            for m in msgs:
                text = m[-1]["content"].split("<<<\n")[1].split("\n>>>")[0]
                ents = [{"text": w, "label": "LOC"} for w in text.split() if w[:1].isupper()]
                res.append(_Res(json.dumps({"entities": ents})))
            return res

    mod = types.ModuleType("vllm")
    mod.LLM, mod.SamplingParams = LLM, lambda **kw: kw
    sp = types.ModuleType("vllm.sampling_params")
    sp.StructuredOutputsParams = lambda **kw: kw
    return mod, sp


def passage(i, text):
    toks, starts, pos = [], [], 0
    for w in text.split(" "):
        toks.append(w); starts.append(pos); pos += len(w) + 1
    return {"id": f"p{i}", "lang": "it", "text": text, "tokens": toks, "token_starts": starts}


def run(monkeypatch, argv, calls):
    mod, sp = fake_vllm(calls)
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    t = load("renner_vllm_teacher")
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    t.main()


def test_guides_out_resume_and_combine(tmp_path, monkeypatch):
    ps = [passage(i, f"da Roma a Napoli {i}") for i in range(5)]
    pf = tmp_path / "passages.jsonl"
    pf.write_text("".join(json.dumps(p) + "\n" for p in ps))
    g2, g3 = tmp_path / "g2.md", tmp_path / "g3.md"
    g2.write_text("RULES V2"); g3.write_text("RULES V3")
    calls = []
    run(monkeypatch, ["--model", "m", "--name", "n", "--passages", str(pf), "--prompts", "fewshot", "--outdir", str(tmp_path / "o"),
                      "--guide", f"v2={g2}", "--guide", f"v3={g3}"], calls)
    assert sorted(f.name for f in (tmp_path / "o").glob("*.jsonl")) == ["n__fewshotv2.jsonl", "n__fewshotv3.jsonl"]
    assert calls[0][0][0]["content"].endswith("RULES V2") and calls[1][0][0]["content"].endswith("RULES V3")
    assert len(calls[0][0]) == 2 * 8 + 2  # system, 8 examples, the passage

    out = tmp_path / "silver.jsonl"
    out.write_text(json.dumps({"id": "p0", "model": "m", "spans": [], "error": None}) + "\n")  # already done
    calls.clear()
    run(monkeypatch, ["--model", "m", "--name", "n", "--passages", str(pf), "--prompts", "fewshot", "--out", str(out),
                      "--guide", str(g3), "--batch", "2"], calls)
    assert [len(c) for c in calls] == [2, 2]
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in recs] == ["p0", "p1", "p2", "p3", "p4"]
    assert recs[1]["guide"] == "g3.md" and recs[1]["prompt"] == "fewshot"
    assert [(s["start"], s["end"], s["label"]) for s in recs[1]["spans"]] == [(1, 2, "LOC"), (3, 4, "LOC")]
    assert (tmp_path / "silver.throughput.json").exists()

    comb = load("ner3_silver_combine")
    raw = comb.load_raw(out)
    assert set(raw) == {f"p{i}" for i in range(5)}
    assert comb.bk.sc.to_pred(raw["p2"]) == {"id": "p2", "LOC": [{"s": 1, "e": 2}, {"s": 3, "e": 4}], "PER": []}
    monkeypatch.setattr(sys, "argv", ["x", "--passages", str(pf), "--raw", str(out), "--out", str(tmp_path / "s.jsonl")])
    comb.main()
    s = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert len(s) == 5 and s[3]["silver_ok"] and len(s[3]["spans"]) == 2 and s[3]["teacher"]["models"] == ["m"]
