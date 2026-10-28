import gzip
import importlib.util
import json
import pathlib

spec = importlib.util.spec_from_file_location("eb", pathlib.Path(__file__).parents[1] / "scripts" / "ner_editor_bench_score.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)

EDITIONS = {
    "exh": {"main_score": True, "loc_tagged": True, "per_tagged": True, "per_match": "head_overlap",
            "measures": {"LOC_editor": "R+P", "LOC_v2": "R+P", "PER": "R+P"}},
    "part": {"main_score": True, "loc_tagged": True, "per_tagged": True, "per_match": "overlap",
             "measures": {"LOC_editor": "R", "LOC_v2": None, "PER": "R"}},
    "sec": {"main_score": False, "loc_tagged": True, "per_tagged": False, "per_match": "overlap",
            "measures": {"LOC_editor": "R (biaisé)", "LOC_v2": None, "PER": None}},
}


def rec(rid, edition, doc, words, loc=(), per=(), lang="de", ignored=(), orgs=(), nested=(), v2=None, mask=(), loc_valid=None):
    text = " ".join(words)
    starts, p = [], 0
    for w in words:
        starts.append(p)
        p += len(w) + 1
    n = len(words)

    def tags(spans):
        t = [0] * n
        for s, e in spans:
            t[s] = 1
            for k in range(s + 1, e):
                t[k] = 2
        return t

    def chars(s, e):
        return starts[s], starts[e - 1] + len(words[e - 1])

    def cspans(spans, lab):
        return [{"start": chars(s, e)[0], "end": chars(s, e)[1], "label": lab} for s, e in spans]

    return {"id": rid, "edition": edition, "doc": doc, "lang": lang, "text": text, "tokens": list(words), "token_starts": starts,
            "spans": cspans(loc, "LOC") + cspans(per, "PER"), "nested": [dict(x, label=l) for (s, e, l) in nested for x in cspans([(s, e)], l)],
            "ignored": [dict(x, label=l) for (s, e, l) in ignored for x in cspans([(s, e)], l)], "orgs": [[x["start"], x["end"]] for x in cspans(orgs, "ORG")],
            "ner_tags_loc_editor": tags(loc), "ner_tags_per_editor": tags(per),
            "loc_gold_v2": [list(x) for x in v2] if v2 is not None else None, "loc_mask": list(mask),
            "loc_valid_v1": loc_valid, "per_valid_v1": None}


W = "a b c d e f g h i j".split()


def write(tmp, recs, preds):
    b = tmp / "bench.jsonl"
    b.write_text("".join(json.dumps(r) + "\n" for r in recs))
    e = tmp / "editions.json"
    e.write_text(json.dumps(EDITIONS))
    args = ["--bench", str(b), "--editions", str(e), "--out", str(tmp / "out.json"), "--boot", "50"]
    for name, rows in preds.items():
        p = tmp / f"{name}.jsonl.gz"
        with gzip.open(p, "wt") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
        args += ["--pred", f"{name}={p}"]
    return args


def test_match_modes():
    g = {(2, 5)}
    assert eb.match(g, {(2, 5)}, "strict") == (1, 1, 1, 1)
    assert eb.match(g, {(3, 4)}, "strict") == (0, 1, 0, 1)
    assert eb.match(g, {(3, 4)}, "overlap") == (1, 1, 1, 1)
    assert eb.match(g, {(3, 4)}, "head") == (0, 1, 0, 1)          # misses the first token of the editor span
    assert eb.match(g, {(2, 3)}, "head") == (1, 1, 1, 1)
    assert eb.match(g, {(2, 3), (6, 7)}, "overlap") == (1, 1, 1, 2)


def test_measure_parsing():
    assert eb.measure("R+P") == (True, True)
    assert eb.measure("R") == (True, False)
    assert eb.measure("R (biaisé)") == (True, False)
    assert eb.measure("R+P (tête du nom, recouvrement)") == (True, True)
    assert eb.measure(None) == (False, False)


def test_teacher_format():
    d = eb.to_pred({"id": "x", "spans": [{"start": 1, "end": 2, "label": "LOC"}, {"start": 3, "end": 5, "label": "PER"}], "error": None})
    assert d == {"LOC": {(1, 2)}, "PER": {(3, 5)}}
    d = eb.to_pred({"id": "x", "LOC": [{"s": 1, "e": 2, "conf": 0.9}], "PER": []})
    assert d == {"LOC": {(1, 2)}, "PER": set()}


def test_recall_everywhere_precision_only_exhaustive(tmp_path):
    recs = [rec("x1", "exh", "d1", W, loc=[(0, 1), (4, 6)], per=[(7, 9)], v2=[(0, 1), (4, 6), (9, 10)]),
            rec("p1", "part", "d2", W, loc=[(2, 3)], per=[(5, 6)]),
            rec("s1", "sec", "d3", W, loc=[(1, 2)])]
    preds = {"m": [{"id": "x1", "LOC": [{"s": 0, "e": 1}, {"s": 4, "e": 5}, {"s": 9, "e": 10}], "PER": [{"s": 7, "e": 8}]},
                   {"id": "p1", "LOC": [{"s": 2, "e": 3}, {"s": 8, "e": 10}], "PER": []},
                   {"id": "s1", "LOC": [], "PER": []}]}
    out = eb.main(write(tmp_path, recs, preds))
    s = out["scores"]["m"]
    le = s["LOC_editor"]["strict"]["main"]
    assert le["recall"]["gold"] == 3 and le["recall"]["found"] == 2          # x1: 1 of 2; p1: 1 of 1; sec not in main
    assert le["exhaustive"]["gold"] == 2 and le["exhaustive"]["pred"] == 3 and le["exhaustive"]["right"] == 1
    assert s["LOC_editor"]["strict"]["all"]["recall"]["gold"] == 4
    assert s["LOC_editor"]["overlap"]["main"]["exhaustive"]["found"] == 2
    assert s["LOC_v2"]["strict"]["main"]["recall"]["gold"] == 3
    # PER: x1 rule head_overlap -> (7, 8) covers the first token of (7, 9)
    assert s["PER"]["strict"]["main"]["exhaustive"]["right"] == 0
    assert s["PER"]["rule"]["main"]["exhaustive"]["right"] == 1
    assert s["PER"]["rule"]["main"]["recall"]["gold"] == 2 and s["PER"]["rule"]["main"]["recall"]["found"] == 1
    assert "edition=sec" not in s["PER"]["strict"] and "edition=sec" in s["LOC_editor"]["strict"]
    nf = out["new_finds"]["m"]
    assert nf["total"]["LOC"]["v2_confirmed"] == 1 and "v2_rejected" not in nf["total"]["LOC"]   # x1 (9, 10): in the v2 gold only
    assert nf["total"]["LOC"]["new"] == 1                                           # p1 (8, 10)


def test_ignored_masked_and_new_find_categories(tmp_path):
    recs = [rec("p1", "part", "d1", W, loc=[(0, 1)], per=[(3, 4)], ignored=[(5, 6, "PER")], orgs=[(7, 8)], nested=[(9, 10, "LOC")])]
    preds = {"m": [{"id": "p1", "LOC": [{"s": 3, "e": 4}, {"s": 7, "e": 8}, {"s": 9, "e": 10}, {"s": 1, "e": 2}],
                    "PER": [{"s": 5, "e": 6}, {"s": 3, "e": 4}]}]}
    args = write(tmp_path, recs, preds) + ["--samples-dir", str(tmp_path / "s")]
    out = eb.main(args)
    t = out["new_finds"]["m"]["total"]
    assert t["LOC"] == {"predicted": 4, "other_label": 1, "org": 1, "new": 1, "no_editor_tag": 3}   # nested (9, 10) is an editor tag
    assert t["PER"] == {"predicted": 2}                                                       # (5, 6) masked, (3, 4) tagged
    assert out["scores"]["m"]["PER"]["strict"]["main"]["recall"]["pred"] == 1
    rows = [json.loads(l) for l in open(tmp_path / "s" / "m__new_finds.jsonl")]
    assert {r["category"] for r in rows} == {"other_label", "org", "new"}
    assert rows[0]["text"] in W


def test_coverage_subset_and_paired(tmp_path):
    recs = [rec(f"x{i}", "exh", f"d{i}", W, loc=[(0, 1)], per=[(2, 3)]) for i in range(6)]
    good = [{"id": f"x{i}", "LOC": [{"s": 0, "e": 1}], "PER": []} for i in range(6)]
    half = [{"id": f"x{i}", "LOC": [{"s": 0, "e": 1}] if i < 3 else [], "PER": []} for i in range(4)]
    out = eb.main(write(tmp_path, recs, {"good": good, "half": half}) + ["--ref", "good"])
    assert out["meta"]["n_passages_scored"] == 4                                     # passages covered by every system
    sub = tmp_path / "ids.txt"
    sub.write_text("\n".join(f"x{i}" for i in range(6)))
    out = eb.main(write(tmp_path, recs, {"good": good, "half": half}) + ["--ref", "good", "--subset", str(sub)])
    assert out["meta"]["n_passages_scored"] == 6
    assert out["scores"]["half"]["LOC_editor"]["strict"]["main"]["recall"]["r"] == 0.5
    d = out["paired_vs_ref"]["diffs"]["half"]["LOC_editor"]["strict"]["main:r"]
    assert d["diff"] == -0.5 and d["ci95"][1] <= 0
