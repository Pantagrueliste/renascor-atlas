"""Silver assembly: one teacher split over several raw files (jobs, GPU shards), +pp, stats, then the train.py files."""
import importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ratlas.ner import nerdata as nd  # noqa: E402
from ratlas.ner.spans import tags_to_spans  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def passage(i, words, coll="C1"):
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    return {"id": f"p{i}", "lang": "la", "collection": coll, "doc": f"d{i}", "text": " ".join(words), "tokens": words, "token_starts": starts}


def raw(i, spans, error=None, unmatched=(), finish="stop"):
    return {"id": f"p{i}", "model": "m", "spans": [{"start": s, "end": e, "label": l} for s, e, l in spans], "unmatched": list(unmatched),
            "error": error, "finish_reason": finish}


def write(p, recs):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def test_multi_file_teacher_pp_stats_and_final(tmp_path, monkeypatch):
    comb, fin = load("ner3_silver_combine"), load("ner3_silver_final")
    ps = [passage(0, ["Romae", "et", "Romae", "vidi", "Petrum", "."]),
          passage(1, ["in", "Lond", ".", "scripsit"]),
          passage(2, ["nihil", "hic"]),
          passage(3, ["Paulus", "dixit"]),
          passage(4, ["Venetiis", "datum"])]
    pf = tmp_path / "pool.jsonl"
    write(pf, ps)
    rd = tmp_path / "raw"
    rd.mkdir()
    write(rd / "t.jsonl", [raw(0, [(0, 1, "LOC"), (4, 6, "PER")]), raw(3, [], error="invalid_json")])
    write(rd / "t.s00.jsonl", [raw(1, [(1, 3, "LOC")], unmatched=["Oxon"]), raw(3, [(0, 1, "PER")])])  # better line for p3 in a later shard
    write(rd / "t.s01.jsonl", [raw(2, []), raw(4, [], error="timeout", finish="length")])
    spec = f"{rd}/t.jsonl,{rd}/t.s*.jsonl"
    assert [pathlib.Path(f).name for f in comb.raw_files(spec)] == ["t.jsonl", "t.s00.jsonl", "t.s01.jsonl"]
    with pytest.raises(FileNotFoundError):
        comb.raw_files(f"{rd}/none*.jsonl")
    out, st = tmp_path / "silver.jsonl", tmp_path / "stats.json"
    monkeypatch.setattr(sys, "argv", ["x", "--passages", str(pf), "--raw", spec, "--out", str(out), "--stats", str(st)])
    comb.main()
    s = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert [(x["start"], x["end"], x["label"]) for x in s["p0"]["spans"]] == [(0, 1, "LOC"), (2, 3, "LOC"), (4, 5, "PER")]  # propagated, dot dropped
    assert [(x["start"], x["end"]) for x in s["p1"]["spans"]] == [(1, 2)]
    assert s["p3"]["silver_ok"] and s["p3"]["spans"][0]["label"] == "PER"
    assert s["p2"]["silver_ok"] and s["p2"]["spans"] == [] and not s["p4"]["silver_ok"]
    t = json.loads(st.read_text())["teachers"]["m"]
    assert (t["ok"], t["error"], t["missing"], t["empty_answer"], t["truncated"], t["unmatched_entities"], t["passages_with_unmatched"]) == (4, 1, 0, 1, 1, 1, 1)

    tr = tmp_path / "silver_balanced_la.jsonl"
    rep = fin.main(["--silver", str(out), "--subset", "balanced", "--out", str(tr)])
    assert (rep["passages"], rep["not_silver_ok_dropped"], rep["LOC"], rep["PER"], rep["passages_without_entity"]) == (4, 1, 3, 2, 1)
    recs = nd.read_slim(tr)
    r0 = next(r for r in recs if r["id"] == "p0")
    assert r0["ner_tags_loc"] == [1, 0, 1, 0, 0, 0] and r0["ner_tags_per"] == [0, 0, 0, 0, 1, 0]
    assert r0["loc_valid"] and r0["per_valid"] and r0["edition"] == "silver/C1" and r0["subset"] == "balanced"
    assert all(tags_to_spans(r["ner_tags_loc"]) is not None for r in recs)


def test_final_rejects_overlap(tmp_path):
    fin = load("ner3_silver_final")
    rec = passage(0, ["a", "b", "c"]) | {"spans": [{"start": 0, "end": 2, "label": "LOC", "text": "a b", "char_start": 0, "char_end": 3},
                                                     {"start": 1, "end": 3, "label": "PER", "text": "b c", "char_start": 2, "char_end": 5}],
                                         "silver_ok": True}
    with pytest.raises(ValueError):
        fin.to_train(rec, "balanced")


def test_verify_exclusion_patterns():
    v = load("ner3_silver_verify")
    assert v.exclusion_hits({"collection": "LondonLives", "id": "x"}) == ["london_lives"]
    assert v.exclusion_hits({"collection": "map_of_early_modern_london", "id": "x"}) == ["map"]
    assert v.exclusion_hits({"collection": "CIDTC", "id": "silver/la/CIDTC/a#1"}) == ["private"]
    assert v.exclusion_hits({"collection": "NOSCEMUS", "doc": "noscemus_1_celtisconradusodae", "id": "silver/la/NOSCEMUS/x#2"}) == []


def test_final_drop_ids(tmp_path):
    fin = load("ner3_silver_final")
    recs = [passage(i, ["Roma", "est"]) | {"silver_ok": True, "spans": [{"start": 0, "end": 1, "label": "LOC", "text": "Roma", "char_start": 0, "char_end": 4}]}
            for i in range(3)]
    sil = tmp_path / "s.jsonl"
    write(sil, recs + [recs[0]])
    drop = tmp_path / "shared.jsonl"
    write(drop, [{"id": "p1", "set": "v1_dev", "shared": 40}, {"id": "p1", "set": "v1_test"}])
    rep = fin.main(["--silver", str(sil), "--subset", "reserve", "--out", str(tmp_path / "o.jsonl"), "--drop-ids", str(drop)])
    assert (rep["passages"], rep["leak_dropped"], rep["duplicate_id_dropped"]) == (2, 1, 1)
    assert rep["leak_dropped_by_set"] == {"v1_dev": 1, "v1_test": 1}
