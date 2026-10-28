import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("sc", pathlib.Path(__file__).parents[1] / "scripts" / "ner3_teacher_score.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

GOLD = {"id": "x", "lang": "it", "spans": [
    {"label": "PER", "start": 2, "end": 3, "alt_start": 1, "certainty": "sure"},
    {"label": "PER", "start": 5, "end": 6, "certainty": "doubt"},
    {"label": "LOC", "start": 8, "end": 9, "certainty": "sure"}]}


def test_alt_form_accepted_only_with_flag():
    g_plain = sc.gold_items(GOLD, "PER")
    g_alt = sc.gold_items(GOLD, "PER", accept_alt=True)
    assert sc.count_items(g_plain, {(1, 3)}, "strict") == (0, 1, 2)
    assert sc.count_items(g_alt, {(1, 3)}, "strict") == (1, 0, 1)


def test_sure_only_ignores_doubt_spans():
    g = sc.gold_items(GOLD, "PER", accept_alt=True)
    assert sc.count_items(g, {(2, 3), (5, 6)}, "strict", sure_only=True) == (1, 0, 0)
    assert sc.count_items(g, {(5, 6)}, "strict", sure_only=False) == (1, 0, 1)
    assert sc.count_items(g, {(5, 7)}, "overlap", sure_only=True) == (0, 0, 1)


def test_teacher_output_format_and_vote():
    d = sc.to_pred({"id": "x", "spans": [{"start": 2, "end": 3, "label": "PER"}, {"start": 8, "end": 9, "label": "LOC"}]})
    assert d["PER"] == [{"s": 2, "e": 3}] and d["LOC"] == [{"s": 8, "e": 9}]
    preds = {"a": {"x": d}, "b": {"x": {"id": "x", "LOC": [], "PER": [{"s": 2, "e": 3}]}}, "c": {"x": {"id": "x", "LOC": [{"s": 8, "e": 9}], "PER": []}}}
    v = sc.combine("vote:a,b,c@2", preds)["x"]
    assert v["PER"] == [{"s": 2, "e": 3}] and v["LOC"] == [{"s": 8, "e": 9}]
