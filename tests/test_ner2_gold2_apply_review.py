import importlib.util, pathlib

spec = importlib.util.spec_from_file_location(
    "apply_review", pathlib.Path(__file__).parents[1] / "scripts" / "ner2_gold2_apply_review.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)

PROP = {"id": "p", "lang": "en", "text": "to Mildenhall River and Ely", "tokens": ["to", "Mildenhall", "River", "and", "Ely"],
        "token_starts": [0, 3, 14, 20, 24],
        "spans": [{"start": 1, "end": 2, "status": "rejected"}, {"start": 1, "end": 3, "status": "agreed"},
                  {"start": 4, "end": 5, "status": "to_check"}]}


def test_review_overrides_and_alternatives():
    rec, c = ar.apply_review(PROP, {"sp": {"1-2": "y"}, "done": True})
    st = {(s["start"], s["end"]): (s["status"], s["entity"]) for s in rec["spans"]}
    assert st[(1, 2)][0] == "place" and st[(1, 3)][0] == "place" and st[(4, 5)][0] == "doubt"
    assert st[(1, 2)][1] == st[(1, 3)][1]
    assert c[("rejected", "place", "owner_changed")] == 1


def test_alternatives_count_once_and_doubt_is_neutral():
    rec, _ = ar.apply_review(PROP, {"sp": {"1-2": "y"}})
    assert ar.score_passage(rec, {(1, 2), (1, 3), (4, 5)}) == (1, 0, 0)
    assert ar.score_passage(rec, set()) == (0, 0, 1)
    assert ar.score_passage(rec, {(0, 1)}) == (0, 1, 1)


def test_added_span():
    rec, c = ar.apply_review(PROP, {"add": [[0, 1]]})
    added = [s for s in rec["spans"] if s["decided_by"] == "owner_added"]
    assert added and added[0]["surface"] == "to" and c[(None, "place", "owner_added")] == 1


def test_owner_rule_overrides_review():
    rec, c = ar.apply_review(PROP, {"sp": {"1-2": "y"}}, {(1, 2): ("n", "patronyme")})
    s = next(s for s in rec["spans"] if (s["start"], s["end"]) == (1, 2))
    assert s["status"] == "not_place" and s["decided_by"] == "owner_rule" and s["rule_reason"] == "patronyme"
    assert c[("rejected", "not_place", "owner_rule")] == 1
