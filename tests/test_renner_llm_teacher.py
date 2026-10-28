import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("t", pathlib.Path(__file__).parents[1] / "scripts" / "renner_llm_teacher.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def passage(text):
    toks, starts, pos = [], [], 0
    for w in text.split(" "):
        toks.append(w); starts.append(pos); pos += len(w) + 1
    return {"id": "x", "lang": "it", "text": text, "tokens": toks, "token_starts": starts}


def test_locate_repeats_and_order():
    p = passage("da Roma a Napoli , poi Roma di nuovo")
    spans, un = t.locate(p, [{"text": "Roma", "label": "LOC"}, {"text": "Napoli", "label": "LOC"}, {"text": "Roma", "label": "LOC"}])
    assert [(s["start"], s["end"]) for s in spans] == [(1, 2), (3, 4), (6, 7)] and not un


def test_locate_multiword_and_unmatched():
    p = passage("il conte di San Marco venne")
    spans, un = t.locate(p, [{"text": "San Marco", "label": "LOC"}, {"text": "Venezia", "label": "LOC"}])
    assert [(s["start"], s["end"]) for s in spans] == [(3, 5)] and un == ["Venezia"]


def test_parse_filters_labels_and_recovers_json():
    raw = 'Here: {"entities": [{"text": "Roma", "label": "LOC"}, {"text": "X", "label": "ORG"}]}'
    assert t.parse(raw) == [{"text": "Roma", "label": "LOC"}]


def test_parse_think_tag_fences_and_last_object():
    raw = 'reasoning {"x": 1} ... </think>```json\n{"entities": [{"text": "Roma", "label": "LOC"}]}\n```'
    assert t.parse(raw) == [{"text": "Roma", "label": "LOC"}]
    raw2 = 'Draft: {"entities": [{"text": "A", "label": "PER"}]} Final: {"entities": [{"text": "B", "label": "PER"}]}'
    assert t.parse(raw2) == [{"text": "B", "label": "PER"}]
    assert t.parse("no json here") is None
    assert t.parse('{"entities": []}') == []


def test_locate_tolerant_whitespace_case_and_split_word():
    p = passage("per Fran conia in  Suevia e roma")
    spans, un = t.locate(p, [{"text": "Franconia", "label": "LOC"}, {"text": "in Suevia", "label": "LOC"}, {"text": "Roma", "label": "LOC"}])
    assert [(s["start"], s["end"]) for s in spans] == [(1, 3), (3, 6), (7, 8)] and not un


def test_build_messages_with_shots():
    shot = {"lang": "la", "text": "Romae", "entities": [{"text": "Romae", "label": "LOC"}]}
    m = t.build_messages({"lang": "it", "text": "Roma"}, "RULES", [shot])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"] and m[0]["content"].endswith("RULES")
