"""Owner-rule view of the editor bench (scripts/ner_owner_rules.py, ner_editor_bench_score.py --owner-rules)."""
import importlib.util
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(HERE))
import ner_owner_rules as OR  # noqa: E402

spec = importlib.util.spec_from_file_location("eb", HERE / "ner_editor_bench_score.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)


def passage(text, lang="it"):
    """Tokenise like the bench (word or single sign); return a record with tokens and token_starts."""
    import re
    ms = list(re.finditer(r"\w+|[^\w\s]", text))
    return {"id": "x", "lang": lang, "text": text, "tokens": [m.group() for m in ms], "token_starts": [m.start() for m in ms]}


def span(r, phrase, start_at=0):
    """Token span of the first occurrence of phrase (by characters)."""
    cs = r["text"].index(phrase, start_at)
    ce = cs + len(phrase)
    ks = [k for k, s in enumerate(r["token_starts"]) if cs <= s < ce]
    return ks[0], ks[-1] + 1


def toks(r, sp):
    return " ".join(r["tokens"][sp[0]:sp[1]]) if sp else None


def per(r, phrase, edition="x"):
    log = Counter()
    m, adds = OR.map_per(r, span(r, phrase), log, edition)
    return toks(r, m), [toks(r, a) for a in adds], log


def loc(r, phrase, edition="x"):
    log = Counter()
    return toks(r, OR.map_loc(r, span(r, phrase), log, edition)), log


def test_period_trimmed():
    r = passage("Scrive da Lond. e da Clemens V. al re.")
    assert loc(r, "Lond.")[0] == "Lond"
    assert per(r, "Clemens V.")[0] == "Clemens V"
    r = passage("Venne M. Luther a Wittenberg.")
    assert per(r, "M. Luther")[0] == "M . Luther"          # internal period kept


def test_title_place_designates_person():
    r = passage("Il Sor Duca di Nevers sta meglio.")
    name, adds, log = per(r, "Sor Duca di Nevers")
    assert name is None and adds == ["Nevers"] and log["title_place"] == 1
    r = passage("le roi d’Angleterre a écrit", "fr")
    assert per(r, "roi d’Angleterre", "ferdinand_i")[:2] == (None, ["Angleterre"])
    r = passage("the L of essex came", "en")
    assert per(r, "L of essex", "reed")[:2] == (None, ["essex"])   # reed: L = Lord; lower-case place at the end


def test_address_is_not_a_dignity():
    r = passage("Monsieur de Montaigne écrit", "fr")
    assert per(r, "Monsieur de Montaigne")[0] == "de Montaigne"
    r = passage("venne don Fadrique ieri", "es")
    assert per(r, "don Fadrique")[0] == "Fadrique"


def test_kinship_and_title_only():
    r = passage("Mein brueder ist hier", "de")
    assert per(r, "Mein brueder")[0] is None
    r = passage("wife of Iohn Burges paid", "en")
    assert per(r, "wife of Iohn Burges")[0] == "Iohn Burges"
    r = passage("Kays Mayt hat befohlen", "de")
    assert per(r, "Kays Mayt")[0] is None


def test_tail_cut_and_tail_place():
    r = passage("Francis, Duke of Anjou, came", "en")
    name, adds, _ = per(r, "Francis, Duke of Anjou")
    assert name == "Francis" and adds == ["Anjou"]
    r = passage("Hans König kam", "de")
    assert per(r, "Hans König")[0] == "Hans König"          # a capitalised title word after a name is a surname
    r = passage("Morgane Pope paid", "en")
    assert per(r, "Morgane Pope")[0] == "Morgane Pope"


def test_koenigsfelden_name_only():
    r = passage("ersamen Heinrich Suter , hofmeister zu Kûngsfeld hat", "de")
    name, adds, _ = per(r, "ersamen Heinrich Suter , hofmeister zu Kûngsfeld", "koenigsfelden")
    assert name == "Heinrich Suter" and adds == ["Kûngsfeld"]
    r = passage("Heini Fricker der jung von Altemburg", "de")
    assert per(r, "Heini Fricker der jung von Altemburg", "koenigsfelden")[0] == "Heini Fricker"
    r = passage("junckher Peter von Griffense", "de")
    assert per(r, "junckher Peter von Griffense", "koenigsfelden")[:2] == ("Peter von Griffense", [])   # lineage name kept


def test_regnal_split():
    r = passage("Felipe II de España mandó", "es")
    name, adds, log = per(r, "Felipe II de España")
    assert name == "Felipe II" and adds == ["España"] and log["regnal"] == 1
    r = passage("Margherita d'Austria venne")
    assert per(r, "Margherita d'Austria")[:2] == ("Margherita d ' Austria", [])   # no numeral: origin name kept


def test_god_personification_people():
    r = passage("dieu le voulut et la fortune aussi", "fr")
    assert per(r, "dieu")[0] is None and per(r, "fortune")[0] is None
    r = passage("le Turc assiège", "fr")
    assert per(r, "le Turc")[0] is None
    r = passage("Iesu Christo nacque")
    assert per(r, "Iesu Christo")[0] == "Iesu Christo"      # Christ stays PER
    r = passage("d ieu tout puissant", "fr")
    name, _, log = per(r, "d ieu", "elan_crenum")
    assert name is None and log["dropcap"] == 1 and log["god"] == 1


def test_people_and_generic_loc():
    r = passage("die franzosen und die Römische kirchen in Wien", "de")
    assert loc(r, "franzosen")[0] is None and loc(r, "Römische")[0] is None and loc(r, "Wien")[0] == "Wien"
    r = passage("in Hispaniarum et Polonicis rebus Paderbornensis", "la")
    assert loc(r, "Hispaniarum")[0] == "Hispaniarum" and loc(r, "Polonicis")[0] is None
    assert loc(r, "Paderbornensis")[0] == "Paderbornensis"   # -ensis kept
    r = passage("at the Church and Paris garden", "en")
    assert loc(r, "Church")[0] is None and loc(r, "Paris garden")[0] == "Paris garden"


def _bench_rec(rid, edition, words, loc=(), per=()):
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
    return {"id": rid, "edition": edition, "doc": rid, "lang": "it", "text": text, "tokens": list(words), "token_starts": starts,
            "spans": [], "nested": [], "ignored": [], "orgs": [], "ner_tags_loc_editor": tags(loc), "ner_tags_per_editor": tags(per),
            "loc_gold_v2": None, "loc_mask": [], "loc_valid_v1": None, "per_valid_v1": None}


def test_scorer_option(tmp_path):
    eds = {"cav": {"main_score": True, "loc_tagged": True, "per_tagged": True, "per_match": "overlap",
                   "measures": {"LOC_editor": "R+P", "LOC_v2": None, "PER": "R+P"}}}
    w = "il Sor Duca di Nevers scrive a Parigi con messer Piero Capello".split()
    recs = [_bench_rec("p1", "cav", w, loc=[(7, 8)], per=[(1, 5), (9, 12)])]
    (tmp_path / "b.jsonl").write_text(json.dumps(recs[0]) + "\n")
    (tmp_path / "e.json").write_text(json.dumps(eds))
    pred = tmp_path / "p.jsonl"
    pred.write_text(json.dumps({"id": "p1", "LOC": [{"s": 4, "e": 5}, {"s": 7, "e": 8}], "PER": [{"s": 10, "e": 12}]}) + "\n")
    base = ["--bench", str(tmp_path / "b.jsonl"), "--editions", str(tmp_path / "e.json"), "--pred", f"m={pred}", "--boot", "20"]
    d = eb.main(base + ["--out", str(tmp_path / "d.json")])
    assert "owner_rules" not in d["meta"]                                    # default unchanged
    assert d["scores"]["m"]["PER"]["strict"]["main"]["exhaustive"]["found"] == 0
    assert d["scores"]["m"]["LOC_editor"]["strict"]["main"]["exhaustive"]["right"] == 1
    o = eb.main(base + ["--out", str(tmp_path / "o.json"), "--owner-rules", "--owner-log", str(tmp_path / "log.jsonl")])
    s = o["scores"]["m"]
    assert s["PER"]["strict"]["main"]["exhaustive"]["gold"] == 1 and s["PER"]["strict"]["main"]["exhaustive"]["found"] == 1   # Piero Capello
    assert s["LOC_editor"]["strict"]["main"]["exhaustive"]["gold"] == 2 and s["LOC_editor"]["strict"]["main"]["exhaustive"]["right"] == 2
    c = o["meta"]["owner_rules"]["counts_per_edition"]["cav"]
    assert c["title_place"] == 1 and c["loc_added"] == 1 and c["title_strip"] == 2
    rows = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert {r["before_text"] for r in rows} == {"Sor Duca di Nevers", "messer Piero Capello"}


def test_owner_log_needs_option(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        eb.main(["--bench", "b", "--editions", "e", "--pred", "m=p", "--out", "o", "--owner-log", "l"])
