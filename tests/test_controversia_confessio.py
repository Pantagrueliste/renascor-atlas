import pytest
from ratlas import common as C
from ratlas.collections import get
from ratlas.model import Mention

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason="raw data / scan not available")


def _rec(path, xml, coll="x"):
    return C.FileRecord(path=path, coll=coll, data=xml.encode("utf8"))


def _setup(slug, suffixes=()):
    col = get(slug)
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = [f for f in files if f.path.endswith(tuple(suffixes))] if suffixes else files
    col.prepare(ctx)
    return col, ctx, files


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="pa1"><teiHeader><fileDesc><titleStmt><title>t</title>
<respStmt><persName>Timo Steyer</persName></respStmt></titleStmt></fileDesc></teiHeader>
<standOff><listPerson><person xml:id="martin_luther" corresp="../register/register-person.xml#martin_luther"/></listPerson>
<listPlace><place xml:id="magdeburg" corresp="../register/register-place.xml#magdeburg"/></listPlace></standOff>
<text><body><p><rs type="person" ref="#martin_luther">Luther</rs> <rs type="person">Flacius</rs> <rs type="peron">Y</rs>
<rs type="place" ref="#magdeburg">Magdeburg</rs> <rs type="bibl" ref="#dwb">DWb</rs></p></body></text></TEI>"""


def test_synthetic():
    col = get("controversia_confessio")
    ctx = col.make_context()
    f = _rec("a/Text_x.xml", SYN)
    ctx.all_files = [f]
    es = col.prepare(ctx)
    # v2: the printed register of volumes 2-7 names the slugs ('Luther, Martin (1483-1546)'); nothing else is taken from it
    byk = {e.local_key: e for e in es}
    assert set(byk) == {"controversia_confessio:martin_luther", "controversia_confessio:magdeburg"}
    assert byk["controversia_confessio:martin_luther"].name == "Luther, Martin"
    assert (byk["controversia_confessio:martin_luther"].birth, byk["controversia_confessio:martin_luther"].death) == ("1483", "1546")
    assert not any(e.auth['gnd'] or e.lat for e in es)
    ms, _ = col.extract(f, ctx)
    assert [(m.kind, m.in_register) for m in ms] == [("person", True), ("person", None), ("person", None), ("place", True)]
    # v2: no midpoint of the 27-year collection range; the range is kept and lies inside the period
    assert (ms[0].doc_year_from, ms[0].doc_year_to, ms[0].date_source) == (1548, 1575, "collection")
    assert ms[0].doc_year is None and ms[0].in_period is True
    ms2, _ = col.extract(_rec("a/0201-Historische_Einleitung.xml", SYN), ctx)
    assert ms2[0].in_period is None                                    # editorial introduction: undated


def test_real_file():
    col, ctx, files = _setup("controversia_confessio", ("0209-Magdeburger_Bekenntnis_Text.xml",))
    assert len(files) == 138 and not any(f.path.endswith("wdbmeta.xml") for f in files)
    f = next(f for f in files if f.path.endswith("0209-Magdeburger_Bekenntnis_Text.xml"))
    ms, _ = col.extract(f, ctx)
    assert len(ms) == 100 and sum(m.kind == "place" for m in ms) == 38 and sum(bool(m.in_register) for m in ms) == 84
