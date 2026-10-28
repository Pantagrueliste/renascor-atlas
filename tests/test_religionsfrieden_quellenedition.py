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


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="t1"><teiHeader><fileDesc><titleStmt>
<title level="s">Serie (1500-1800)</title><title level="a">Edikt von Amboise (19. Maerz 1563)</title></titleStmt>
<publicationStmt><date when="2013">2013</date></publicationStmt></fileDesc>
<profileDesc><particDesc><listPerson><person xml:id="karl_ix"><persName ref="http://d-nb.info/gnd/118560131">Karl IX.</persName></person></listPerson></particDesc>
<settingDesc><listPlace><place xml:id="amboise"><placeName ref="http://www.geonames.org/3037177/amboise.html">Amboise</placeName></place></listPlace></settingDesc></profileDesc></teiHeader>
<text><body><p><rs type="person" ref="#karl_ix">Koenig</rs> in <rs type="place" ref="#amboise">Amboise</rs> <rs type="bibl" ref="#x">Quelle</rs>.</p></body></text></TEI>"""


def test_synthetic():
    col = get("religionsfrieden_quellenedition")
    ctx = col.make_context()
    ms, es = col.extract(_rec("a/b.xml", SYN), ctx)
    assert [(m.kind, m.in_register) for m in ms] == [("person", True), ("place", True)]
    assert ms[0].auth["gnd"] == "118560131" and ms[1].auth["geonames"] == "3037177"
    assert (ms[0].doc_year_from, ms[0].date_source, ms[0].in_period) == (1563, "header", True)
    assert {e.kind for e in es} == {"person", "place"}
    Mention(**{k: v for k, v in ms[0].__dict__.items()}).to_row()


def test_real_file():
    col, ctx, files = _setup("religionsfrieden_quellenedition", ("pa000001-0039.xml", "pa000001-0080.xml"))
    by = {f.name: col.extract(f, ctx) for f in files if f.path.endswith(("pa000001-0039.xml", "pa000001-0080.xml"))}
    ms, es = by["pa000001-0039.xml"]
    assert len(ms) == 62 and sum(m.kind == "person" for m in ms) == 35
    assert all(m.doc_year == 1563 and m.in_period for m in ms)
    assert sum(1 for m in ms if m.auth["gnd"] or m.auth["geonames"]) > 40
    ms80, _ = by["pa000001-0080.xml"]
    assert ms80[0].doc_year == 1712 and ms80[0].in_period is False      # a post-1700 file is flagged, not dropped
    assert all(not f.path.endswith(".html") for f in files)
