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


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="d1"><teiHeader><fileDesc><titleStmt>
<title>Tagebuch: <date when="1635-02">Februar 1635</date></title></titleStmt><sourceDesc>
<listPerson><person xml:id="winss_andreas"><persName ref="https://d-nb.info/gnd/123456789"><forename>Andreas</forename><surname>Winss</surname></persName>
<persName type="display">Winss, Andreas</persName><note><rs type="person" ref="#other">someone</rs></note></person></listPerson>
<listPlace><place xml:id="bernburg"><placeName ref="https://www.geonames.org/2950073/bernburg.html">Bernburg</placeName></place></listPlace></sourceDesc></fileDesc></teiHeader>
<text><body><div type="entry"><index indexName="place"><term ref="#bernburg">Bernburg</term></index>
<index indexName="subject"><term ref="#gast"/></index>
<p>Jn die predigt <rs type="person" ref="#winss_andreas">Winsij</rs>, <rs type="place" ref="#bernburg">alhjer</rs>,
<rs type="person" ref="#dangling_one">X</rs> <rs type="symbol" ref="#sonntag">S</rs> <rs type="bibl" ref="#b">B</rs> <rs type="peron" ref="#winss_andreas">Y</rs></p></div></body></text></TEI>"""


def test_synthetic():
    col = get("tagebuecher_christian_ii_anhalt")
    ctx = col.make_context()
    f = _rec("renascor_2026-10/tagebuecher-christian-ii-anhalt/files/1635_02.xml", SYN)
    ctx.all_files = [f]
    es = col.prepare(ctx)
    assert {e.local_key for e in es} == {"tagebuecher_christian_ii_anhalt:winss_andreas",
                                         "tagebuecher_christian_ii_anhalt:bernburg"}
    ms, _ = col.extract(f, ctx)
    assert [(m.kind, m.role, m.in_register) for m in ms] == [
        ("person", "mention", True), ("place", "mention", True), ("person", "mention", False),
        ("person", "mention", True), ("place", "entry_place", True)]          # symbol and bibl rs are not entities
    assert ms[0].auth["gnd"] == "123456789" and ms[1].auth["geonames"] == "2950073"
    assert ms[2].local_key.endswith(":dangling_one") and ms[0].doc_year == 1635 and ms[0].date_source == "header"


def test_real_file():
    col, ctx, files = _setup("tagebuecher_christian_ii_anhalt", ("1635_02.xml", "1621_11.xml"))
    assert len(files) == 338 and not any(f.path.endswith("mets.xml") for f in files)
    f = next(f for f in files if f.path.endswith("1635_02.xml"))
    ms, _ = col.extract(f, ctx)
    assert len(ms) == 440 and sum(m.role == "entry_place" for m in ms) == 37
    assert all(m.in_register for m in ms) and all(m.doc_year == 1635 for m in ms)
    assert any(m.auth["gnd"] for m in ms if m.kind == "person") and any(m.auth["geonames"] for m in ms if m.kind == "place")
