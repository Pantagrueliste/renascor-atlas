import pytest

from ratlas import classify, common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason="raw data / scan not available")

XML = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="pa1"><teiHeader><fileDesc><titleStmt><title>t</title></titleStmt></fileDesc></teiHeader>
<text><body><p>Gedruckt zu Magdeburg 1550. <rs type="person" ref="#a">Luther</rs>
<note type="annotation">Der Reformator <rs type="person" ref="#a">Luther</rs> in <rs type="place" ref="#b">Wittenberg</rs></note>
<note type="app1"><rs type="person" ref="#c">Flacius</rs></note>
<note place="outer-margin"><rs type="person" ref="#d">Amsdorf</rs></note>
<note type="footnote"><rs type="place" ref="#b">Magdeburg</rs></note></p></body></text></TEI>"""


def test_zones_and_text_date():
    col = get("controversia_confessio")
    ctx = col.make_context()
    f = C.FileRecord(path="a/0209_Text.xml", coll="x", data=XML.encode())
    ctx.all_files = [f]
    col.prepare(ctx)
    ms, _ = col.extract(f, ctx)
    for m in ms:
        classify.finalize(m)
    z = [(m.surface, m.zone_class, m.excluded) for m in ms]
    assert z == [("Luther", "text", False), ("Luther", "editorial_commentary", True), ("Wittenberg", "editorial_commentary", True),
                 ("Flacius", "apparatus", True), ("Amsdorf", "text", False), ("Magdeburg", "editorial_commentary", True)]
    assert {m.doc_year for m in ms} == {1550} and {m.date_source for m in ms} == {"text"}
