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


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><msDesc><msContents><p>
<origDate from="1563-01-03" to="1569-12-04"/></p></msContents></msDesc></sourceDesc></fileDesc></teiHeader>
<text><body><listPlace><place xml:id="Kleinbasel"><placeName>Kleinbasel</placeName></place></listPlace>
<div ana="#uf_Eintrag"><p><persName ana="#uf_PersonTäter"><forename>Hans</forename></persName> aus
<placeName ana="#uf_OrtHerkunft" ref="#Kleinbasel">Basel</placeName>, <persName ana="#uf_PersonNotar">Im Hoff</persName></p></div></body></text></TEI>"""


def test_synthetic_roles():
    col = get("urfehdebuecher_basel_x")
    ctx = col.make_context()
    root = C.parse_xml(SYN.encode("utf8"))
    ents = [e for e, a in C.parse_register(root, ctx, "p", None)[0]]
    for e, a in C.parse_register(root, ctx, "p", None)[0]:
        ctx.shared.add(e, a)
    ms, es = col.extractor.extract_root(root, _rec("a/b.xml", SYN), ctx)
    assert [(m.kind, m.role) for m in ms] == [("person", "offender"), ("place", "origin"), ("person", "notary")]
    assert ms[1].local_key == "urfehdebuecher_basel_x:Kleinbasel" and ms[0].local_key is None
    assert (ms[0].doc_year_from, ms[0].doc_year_to) == (1563, 1569) and es == []


def test_real_file():
    col, ctx, files = _setup("urfehdebuecher_basel_x")
    ms, es = col.extract(files[0], ctx)
    assert len(ms) == 3599 and sum(m.role == "offender" for m in ms) == 776
    assert sum(m.role == "origin" for m in ms) == 565 and all(m.in_period for m in ms)
    assert len(ctx.shared.entities) == 369 and all(m.tier == "T2" and m.context for m in ms[:50])
