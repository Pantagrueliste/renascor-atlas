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


SYN = """<TEI><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><country>Suiza</country><settlement>Lausana</settlement></msIdentifier>
<msContents><msItem><editor><persName type="escribano">Francisco Santos</persName></editor></msItem></msContents>
<history><origin><origDate when="1688">1688</origDate><origPlace><placeName type="settlement">Valladolid</placeName><region>Valladolid</region>
<country>Espana</country><geo>41.651981 -4.728561</geo></origPlace></origin></history></msDesc></sourceDesc></fileDesc></teiHeader><text><body/></text></TEI>"""


def test_synthetic():
    col = get("cosuiza")
    ctx = col.make_context()
    ms, es = col.extract(_rec("a/COSUIZA-0001.xml", SYN), ctx)
    assert [(m.kind, m.role, m.surface) for m in ms] == [("person", "author", "Francisco Santos"),
                                                          ("place", "sent_from", "Valladolid")]
    assert ms[1].local_key and len(es) == 1 and abs(es[0].lat - 41.651981) < 1e-6 and es[0].coord_source == "tei:geo"
    assert all(m.context is None and m.tier == "T3" for m in ms)       # T3: no snippets
    assert ms[0].doc_year == 1688


def test_real_files():
    col, ctx, files = _setup("cosuiza", ("COSUIZA-0007.xml",))
    f = next(f for f in files if f.path.endswith("COSUIZA-0007.xml"))
    ms, es = col.extract(f, ctx)
    assert len(ms) == 3 and ms[0].doc_year == 1771 and ms[0].in_period is False
    assert any(e.lat for e in es)
