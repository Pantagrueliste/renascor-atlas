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



def test_synthetic():
    col = get("biblioteca_italiana")
    ctx = col.make_context()
    xml = "<TEI.2><teiHeader/><text><body><l>ad <rs type='name'>Arrio</rs> in <rs type='luogo'>Siviglia</rs> <name>Untyped</name></l></body></text></TEI.2>"
    ms, _ = col.extract(_rec("bibit_tei/bibit000316.xml", xml), ctx)
    assert [(m.kind, m.surface) for m in ms] == [("person", "Arrio"), ("place", "Siviglia")]
    assert ms[0].doc_year == 1550 and ms[0].tier == "T2"


def test_real_file():
    col, ctx, files = _setup("biblioteca_italiana")
    assert sorted(f.stem for f in files) == ["bibit000316", "bibit000996"]
    ms, _ = col.extract(files[0], ctx)
    assert len(ms) == 81 and sum(m.kind == "place" for m in ms) == 26
