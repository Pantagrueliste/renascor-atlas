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



def test_real_file():
    col, ctx, files = _setup("dibilit_german")
    assert len(files) == 47
    f = next(f for f in files if f.path.endswith("ayrer_sidea_1865.txt.xml"))
    ms, es = col.extract(f, ctx)
    assert len(ms) == 1 and ms[0].role == "author" and ms[0].auth["gnd"] == "118651307"
    assert ms[0].doc_year == 1618 and ms[0].date_source == "header" and ms[0].in_register is True   # v2: firstPublication, not the 1600 period year
    assert ms[0].context is None
    assert len(es) == 1 and es[0].auth["gnd"] == "118651307"
