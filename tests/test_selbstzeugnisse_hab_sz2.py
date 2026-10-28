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


def test_real_zip():
    col, ctx, files = _setup("selbstzeugnisse_hab_sz2")
    members = {f.member for f in files}
    assert "sz2/register/listPerson_s2.xml" in members and "sz2/tagebuecher/listPerson_s2.xml" not in members
    assert "sz2/register/itinerar_s2_tab.xml" not in members
    assert len(ctx.shared.entities) >= 1340
    assert sum(e.lat is not None for e in ctx.shared.entities.values()) > 200
    f = next(f for f in files if f.member.endswith("cod_guelf_211_1_extrav.xml"))
    ms, es = col.extract(f, ctx)
    assert es == [] and len(ms) == 1405 and sum(m.kind == "person" for m in ms) == 1002
    assert sum(m.role == "visited" for m in ms) == 222 and (ms[0].doc_year_from, ms[0].doc_year_to) == (1698, 1698)   # v2: dated by the preceding <date>
    assert {m.doc_year for m in ms} >= {1698, 1699}
    assert any(m.auth["gnd"] for m in ms) and any(m.auth["geonames"] for m in ms)
    reg = next(f for f in files if f.member == "sz2/register/listPerson_s2.xml")
    assert col.extract(reg, ctx) == ([], [])
