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


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc/></teiHeader><text><body>
<entry><orth><rs type="person" ref="#midas"><w lemma="Midas" lemmaRef="http://en.wikipedia.org/wiki/Midas">Midas</w></rs></orth></entry>
<entry><orth><rs type="place" ref="vesuvius"><w lemma="V">Vesuvius</w></rs></orth></entry>
<entry><orth><rs type="corporate" ref="#remi"><w lemma="R">Remi</w></rs></orth></entry>
<entry><orth><rs type="animal" ref="#cerberus"><w lemma="C">Cerberus</w></rs></orth></entry></body></text></TEI>"""


def test_synthetic():
    col = get("dasypodius_dictionarium_latinogermanicum")
    ctx = col.make_context()
    ms, es = col.extract(_rec("a/b.xml", SYN), ctx)
    assert [m.kind for m in ms] == ["person", "place", "org"]            # animal is skipped
    assert ms[0].auth["other"] == ["wikipedia:en:Midas"] and ms[0].local_key == "dasypodius_dictionarium_latinogermanicum:midas"
    assert ms[0].in_register is False and ms[1].local_key.endswith(":vesuvius")
    assert ms[0].doc_year == 1535 and ms[0].date_source == "collection"


def test_real_file():
    col, ctx, files = _setup("dasypodius_dictionarium_latinogermanicum", ("03-ch.xml",))
    f = next(f for f in files if f.path.endswith("03-ch.xml"))
    ms, es = col.extract(f, ctx)
    assert len(ms) == 332 and {m.kind for m in ms} == {"person", "place", "org"}
    assert sum(1 for m in ms if m.auth["other"]) > 200
