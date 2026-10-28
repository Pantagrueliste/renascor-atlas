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


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body>
<p><date when="1594-05-01">1. Mai</date> <rs type="place" ref="#Abano_Terme">Abano</rs> <rs type="person" ref="register_personen.xml#bocer_heinrich">Bocer</rs>
<rs type="libri">Buch</rs> <rs type="placec" ref="#Fugger">F</rs></p></body>
<back><listPerson><person xml:id="bocer_heinrich"><persName ref="http://d-nb.info/gnd/117608750"><forename>Heinrich</forename><surname>Bocer</surname></persName></person></listPerson>
<listPlace><place xml:id="Abano_Terme"><placeName>Abano Terme</placeName><location><geo>45.35 11.783333</geo></location><idno type="URI">http://vocab.getty.edu/tgn/1043076</idno></place></listPlace>
<listPlace><place xml:id="Abano_Terme"><placeName>Abano Terme</placeName></place></listPlace></back></text></TEI>"""


def test_synthetic():
    col = get("selbstzeugnisse_hab_diarium")
    ctx = col.make_context()
    f = _rec("a/diarium.xml", SYN)
    ctx.all_files = [f]
    es = col.prepare(ctx)
    assert len(es) == 2                                                       # duplicated listPlace: first wins
    place = next(e for e in es if e.kind == "place")
    assert place.lat == 45.35 and place.auth["tgn"] == "1043076"
    ms, _ = col.extract(f, ctx)
    assert [(m.kind, m.in_register) for m in ms] == [("place", True), ("person", True), ("place", False)]
    assert ms[1].auth["gnd"] == "117608750" and ms[0].doc_year == 1594 and ms[0].date_source == "text"


def test_real_file():
    col, ctx, files = _setup("selbstzeugnisse_hab_diarium")
    assert len(files) == 1 and len(ctx.shared.entities) == 1554
    ms, _ = col.extract(files[0], ctx)
    assert len(ms) == 3479
    # v2: every mention is dated by the last <date> before it, not by the 1594-1635 span of the diary (one midpoint, 1614)
    assert ms[0].doc_year_from == ms[0].doc_year_to and len({m.doc_year for m in ms}) > 30
    assert min(m.doc_year for m in ms) == 1594 and max(m.doc_year for m in ms) == 1635
    assert sum(m.in_register is True for m in ms) == 3009   # v2: 5 tags whose register entry has another kind keep no entry (metonymy rule)
