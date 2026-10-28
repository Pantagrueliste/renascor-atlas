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


SYN = """<TEI.2><teiHeader><fileDesc><sourceDesc><bibl>Oxoniae: Henricus Clements, 1703.</bibl></sourceDesc></fileDesc></teiHeader>
<text><body><p><name type="person">Ascham</name> <name type="place">Oxonia</name> <name>x</name></p></body></text></TEI.2>"""


def test_synthetic_post_1700_imprint_is_undated():
    # v2: the imprint of a later edition (1703) says nothing about the letters: undated, not 'out of period'
    col = get("camena_cera")
    ctx = col.make_context()
    ms, es = col.extract(_rec("a/b.xml", SYN), ctx)
    assert len(ms) == 2 and ms[0].doc_year is None and ms[0].in_period is None


HEAD = """<TEI.2><teiHeader><fileDesc><sourceDesc><bibl>Lugd. Bat. 1599.</bibl></sourceDesc></fileDesc></teiHeader>
<text><body><div1 type="book"><div2 type="letter"><head>EPISTOLA I. <name type='person'>STEPH. PIGHIUS</name> <name type='person'>Justo Lipsio</name> S. P. D.
<name type='place'>Jenam</name></head><p>Salutem <name type='person'>Crajum</name></p></div2></div1></body></text></TEI.2>"""


def test_letter_head_roles():
    col = get("camena_cera")
    ctx = col.make_context()
    ms, _ = col.extract(_rec("a/b.xml", HEAD), ctx)
    assert [(m.surface, m.role) for m in ms] == [("STEPH. PIGHIUS", "sender"), ("Justo Lipsio", "addressee"),
                                                 ("Jenam", "sent_to"), ("Crajum", "mention")]


def test_real_file():
    col, ctx, files = _setup("camena_cera", ("Goldast_epistolae.xml",))
    f = next(f for f in files if f.path.endswith("Goldast_epistolae.xml"))
    ms, _ = col.extract(f, ctx)
    assert len(ms) == 432 and ms[0].doc_year == 1688 and ms[0].in_period and ms[0].kind == "person"
