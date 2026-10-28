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


SYN = """<TEI.2><teiHeader><fileDesc><titleStmt><title type='short'>X: Y. - Venedig, 1640.</title></titleStmt>
<sourceDesc><bibl>Venedig: apud Juntas, 1640.</bibl></sourceDesc></fileDesc><revisionDesc><change><date>01/2011</date></change></revisionDesc></teiHeader>
<text><body><p><name type='person'>Horatius</name> <name type='place'>Roma</name> <name type='myth'>Phoebus</name>
<name type='nation'>Graeci</name> <name>Anon</name> <name type='religion'>Christus</name></p></body></text></TEI.2>"""


def test_synthetic():
    col = get("camena_thesaurus")
    ctx = col.make_context()
    ms, es = col.extract(_rec("a/b.xml", SYN), ctx)
    assert [(m.kind, m.tag) for m in ms] == [("person", "name@person"), ("place", "name@place"),
                                              ("person", "name@myth"), ("person", "name@religion")]
    assert ms[0].doc_year == 1640 and ms[0].date_source == "header" and ms[0].local_key is None


def test_real_file():
    col, ctx, files = _setup("camena_thesaurus", ("pontanusinstitutiones_2.xml",))
    f = next(f for f in files if f.path.endswith("pontanusinstitutiones_2.xml"))
    ms, _ = col.extract(f, ctx)
    assert len(ms) == 782 and ms[0].doc_year == 1594 and ms[0].in_period
    assert sum(m.tag == "name@place" for m in ms) == 122 and sum(m.tag == "name@myth" for m in ms) == 82
