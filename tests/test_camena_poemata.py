from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import camena_poemata as cp

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

BIS = 'originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/poemata/bis_1.xml'


def test_synthetic_types():
    xml = ('<TEI.2><teiHeader><fileDesc><titleStmt><respStmt><name>Modern Editor</name></respStmt></titleStmt>'
           '<sourceDesc><bibl>Printer, Heidelberg, 1584.</bibl></sourceDesc></fileDesc></teiHeader><text><body><l>'
           '<name type="myth">Phoebus</name> <name type="religion">Deus</name> <name type="nation">Galli</name>'
           ' <name type="place">Roma</name> <name type="?">Sirius</name> <name type="person">Davidis</name> <name>x</name>'
           '</l></body></text></TEI.2>')
    ctx = C.Context(slug='camena_poemata', licence=cp.LICENCE, tier='T1', type_map=cp.OPTIONS['type_map'])
    ms, _ = cp.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/a.xml', coll='x'), ctx)
    assert Counter((m.kind, m.tag) for m in ms) == {('person', 'name@myth'): 1, ('person', 'name@religion'): 1,
                                                  ('org', 'name@nation'): 1, ('place', 'name@place'): 1,
                                                  ('person', 'name@person'): 1}
    assert {m.doc_year for m in ms} == {1584} and all(m.local_key is None for m in ms)


def test_real_file():
    ms, ents, _ = H.extract_paths('camena_poemata', [BIS])
    assert len(ms) == pytest.approx(599, abs=15) and not ents
    assert {m.doc_year for m in ms} == {1584}
    assert max(len(m.context or '') for m in ms) <= 120
