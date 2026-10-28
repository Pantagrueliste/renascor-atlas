import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><principal><persName xml:id="m"><surname>Mounier</surname></persName></principal></titleStmt>
<sourceDesc><listPerson><person xml:id="palamon"><persName>Palamon</persName></person></listPerson></sourceDesc></fileDesc></teiHeader>
<text xml:id="poem"><body><head>de <persName ref="#palamon">Palamon</persName> et <persName ref="#nobody">X</persName></head></body></text></TEI>'''


def test_synthetic_in_file_register():
    col = get('elan_fnvr')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYN), C.FileRecord(path='e/data__tei__texts__palamon.xml', coll='e'), ctx)
    assert len(es) == 1 and es[0].local_key == 'elan_fnvr:data__tei__texts__palamon#palamon'
    assert [(m.surface, m.local_key is not None) for m in ms] == [('Palamon', True), ('X', False)]   # principal ignored
    assert all(m.licence == 'CC-BY-NC-SA-4.0' and m.tier == 'T2' for m in ms)


def test_real_files():
    col = get('elan_fnvr')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 6
    tot, ents, years = collections.Counter(), 0, set()
    for rec in files:
        ms, es = col.extract(rec, ctx)
        tot[rec.stem.rsplit('__', 1)[-1]] += len(ms)
        ents += len(es)
        years |= {m.date_source for m in ms}
        assert all(m.in_register for m in ms)
    assert tot['palamon'] == 568 and tot['eurialus'] == 862 and ents == 206
    assert years == {'collection', 'header'}
