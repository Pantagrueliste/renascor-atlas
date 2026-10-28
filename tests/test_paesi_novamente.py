import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><biblStruct><monogr><imprint><pubPlace>Vicenza</pubPlace><date>1507</date></imprint></monogr></biblStruct></sourceDesc></fileDesc></teiHeader>
<text><body><p>da <rs type="person" ref="#vespucci">Alberico ve<choice><orig>\xc5\xbf</orig><reg>s</reg></choice>putio</rs> a
<rs type="place" ref="#kozhikode">Calicut</rs> <rs type="nom2" ref="#x">z</rs></p></body></text></TEI>'''


def test_synthetic_diplomatic_surface_and_opaque_keys():
    col = get('paesi_novamente')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYN), C.FileRecord(path='p/x.xml', coll='p'), ctx)
    assert es == [] and [m.surface for m in ms] == ['Alberico veſputio', 'Calicut']
    assert [m.local_key for m in ms] == ['paesi_novamente:vespucci', 'paesi_novamente:kozhikode']
    assert not any(m.in_register for m in ms) and {m.doc_year for m in ms} == {1507}


def test_real_files():
    col = get('paesi_novamente')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 2
    tot = collections.Counter()
    for rec in files:
        ms, _ = col.extract(rec, ctx)
        tot.update(m.kind for m in ms)
        assert {m.doc_year for m in ms} in ({1507}, {1508}) and all(m.local_key for m in ms)
    assert tot == {'person': 778, 'place': 2820}
    assert all(r for r in [col.tier == 'T3'])
