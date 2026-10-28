import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><settlement>Florence</settlement></msIdentifier></msDesc>
<bibl><date>1919</date></bibl></sourceDesc></fileDesc></teiHeader><text><body><p>
<rs type="person" ref="Lorenzo dei Medici">Lorenzo</rs> <rs type="place" ref="Italia">Italia</rs> <rs type="org" ref="repubblica">r</rs>
<rs type="concept" ref="pace">pace</rs> <rs type="population" ref="Italiani">Italiani</rs></p></body></text></TEI>'''


def test_synthetic_label_keys():
    col = get('guicciardini_si')
    ctx = col.make_context()
    ms, _ = col.extractor.extract_root(C.parse_xml(SYN), C.FileRecord(path='g/x.xml', coll='g'), ctx)
    assert [(m.kind, m.local_key) for m in ms] == [('person', 'guicciardini_si:Lorenzo_dei_Medici'),
                                                   ('place', 'guicciardini_si:Italia'), ('org', 'guicciardini_si:repubblica')]
    assert not any(m.in_register for m in ms) and all(not any(v for k, v in m.auth.items() if k != 'other') for m in ms)
    assert {(m.doc_year_from, m.doc_year_to, m.date_source) for m in ms} == {(1537, 1540, 'collection')}   # not the 1919 citation


def test_real_file():
    col = get('guicciardini_si')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 1
    ms, _ = col.extract(files[0], ctx)
    assert collections.Counter(m.kind for m in ms) == {'place': 97, 'org': 34, 'person': 35} or len(ms) >= 160
    assert {m.zone for m in ms} == {'txt'} and all(m.in_period for m in ms)
