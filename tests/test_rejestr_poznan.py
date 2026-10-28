import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body><p><rs type="person" ref="Q207">Sigismundo</rs>
<rs type="institution" ref="Q930">canonicus</rs> <rs type="place" ref="Q2910">Posnaniensis</rs> <rs ref="Q1">untyped</rs></p></body></text></TEI>'''


def test_synthetic_q_ids_are_not_wikidata():
    col = get('rejestr_poznan')
    ctx = col.make_context()
    ms, _ = col.extractor.extract_root(C.parse_xml(SYN), C.FileRecord(path='r/x.xml', coll='r'), ctx)
    assert [(m.kind, m.local_key) for m in ms] == [('person', 'rejestr_poznan:Q207'), ('org', 'rejestr_poznan:Q930'),
                                                   ('place', 'rejestr_poznan:Q2910')]
    assert all(m.auth['wikidata'] is None and m.in_register is False for m in ms)
    assert {(m.doc_year, m.date_source) for m in ms} == {(1561, 'collection')}


def test_real_file():
    col = get('rejestr_poznan')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 1
    ms, _ = col.extract(files[0], ctx)
    c = collections.Counter(m.kind for m in ms)
    assert c['person'] == 811 and c['place'] == 112 and 1730 <= c['org'] <= 1733
    assert all(m.auth['wikidata'] is None for m in ms) and all(m.local_key for m in ms)
