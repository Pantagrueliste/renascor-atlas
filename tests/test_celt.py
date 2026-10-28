import collections

import pytest

from ratlas import common as C
from ratlas.collections import get
from ratlas.collections.celt import classify_type
from ratlas.helpers_batch2 import sgml_to_xml_bytes

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = ('<?xml version="1.0" encoding="iso-8859-1"?>\n<!DOCTYPE TEI.2 SYSTEM "x.dtd" [\n<!ENTITY % TEIbase "TEI.prose">\n]>\n'
       '<TEI.2 id="G1"><teiHeader><fileDesc><titleStmt><title>F&auml;rber &Eacute;irinn</title>'
       '<respStmt><name id="BF">Beatrix F&auml;rber</name></respStmt></titleStmt></fileDesc>'
       '<profileDesc><creation>1630s <dateRange from="1628" to="1634">Between 1628 and 1634</dateRange></creation></profileDesc></teiHeader>'
       '<text><body><p><name type="person:saint">Pádraig</name> in <name type="place">&Eacute;ire</name>, <name type="deity">Tighearna</name>, '
       '<name type="book">Leabhar</name> <name type="person:collective">Ui N&eacute;ill</name> <name>untyped</name></p></body></text></TEI.2>').encode('latin-1')


def test_sgml_entities_fixed():
    root = C.parse_xml(sgml_to_xml_bytes(SYN))
    assert root is not None and root.get('id') == 'G1'
    assert 'Éirinn' in C.surface_text(next(root.iter('title')))


def test_classify_type():
    assert classify_type('person:saint') == 'person' and classify_type('PERSON:collective') == 'org'
    assert classify_type('place:country') == 'place' and classify_type('deity') == 'person'
    assert classify_type('nomen sacrum') == 'person' and classify_type('book') is None and classify_type('') is None


def test_synthetic_extract():
    col = get('celt')
    ctx = col.make_context()
    rec = C.FileRecord(path='celt/xml/G1.xml', coll='celt', data=SYN)
    ms, _ = col.extract(rec, ctx)
    assert [(m.kind, m.tag, m.surface) for m in ms] == [
        ('person', 'name@person:saint', 'Pádraig'), ('place', 'name@place', 'Éire'), ('person', 'name@deity', 'Tighearna'),
        ('org', 'name@person:collective', 'Ui Néill')]
    assert {(m.doc_year_from, m.doc_year_to, m.date_source) for m in ms} == {(1628, 1634, 'header')}
    assert all(m.tier == 'T3' and m.context is None for m in ms)


def test_real_file_g100070():
    col = get('celt')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    rec = next(r for r in files if r.path == 'celt/xml/G100070.xml')
    ms, _ = col.extract(rec, ctx)
    assert collections.Counter(m.kind for m in ms) == {'person': 643, 'place': 641}
    assert {m.doc_year_from for m in ms} == {1607} and all(m.in_period for m in ms)
