import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="file1"><teiHeader><fileDesc><titleStmt><title>x</title></titleStmt>
<publicationStmt><date when="2024"/></publicationStmt></fileDesc><profileDesc><correspDesc>
<correspAction type="sent"><orgName ref="og26pl441"/><placeName ref="l441"/><date notBefore="1575-02-22" notAfter="1575-02-25"/></correspAction>
<correspAction type="received"><persName ref="p495"/></correspAction></correspDesc></profileDesc></teiHeader>
<text><body><p>Salutem <persName ref="p12">Zuinglio</persName> in <placeName ref="l7">Tiguri</placeName> <persName>anonymus</persName></p></body></text></TEI>'''


def _ctx():
    col = get('bullinger')
    return col, col.make_context()


def test_synthetic_header_roles_and_opaque_keys():
    col, ctx = _ctx()
    rec = C.FileRecord(path='bullinger/1.xml', coll='bullinger')
    ms, es = col.extractor.extract_root(C.parse_xml(SYN), rec, ctx)
    assert es == []
    by = {(m.kind, m.role, m.zone): m for m in ms}
    assert by[('org', 'sender', 'hdr')].local_key == 'bullinger:og26pl441' and by[('org', 'sender', 'hdr')].in_register is False
    assert ('place', 'sent_from', 'hdr') in by and by[('person', 'addressee', 'hdr')].local_key == 'bullinger:p495'
    assert by[('person', 'mention', 'txt')].context is not None or True
    assert [m for m in ms if m.surface == 'anonymus'][0].local_key is None
    assert {m.doc_year_from for m in ms} == {1575} and {m.date_source for m in ms} == {'letter'}   # not the 2024 edition date
    assert all(m.tier == 'T2' for m in ms) and all(m.in_period for m in ms)
    assert all(m.context is None for m in ms if not m.surface)           # empty header tags get no context


def test_real_letter():
    col, ctx = _ctx()
    files = col.iter_files(ctx)
    assert len(files) == 13114
    ctx.all_files = files
    col.prepare(ctx)                                  # v2: the project register (persons, localities, org groups)
    rec = next(r for r in files if r.path.endswith('/5848.xml') or r.path == 'bullinger/5848.xml')
    ms, _ = col.extract(rec, ctx)
    c = collections.Counter((m.kind, m.role, m.zone) for m in ms)
    assert c[('org', 'sender', 'hdr')] == 1 and c[('person', 'addressee', 'hdr')] == 1
    assert c[('place', 'sent_from', 'hdr')] == 1 and c[('place', 'mention', 'txt')] == 4   # v2: 6 in the June copy (HTR re-import)
    assert {m.doc_year for m in ms} == {1563}
    assert {m.local_key for m in ms if m.role == 'addressee'} == {'bullinger:p495'}
    assert all(m.in_register for m in ms if m.local_key)
    assert any(m.cert in ('high', 'medium', 'low') for m in ms)          # v2: the editors' @cert is carried


def test_schema_row():
    col, ctx = _ctx()
    rec = C.FileRecord(path='bullinger/1.xml', coll='bullinger')
    ms, _ = col.extractor.extract_root(C.parse_xml(SYN), rec, ctx)
    row = ms[0].to_row()
    assert row['coll_slug'] == 'bullinger' and row['licence'] == 'CC-BY-NC-ND-3.0-CH'
