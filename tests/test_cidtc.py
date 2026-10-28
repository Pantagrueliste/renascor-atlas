import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS} xml:id="IDL9999"><teiHeader><fileDesc><titleStmt><title>
<persName type="sender"><span>A</span></persName> to <persName type="addressee"><span>B</span></persName></title></titleStmt></fileDesc>
<profileDesc><correspDesc><correspAction type="sent">
<persName ref="Pe1" role="sender" ana="Dantiscus, Ioannes"/><persName ref="Pe7" role="addressee" ana="Bona Sforza"/>
<placeName ref="Pl1">Heilsberg</placeName><date type="part" notBefore="1542-04-13" notAfter="1542-04-13"/>
</correspAction></correspDesc></profileDesc></teiHeader>
<text><body><p><rs type="person" key="24 12 7">maiestates</rs> <name type="place" key="1">Heilsberg</name></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, ents = run('cidtc', [('cidtc/xml/IDL9999.xml', SYN)])
    # v2: a local register built from the header @ana / text forms names the entities (not the commonest epithet)
    assert {e.local_key: e.name for e in ents} == {'cidtc:pe1': 'Dantiscus, Ioannes', 'cidtc:pe7': 'Bona Sforza', 'cidtc:pl1': 'Heilsberg'}
    # the titleStmt persName are skipped; header roles come from @role, the surface from @ana
    hdr = [m for m in ms if m.zone == 'hdr']
    assert {(m.role, m.surface, m.local_key) for m in hdr} == {
        ('sender', 'Dantiscus, Ioannes', 'cidtc:pe1'), ('addressee', 'Bona Sforza', 'cidtc:pe7'),
        ('sent_from', 'Heilsberg', 'cidtc:pl1')}
    # plural key -> one row per person, places and persons have separate key spaces
    assert sorted(m.local_key for m in ms if m.tag == 'rs@person') == ['cidtc:pe12', 'cidtc:pe24', 'cidtc:pe7']
    assert [m.local_key for m in ms if m.tag == 'name@place'] == ['cidtc:pl1']
    assert {m.doc_year for m in ms} == {1542} and all(m.in_period and m.tier == 'T3' for m in ms)
    schema_ok(ms)


def test_epithet_only_entity_is_flagged():
    x = SYN.replace(b'<persName ref="Pe7" role="addressee" ana="Bona Sforza"/>', b'<persName ref="Pe7" role="addressee"/>')
    x = x.replace(b'key="24 12 7">maiestates', b'key="7">maiestatem regiam')
    ms, ents = run('cidtc', [('cidtc/xml/IDL9999.xml', x)])
    e = next(e for e in ents if e.local_key == 'cidtc:pe7')
    assert e.name == 'maiestatem regiam' and e.note_type == 'epithet_only'


def test_real_letter():
    ms, _ = run_real('cidtc', ['cidtc/xml/IDL0001.xml'])
    assert len(ms) == 7 and count(ms, kind='person') == 5 and count(ms, kind='place') == 2
    assert count(ms, role='sender', surface='Dantiscus, Ioannes') == 1
    assert count(ms, zone='txt', local_key='cidtc:pe24') == 1 and all(m.context is None for m in ms)
    assert {m.date_source for m in ms} == {'letter'}
    schema_ok(ms)
