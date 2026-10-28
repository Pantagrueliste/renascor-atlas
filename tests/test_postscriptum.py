import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><country>España</country>
<settlement>Cuenca</settlement></msIdentifier></msDesc></sourceDesc></fileDesc>
<profileDesc><correspDesc><correspAction type="sent"><persName ref="pscdd:PO1" xml:id="PO1">Pedro</persName>
<location type="origin"><placeName>España, Guadalajara, Atienza</placeName><geo>41.1981179 -2.8708039</geo></location>
<date when="1538"/></correspAction>
<correspAction type="received"><persName ref="pscdd:JR8">Juan</persName>
<location type="destination"><placeName>España, Soria, Berlanga</placeName><geo></geo></location></correspAction>
</correspDesc></profileDesc></teiHeader><text><body><p><placeName><w>En</w> <w>Atienza</w></placeName></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, ents = run('postscriptum', [('postscriptum/ES1500/x/PS1_TEIP5.xml', SYN)])
    assert {e.name: (e.lat, e.coord_source) for e in ents if e.kind == 'place'} == {
        'España, Guadalajara, Atienza': (41.1981179, 'tei:geo'), 'España, Soria, Berlanga': (None, None)}
    keys = {(m.role, m.local_key) for m in ms if m.zone == 'hdr'}
    assert keys == {('sender', 'postscriptum:PO1'), ('addressee', 'postscriptum:JR8'),
                    ('sent_from', 'postscriptum:España, Guadalajara, Atienza'),
                    ('sent_to', 'postscriptum:España, Soria, Berlanga')}
    assert count(ms, zone='txt', kind='place') == 1                     # body placeName, no key
    assert all(m.doc_year == 1538 and m.tier == 'T3' and m.context is None for m in ms)
    assert count(ms, surface='Cuenca') == 0                              # archive location is not a mention
    schema_ok(ms, ents)


def test_real_letter():
    ms, ents = run_real('postscriptum', ['postscriptum/ES1500/ES1500_XML-TEI_P5/PSCR3017_TEIP5.xml'])
    assert len(ms) == 5 and {m.local_key for m in ms if m.kind == 'person'} == {'postscriptum:MG28', 'postscriptum:JA30'}
    amb = next(e for e in ents if e.name == 'Bélgica, Amberes')
    assert round(amb.lat, 3) == 51.219 and amb.n_mentions is None
    assert {m.doc_year for m in ms} == {1556}
