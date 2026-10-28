import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><titleStmt><author><persName xml:id="johsch">J. Schmid</persName></author>
<respStmt><persName xml:id="chw">Editor</persName></respStmt></titleStmt></fileDesc>
<profileDesc><correspDesc><correspAction type="sent"><persName ref="https://d-nb.info/gnd/108194221">Schmid</persName>
<placeName ref="https://www.geonames.org/3054643">Ofen</placeName><date when="1649-01-20"/></correspAction>
<correspAction type="received"><persName ref="https://d-nb.info/gnd/118532529">Ferdinand III.</persName></correspAction>
<correspContext><ref type="prev"><rs ref="#johsch" type="person">Schmid</rs></ref></correspContext></correspDesc></profileDesc></teiHeader>
<text><body><p><rs ref="#ferd3" type="person">kayser</rs> in <rs ref="#ofen" type="place">Offen</rs> <rs ref="#x" type="event">Krieg</rs></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, ents = run('dipko_schwarzenhorn_1649', [('renascor_2026-10/dipko-schwarzenhorn-1649/files/l.490120.xml', SYN)])
    assert ents == [] and len(ms) == 6          # author, sender, place, addressee, 2 text rs; event + correspContext skipped
    s = next(m for m in ms if m.role == 'sender')
    assert s.auth['gnd'] == '108194221' and s.local_key is None
    assert next(m for m in ms if m.role == 'sent_from').auth['geonames'] == '3054643'
    assert next(m for m in ms if m.role == 'author').local_key == 'dipko_schwarzenhorn_1649:johsch'
    assert [m.local_key for m in ms if m.kind == 'place' and m.zone == 'txt'] == ['dipko_schwarzenhorn_1649:ofen']
    schema_ok(ms)


def test_real_letter():
    ms, _ = run_real('dipko_schwarzenhorn_1649', ['renascor_2026-10/dipko-schwarzenhorn-1649/files/l.490120.xml'])
    assert count(ms, role='addressee') == 1 and count(ms, kind='place', role='sent_from') == 1
    assert {m.doc_year for m in ms} == {1649} and all(m.in_period for m in ms)
    assert count(ms, local_key='dipko_schwarzenhorn_1649:ferd3') >= 2
    schema_ok(ms)
