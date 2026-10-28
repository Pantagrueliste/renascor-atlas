import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><titleStmt><title>550308B Flacius an Nidbruck</title></titleStmt>
<sourceDesc><msDesc><msIdentifier><settlement>Wien</settlement></msIdentifier><head><title>
<rs type="person" ref="#matthias_flacius">Matthias Flacius</rs> (Köln) an <rs type="person" ref="#caspar_nidbruck">Caspar von Nidbruck</rs>
</title></head></msDesc></sourceDesc></fileDesc></teiHeader>
<text><body><p><rs type="person" ref="#hubertus_languet">Languetum</rs> <rs type="author" ref="'ambrosius">Ambrosius</rs></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, _ = run('magdeburger_zenturien_methode_hab', [('renascor_2026-10/magdeburger-zenturien-methode-hab/files/Wien/OeNB/550308B_Flacius_an_Nidbruck.xml', SYN)])
    assert len(ms) == 4
    assert [m.surface for m in ms if m.role == 'sender'] == ['Matthias Flacius']
    assert [m.surface for m in ms if m.role == 'addressee'] == ['Caspar von Nidbruck']
    assert next(m for m in ms if m.tag == 'rs@author').local_key.endswith(':ambrosius')      # stray quote removed
    assert {(m.doc_year, m.date_source) for m in ms} == {(1555, 'filename')}                 # no origDate: YYMMDD prefix
    schema_ok(ms)


def test_real_letter():
    ms, _ = run_real('magdeburger_zenturien_methode_hab', ['renascor_2026-10/magdeburger-zenturien-methode-hab/files/Wien/OeNB/550308B_Flacius_an_Nidbruck.xml'])
    assert count(ms, role='sender', local_key='magdeburger_zenturien_methode_hab:matthias_flacius') == 1
    assert count(ms, role='addressee') == 1 and {m.doc_year for m in ms} == {1555}
