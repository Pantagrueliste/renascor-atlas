import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><profileDesc><creation><date when="2012-03-29"/><placeName>Graz, Austria</placeName></creation></profileDesc></teiHeader>
<text><body><div1><ab><persName key="info:fedora/o:rpb-c-0001">Vor dem Datum</persName></ab>
<ab><date when="1537">1537</date> <persName key="info:fedora/o:rpb-c-0010"><forename>Paulo</forename> <surname>Mämmingern</surname></persName></ab>
<ab><date when="1710">1710</date> <persName key="info:fedora/o:rpb-c-0011">Spät</persName></ab></div1></body></text></TEI>'''.encode()


def test_synthetic():
    ms, _ = run('regensburger_wappen_portraetbuecher', [('renascor_2026-10/regensburger-wappen-portraetbuecher/files/rpb-06-text.xml', SYN)])
    assert [(m.local_key.split(':')[1], m.doc_year, m.in_period) for m in ms] == [
        ('rpb-c-0001', None, None), ('rpb-c-0010', 1537, True), ('rpb-c-0011', 1710, False)]
    assert ms[1].surface == 'Paulo Mämmingern' and ms[1].date_source == 'text' and ms[1].tier == 'T3'
    schema_ok(ms)


def test_real_file():
    ms, _ = run_real('regensburger_wappen_portraetbuecher', ['renascor_2026-10/regensburger-wappen-portraetbuecher/files/rpb-06-text.xml'])
    assert len(ms) == 1674 and abs(count(ms, in_period=True) - 1256) <= 5     # colls.json: 1,256 before folio 087r
    assert len({m.local_key for m in ms if m.in_period}) in range(115, 125)
