import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

HDR = '''<correspDesc><correspAction type="sent"><rs type="person" ref="#buelow_busse">Busse</rs><rs type="place"/>%s</correspAction>
<correspAction type="received"><rs type="person" ref="#lueneburg">Rat</rs><rs type="place" ref="#lueneburg">Stadt</rs></correspAction></correspDesc>'''
SYN = f'''<TEI {NS} xml:id="hs15-brief073"><teiHeader><fileDesc><titleStmt><title>Hs. 15, Brief 73: X an Y, 2. April 1469</title>
<respStmt><name type="person">Editor</name></respStmt></titleStmt>
<sourceDesc><msDesc><msIdentifier><settlement>Lüneburg</settlement></msIdentifier></msDesc></sourceDesc></fileDesc>
<profileDesc>{HDR % '<date when-custom="1469-04-02">2. April 1469</date>'}</profileDesc></teiHeader>
<text><body><p><rs type="place" ref="#bleckede">Blekede</rs> <rs type="person">Nonne</rs></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, _ = run('luene_nonnenbriefe', [('renascor_2026-10/luene-nonnenbriefe/files/texts/Brief073-D_tei-transcript.xml', SYN)])
    assert len(ms) == 5          # empty <rs type=place/> and msIdentifier/settlement dropped
    keys = {m.local_key for m in ms}
    assert 'luene_nonnenbriefe:person:lueneburg' in keys and 'luene_nonnenbriefe:place:lueneburg' in keys
    assert {m.doc_year for m in ms} == {1469} and {m.date_source for m in ms} == {'letter'}
    assert count(ms, role='sender') == 1 and count(ms, role='sent_to') == 1
    schema_ok(ms)


def test_undated_letter_gets_collection_span_and_title_year():
    x = SYN.replace(b'<date when-custom="1469-04-02">2. April 1469</date>', b'<date>undatiert</date>')
    ms, _ = run('luene_nonnenbriefe', [('renascor_2026-10/luene-nonnenbriefe/files/texts/Brief073-D_tei-transcript.xml', x)])
    assert {(m.doc_year, m.date_source) for m in ms} == {(1469, 'header')}      # year of the title
    y = x.replace(b'2. April 1469', b'undatiert')
    ms, _ = run('luene_nonnenbriefe', [('renascor_2026-10/luene-nonnenbriefe/files/texts/Brief001.xml', y)])
    assert {(m.doc_year_from, m.doc_year_to, m.date_source) for m in ms} == {(1460, 1555, 'collection')}


def test_real_letter():
    ms, _ = run_real('luene_nonnenbriefe', ['renascor_2026-10/luene-nonnenbriefe/files/texts/Brief073-D_tei-transcript.xml'])
    assert count(ms, local_key='luene_nonnenbriefe:place:bleckede') == 2 and {m.doc_year for m in ms} == {1469}
