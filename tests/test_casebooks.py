from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import casebooks as cb

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

CASE = 'eng_dh/casebooks-data-master/cases/CASE30579.xml'
NS = 'xmlns="http://www.tei-c.org/ns/1.0" xmlns:cb="http://www.magicandmedicine.hps.cam.ac.uk/ns/1.0"'


def _ctx():
    return C.Context(slug='casebooks', licence=cb.LICENCE, tier='T2', default_years=(1596, 1634), opaque_keys=True,
                     type_map={'address': 'place'})


def test_master_links_and_roles():
    master = (f'<TEI {NS}><text><body><listPerson><person xml:id="PERSON1"><persName><forename>William</forename> '
              '<surname>Cooper</surname></persName><link sameAs="CASE7#wc"/></person></listPerson></body></text></TEI>')
    ctx = _ctx()
    for ent, al in cb.EXTRACTOR.parse_register_file(C.FileRecord(path='m/names-x.xml', coll='x'), ctx,
                                                    C.parse_xml(master.encode())):
        ctx.shared.add(ent, al)
    case = (f'<TEI {NS}><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><settlement>Oxford</settlement>'
            '</msIdentifier></msDesc></sourceDesc></fileDesc><profileDesc><particDesc><listPerson>'
            '<person xml:id="wc"><persName><forename>William</forename> <surname>Cooper</surname></persName></person>'
            '<person xml:id="rc"><persName><forename>Richard</forename> <surname>Cooper</surname></persName></person>'
            '<person xml:id="rn"><persName><forename>Richard</forename> <surname>Napier</surname></persName></person>'
            '</listPerson></particDesc><settingDesc><p><date when="1607-02-04"/></p><cb:consultation>'
            '<cb:astrologer ref="#rn"/><cb:querent ref="#rc"/><cb:subject ref="#wc"/></cb:consultation></settingDesc>'
            '</profileDesc></teiHeader><text><body><p><rs type="address" xml:id="a1" key="PLACE815">of Stantone</rs> '
            '<rs type="address" sameAs="#a1">there</rs><rs type="date" key="X">Wednesday</rs></p></body></text></TEI>')
    rec = C.FileRecord(path='cases/CASE7.xml', coll='x')
    ms, ents = cb.EXTRACTOR.extract_root(C.parse_xml(case.encode()), rec, ctx)
    by = {(m.role, m.surface): m for m in ms}
    assert by[('subject', 'William Cooper')].local_key == 'casebooks:PERSON1'      # master key through the link
    assert by[('querent', 'Richard Cooper')].local_key == 'casebooks:CASE7#rc'     # unlinked: case-local entity
    assert by[('astrologer', 'Richard Napier')].kind == 'person'
    assert {e.local_key for e in ents} == {'casebooks:CASE7#rc', 'casebooks:CASE7#rn'}
    pl = [m for m in ms if m.kind == 'place']
    assert [m.local_key for m in pl] == ['casebooks:PLACE815', 'casebooks:PLACE815']
    assert all(m.doc_year == 1607 and m.date_source == 'header' for m in ms)
    assert not any(m.surface == 'Oxford' or m.surface == 'Wednesday' for m in ms)


def test_real_case_without_master():
    ms, ents, _ = H.extract_paths('casebooks', [CASE], with_registers=False)
    c = Counter((m.kind, m.role) for m in ms)
    assert c[('person', 'subject')] == 1 and c[('person', 'querent')] == 1 and c[('person', 'astrologer')] == 1
    assert c[('person', 'mention')] == 1 and c[('place', 'mention')] == 1      # John Bird; of Stantone
    assert next(m for m in ms if m.kind == 'place').local_key == 'casebooks:PLACE815'
    assert {m.doc_year for m in ms} == {1607}


def test_name_order_and_anonymous():
    x = C.parse_xml(f'<persName xmlns="http://www.tei-c.org/ns/1.0"><roleName type="prefix">Mrs</roleName> '
                    '<surname>Carter</surname> <forename>Anne</forename></persName>'.encode())
    assert cb.cb_name(x) == 'Mrs Anne Carter'
    assert cb.cb_name(None) == ''


def test_birth_dates_do_not_widen_the_consultation_date():
    xml = (f'<TEI {NS}><teiHeader><profileDesc><particDesc><listPerson><person xml:id="sf"><persName>Simon Forman</persName>'
           '<birth><date when="1552-12-31"/></birth></person></listPerson></particDesc><settingDesc>'
           '<p><date notBefore="1597-11-23" notAfter="1599-02-08"/></p></settingDesc></profileDesc></teiHeader>'
           '<text><body><p/></body></text></TEI>')
    ms, _ = cb.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='cases/CASE9.xml', coll='x'), _ctx())
    assert {(m.doc_year_from, m.doc_year_to, m.date_source) for m in ms} == {(1597, 1599, 'header')}
    undated = '<TEI %s><teiHeader/><text><body><p/></body></text></TEI>' % NS
    ms2, _ = cb.EXTRACTOR.extract_root(C.parse_xml(undated.encode()), C.FileRecord(path='cases/CASE8.xml', coll='x'), _ctx())
    assert ms2 == []       # nothing tagged: no rows
