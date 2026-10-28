import pytest

from b5util import REAL, count, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

P = 'originals_2026-10-15/web-editions-apis/sozinianische-briefwechsel/LETTER_MAIN_ed_bsp_ydl_hmb.xml'


def test_real_letter():
    ms, ents = run_real('sozinianische_briefwechsel', [P])
    # one letter: 4 persons + 1 place in the standOff register
    assert sorted((e.kind, e.name) for e in ents)[:2] == [('person', 'Boulliau, Ismaël'), ('person', 'Broussel, Pierre d. J.')]
    assert len([e for e in ents if e.kind == 'place']) == 1
    send = [m for m in ms if m.role == 'sender']
    assert send and send[0].local_key == 'sozinianische_briefwechsel:Lubieniecki_Stanislaw' and send[0].in_register
    assert send[0].context is None                      # correspDesc entry: no header-neighbour snippet
    add = next(m for m in ms if m.role == 'addressee')
    assert add.auth['gnd'] == '119277379' and add.doc_year == 1664 and add.date_source == 'letter'
    assert count(ms, role='sent_from', local_key='sozinianische_briefwechsel:Hamburg') >= 1
    # regest (zone hdr) and transcription (txt) rs mentions are both present; rs type=term/source are not
    assert count(ms, zone='txt', kind='person') >= 3 and count(ms, zone='hdr', tag='rs@person') >= 2
    assert all(m.tier == 'T1' and m.context is None or len(m.context) <= 120 for m in ms)
    schema_ok(ms, ents)


def test_ancient_note_type():
    from b5util import run
    x = b'''<teiCorpus xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><standOff><listPerson>
<person xml:id="Arist"><persName type="reg"><surname>Aristoteles</surname></persName><birth>384 v. Chr.</birth></person>
<person xml:id="Boul"><persName type="reg"><surname>Boulliau</surname></persName><birth>1605</birth></person></listPerson></standOff></teiCorpus>'''
    _, ents = run('sozinianische_briefwechsel', [('originals_2026-10-15/web-editions-apis/sozinianische-briefwechsel/x.xml', x)])
    assert {e.name: e.note_type for e in ents} == {'Aristoteles': 'ancient', 'Boulliau': None}
