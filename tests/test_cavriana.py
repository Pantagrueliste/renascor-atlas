from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import cavriana

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

from ratlas import helpers_v2 as V

_B = (V.NEW + '/cavriana') if V.use_new() else 'cavriana'
LETTER = _B + '/letters/1570-09-01.xml'
HEADER_ONLY = _B + '/letters/1570-09-24.xml'


def test_synthetic_fixes():
    from ratlas.model import Entity
    ctx = C.Context(slug='cavriana', licence='CC-BY-4.0', tier='T1')
    ctx.shared.add(Entity(local_key='cavriana:place-firenze', coll_slug='cavriana', kind='place', name='Florence',
                          variants=['Firenze'], source_path='x'), ['place-firenze'])
    ctx.shared.add(Entity(local_key='cavriana:pers-concini-b', coll_slug='cavriana', kind='person', name='B. Concini',
                          source_path='x'), ['pers-concini-b'])
    xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier>'
           '<placeName>Florence</placeName></msIdentifier></msDesc></sourceDesc></fileDesc><profileDesc><correspDesc>'
           '<correspAction type="sent"><persName ref="#pers-concino-b">C</persName><placeName>Firenze</placeName>'
           '<date when="1570-01-02"/></correspAction></correspDesc></profileDesc></teiHeader><text><body>'
           '<p><placeName ref="#place-florence">Fiorenza</placeName></p></body></text></TEI>')
    rec = C.FileRecord(path='cavriana/letters/x.xml', coll='cavriana')
    cavriana.EXTRACTOR._ctx = ctx
    ms, _ = cavriana.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), rec, ctx)
    by = {(m.role, m.kind): m for m in ms}
    assert by[('repository', 'place')].local_key == 'cavriana:place-firenze'
    assert by[('sender', 'person')].local_key == 'cavriana:pers-concini-b'
    assert by[('sent_from', 'place')].local_key == 'cavriana:place-firenze'
    assert by[('mention', 'place')].local_key == 'cavriana:place-firenze'
    assert all(m.context is None for m in ms if m.zone == 'hdr')


def test_real_letters():
    ms, ents, ctx = H.extract_paths('cavriana', [LETTER, HEADER_ONLY])
    c = Counter((m.kind, m.zone) for m in ms)
    assert c[('person', 'txt')] == 79 + 0 and c[('place', 'txt')] >= 31
    sender = [m for m in ms if m.role == 'sender']
    assert {m.local_key for m in sender} == {'cavriana:pers-cavriana-f'}
    assert any(m.role == 'sent_from' and m.local_key == 'cavriana:place-paris' and m.auth['tgn'] == '7008038' for m in ms)   # v2: TGN re-derived upstream (was 7008882)
    assert any(m.role == 'addressee' and m.auth['viaf'] == '59259382' for m in ms)
    assert {m.doc_year for m in ms} == {1570} and {m.date_source for m in ms} == {'letter'}
    paris = next(e for e in ents if e.local_key == 'cavriana:place-paris')
    assert paris.name == 'Paris' and 'Parigi' in paris.variants and paris.lat == 48.8566
