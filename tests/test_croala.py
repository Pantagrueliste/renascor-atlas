import pytest

from ratlas import common as C
from ratlas import helpers_batch1 as H

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def _rows_ok(ms):
    for m in ms[:50]:
        r = m.to_row()
        assert r['kind'] in ('person', 'place', 'org') and r['part'] in ('txt', 'hdr')
        assert set(r['auth']) >= {'gnd', 'viaf', 'wikidata', 'geonames', 'other'}
        if r['context']:
            assert len(r['context']) <= 120


def test_synthetic():
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><author ref="http://www.wikidata.org/entity/Q512372"><persName>Vrancic, Antun</persName><date>1504-1573</date></author>
    <respStmt><persName><ref type="viaf" target="6913774">Editor</ref></persName></respStmt></titleStmt>
    <publicationStmt><p><placeName ref="3186886">Zagreb</placeName></p></publicationStmt></fileDesc>
    <profileDesc><settingDesc><place ana="origin"><placeName ref="https://www.wikidata.org/wiki/Q31487">Cracovia</placeName></place>
    <place ana="destination"><placeName>Unknown</placeName></place></settingDesc>
    <particDesc><person role="sender"><persName ref="vranc01">Vrancic, Antun</persName></person>
    <person role="recipient"><persName ref="36908544">Ruiz de Moros, Pedro</persName></person></particDesc>
    <creation><date when="1542-01-10">1542</date><placeName>Krakow</placeName></creation></profileDesc></teiHeader>
    <text><body><p><persName ref="http://viaf.org/viaf/12345">Erasmus</persName> in <placeName ref="2988507">Paris</placeName></p></body></text></TEI>"""
    ms, es = H.synthetic('croala', xml)
    assert es == []
    r = {(m.role, m.surface): m for m in ms}
    assert r[('author', 'Vrancic, Antun')].auth['wikidata'] == 'Q512372'          # ref taken from <author>
    assert r[('sender', 'Vrancic, Antun')].local_key == 'croala:vranc01'
    pm = r[('addressee', 'Ruiz de Moros, Pedro')]
    assert pm.auth['viaf'] is None and 'num:36908544' in pm.auth['other']        # bare number on a person: not asserted as VIAF
    assert r[('sent_from', 'Cracovia')].auth['wikidata'] == 'Q31487'
    assert not any(k[1] in ('Unknown', 'Krakow', 'Editor', 'Zagreb') for k in r)  # placeholder, repeated creation place, editor, publisher
    assert r[('mention', 'Paris')].auth['geonames'] == '2988507' and r[('mention', 'Erasmus')].auth['viaf'] == '12345'
    assert {m.doc_year for m in ms} == {1542} and {m.date_source for m in ms} == {'header'}   # not the author's life dates


def test_real_file():
    ms, _ = H.extract_paths('croala', 'vrancic-a-epist-1542-01-10.xml')
    assert len(ms) == 5 and {m.role for m in ms} == {'author', 'sender', 'addressee', 'sent_from', 'sent_to'}
    assert {m.doc_year for m in ms} == {1542}
    _rows_ok(ms)


def test_only_txts():
    from ratlas.collections import get
    col = get('croala')
    fs = col.iter_files(col.make_context())
    assert fs and all(r.path.startswith('croala/txts/') for r in fs)
