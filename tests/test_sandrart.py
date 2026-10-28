import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import sandrart

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

ICONO = 'dtae/dta_1600-1699.zip!dta_komplett_1600-1699_2026-02-10/sandrart_iconologia_1680.TEI-P5.xml'


def test_synthetic_ref_lists():
    xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><author><persName '
           'ref="http://d-nb.info/gnd/118794396"><surname>Sandrart</surname></persName></author></titleStmt>'
           '<sourceDesc><biblFull><publicationStmt><date type="publication">1675</date></publicationStmt></biblFull>'
           '</sourceDesc></fileDesc></teiHeader><text><body><p>'
           '<persName ref="http://ta.sandrart.net/-person-4 http://d-nb.info/gnd/118794396 '
           'http://www.getty.edu/vow/ULANFullDisplay?find=&amp;role=&amp;nation=&amp;subjectid=500014974 '
           'http://viaf.org/viaf/66562250">Sandrart</persName> '
           '<placeName ref="http://ta.sandrart.net/-place-108 http://www.geonames.org/2861650/ '
           'http://www.getty.edu/vow/TGNFullDisplay?find=&amp;place=&amp;nation=&amp;subjectid=7004334">Tyber</placeName> '
           '<name type="artificialWork" ref="http://ta.sandrart.net/-artwork-5">Laocoon</name></p></body></text></TEI>')
    ctx = C.Context(slug='sandrart', licence=sandrart.LICENCE, tier='T2')
    ms, ents = sandrart.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/s.xml', coll='x'), ctx)
    assert not ents and len(ms) == 3          # author + person + place; the artwork is not read
    p = next(m for m in ms if m.surface == 'Sandrart' and m.zone == 'txt')
    assert p.local_key == 'sandrart:person-4' and p.in_register is False
    assert (p.auth['gnd'], p.auth['viaf'], p.auth['ulan']) == ('118794396', '66562250', '500014974')
    pl = next(m for m in ms if m.kind == 'place')
    assert (pl.auth['geonames'], pl.auth['tgn']) == ('2861650', '7004334') and 'sandrart:place-108' in pl.auth['other']
    assert {m.doc_year for m in ms} == {1675}


def test_real_file():
    ms, _, _ = H.extract_paths('sandrart', [ICONO])
    assert len(ms) > 3000 and {m.doc_year for m in ms} == {1680} and {m.tier for m in ms} == {'T2'}
    assert any(m.auth['gnd'] and m.kind == 'person' for m in ms) and all(len(m.context or '') <= 120 for m in ms)
