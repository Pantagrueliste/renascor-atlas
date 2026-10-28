from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import dracor_ita, dracor_eng

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

ORFEO = 'originals_2026-10-15/dracor_api_tei/ita/poliziano-orfeo.xml'


def test_synthetic_cast_author_events():
    xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="ita9"><teiHeader><fileDesc><titleStmt><author>'
           '<persName><forename>Angelo</forename> <surname>Poliziano</surname></persName><idno type="wikidata">Q250414</idno>'
           '</author><author><persName>(Anonimo)</persName></author></titleStmt><publicationStmt><publisher>Biblioteca Italiana'
           '</publisher></publicationStmt></fileDesc><profileDesc><particDesc><listPerson>'
           '<person xml:id="orpheo" ana="http://www.wikidata.org/entity/Q174353"><persName>Orpheo</persName></person>'
           '<personGrp xml:id="coro"><name>Coro</name></personGrp></listPerson></particDesc></profileDesc></teiHeader>'
           '<standOff><listEvent><event type="print" when="1494"/><event type="written" notBefore="1479" notAfter="1480"/>'
           '</listEvent></standOff><text><body><p><name>Giove</name></p></body></text></TEI>')
    ctx = C.Context(slug='dracor_ita', licence=dracor_ita.LICENCE, tier='T2')
    ms, ents = dracor_ita.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/a.xml', coll='x'), ctx)
    c = Counter((m.role, m.surface) for m in ms)
    assert c == {('author', 'Angelo Poliziano'): 1, ('cast', 'Orpheo'): 1, ('cast', 'Coro'): 1}   # no (Anonimo), no <name>
    author = next(m for m in ms if m.role == 'author')
    assert author.auth['wikidata'] == 'Q250414' and (author.doc_year_from, author.doc_year_to) == (1479, 1480)
    orph = next(e for e in ents if e.name == 'Orpheo')
    assert orph.auth['wikidata'] == 'Q174353' and orph.note_type == 'character'
    assert next(m for m in ms if m.surface == 'Orpheo').local_key == 'dracor_ita:ita9#orpheo'


def test_real_play():
    ms, ents, _ = H.extract_paths('dracor_ita', [ORFEO])
    assert sum(m.role == 'cast' for m in ms) == 11 and sum(m.role == 'author' for m in ms) == 1
    assert {m.doc_year_from for m in ms} == {1479} and any(e.auth['wikidata'] == 'Q174353' for e in ents)
    assert {m.licence for m in ms} == {'LicenseRef-BibliotecaItaliana-NonCommercial'} and 'biblioteca-italiana' in ms[0].renascor_id


def test_eng_registry():
    from ratlas.collections import discover
    d = discover()
    assert d['dracor_eng'].tier == 'T2' and d['dracor_ita'].licence.startswith('LicenseRef-Biblioteca')
