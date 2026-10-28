import pytest

from ratlas import common as C
from ratlas.collections import get
from ratlas.collections.dta_1600 import licence_of

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>t</title>
<author><persName ref="http://d-nb.info/gnd/12457467X"><surname>Seiler</surname><forename>Tobias</forename></persName></author>
<editor><persName ref="http://d-nb.info/gnd/115266127"><surname>Geyken</surname></persName></editor></titleStmt>
<publicationStmt><date type="publication">2025-10-24T10:12:37Z</date><availability>
<licence target="http://creativecommons.org/licenses/by-nc/3.0/de/"><p>BY-NC</p></licence></availability></publicationStmt>
<sourceDesc><biblFull><titleStmt><author><persName ref="http://d-nb.info/gnd/12457467X"><surname>Seiler</surname><forename>Tobias</forename></persName></author></titleStmt>
<publicationStmt><date type="publication">1605</date></publicationStmt></biblFull></sourceDesc></fileDesc></teiHeader>
<text><body><p>Herr <persName ref="https://www.wikidata.org/wiki/Q505548">Churf</persName> in <placeName>Dresden</placeName></p></body></text></TEI>'''


def test_synthetic_author_once_licence_and_date():
    col = get('dta_1600')
    ctx = col.make_context()
    rec = C.FileRecord(path='dtae/dta_1600-1699.zip!x/343014.TEI-P5.xml', coll='dtae')
    ms, _ = col.extractor.extract_root(C.parse_xml(SYN), rec, ctx)
    auth = [m for m in ms if m.role == 'author']
    assert len(auth) == 1 and auth[0].auth['gnd'] == '12457467X' and auth[0].zone == 'hdr' and auth[0].context is None
    assert not any('Geyken' in m.surface for m in ms)                     # modern editor
    assert {m.doc_year for m in ms} == {1605}                              # imprint, not the 2025 publication
    assert [m.auth['wikidata'] for m in ms if m.surface == 'Churf'] == ['Q505548']
    assert licence_of(C.parse_xml(SYN)) == ('CC-BY-NC-3.0-DE', 'T2')


def test_licence_variants():
    mk = lambda t: C.parse_xml(f'<TEI><teiHeader><licence target="{t}"/></teiHeader></TEI>'.encode())
    assert licence_of(mk('http://creativecommons.org/licenses/by-sa/4.0/deed.de')) == ('CC-BY-SA-4.0', 'T1')
    assert licence_of(mk('http://creativecommons.org/licenses/by-sa/3.0/de/')) == ('CC-BY-SA-3.0-DE', 'T1')
    assert licence_of(mk('http://creativecommons.org/publicdomain/zero/1.0/')) == ('CC0-1.0', 'T1')


def test_real_files():
    col = get('dta_1600')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 1107 and not any('sandrart' in r.name.lower() for r in files)
    rec = next(r for r in files if r.name.startswith('dresden_hofdiarium_1665'))
    ms, _ = col.extract(rec, ctx)
    assert 3200 < len(ms) < 3220 and {m.doc_year for m in ms} == {1665}
    assert sum(1 for m in ms if m.auth['wikidata']) > 1500
