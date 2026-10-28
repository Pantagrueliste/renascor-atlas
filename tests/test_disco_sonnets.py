import pytest

from ratlas import common as C
from ratlas.collections import get

BASE = 'queue5/disco-main/'
real = pytest.mark.skipif(not (C.raw_root() / BASE / 'ancillary-files/prosopography.xml').exists(), reason='raw data not available')

PROSO = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><listPerson>
<person xml:id="disco_009g"><idno cert="high" resource="https://viaf.org/viaf/7399844"/><persName type="short">Lara</persName>
<persName type="source">Ana de Lara</persName><birth><location><placeName><settlement>Toledo</settlement><country>Espa\xc3\xb1a</country></placeName></location>
<date type="century">16</date></birth><death><location><placeName><settlement>Sevilla</settlement></placeName></location><date type="century">17</date></death></person>
<person xml:id="disco_010n"><persName type="source">Nope</persName></person></listPerson></body></text></TEI>'''
SONNET = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>S</title><author>Ana de Lara</author></titleStmt></fileDesc>
<profileDesc><particDesc><listPerson><person xml:id="disco_009g"><persName type="source">Ana de Lara</persName>
<birth><location><placeName><settlement>Toledo</settlement><country>Espa\xc3\xb1a</country></placeName></location><date type="century">16</date></birth>
<death><location><placeName><settlement>Sevilla</settlement></placeName></location><date type="century">17</date></death></person></listPerson></particDesc></profileDesc></teiHeader>
<text><body><lg><l>x <name>[Ser]</name></l></lg></body></text></TEI>'''


def test_synthetic():
    col = get('disco_sonnets')
    ctx = col.make_context()
    pr = C.FileRecord(path=BASE + 'ancillary-files/prosopography.xml', coll='x', data=PROSO)
    ctx.all_files = [pr]
    ents = col.prepare(ctx)
    assert sorted((e.kind, e.name) for e in ents) == [('person', 'Ana de Lara'), ('place', 'Sevilla'), ('place', 'Toledo')]
    assert next(e for e in ents if e.kind == 'person').auth['viaf'] == '7399844'
    ms, es = col.extractor.extract_root(C.parse_xml(SONNET), C.FileRecord(path=BASE + 'tei/15th-17th/per-sonnet/disco009g_0001.xml', coll='x'), ctx)
    assert es == [] and [(m.role, m.surface, m.in_register) for m in ms] == [
        ('author', 'Ana de Lara', True), ('birth_place', 'Toledo', True), ('death_place', 'Sevilla', True)]
    assert ms[0].auth['viaf'] == '7399844' and (ms[0].doc_year_from, ms[0].doc_year_to) == (1501, 1700)   # centuries 16 -> 17
    assert all(m.in_period and m.date_source == 'author_lifespan' for m in ms)
    assert ms[0].doc_year is None                      # v2: 1501-1700 is a century range, not a year


@real
def test_real():
    col = get('disco_sonnets')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    ents = col.prepare(ctx)
    assert sum(e.kind == 'person' for e in ents) == 477
    # v2: only @cert=high VIAF ids are authority ids (166); the 36 medium / low ones are suggestions in auth.other
    assert sum(bool(e.auth['viaf']) for e in ents) == 166
    assert sum(any(x.startswith('viaf-suggested:') for x in e.auth['other']) for e in ents) == 36
    first = ctx.options['_first_sonnet']['disco_002g']
    rec = next(r for r in files if r.path == first)
    ms, _ = col.extract(rec, ctx)
    other = next(r for r in files if r.path != first and 'disco002g_' in r.path)
    assert col.extract(other, ctx)[0] == []           # the author's rows are emitted once, with the first sonnet
    assert [m.role for m in ms] == ['author', 'birth_place', 'death_place']
    assert (ms[0].doc_year_from, ms[0].doc_year_to) == (1527, 1598) and ms[0].doc_year is None and ms[0].local_key == 'disco_sonnets:disco_002g'
    ms[0].to_row()
