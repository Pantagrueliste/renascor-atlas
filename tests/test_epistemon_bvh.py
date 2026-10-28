import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'originals_2026-10-15/github-neolatin-french-tei/tei-fr/Projects/BVH/Corpus/XUVA_Gordon1563_R656_tei.xml'
real = pytest.mark.skipif(not (C.raw_root() / RAW).exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="t1"><teiHeader><fileDesc><titleStmt><title>T</title>
<respStmt><persName key="MLD">Marie-Luce</persName></respStmt></titleStmt>
<sourceDesc><biblStruct><monogr><author><persName key="pers1">Elie Vinet</persName></author>
<imprint><date when="1563">1563</date></imprint></monogr></biblStruct></sourceDesc></fileDesc></teiHeader>
<text><body><p><persName key="#persf_pantagruel">Pantagruel</persName> et <name>Panurge</name> et <name type="allegorie">Folie</name>
vont a <placeName type="ville" key="#loc_paris">Paris</placeName> chez les <orgName type="peuple">Dipsodes</orgName>
<name key="antoine_loisel">Loisel</name> <geogName type="hydro">Seine</geogName></p></body></text></TEI>'''


def test_synthetic():
    col = get('epistemon_bvh')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=RAW, coll='x'), ctx)
    d = {m.surface: m for m in ms}
    assert d['Pantagruel'].local_key == 'epistemon_bvh:persf_pantagruel' and d['Pantagruel'].in_register is False
    assert d['Panurge'].kind == 'person' and d['Panurge'].tag == 'name@person' and d['Panurge'].local_key is None
    assert d['Folie'].tag == 'name@allegorie'
    assert d['Paris'].kind == 'place' and d['Paris'].local_key == 'epistemon_bvh:loc_paris'
    assert d['Dipsodes'].kind == 'org' and d['Seine'].kind == 'place'
    assert d['Loisel'].local_key == 'epistemon_bvh:t1#antoine_loisel'
    assert 'MLD' not in {m.ref_raw for m in ms}                                   # respStmt skipped
    assert d['Elie Vinet'].zone == 'hdr' and d['Elie Vinet'].role == 'author' and d['Elie Vinet'].local_key is None
    assert all(m.doc_year == 1563 for m in ms) and all(len(m.context) <= 120 for m in ms)


@real
def test_real():
    col = get('epistemon_bvh')
    ctx = col.make_context()
    rec = next(r for r in col.iter_files(ctx) if r.path == RAW)
    ms, _ = col.extract(rec, ctx)
    want = rec.count('txt|persName', 'txt|placeName', 'txt|geogName', 'txt|orgName', 'txt|name')
    assert 0.95 * want <= len([m for m in ms if m.zone == 'txt']) <= want
    assert ms[0].doc_year == 1563 and ms[0].tier == 'T2'
    ms[0].to_row()
