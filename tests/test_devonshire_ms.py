import pytest

from ratlas import common as C
from ratlas.collections import get

ARCH = 'renascor_2026-10/devonshire-manuscript/files/devonshire-ms.tar.gz'
MEMBER = ARCH + '!devonshire-ms-main/Devonshire Manuscript poems/LDev001-TM1479.xml'
real = pytest.mark.skipif(not (C.raw_root() / ARCH).exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><respStmt><name key="BB">Barbara Bond</name></respStmt></titleStmt></fileDesc>
<profileDesc><particDesc><listPerson type="historical"><person xml:id="WYATT" role="poet"><persName><ref type="LOC">n 50019425</ref>
<ref type="DNB">http://www.oxforddnb.com/view/article/30111</ref><roleName type="honorific">Sir</roleName><forename>Thomas</forename><surname>Wyatt</surname></persName>
<birth when="1503">1503</birth><death when="1542">1542</death></person></listPerson>
<listPerson type="modern"><person xml:id="NOTT" role="scholar"><persName><surname>Nott</surname></persName></person></listPerson></particDesc></profileDesc></teiHeader>
<text><body><div type="poem"><head>by <name key="WYATT">Sir Thomas Wyatt</name>, ed. <name key="NOTT">Nott</name></head>
<l><name type="author" key="WYATT">Th W</name> and <name key="DOUGLAS">Margaret</name></l></div></body></text></TEI>'''


def test_synthetic():
    col = get('devonshire_ms')
    ctx = col.make_context()
    rec = C.FileRecord(path=MEMBER, coll='x', c={'reg|person': 2})
    ctx.all_files = [rec]
    rec.data = SYNTH
    ents = col.prepare(ctx)
    assert [e.local_key for e in ents] == ['devonshire_ms:WYATT']
    e = ents[0]
    assert e.name == 'Sir Thomas Wyatt' and e.birth == '1503' and e.note_type == 'poet'
    assert 'loc:n50019425' in e.auth['other'] and 'odnb:30111' in e.auth['other']
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), rec, ctx)
    assert es == [] and [(m.surface, m.role, m.local_key) for m in ms] == [
        ('Sir Thomas Wyatt', 'mention', 'devonshire_ms:WYATT'), ('Th W', 'author', 'devonshire_ms:WYATT'),
        ('Margaret', 'mention', None)]                                   # NOTT (modern scholar) dropped, DOUGLAS unregistered here
    assert all(m.doc_year_from == 1532 and m.date_source == 'collection' and m.context is None for m in ms)


@real
def test_real():
    col = get('devonshire_ms')
    ctx = col.make_context()
    ctx.all_files = col.iter_files(ctx)
    assert len(ctx.all_files) >= 195
    ents = col.prepare(ctx)
    assert len(ents) == 25 and sum(bool(e.auth['other']) for e in ents) >= 16
    rec = next(r for r in ctx.all_files if 'LDev001-TM1479' in r.path)
    ms, es = col.extract(rec, ctx)
    assert es == [] and len(ms) == rec.count('txt|name')
    assert all(m.local_key == 'devonshire_ms:WYATT' for m in ms if m.ref_raw == 'WYATT')
