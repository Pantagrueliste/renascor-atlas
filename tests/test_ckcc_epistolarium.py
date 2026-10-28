import pytest

from ratlas import common as C
from ratlas.collections import get

B = 'originals_2026-10-15/repository-bulk-downloads/CKCC_Epistolarium/'
real = pytest.mark.skipif(not (C.raw_root() / B / 'data/corpus/meta/corpora.xml').exists(), reason='raw data not available')

PERSONS = b'''<TEI><listPerson>
<person id="bacon.francis.1561-1626"><persName><forename>Francis</forename><surname>Bacon</surname><addName>viscount St. Alban</addName></persName>
<sex value="1">male</sex><birth when="1561-01-22"/><death when="1626-04-09"/><xref subtype="ppn" type="CEN">068482817</xref><xref type="CKCC">bacon.francis.1561-1626</xref></person>
</listPerson></TEI>'''
ORGS = b'''<TEI><listPerson><person id="royal-society.london"><persName><surname type="complex">Royal Society (London)</surname></persName></person></listPerson></TEI>'''
PLACES = b'''<TEI><listPlace><place id="delft.nld"><placeName><settlement>Delft</settlement><country ref="#nld"/></placeName>
<location><geo>52.0 4.36</geo></location><xref type="CNW">delft.nl</xref></place></listPlace></TEI>'''
VOLUME = b'''<TEI.2><teiHeader><fileDesc><titleStmt><title>V</title></titleStmt></fileDesc></teiHeader><text><body>
<div type="chapter" subtype="introduction"><p>Intro <persName key="bacon.francis.1561-1626">Bacon</persName></p></div>
<div type="chapter" subtype="artifact"><interpGrp><interp type="id" value="0001"/><interp type="date" value="1650-03-01"/>
<interp type="sender" value="bacon.francis.1561-1626"/><interp type="senderloc" value="delft.nld"/><interp type="recipient" value="royal-society.london"/>
<interp type="recipientloc" value="?"/></interpGrp>
<p>Hij sprak <persName key="royal-society.london">Royal Society</persName> en <persName key="@Jan, Pieter">Jan</persName> te <placeName key="amste001">Amsterdam</placeName>
en <placeName>Leiden</placeName>.</p></div>
<div type="chapter" subtype="index"><p><placeName key="delft.nld">Delft</placeName></p></div></body></text></TEI.2>'''
TXT = b'''### 0001.xml
<TEI><teiHeader><meta type="id" value="0001"/><meta type="date" value="1615-08-31"/><meta type="sender" value="swam001"/><meta type="senderloc" value="?"/>
<meta type="recipient" value="bacon.francis.1561-1626"/><meta type="language" value="la"/></teiHeader>
<text><body><p>Cum <persName key="sten901">Stenonio</persName> &amp; <persName key="bacon.francis.1561-1626">Baconi</persName></p></body></text></TEI>
### 0002.xml
<TEI><teiHeader><meta type="id" value="0002"/><meta type="date" value="1710"/></teiHeader><text><body><p>x</p></body></text></TEI>
'''


def _ctx():
    col = get('ckcc_epistolarium')
    ctx = col.make_context()
    recs = [C.FileRecord(path=B + 'data/corpus/meta/persons/CKCC-persons-b.xml', coll='x', data=PERSONS),
            C.FileRecord(path=B + 'data/corpus/meta/persons/CKCC-organizations.xml', coll='x', data=ORGS),
            C.FileRecord(path=B + 'data/corpus/meta/places/CKCC-places-d.xml', coll='x', data=PLACES)]
    ctx.all_files = recs
    ents = col.prepare(ctx)
    return col, ctx, ents


def test_registers():
    col, ctx, ents = _ctx()
    d = {e.local_key: e for e in ents}
    b = d['ckcc_epistolarium:bacon.francis.1561-1626']
    assert b.name == 'Francis Bacon' and b.birth == '1561-01-22' and b.auth['other'] == ['ppn:068482817']
    assert 'Francis Bacon viscount St. Alban' in b.variants
    assert d['ckcc_epistolarium:royal-society.london'].kind == 'org'
    p = d['ckcc_epistolarium:delft.nld']
    assert (p.lat, p.lon, p.coord_source) == (52.0, 4.36, 'tei:geo') and p.auth['other'] == ['cnw:delft.nl']


def test_volume():
    col, ctx, _ = _ctx()
    rec = C.FileRecord(path=B + 'data/corpus/data/groo001/groo001brie01_01.xml', coll='x', data=VOLUME)
    ms, es = col.extract(rec, ctx)
    txt = [m for m in ms if m.zone == 'txt']
    assert [m.surface for m in txt] == ['Bacon', 'Royal Society', 'Jan', 'Amsterdam', 'Leiden']      # index chapter skipped
    d = {m.surface: m for m in txt}
    assert d['Royal Society'].kind == 'org' and d['Royal Society'].in_register
    assert d['Jan'].local_key is None and d['Jan'].ref_raw == '@Jan, Pieter'
    assert d['Amsterdam'].local_key == 'ckcc_epistolarium:amste001' and d['Amsterdam'].in_register is False
    assert d['Leiden'].local_key is None and d['Leiden'].ref_raw is None
    assert d['Royal Society'].doc_id == 'groo001:0001' and d['Royal Society'].doc_year == 1650
    assert d['Royal Society'].date_source == 'letter'
    assert d['Bacon'].date_source == 'collection' and d['Bacon'].doc_year == 1650        # introduction: volume span
    hdr = {(m.role): m for m in ms if m.zone == 'hdr'}
    assert set(hdr) == {'sender', 'addressee', 'sent_from'}
    assert hdr['sender'].surface == 'Francis Bacon' and hdr['sent_from'].surface == 'Delft' and hdr['addressee'].kind == 'org'
    assert all(len(m.context or '') <= 120 for m in ms)


def test_txt_corpus():
    col, ctx, _ = _ctx()
    rec = C.FileRecord(path=B + 'data/corpus/data/swam001/swam001-corpus.txt', coll='x', data=TXT)
    ms, es = col.extract(rec, ctx)
    assert es == []
    d = {m.surface: m for m in ms if m.zone == 'txt'}
    assert d['Stenonio'].local_key == 'ckcc_epistolarium:sten901' and d['Stenonio'].doc_id == 'swam001:0001'
    assert d['Baconi'].local_key == 'ckcc_epistolarium:bacon.francis.1561-1626' and d['Baconi'].doc_year == 1615
    s = next(m for m in ms if m.role == 'sender')
    assert s.local_key == 'ckcc_epistolarium:swam001' and s.in_register is False
    late = [m for m in ms if m.doc_id == 'swam001:0002']
    assert late == []                         # no tags and no metadata ids besides date: nothing to emit
    ms[0].to_row()


@real
def test_real_files():
    col = get('ckcc_epistolarium')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    ents = col.prepare(ctx)
    assert len(ents) == 4582 and sum(e.lat is not None for e in ents) > 700
    rec = next(r for r in files if r.path.endswith('groo001brie18_01.xml'))
    ms, _ = col.extract(rec, ctx)
    assert len([m for m in ms if m.zone == 'txt']) == rec.count('txt|persName', 'txt|placeName', 'txt|orgName', 'txt|geogName')
    assert all(m.doc_year == 1629 for m in ms)
    txt = next(r for r in files if r.path.endswith('leeu027-corpus.txt'))
    ms, _ = col.extract(txt, ctx)
    assert len(ms) == 332 and not any(m.in_period for m in ms)       # 83 letters 1707-1721: out of period
