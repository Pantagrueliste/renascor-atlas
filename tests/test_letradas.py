import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'renascor_2026-10/letradas/files/LETRADAS-0109.xml'
real = pytest.mark.skipif(not (C.raw_root() / RAW).exists(), reason='raw data not available')

SYNTH = '''<TEI><teiHeader><fileDesc><publicationStmt><idno type="corpus-num">Letradas-0001</idno></publicationStmt><sourceDesc><msDesc>
<history><origin><origDate when="1650">1650 abril 09</origDate><origPlace><placeName>Córdoba</placeName><region type="province">Córdoba</region>
<country>España</country><geo>37.89 -4.78</geo></origPlace></origin></history></msDesc></sourceDesc></fileDesc>
<profileDesc><particDesc><person role="emisor"><persName>Ana Pérez</persName></person><person role="signer"><persName>Ana Pérez</persName></person>
<person role="scriptor"><persName>Luis Gil</persName></person>
<person role="addressee"><persName>Juan Ruiz</persName><origPlace><placeName/><country>España</country></origPlace></person></particDesc></profileDesc></teiHeader>
<text><body><p>x</p></body></text></TEI>'''.encode()


def test_synthetic():
    col = get('letradas')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=RAW, coll='x'), ctx)
    assert [(m.role, m.surface) for m in ms] == [('sender', 'Ana Pérez'), ('scribe', 'Luis Gil'), ('addressee', 'Juan Ruiz'),
                                                  ('sent_from', 'Córdoba')]
    assert len(es) == 1 and es[0].lat == pytest.approx(37.89) and es[0].coord_source == 'tei:geo'
    assert ms[-1].local_key == es[0].local_key and ms[-1].doc_year == 1650 and ms[-1].in_period is True
    assert all(m.context is None and m.tier == 'T3' and m.zone == 'hdr' for m in ms)


@real
def test_real_out_of_period():
    col = get('letradas')
    ctx = col.make_context()
    rec = next(r for r in col.iter_files(ctx) if r.path == RAW)
    ms, es = col.extract(rec, ctx)
    assert {m.role for m in ms} >= {'sender', 'addressee', 'sent_from'}
    assert all(m.doc_year == 1766 and m.in_period is False for m in ms)       # 1766: out of period, kept and flagged
    assert es and es[0].name == 'Córdoba' and es[0].lat is not None
    ms[0].to_row()
