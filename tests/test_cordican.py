import pytest

from ratlas import common as C
from ratlas.collections import discover, get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = '''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><msDesc>
<msIdentifier><settlement>Las Palmas de Gran Canaria</settlement><repository>AHPLP</repository></msIdentifier>
<msContents><summary>Arrendamiento ... zafra del año 1518.</summary></msContents>
</msDesc>
<particDesc><person role="author"><persName>Cristóbal de San Clemente</persName></person></particDesc>
</sourceDesc>
<bibl><date when="1517" when-custom="XVI">XVI</date></bibl>
</fileDesc></teiHeader><text><body><p>otorgo a <placeName type="city">Gran Canaria</placeName> por <persName>Juan</persName>.</p></body></text></TEI>'''


def _rec(path, xml, coll='x'):
    return C.FileRecord(path=path, coll=coll, data=xml.encode('utf8'))


def test_synthetic():
    col = get('cordican')
    ctx = col.make_context()
    ms, es = col.extract(_rec('cordican/000001.xml', SYN), ctx)
    kinds = [m.kind for m in ms]
    assert 'person' in kinds and 'place' in kinds
    assert all(m.tier == 'T3' and m.context is None for m in ms)   # T3: no context snippets
    assert {m.doc_year for m in ms} == {1517}
    assert all(m.renascor_id == 'cordican' for m in ms)


def test_registry():
    d = discover()
    assert 'cordican' in d and d['cordican'].tier == 'T3'
    assert d['cordican'].module.RENASCOR_IDS == ['cordican']
