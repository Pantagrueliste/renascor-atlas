import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'elan/elan__polygrenaille/data__tei__grenaille_la_mode.xml'
real = pytest.mark.skipif(not (C.raw_root() / RAW).exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><principal><persName sameAs="#ed"/></principal></titleStmt>
<publicationStmt><date when="2017"/></publicationStmt><sourceDesc><msDesc><msIdentifier><settlement>[A REMPLIR]</settlement></msIdentifier></msDesc></sourceDesc></fileDesc></teiHeader>
<text><front><titlePage><docTitle><titlePart>Par <persName corresp="#grenaille">FRANCOIS DE GRENAILLE</persName></titlePart></docTitle></titlePage></front>
<body><p>Voyez <persName>Seneque</persName>.</p></body></text></TEI>'''


def test_synthetic():
    col = get('elan_polygrenaille')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=RAW, coll='x'), ctx)
    assert [(m.surface, m.role, m.local_key) for m in ms] == [('FRANCOIS DE GRENAILLE', 'author', 'elan_polygrenaille:grenaille'),
                                                              ('Seneque', 'mention', None)]
    assert all(m.doc_year == 1642 and m.tier == 'T2' for m in ms)


@real
def test_real():
    col = get('elan_polygrenaille')
    ctx = col.make_context()
    rec = col.iter_files(ctx)[0]
    ms, _ = col.extract(rec, ctx)
    assert len(ms) == 1 and ms[0].role == 'author' and ms[0].context
    ms[0].to_row()
