import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'gongoraobra/gongora_epistolario.xml'
real = pytest.mark.skipif(not (C.raw_root() / RAW).exists(), reason='raw data not available')

SYNTH = '''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>E</title>
<author key="G">G</author></titleStmt></fileDesc><profileDesc><creation><date notAfter="1625" notBefore="1580"/></creation></profileDesc></teiHeader>
<text><body><div><head>1. A <persName type="destinatary" key="don F">don F</persName></head>
<p>Dice <persName>el marqués</persName> a <persName type="author" key="G" ref="https://viaf.org/viaf/12309145/">Luis</persName>.
<placeName key="Madrid">Madrid</placeName>, <date>julio 4 de 1617 años</date>.</p></div>
<div><head>2. x</head><p><placeName>Toledo</placeName> sin fecha</p></div></body></text></TEI>'''.encode()


def test_synthetic():
    col = get('gongora_epistolario')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=RAW, coll='x'), ctx)
    assert es == [] and len(ms) == 5
    by = {m.surface: m for m in ms}
    assert by['don F'].role == 'addressee' and by['Luis'].role == 'author' and by['Luis'].auth['viaf'] == '12309145'
    assert by['Madrid'].role == 'sent_from' and by['Madrid'].doc_year == 1617 and by['Madrid'].date_source == 'letter'
    assert by['Toledo'].role == 'mention' and (by['Toledo'].doc_year_from, by['Toledo'].doc_year_to) == (1580, 1625)
    assert by['Toledo'].date_source == 'header' and by['el marqués'].local_key is None


@real
def test_real():
    col = get('gongora_epistolario')
    ctx = col.make_context()
    recs = col.iter_files(ctx)
    assert len(recs) == 1
    ms, _ = col.extract(recs[0], ctx)
    assert abs(len(ms) - recs[0].count('txt|persName', 'txt|placeName')) <= 5
    assert sum(m.role == 'addressee' for m in ms) >= 100
    assert all(m.tier == 'T2' and (m.context is None or len(m.context) <= 120) for m in ms)
    ms[0].to_row()
