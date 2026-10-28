import importlib.util
from pathlib import Path

from lxml import etree

_spec = importlib.util.spec_from_file_location('ner2_layer_extract', Path(__file__).resolve().parent.parent / 'scripts' / 'ner2_layer_extract.py')
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


def test_stem_and_keys():
    assert X.stem_of('eebo_phase2/xml/A00001.xml') == 'A00001'
    assert X.stem_of('A00001.P5.txt') == 'A00001'
    assert X.stem_of(r'CorpusCorp/latin_xmls\10235.xml.txt') == '10235'
    assert X._noscemus_id('Noscemus_Digital_Sourcebook/668513/Schleusinger_pdf.txt') == '668513'


def test_tei_body_text_and_markup():
    xml = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc/></teiHeader><text><body><div><p>The <hi>L</hi>ord went to Lon<g ref="char:EOLhyphen">-</g>\xe2\x88\xa3don and <note>a marginal note</note>Yorke.</p></div></body></text></TEI>'''
    root = etree.fromstring(xml)
    txt, place, has_body = X.tei_body_text(root)
    assert has_body and not place
    assert 'marginal' not in txt and 'London' in txt and 'Lord' in txt
    xml2 = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>At <placeName>Rome</placeName></p></body></text></TEI>'
    assert X.tei_body_text(etree.fromstring(xml2))[1] is True
    xml3 = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>At <rs type="place">Rome</rs></p></body></text></TEI>'
    assert X.tei_body_text(etree.fromstring(xml3))[1] is True


def test_osta_noscemus_clean():
    assert 'RMK' not in X.osta_text('{RMK: x} Carta de relacio<n> e<m>biada')
    assert 'relacion' in X.osta_text('Carta de relacio<n> e<m>biada')
    assert X.noscemus_text('----\nMetadata\n-----\nEnd of metadata section\n-----\n\nTEXT').strip() == 'TEXT'
    assert X.clean_text('natu¬\nrales plurimos') == 'naturales plurimos'


def test_dropcaps():
    t = 'the city of L ondon and f rance ; the L ondon road'
    out, n = X.repair_dropcaps(t, 'en', X.doc_vocab(t))
    assert 'London' in out and n >= 2
    out, n = X.repair_dropcaps('a man and I am', 'en')
    assert n == 0


def test_dropcaps_conservative():
    t = 'the city of L ondon and the h ad of it ; J pray you ; the London road and had'
    out, n = X.repair_dropcaps(t, 'en', X.doc_vocab(t))
    assert 'City' not in out and 'London and' in out
    assert 'J pray' in out                    # J is a one-letter word
    assert 'h ad' not in out and 'had of' in out      # "had" occurs elsewhere in the document
    t2 = 'wherin h oughout was said'
    out2, n2 = X.repair_dropcaps(t2, 'en', X.doc_vocab(t2))
    assert 'h oughout' in out2                # lower case, joined form unknown: left alone
