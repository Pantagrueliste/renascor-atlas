import importlib.util
import random
from pathlib import Path

import numpy as np
from lxml import etree

_D = Path(__file__).resolve().parent.parent / 'scripts'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _D / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load('ner2_tei_text')
X = _load('ner2_layer_extract')
NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def tei(body):
    return etree.fromstring(f'<TEI {NS}><teiHeader><fileDesc>header text</fileDesc></teiHeader><text><body>{body}</body></text></TEI>'.encode('utf-8'))


def surfaces(r):
    return [(t['tag'], r['text'][t['char_start']:t['char_end']]) for t in r['tags']]


def test_ranges_follow_cleaning():
    body = '<div><p>Wir  kamen   nach  <placeName ref="l1">Lon<g ref="char:EOLhyphen">-</g><lb break="no"/>\n  don</placeName> und zu <persName>Hein-\nrich</persName>.</p></div>'
    r = T.extract_tei(tei(body), 'de')
    assert surfaces(r) == [('placeName', 'London'), ('persName', 'Heinrich')]
    assert r['tags'][0]['ref'] == 'l1' and r['place'] is True


def test_ranges_follow_dropcap_repair():
    body = '<p><placeName>L ondon</placeName> is a city. London is large. London London London London.</p>'
    r = T.extract_tei(tei(body), 'en')
    assert surfaces(r) == [('placeName', 'London')]
    assert r['n_dropcap'] == 1


def test_nested_and_whitespace_only_tags():
    body = '<p><persName>Jean de <placeName>Paris</placeName></persName> <name> </name>x</p>'
    r = T.extract_tei(tei(body), 'fr', repair=False)
    s = surfaces(r)
    assert ('persName', 'Jean de Paris') in s and ('placeName', 'Paris') in s and len(s) == 2
    assert [t['nested'] for t in r['tags']] == [False, True]


def test_exclusions_and_counts():
    body = ('<p>Salutem <note type="footnote">editor says <placeName>Rome</placeName></note><note type="margin">Roma <placeName>Romae</placeName></note>'
            '<note>unknown kind</note><app><lem>Venetiis</lem><rdg>Venetijs</rdg></app> <choice><orig>vrbe</orig><reg>urbe</reg></choice>'
            '<div type="translation"><p>Greetings from <placeName>Venice</placeName></p></div></p><head resp="#ed">Titre ajouté</head>')
    r = T.extract_tei(tei(body), 'la', repair=False)
    t = r['text']
    assert 'editor says' not in t and 'unknown kind' not in t and 'Venetijs' not in t and 'urbe' not in t.replace('vrbe', '') and 'Greetings' not in t
    assert 'Roma Romae' in t and 'Venetiis' in t and 'vrbe' in t and 'Titre' not in t
    assert [x for x in surfaces(r) if x[0] == 'placeName'] == [('placeName', 'Romae')]
    assert r['dropped_tags'] == {'placeName': 2}
    assert r['ex']['note_editorial'] > 0 and r['ex']['note_other'] > 0 and r['ex']['apparatus'] > 0 and r['ex']['div_editorial'] > 0 and r['ex']['head_editorial'] > 0


def test_note_kinds():
    def n(attrs):
        return T.note_kind(etree.fromstring(f'<note {NS} {attrs}>x</note>'))
    assert n('type="margin"') == 'keep' and n('type="Address"') == 'keep' and n('place="margin"') == 'keep' and n('place="foot"') == 'keep'
    assert n('type="footnote"') == 'editorial' and n('resp="#editor"') == 'editorial' and n('type="margin" resp="#ed"') == 'editorial' and n('type="margin" resp="#author"') == 'keep'
    assert n('type="editorial" place="margin"') == 'editorial' and n('') == 'other' and n('resp="author"') == 'other'


def test_no_body():
    assert T.extract_tei(etree.fromstring(f'<TEI {NS}><teiHeader/></TEI>'), 'en') is None


def test_equivalent_to_first_pass_reading():
    """Without notes / editorial divisions the text equals ner2_layer_extract: tei_body_text + clean_text + repair_dropcaps."""
    bodies = [
        '<div><p>The <hi>L</hi>ord went to Lon<g ref="char:EOLhyphen">-</g>∣don and <placeName>Yorke</placeName>.</p><p>S ome  tex­ t ​with\ttabs<lb/>and\n\n\n\nlines.</p></div>',
        '<p>L ondon is a city of E ngland. London England London England.<lb break="no"/> continued &amp;amp; end&#160;x</p><lg><l>Vers un</l><l>vers deux</l></lg>',
        '<p>Haec <choice><abbr>ciu.</abbr><expan>civitas</expan></choice> <del>x</del><supplied>y</supplied> <sic>sit</sic><corr>est</corr> <pb n="3"/>Roma</p>',
    ]
    for lang in ('en', 'la', 'de'):
        for b in bodies:
            root = tei(b)
            txt, _, _ = X.tei_body_text(root)
            old = X.clean_text(txt)
            old, _ = X.repair_dropcaps(old, lang, X.doc_vocab(old))
            new = T.extract_tei(root, lang, norm_newlines=False)
            assert new['text'] == old, (lang, b, new['text'], old)


def test_equivalence_random_whitespace():
    rnd = random.Random(3)
    words = ['Rome', 'Lon-', 'don', 'a', 'L', 'ondon', 'sermo', '\n', ' ', '  ', '\t', '-\n', '­', 'E', 'ngland', '\n\n', 'qui']
    for _ in range(60):
        raw = ''.join(rnd.choice(words) + rnd.choice(['', ' ', '\n']) for _ in range(rnd.randint(5, 60)))
        old = X.clean_text(raw)
        old, _ = X.repair_dropcaps(old, 'en', X.doc_vocab(old))
        new = T.extract_plain(raw, 'en')
        assert new['text'] == old, (raw, new['text'], old)


def test_plain_spans():
    r = T.extract_plain('Visit  Rome and Lon-\ndon now', 'en', repair=False, spans=[(7, 11, 'placeName', '', ''), (16, 24, 'placeName', '', '')])
    assert [r['text'][t['char_start']:t['char_end']] for t in r['tags']] == ['Rome', 'London']


def test_shingles_duplicates():
    rnd = random.Random(1)
    vocab = [f'w{i}' for i in range(3000)]
    a = ' '.join(rnd.choice(vocab) for _ in range(4000))
    b = a.replace('w1', 'Wé').upper() if False else a.upper()
    c = ' '.join(rnd.choice(vocab) for _ in range(4000))
    ha, hb, hc = T.shingle_hashes(a), T.shingle_hashes(b), T.shingle_hashes(c)
    assert len(ha) > 300 and np.array_equal(ha, hb)
    assert len(np.intersect1d(ha, hc)) / len(ha) < 0.02
    half = T.shingle_hashes(' '.join(a.split()[:2000]))
    share = len(np.intersect1d(ha, half)) / len(ha)
    assert 0.4 < share < 0.6


def test_layout_newlines_are_spaces():
    body = '<p>\n  <w>yo</w>\n  <w>he</w>\n  <w>estado</w><lb/>\n  <w>seys</w>\n</p>'
    r = T.extract_tei(tei(body), 'es', repair=False)
    assert r['text'] == 'yo he estado\nseys'


def test_generated_headings_left_out():
    r = T.extract_tei(tei('<div><head>main text of letter</head><p>Serenissimo Principe</p></div>'), 'it', repair=False)
    assert r['text'] == 'Serenissimo Principe' and r['ex']['head_editorial'] > 0
    r = T.extract_tei(tei('<seg decls="#translation" xml:lang="en">alive or dead</seg><p>Vivus aut mortuus</p>'), 'la', repair=False)
    assert r['text'] == 'Vivus aut mortuus'
