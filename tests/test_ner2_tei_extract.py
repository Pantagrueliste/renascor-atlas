import importlib.util
import sys
from pathlib import Path

import numpy as np

_D = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(_D))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _D / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load('ner2_tei_extract')
A = _load('ner2_tei_assemble')
T = _load('ner2_tei_text')


def test_member_and_row_keys_agree():
    S = E.SOURCES
    assert S['bullinger']['mkey']('bullinger/10000.xml') == S['bullinger']['rkey']({'path': 'bullinger/10000.xml'})
    assert S['bullinger']['mkey']('bullinger/stdout.log') is None
    r = {'filename': '523607.TEI-P5.txt'}
    assert S['dtae']['mkey']('dtae/dta_komplett_1600-1699_2026-02-10/523607.TEI-P5.xml') == S['dtae']['rkey'](r)
    r = {'filename': 'nn_altonaischer0094_1698.TEI-P5.xml'}
    assert S['dtae']['mkey']('dtae/dta_komplett_1600-1699_2026-02-10/nn_altonaischer0094_1698.TEI-P5.xml') == S['dtae']['rkey'](r)
    assert S['postscriptum']['mkey']('postscriptum/ES1500/ES1500_XML-TEI_P5/PS4000_TEIP5.xml') == S['postscriptum']['rkey']({'path': 'postscriptum/ES1500/PS4000_TEIP5.xml'})
    assert S['qhod']['mkey']('qhod/qhod/QhoDstandOff.xml') == S['qhod']['rkey']({'path': 'qhod/QhoDstandOff.xml'})
    assert S['qhod']['mkey']('qhod/sema/Selim_II_Urkunde_1.xml') == S['qhod']['rkey']({'path': 'sema/Selim_II_Urkunde_1.xml'})
    assert S['germanc']['mkey']('de_dh/germanc/TEI/DRAM_P1_NoD_1673_Leonilda.xml') == S['germanc']['rkey']({'filename': 'DRAM_P1_NoD_1673_LeonildaRAW.txt'})
    assert S['croala']['mkey']('croala/txts/aa-vv-carmina-vgc.xml') == S['croala']['rkey']({'filename': 'aa-vv-carmina-vgc.txt'})
    assert S['suriano']['mkey']('suriano/datasource/tei/03/001.xml') == S['suriano']['rkey']({'filename': 'suriano_vol03.txt'}) == '03'
    assert S['dtae_dumps']['mkey']('dtae_dumps/1500-1599/staden_landschafft_1557.txt') == S['dtae_dumps']['rkey']({'filename': 'staden_landschafft_1557.TEI-P5.txt'})
    assert S['bdpn']['rkey']({'filename': 'bdpn_almanaqueylunarioultimarevision.txt'}) == S['bdpn']['mkey']('bdpn/archivosXMLDocumentos/Almanaque y Lunario (última revisión) (1).xml')


def test_v1_stem_keys():
    assert E.stem_key('originals_2026-10-17/bullinger/1.xml') == '1'
    assert E.stem_key('dtae/dta_1600-1699.zip!dta_komplett_1600-1699_2026-02-10/343012.TEI-P5.xml') == '343012'
    assert E.stem_key('de_dh/germanc.zip!TEI/NEWS_P1_NoD_1666_berlin1.xml') == E.stem_key('NEWS_P1_NoD_1666_berlin1RAW.txt')
    assert E.stem_key('originals/github/B330636101_D55869_tei.xml') == E.stem_key('B330636101_D55869')


def test_ckcc_html_spans():
    raw, spans = E.html_to_marked('<div class="letter"><p n="p1">Ad <span class="rpername" title="Bogerman, Johannes">Bogermanni</span> et <span class="i">alios</span>.</p></div>')
    assert [(raw[s:e], tag, ty, ref) for s, e, tag, ty, ref in spans] == [('Bogermanni', 'persName', 'rpername', 'Bogerman, Johannes')]
    res = T.extract_plain(raw, 'la', repair=False, spans=spans)
    assert [res['text'][t['char_start']:t['char_end']] for t in res['tags']] == ['Bogermanni']


def test_tag_kind():
    assert A.tag_kind('placeName', '') == 'place' and A.tag_kind('rs', 'address') == 'place' and A.tag_kind('name', 'person') == 'person'
    assert A.tag_kind('persName', 'author') == 'person' and A.tag_kind('orgName', '') == 'org' and A.tag_kind('name', 'artificialWork') == 'other'


def test_tag_stats():
    tags = [{'tag': 'placeName', 'type': ''}, {'tag': 'rs', 'type': 'place'}, {'tag': 'persName', 'type': 'x'}, {'tag': 'name', 'type': 'artificialWork'}, {'tag': 'orgName', 'type': ''}]
    assert dict(E.tag_stats(tags)) == {'place': 2, 'pers': 1, 'other': 1, 'org': 1}


def test_marked_edit_mark_positions():
    m = T.Marked('ab  cd', [(2, 1, 0), (4, 0, 1)])           # an end mark at 2 (before the blanks), a start mark at 4 (after them)
    m.sub(T._SP, ' ')
    assert m.text == 'ab cd' and m.pos == [2, 3]
    m = T.Marked('ab  cd', [(3, 0, 0), (3, 1, 1)])           # both inside the replaced blanks: start after, end before
    m.sub(T._SP, ' ')
    assert m.pos == [3, 2]


def test_dedup_containment_numbers():
    rng = np.random.default_rng(0)
    words = [f'w{i}' for i in range(5000)]
    a = ' '.join(rng.choice(words, 6000))
    b = ' '.join(a.split()[:3000]) + ' ' + ' '.join(rng.choice(words, 3000))
    ha, hb = T.shingle_hashes(a), T.shingle_hashes(b)
    share = len(np.intersect1d(ha, hb)) / len(hb)
    assert 0.4 < share < 0.6
    assert T.fingerprint(T.word_ids(a)) == T.fingerprint(T.word_ids(a.upper()))
    assert T.fingerprint(T.word_ids(a)) != T.fingerprint(T.word_ids(b))
