"""NER v2 test set: layout repairs of ner.tei (default off) and the re-cutting of the v1 passage regions."""
from __future__ import annotations

import dataclasses

from ratlas.ner import tei
from ratlas.ner.ner2_testset import Aligner, detached, norm_stream

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def doc(body: str) -> bytes:
    return f'<TEI {NS}><teiHeader/><text><body><p>{body}</p></body></text></TEI>'.encode('utf-8')


CHOICE = ('de <placeName><choice>\n  <orig>f</orig>\n  <reg>F</reg>\n </choice>rance</placeName> et tré<choice>\n  <orig>ſ</orig>\n  <reg>s</reg>\n'
          ' </choice>aulte et trés bon roy et ses gens')


def texts(data: bytes, **kw) -> list:
    return [p['text'] for p in tei.document_passages(data, opts=tei.Options(**kw))['passages']]


def test_default_is_unchanged():
    t = texts(doc(CHOICE))
    assert t == ['de f rance et tré ſ aulte et trés bon roy et ses gens']


def test_repair_ws_joins_the_halves_and_keeps_the_gold_span():
    d = tei.document_passages(doc(CHOICE), opts=tei.Options(repair_ws=True))
    p = d['passages'][0]
    assert p['text'] == 'de france et tréſaulte et trés bon roy et ses gens'
    (sp,) = [s for s in p['spans'] if s['label'] == 'LOC']
    assert p['text'][sp['start']:sp['end']] == 'france'
    assert 'ws_repair' in p['flags']


def test_space_after_an_abbreviation_point_is_kept():
    body = 'contra <choice><abbr>sign. ser.</abbr> <expan>signorum seriem</expan></choice>item orbita hujus continuet orbitam'
    assert texts(doc(body), repair_ws=True) == ['contra sign. ser. item orbita hujus continuet orbitam']
    body = 'vous <choice><abbr>M\'r</abbr> <expan>Monsieur</expan></choice>, J\'ay receu les lettres de nostre bon amy'
    assert texts(doc(body), repair_ws=True) == ["vous M'r, J'ay receu les lettres de nostre bon amy"]


def test_whitespace_outside_choice_is_kept():
    t = texts(doc('un deux <choice><orig>f</orig><reg>F</reg></choice> trois quatre cinq'), repair_ws=True)
    assert t == ['un deux f trois quatre cinq']


def test_dehyph_lower_case_drops_the_hyphen_and_closes_the_span():
    body = 'ich reise nach <placeName>Straßbur-<lb/>g</placeName> und dann weiter nach hause heute'
    assert texts(doc(body)) == ['ich reise nach Straßbur- g und dann weiter nach hause heute']
    d = tei.document_passages(doc(body), opts=tei.Options(dehyph=True))
    p = d['passages'][0]
    assert p['text'] == 'ich reise nach Straßburg und dann weiter nach hause heute'
    (sp,) = [s for s in p['spans'] if s['label'] == 'LOC']
    assert p['text'][sp['start']:sp['end']] == 'Straßburg'
    assert 'dehyph' in p['flags']


def test_dehyph_before_a_capital_keeps_the_hyphen_without_space():
    body = 'der Kriegs-<lb/>Rath und die Bau-<lb break="no"/>kosten sind hoch genug'
    t = texts(doc(body), dehyph=True)
    assert t == ['der Kriegs-Rath und die Baukosten sind hoch genug']


def test_dehyph_needs_a_word_before_the_hyphen():
    t = texts(doc('a - <lb/>b und noch ein paar weitere worte hier'), dehyph=True)
    assert t == ['a - b und noch ein paar weitere worte hier']


def test_regions_of_v1_are_kept_exactly():
    body = ' '.join(f'w{i}' for i in range(30)) + '</p><p>' + ' '.join(f'u{i}' for i in range(30)) + ' <placeName>Straßbur-<lb/>g</placeName> ' \
        + '</p><p>' + ' '.join(f'v{i}' for i in range(30))
    v1 = tei.document_passages(doc(body), opts=tei.Options(), max_words=40)['passages']
    assert len(v1) >= 3
    al = Aligner([p['text'] for p in v1])
    d = tei.document_passages(doc(body), opts=tei.Options(dehyph=True), max_words=40, group_fn=al)
    assert not al.failed and len(d['passages']) == len(v1) and al.kept == list(range(len(v1)))
    for a, b in zip(v1, d['passages']):
        assert norm_stream(a['text'])[0] == norm_stream(b['text'])[0]
    assert any('Straßburg' in p['text'] for p in d['passages'])


def test_control_run_without_repair_gives_back_v1():
    body = ' '.join(f'w{i}' for i in range(30)) + '</p><p>' + CHOICE + '</p><p>' + ' '.join(f'v{i}' for i in range(30))
    v1 = tei.document_passages(doc(body), opts=tei.Options(), max_words=50)['passages']
    al = Aligner([p['text'] for p in v1])
    d = tei.document_passages(doc(body), opts=tei.Options(), max_words=50, group_fn=al)
    assert [p['text'] for p in d['passages']] == [p['text'] for p in v1]
    assert [p['spans'] for p in d['passages']] == [p['spans'] for p in v1]


def test_options_default_off():
    o = tei.Options()
    assert not o.repair_ws and not o.dehyph
    assert {f.name for f in dataclasses.fields(o)} >= {'repair_ws', 'dehyph'}


def test_detached_counter():
    assert detached('de f rance et p aris') == 2
    assert detached('de france et paris') == 0
