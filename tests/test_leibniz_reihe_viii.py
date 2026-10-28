import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><text><body><div><head>1. TITLE</head><head>[Frühjahr - Dezember 1675]</head>
<p>Hugenius<index indexName="Namensregister"><term>Huygens (Hugenius), Christiaan 1629-1695</term></index> in
<index indexName="Ortsregister"><term>London (Londinum)</term></index> und
<index indexName="Sachverzeichnis"><term>antlia</term><index><term>pneumatica</term></index></index></p></div></body></text></TEI>'''.encode()


def test_synthetic():
    ms, ents = run('leibniz_reihe_viii', [('originals_2026-10-15/oct-binary/oct__leibniz-reihe-viii/files/band02/x.xml', SYN)])
    assert ents == [] and len(ms) == 2                       # the subject index (Sachverzeichnis) is not an entity
    p = next(m for m in ms if m.kind == 'person')
    # v2: the key drops the parenthetical variant and the life dates
    assert p.tag == 'index@Namensregister' and p.local_key == 'leibniz_reihe_viii:person:huygens christiaan'
    assert p.in_register is False and p.context and '|' in p.context and len(p.context) <= 120
    assert (p.doc_year_from, p.doc_year_to, p.date_source) == (1675, 1675, 'header')


def test_undated_chapter_gets_volume_span():
    x = SYN.replace(b'[Fr\xc3\xbchjahr - Dezember 1675]', b'')
    ms, _ = run('leibniz_reihe_viii', [('originals_2026-10-15/oct-binary/oct__leibniz-reihe-viii/files/band01/x.xml', x)])
    assert {(m.doc_year_from, m.doc_year_to, m.date_source) for m in ms} == {(1668, 1676, 'collection')}


def test_real_chapter():
    ms, _ = run_real('leibniz_reihe_viii', ['originals_2026-10-15/oct-binary/oct__leibniz-reihe-viii/files/band01/tei/Einzeldateien/L_chapter_39.xml'])
    assert count(ms, kind='person') >= 10 and all(m.doc_year == 1672 for m in ms)
    assert any('Huygens' in m.surface for m in ms)
    schema_ok(ms)
