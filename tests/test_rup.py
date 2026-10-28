from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

from ratlas import helpers_v2 as V

_ZIP = V.NEW + '/rup/rup.zip' if V.use_new() else 'eng_dh/rup.zip'
POEM = _ZIP + '!Rare-or-Unique-Poems-from-manuscripts-1500-1660-FOLGER-main/poems/rup_000003_va103_1_005r.xml'


def test_poem_with_attributed_author():
    ms, ents, ctx = H.extract_paths('rup', [POEM])
    by = Counter((m.kind, m.role, m.zone) for m in ms)
    assert by[('person', 'author', 'hdr')] == 1          # respStmt 'attributed author' -> P0003
    author = next(m for m in ms if m.role == 'author')
    assert author.local_key == 'rup:P0003' and author.doc_year == 1618 and author.date_source == 'header'
    assert any(m.surface == 'Lady Vayne' and m.local_key == 'rup:P0004' for m in ms)
    assert any(m.kind == 'org' and m.surface == 'Westminster Abbey' for m in ms)
    assert len([e for e in ents if e.kind == 'person']) == 22


def test_v2_one_more_poem_and_dangling_ref():
    new = _ZIP + '!Rare-or-Unique-Poems-from-manuscripts-1500-1660-FOLGER-main/poems/rup_000021_add11811_3v.xml'
    ms, ents, _ = H.extract_paths('rup', [new])
    from ratlas import classify
    assert ms and all(classify.finalize(m).zone_class in ('text', 'header') for m in ms)
    # persons.xml#P0023 dangles upstream (the register ends at P0022): unresolved, not invented
    assert any(m.ref_raw and 'P0023' in m.ref_raw and m.local_key is None for m in ms)


def test_register_auth():
    _, ents, _ = H.extract_paths('rup', [POEM])
    anne = next(e for e in ents if e.local_key == 'rup:P0002')
    assert anne.birth == '12 December 1574' and 'loc:n85355622' in anne.auth['other']


def test_synthetic_respstmt():
    xml = (b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><respStmt><resp>attributed author</resp>'
           b'<name>Mr W.J.</name></respStmt><respStmt><resp>TEI encoding</resp><name>Modern Editor</name></respStmt>'
           b'</titleStmt></fileDesc></teiHeader><text><body><p>x</p></body></text></TEI>')
    hits = H.author_respstmt_hits(C.parse_xml(xml))
    assert [h.surface for h in hits] == ['Mr W.J.'] and hits[0].role == 'author'


def test_cited_modern_editors_are_dropped():
    ms, _, _ = H.extract_paths('rup', [_ZIP + '!Rare-or-Unique-Poems-from-manuscripts-1500-1660-FOLGER-main/poems/'
                                      'rup_000013_va103_1_009v.xml'])
    assert not any(m.surface in ('Jonathan Barry', 'Peter Elmer') for m in ms)
    assert any(m.surface == 'Dr Johnson' for m in ms)
