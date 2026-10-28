import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

REG = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body><listPerson type="bibl">
<person xml:id="jesus" role="bibl"><persName type="main" key="jesus">Jesus</persName><persName type="variation">JEsus</persName></person>
<person xml:id="a" role="hist"><persName type="main">A</persName></person><person xml:id="b" role="myth"><persName type="main">B</persName></person>
<person xml:id="c" role="hist"><persName type="main">C</persName></person><person xml:id="d" role="hist"><persName type="main">D</persName></person>
</listPerson></body></text></TEI>'''
ED = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><settlement>Graz</settlement></msIdentifier></msDesc>
<bibl><date when="1680">1680</date></bibl></sourceDesc></fileDesc></teiHeader><text><body><p><persName type="bibl" key="jesus"><w>JEsus</w></persName>
<placeName key="vien" type="city">Wien</placeName> <rs type="death" subtype="figu">tod</rs></p></body></text></TEI>'''


def test_synthetic_register_and_resolution():
    col = get('abacus')
    ctx = col.make_context()
    root = C.parse_xml(REG)
    ents = col.extractor.parse_register_file(C.FileRecord(path='x', coll='x'), ctx, root)
    notes = {e.local_key: e.note_type for e, _ in ents}
    assert notes['abacus:jesus'] == 'biblical' and notes['abacus:b'] == 'mythological' and notes['abacus:a'] == 'person'
    for e, al in ents:
        ctx.shared.add(e, al)
    ms, _ = col.extractor.extract_root(C.parse_xml(ED), C.FileRecord(path='x/ed.xml', coll='x'), ctx)
    assert [(m.kind, m.local_key) for m in ms] == [('person', 'abacus:jesus'), ('place', None)]   # rs and msIdentifier ignored
    assert {m.doc_year for m in ms} == {1680} and all(m.tier == 'T3' and m.context is None for m in ms)


def test_real_edition():
    col = get('abacus')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    ents = col.prepare(ctx)
    assert collections.Counter(e.kind for e in ents) == {'person': 998, 'place': 338}
    rec = next(r for r in files if r.path.endswith('Abraham-Mercks_Wienn.xml'))
    ms, _ = col.extract(rec, ctx)
    c = collections.Counter(m.kind for m in ms)
    assert c['person'] == 794 and c['place'] == 461
    assert sum(1 for m in ms if m.in_register) / len(ms) > 0.7
    assert {m.doc_year for m in ms} == {1680}
