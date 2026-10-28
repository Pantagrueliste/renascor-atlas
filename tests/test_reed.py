import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="eng"><teiHeader><fileDesc><titleStmt><title>x</title>
<respStmt><persName>Modern Editor</persName></respStmt></titleStmt></fileDesc></teiHeader><text><group>
<text type="record" xml:id="r1"><body><head><rs ref="eats:2471" type="Place_County">Yorkshire</rs> <date from-iso="1515" to-iso="1516">1515-6</date></head>
<div><p><rs ref="http://ereed.library.utoronto.ca/eats/entity/394333/" type="Person_Male">Will<ex>elm</ex>i</rs>
<rs ref="eats:1" type="Drama_Character">Deus</rs> <rs ref="eats:393980" type="Collective_GuildsOccupational">Tannours</rs>
<rs ref="eats:5" type="Collective_Office">mayor</rs></p></div></body></text>
<text type="record" xml:id="r2"><body><head>undated</head><p><rs ref="eats:2471" type="Place_County">Yorkshire</rs></p></body></text></group></text></TEI>'''


def test_synthetic_records_kinds_and_urls():
    col = get('reed')
    ctx = col.make_context()
    rec = C.FileRecord(path='eng_dh/reed.zip!Collections-master/tei/records/x.xml', coll='eng_dh')
    ms, _ = col.extractor.extract_root(C.parse_xml(SYN), rec, ctx)
    r1 = [m for m in ms if m.doc_id == 'r1']
    assert collections.Counter(m.kind for m in r1) == {'place': 1, 'person': 1, 'org': 1}   # Drama_*, Collective_Office skipped
    p = next(m for m in r1 if m.kind == 'person')
    assert p.surface == 'Willelmi' and p.ref_raw.endswith('/394333/')
    assert (p.doc_year_from, p.doc_year_to, p.in_period) == (1515, 1516, True)
    r2 = [m for m in ms if m.doc_id == 'r2']
    assert len(r2) == 1 and r2[0].in_period is None and r2[0].date_source is None


def test_real_records_and_register():
    col = get('reed')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 10
    ctx.all_files = files
    ents = col.prepare(ctx)
    kinds = collections.Counter(e.kind for e in ents)
    assert kinds['person'] > 4000 and kinds['place'] > 1700 and kinds['org'] > 100
    assert sum(e.lat is not None for e in ents if e.kind == 'place') > 1000
    assert any(e.auth.get('viaf') for e in ents) and any(e.birth for e in ents)
    rec = next(r for r in files if r.path.endswith('yorkc.xml'))
    ms, _ = col.extract(rec, ctx)
    assert len(ms) == 121 and sum(m.in_period for m in ms if m.in_period is not None) == 14
    assert sum(m.local_key is not None for m in ms) >= 118
    assert all(m.tier == 'T2' for m in ms) and max(len(m.context or '') for m in ms) <= 120
