import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="P"><teiHeader><fileDesc><sourceDesc>
<listPerson><person xml:id="Q7198"><persName><ref target="https://wikidata.org/wiki/Q7198">Ovidius</ref></persName></person>
<person xml:id="Leena"><persName>Leena, personaggio del "Dialogo"</persName></person></listPerson>
<listPlace><place xml:id="Q16990"><placeName><ref target="https://wikidata.org/wiki/Q16990">Aetna</ref></placeName></place></listPlace></sourceDesc></fileDesc>
<profileDesc><creation><date>1477-1481</date></creation></profileDesc></teiHeader>
<text><body><p><persName ref="#Q7198">Ovidius</persName> <persName ref="Q7198">Ov.</persName> <persName ref="#Leena">L</persName>
<persName>nobody</persName> <placeName ref="#Q16990">Aetna</placeName></p></body></text></TEI>'''


def test_synthetic_wikidata_register():
    col = get('poliziano_sappho')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYN), C.FileRecord(path='p/x.xml', coll='p'), ctx)
    ent = {e.local_key.rsplit('#', 1)[1]: e for e in es}
    assert ent['Q7198'].auth['wikidata'] == 'Q7198' and ent['Q16990'].kind == 'place'
    assert ent['Leena'].note_type == 'fictional' and not ent['Leena'].auth['wikidata']
    keys = [m.local_key and m.local_key.rsplit('#', 1)[1] for m in ms]
    assert keys == ['Q7198', 'Q7198', 'Leena', None, 'Q16990']            # the bare ref="Q7198" resolves too
    assert ms[0].auth['wikidata'] == 'Q7198' and (ms[0].doc_year_from, ms[0].doc_year_to) == (1477, 1481)


def test_real_file():
    col = get('poliziano_sappho')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 1
    ms, es = col.extract(files[0], ctx)
    assert collections.Counter(e.kind for e in {e.local_key: e for e in es}.values()) == {'person': 72, 'place': 3}   # one id twice
    assert sum(1 for e in es if e.auth['wikidata']) >= 50
    assert collections.Counter(m.kind for m in ms) == {'person': 230, 'place': 8}
    assert sum(m.local_key is not None for m in ms) >= 95 and all(m.tier == 'T2' for m in ms)
