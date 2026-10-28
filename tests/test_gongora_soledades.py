import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><sourceDesc><listWit><witness><msDesc><msIdentifier><country>España</country>
<settlement>Madrid</settlement></msIdentifier></msDesc></witness></listWit></sourceDesc></fileDesc>
<profileDesc><particDesc><listPerson><person xml:id="act001"><persName>Acteón</persName><note type="bio">Cazador de la mitología griega.</note></person>
<person xml:id="cat017"><persName>Catón</persName><note type="bio">Político romano.</note></person></listPerson></particDesc></profileDesc></teiHeader>
<text><body><l><persName ref="#act001">Acteon</persName> <persName ref="#cat017">Caton</persName> <geogName ref="#tor131">Tormes</geogName></l>
<listPlace><place xml:id="tor131"><geogName>Tormes</geogName></place></listPlace></body></text></TEI>'''.encode()


def test_synthetic():
    ms, ents = run('gongora_soledades', [('x/anotada/input/soledades_anotada.xml', SYN)], prepare=False)
    assert {e.name: e.note_type for e in ents} == {'Acteón': 'mythological', 'Catón': 'historical', 'Tormes': 'place'}
    assert len(ms) == 3 and all(m.in_register for m in ms)
    assert {(m.doc_year_from, m.doc_year_to, m.date_source) for m in ms} == {(1612, 1613, 'collection')}
    schema_ok(ms, ents)


def test_real_file():
    ms, ents = run_real('gongora_soledades', ['originals_2026-10-15/github-lowresource-tei/GitHubTEI/arojascastro__soledades/edicion/anotada/input/soledades_anotada.xml'], prepare=False)
    assert len(ms) == 308 and count(ms, kind='person') == 256
    assert sum(m.local_key is not None for m in ms) >= 306
    assert sum(e.note_type == 'mythological' for e in ents) > 70 and len(ents) == 115
