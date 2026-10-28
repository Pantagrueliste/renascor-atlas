"""v2 rules of individual collection modules (synthetic fragments; one real check for BDPN)."""
import pytest

from ratlas import classify, common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not C.raw_root().exists(), reason='raw data not available')

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def run(slug, xml, path='x/doc.xml', prepare=True, **ctxopts):
    col = get(slug)
    ctx = col.make_context()
    rec = C.FileRecord(path=path, coll='x', data=xml.encode())
    ctx.all_files = [rec]
    ents = list(col.prepare(ctx)) if prepare else []
    ms, es = col.extract(rec, ctx)
    for m in ms:
        classify.finalize(m, ctx.shared.entities.get(m.local_key) if m.local_key else None)
    return ms, ents + es, ctx


def test_casebooks_one_mention_per_case_and_person():
    xml = (f'<TEI {NS} xmlns:cb="http://www.magicandmedicine.hps.cam.ac.uk/ns/1.0"><teiHeader><profileDesc><particDesc><listPerson>'
           '<person xml:id="rn"><persName><forename>Richard</forename> <surname>Napier</surname></persName></person>'
           '<person xml:id="wc"><persName><forename>William</forename> <surname>Cooper</surname></persName></person>'
           '</listPerson></particDesc><settingDesc><p><date when="1607-02-04"/></p><cb:consultation>'
           '<cb:astrologer ref="#rn"/><cb:consultant ref="#rn"/><cb:querent ref="#wc"/><cb:subject ref="#wc"/></cb:consultation>'
           '</settingDesc></profileDesc></teiHeader><text><body><p/></body></text></TEI>')
    ms, _, _ = run('casebooks', xml, 'eng_dh/casebooks-data-master/cases/CASE7.xml', prepare=False)
    assert len(ms) == 2                                   # 4 role rows in v1
    by = {m.surface: m for m in ms}
    assert by['Richard Napier'].role == 'astrologer' and by['Richard Napier'].tag == 'person@particDesc+consultant'
    assert by['William Cooper'].role == 'subject' and by['William Cooper'].tag == 'person@particDesc+querent'


def test_cavriana_groups_and_ethnics_are_org_people():
    from ratlas.collections import cavriana
    xml = (f'<TEI {NS}><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><placeName>Florence</placeName></msIdentifier></msDesc>'
           '</sourceDesc></fileDesc><profileDesc><correspDesc><correspAction type="sent"><date when="1570-01-02"/></correspAction></correspDesc>'
           '</profileDesc></teiHeader><text><body><p><rs type="group" ref="#grp-mori">Mori</rs> <rs type="ethnic" ref="#place-svizzera">svizzeri</rs>'
           '</p></body></text></TEI>')
    col = get('cavriana')
    ctx = col.make_context()
    grp = (f'<TEI {NS}><text><body><listPerson><personGrp xml:id="grp-mori"><persName>Moors</persName><idno type="WIKIDATA">Q169260</idno>'
           '</personGrp></listPerson></body></text></TEI>')
    rec_g = C.FileRecord(path='x/letters/groupNames.xml', coll='x', data=grp.encode())
    for e, al in col.extractor.parse_register_file(rec_g, ctx, C.parse_xml(grp.encode())):
        ctx.shared.add(e, al)
    cavriana.EXTRACTOR._ctx = ctx
    ms, _ = cavriana.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/letters/1570-01-02.xml', coll='x'), ctx)
    for m in ms:
        classify.finalize(m, ctx.shared.entities.get(m.local_key) if m.local_key else None)
    by = {m.surface: m for m in ms}
    assert by['Mori'].kind == 'org' and by['Mori'].person_class == 'people' and by['Mori'].local_key == 'cavriana:grp-mori'
    assert by['svizzeri'].kind == 'org' and by['svizzeri'].person_class == 'people'
    assert by['svizzeri'].auth['geonames'] is None          # no place ids across kinds
    assert by['Florence'].zone_class == 'repository' and by['Florence'].excluded


def test_hearth_tax_rows_are_not_name_piles():
    xml = (f'<TEI {NS}><text><body><list><item><persName>John Smith</persName></item><item><persName>John Smith</persName></item>'
           '<item><persName>...</persName></item><item><persName>A vacant</persName></item>'
           '<item><persName>Willm Richinson <note>Supplied from exemption certificate</note></persName></item></list></body></text></TEI>')
    ms, _, _ = run('hearth_tax', xml, 'renascor_2026-10/hearth-tax-digital/files/x.xml', prepare=False)
    assert [m.surface for m in ms] == ['John Smith', 'John Smith', 'A vacant', 'Willm Richinson']      # '...' has no letter: dropped
    assert len({m.local_key for m in ms}) == 4 and all(m.in_register is False for m in ms)
    assert ms[2].excluded and ms[2].excluded_reason == 'vacant'


def test_reed_record_heading_is_provenance():
    from ratlas.collections import reed
    xml = (f'<TEI {NS}><text type="record" xml:id="rec1"><body><head><rs type="Place_County" ref="eats:1">Berkshire</rs>, '
           '<rs type="Place_PA_Settlement" ref="eats:2">Reading</rs> <date when="1566"/></head><p>at <rs type="Place_PA_Settlement" ref="eats:2">Reading</rs>'
           '</p></body></text></TEI>')
    col = get('reed')
    ctx = col.make_context()
    ms, _ = col.extractor.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/records/r.xml', coll='x'), ctx)
    for m in ms:
        classify.finalize(m)
    assert [(m.role, m.zone_class, m.excluded) for m in ms] == [('provenance', 'header', True), ('provenance', 'header', True),
                                                                ('mention', 'text', False)]


def test_rejestr_poznan_parallel_witness():
    xml = (f'<TEI {NS}><text><body><div xml:id="ASK91"><rs type="person" ref="Q10">A</rs></div>'
           '<div xml:id="ASK3"><rs type="person" ref="Q10">A</rs></div></body></text></TEI>')
    ms, _, _ = run('rejestr_poznan', xml, 'renascor_2026-10/rejestr-kontrybucji-poznan-1561/files/spisy_1561.xml', prepare=False)
    assert [(m.doc_id, m.excluded) for m in ms] == [('ASK91', False), ('ASK3', True)]
    assert ms[1].excluded_reason == 'parallel_witness'


def test_polemica_gongorina_quote_markers():
    xml = (f'<TEI {NS}><text><body><p><persName type="authority" key="Propercio" unit="quote" quantity="1"/> '
           '<name type="polemista">vuesa merced</name> <name type="polemista">Faría</name></p></body></text></TEI>')
    ms, _, _ = run('polemicagongorina_obvil', xml, 'polemos/x.xml', prepare=False)
    assert [(m.surface, m.role, m.excluded) for m in ms] == [('', 'quote_source', True), ('Faría', 'mention', False)]


def test_lettres_calvin_envoyee_a_is_sent_to():
    xml = (f'<TEI {NS}><teiHeader><fileDesc><sourceDesc/></fileDesc></teiHeader><text><body><opener><dateline>De '
           '<placeName ref="https://www.wikidata.org/wiki/Q6602">Strasbourg</placeName>. Envoyée à '
           '<placeName ref="https://www.wikidata.org/wiki/Q90">Paris</placeName>.</dateline></opener></body></text></TEI>')
    ms, _, _ = run('lettres_calvin', xml, 'renascor_2026-10/lettres-calvin/files/x.xml', prepare=False)
    assert [(m.surface, m.role) for m in ms] == [('Strasbourg', 'sent_from'), ('Paris', 'sent_to')]


def test_leibniz_apparatus_and_empty_terms():
    xml = (f'<TEI {NS}><text><body><div><head>[1675]</head><p>x<index indexName="Namensregister"><term>Huygens (Hugenius), Christiaan 1629-1695</term></index>'
           '<app><lem>y<index indexName="Namensregister"><term>Huygens, Christiaan</term></index></lem><rdg>z</rdg></app>'
           '<index indexName="Namensregister"><term>,</term></index></p></div></body></text></TEI>')
    ms, _, _ = run('leibniz_reihe_viii', xml, 'originals_2026-10-15/oct-binary/oct__leibniz-reihe-viii/files/band01/x.xml', prepare=False)
    assert [(m.zone_class, m.excluded) for m in ms] == [('text', False), ('apparatus', True)]
    assert len({m.local_key for m in ms}) == 1


def test_pceec2_decade_date(tmp_path):
    from ratlas.collections import pceec2
    ctx = C.Context(slug='pceec2', licence='x', tier='T3')
    txt = '(METADATA (AUTHOR A_B:m:_:1500:30) (RECIPIENT C_D:f:_:1510:20) (LETTER L1:x:1570S:y:z))'
    rec = C.FileRecord(path='p/x.psd', coll='x', data=txt.encode('latin-1'))
    ms, _ = pceec2.extract(rec, ctx)
    assert {(m.doc_year_from, m.doc_year_to, m.doc_year, m.in_period) for m in ms} == {(1570, 1579, 1574, True)}


def test_elan_crenum_split_initial_is_joined():
    xml = (f'<TEI {NS}><text><body><l>Comme heritier de la <placeName><choice>\n  <orig>f</orig>\n  <reg>F</reg>\n </choice>rance</placeName> '
           '<persName ref="#charlemagne"><choice><orig>c</orig><reg>C</reg></choice>harles</persName></l></body></text></TEI>')
    ms, _, _ = run('elan_crenum', xml, 'elan/elan__crenum/x.xml', prepare=False)
    assert [m.surface for m in ms] == ['France', 'Charles']


def test_tagebuecher_metonymy_keeps_person_kind():
    xml = (f'<TEI {NS} xml:id="d1"><teiHeader><fileDesc><titleStmt><title><date when="1635-02"/></title></titleStmt><sourceDesc>'
           '<listPlace><place xml:id="cassel"><placeName ref="https://www.geonames.org/2886242/kassel.html">Kassel</placeName></place></listPlace>'
           '</sourceDesc></fileDesc></teiHeader><text><body><p><rs type="person" ref="#cassel">Cassel</rs> <rs type="place" ref="#cassel">Cassel</rs></p>'
           '</body></text></TEI>')
    ms, _, _ = run('tagebuecher_christian_ii_anhalt', xml, 'renascor_2026-10/tagebuecher-christian-ii-anhalt/files/1635_02.xml')
    p, pl = ms
    assert p.kind == 'person' and p.person_class == 'metonym' and p.auth['geonames'] is None and p.in_register is False
    assert pl.kind == 'place' and pl.auth['geonames'] == '2886242' and pl.in_register is True


def test_croala_author_variants_collapse():
    xml = (f'<TEI {NS}><teiHeader><fileDesc><titleStmt><author ref="https://www.wikidata.org/entity/Q123">'
           '<persName>Ciriaco d\'Ancona</persName><persName>Kyriacus Anconitanus</persName></author></titleStmt></fileDesc></teiHeader>'
           '<text><body><p>x</p></body></text></TEI>')
    ms, _, _ = run('croala', xml, 'croala/txts/x.xml', prepare=False)
    assert [m.role for m in ms].count('author') <= 1


def test_paesi_novamente_entities_named_by_slug():
    xml = (f'<TEI {NS}><text><body><p><rs type="place" ref="#cap_vert">Capouerde</rs> <rs type="place" ref="#kozhikode">Calichuth</rs></p>'
           '</body></text></TEI>')
    ms, ents, _ = run('paesi_novamente', xml, 'renascor_2026-10/paesi-novamente-retrovati-hab/files/x.xml')
    assert {e.name for e in ents} == {'Cap Vert', 'Kozhikode'} and all(m.in_register for m in ms)
    assert 'Capouerde' in next(e for e in ents if e.name == 'Cap Vert').variants


def test_epistemon_classes_from_key_prefix_and_lexicon():
    xml = (f'<TEI {NS}><text><body><p><persName key="#persf_panurge">Panurge</persName> <name>Vénus</name> <name>Dieu</name> '
           '<persName key="#persh_homer">Homère</persName> <orgName type="peuple">Gaulois</orgName></p></body></text></TEI>')
    ms, _, _ = run('epistemon_bvh', xml, 'originals_2026-10-15/github-neolatin-french-tei/tei-fr/Projects/BVH/x.xml', prepare=False)
    assert [(m.surface, m.person_class) for m in ms] == [('Panurge', 'character'), ('Vénus', 'mythological'), ('Dieu', 'biblical'),
                                                         ('Homère', 'human'), ('Gaulois', 'people')]


def test_bdpn_celestial_bodies_are_classed_and_excluded():
    from ratlas import helpers_batch0 as H0
    ALMANAQUE = 'bdpn/archivosXMLDocumentos/Discurso_y_relacion_cometographa.xml'
    ms, ents, ctx = H0.extract_paths('bdpn', [ALMANAQUE])
    byk = {e.local_key: e for e in ents}
    for m in ms:
        classify.finalize(m, byk.get(m.local_key))
    cel = [m for m in ms if m.kind == 'place' and m.person_class == 'celestial']
    assert cel and all(m.excluded and 'celestial' in m.excluded_reason for m in cel)
    assert any(m.kind == 'place' and not m.excluded for m in ms)          # real places stay in the counts


def test_ckcc_letters_in_two_corpora_are_counted_once():
    from ratlas.collections import ckcc_epistolarium as K
    a = ('<TEI.2><text><body><div type="chapter" subtype="artifact"><interpGrp><interp type="id" value="1"/><interp type="date" value="1640-01-02"/>'
         '<interp type="sender" value="a.b.1600-1650"/><interp type="recipient" value="c.d.1600-1650"/></interpGrp><p>x</p></div></body></text></TEI.2>')
    b = a.replace('value="1"', 'value="77"')
    r1 = C.FileRecord(path='x/data/corpus/data/groo001/groo001brie01_01.xml', coll='x', data=a.encode())
    r2 = C.FileRecord(path='x/data/corpus/data/huyg003/huyg003oeuv01_01.xml', coll='x', data=b.encode())
    docs = {}
    for r in (r1, r2):
        for doc_id, date, s, rc in K._letters_of(r):
            docs.setdefault((date, s, rc), []).append(doc_id)
    assert list(docs.values()) == [['groo001:1', 'huyg003:77']]
