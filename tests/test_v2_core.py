"""Tests of the v2 rules (audit of 2026-10-16): zones, classes, default exclusion, tag/entry kind mismatch, dates, keys, surfaces.

Synthetic data only, plus a few real files when the raw data is present.
"""
from collections import Counter

import pytest

from ratlas import classify, common as C
from ratlas import helpers_v2 as V
from ratlas.model import Entity, Mention, best_year, period_overlap

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def _ctx(**kw):
    return C.Context(slug='t', licence='CC-BY-4.0', tier='T1', **kw)


def _mentions(xml, ctx=None, local=None, **kw):
    ctx = ctx or _ctx()
    root = C.parse_xml(xml.encode())
    rec = C.FileRecord(path='x/doc.xml', coll='x')
    ms = C.build_mentions(root, rec, ctx, local=local, **kw)
    return [classify.finalize(m, None) for m in ms]


# ---- dates (audit point 12) ---------------------------------------------------------------------------------------------
def test_wide_ranges_have_no_year_and_straddling_wide_ranges_are_unknown():
    assert best_year(1500, 1500) == 1500 and best_year(1500, 1510) == 1505
    assert best_year(1500, 1521) is None and best_year(1395, 1619) is None and best_year(None, None) is None
    assert period_overlap(1500, 1600) is True            # a wide range inside the period is in the period
    assert period_overlap(1395, 1619) is None            # straddles 1450: we do not know
    assert period_overlap(550, 1616) is None             # CELT Annals of the Four Masters
    assert period_overlap(1690, 1705) is True            # narrow: overlap, as in v1
    assert period_overlap(1701, 1799) is False and period_overlap(1200, 1300) is False and period_overlap(None, None) is None


# ---- zones and classes (audit points 7-8) ----------------------------------------------------------------------------
XML_ZONES = f'''<TEI {NS}><teiHeader><fileDesc><titleStmt><title>t</title></titleStmt>
<sourceDesc><msDesc><msIdentifier><settlement>Florence</settlement></msIdentifier></msDesc></sourceDesc></fileDesc>
<profileDesc><abstract><p>Regest: <rs type="person" ref="#a">Luther</rs></p></abstract></profileDesc></teiHeader>
<text><body><p><rs type="person" ref="#a">Luther</rs>
<note type="footnote"><rs type="person" ref="#a">Luther</rs></note>
<note place="margin"><rs type="person" ref="#a">Luther</rs></note>
<note place="foot"><rs type="person" ref="#a">Luther</rs></note>
<note type="crit_app"><rs type="person" ref="#a">Luther</rs></note>
<note resp="#editor"><rs type="person" ref="#a">Luther</rs></note>
<app><lem><rs type="person" ref="#a">Luther</rs></lem><rdg><rs type="person" ref="#a">Luther</rs></rdg></app></p></body></text></TEI>'''


def test_zone_classes():
    ms = _mentions(XML_ZONES)
    got = [(m.zone, m.zone_class, m.excluded) for m in ms]
    assert got == [('hdr', 'repository', True),          # settlement of the msIdentifier
                   ('hdr', 'regest', True),               # profileDesc/abstract
                   ('txt', 'text', False),
                   ('txt', 'footnote', True),
                   ('txt', 'text', False),                # note place=margin: the manuscript's own marginalia
                   ('txt', 'text', False),                # note place=foot alone: the original's footnote (DTA), not evidence of an editor
                   ('txt', 'apparatus', True),
                   ('txt', 'footnote', True),             # note resp=#editor
                   ('txt', 'text', False),                # <lem> is the base reading
                   ('txt', 'apparatus', True)]            # <rdg>
    assert ms[0].excluded_reason == 'repository' or 'repository' in ms[0].excluded_reason


def test_class_of_note_type_and_tag():
    assert classify.class_of_note_type('mythological', 'person') == 'mythological'
    assert classify.class_of_note_type('biblical', 'person') == 'biblical'
    assert classify.class_of_note_type('fictional', 'person') == 'character'
    assert classify.class_of_note_type('character', 'person') == 'character'
    assert classify.class_of_note_type('dynasty', 'person') == 'dynasty'
    assert classify.class_of_note_type('celestial', 'place') == 'celestial'
    assert classify.class_of_note_type('place', 'place') is None
    assert classify.class_of_note_type('person', 'person') == 'human' and classify.class_of_note_type(None, 'person') == 'human'
    assert classify.class_of_note_type('nation', 'org') == 'people'
    assert classify.class_of_tag('name@myth', 'person') == 'mythological'
    assert classify.class_of_tag('name@religion', 'person') == 'biblical'
    assert classify.class_of_tag('name@nomen sacrum', 'person') == 'mythological'
    assert classify.class_of_tag('name@nation', 'org') == 'people'
    assert classify.class_of_tag('persName', 'person') is None


def test_placeholders_and_empty_surfaces_are_excluded_not_dropped_when_referenced():
    for s in ('S.l.', 's.l.', 'Pl0', 'UNKNOWN', 'Anónimo-12', 'Anonimo 3', 'N. N.', 'Anonymus', 'unbekanntes Kloster', '?'):
        assert classify.is_placeholder(s), s
    for s in ('Paris', 'Anna', 'Slovenia', 'Nonne'):
        assert not classify.is_placeholder(s), s
    xml = f'<TEI {NS}><text><body><p><persName>Anónimo-4</persName> <persName/> <persName ref="#a"/> <placeName>S.l.</placeName></p></body></text></TEI>'
    ms = _mentions(xml)
    assert [m.surface for m in ms] == ['Anónimo-4', '', 'S.l.']       # the tag with no letter and no ref is dropped, the one with a ref kept
    assert [m.excluded_reason for m in ms] == ['placeholder', None, 'placeholder'] or ms[1].excluded is False


def test_cast_and_celestial_are_excluded_but_kept():
    m = Mention(coll_slug='t', source_path='x', doc_id='d', kind='person', tag='person@cast', surface='Servant', role='cast', zone='hdr')
    classify.finalize(m)
    assert m.person_class == 'character' and m.excluded and m.excluded_reason == 'character' and m.zone_class == 'header'
    ent = Entity(local_key='t:sol', coll_slug='t', kind='place', name='Sol', source_path='x', note_type='celestial')
    p = Mention(coll_slug='t', source_path='x', doc_id='d', kind='place', tag='placeName', surface='Sol', local_key='t:sol')
    classify.finalize(p, ent)
    assert p.person_class == 'celestial' and p.excluded and 'celestial' in p.excluded_reason
    h = Mention(coll_slug='t', source_path='x', doc_id='d', kind='person', tag='persName', surface='Anna')
    classify.finalize(h)
    assert h.person_class == 'human' and not h.excluded


def test_roles_provenance_and_quote_source_are_excluded():
    for role in ('provenance', 'quote_source'):
        m = Mention(coll_slug='t', source_path='x', doc_id='d', kind='place', tag='rs', surface='Berkshire', role=role)
        classify.finalize(m)
        assert m.excluded and m.excluded_reason == role


def test_module_set_reasons_survive_finalize():
    m = Mention(coll_slug='t', source_path='x', doc_id='d', kind='person', tag='persName', surface='Anna', excluded_reason='duplicate_letter')
    classify.finalize(m)
    classify.finalize(m)
    assert m.excluded and m.excluded_reason == 'duplicate_letter'


# ---- tag type vs register entry type (audit point 7) --------------------------------------------------------------------
def test_kind_mismatch_keeps_the_kind_and_inherits_nothing():
    ctx = _ctx()
    place = Entity(local_key='t:cassel', coll_slug='t', kind='place', name='Kassel', source_path='x',
                   auth=C.parse_auth(['https://www.geonames.org/2886242', 'https://d-nb.info/gnd/4030691-2']),
                   lat=51.3, lon=9.5)
    ctx.shared.add(place, ['cassel'])
    xml = f'<TEI {NS}><text><body><p><rs type="person" ref="#cassel" cert="medium">Cassel</rs> <rs type="place" ref="#cassel">Cassel</rs>' \
          f' <rs type="person" ref="https://d-nb.info/gnd/118540238 #cassel">Goethe</rs></p></body></text></TEI>'
    ms = _mentions(xml, ctx=ctx)
    pers, plc, own = ms
    assert pers.kind == 'person' and pers.auth['geonames'] is None and pers.auth['gnd'] is None
    assert pers.local_key == 't:cassel~person' and pers.in_register is False and pers.person_class == 'metonym'
    assert pers.cert == 'medium'
    assert plc.local_key == 't:cassel' and plc.auth['geonames'] == '2886242'
    assert own.auth['gnd'] == '118540238' and own.auth['geonames'] is None      # the tag's own id stays


# ---- surfaces (audit issues 71-72, 108) -----------------------------------------------------------------------------------
def test_surface_cleaning_and_nested_notes():
    xml = f'''<TEI {NS}><text><body><p><placeName>Lon- dres</placeName> <placeName>Hagam.</placeName> <placeName>Frf.</placeName>
    <persName ref="#x">Rittershusio<note>Der gelehrte Jurist Konrad Rittershausen gehörte zu</note></persName> <persName>Marquis de Po- linville,</persName></p></body></text></TEI>'''
    ms = _mentions(xml)
    assert [m.surface for m in ms] == ['Londres', 'Hagam', 'Frf.', 'Rittershusio', 'Marquis de Polinville']


def test_term_key_and_clean_surface():
    assert V.term_key('Huygens (Hugenius), Christiaan 1629-1695') == V.term_key('Huygens, Christiaan 1629–1695') == 'huygens christiaan'
    assert V.term_key(',') == ''
    assert V.clean_surface('Lon- dres') == 'Londres' and V.clean_surface('Paris, France.') == 'Paris, France'


def test_running_date_ignores_typos_and_uses_the_preceding_date():
    xml = f'''<TEI {NS}><text><body>
    <div><date when="1563-01-02"/><p><persName>A</persName></p></div>
    <div><date when="1564-03-04"/><p><persName>B</persName></p></div>
    <div><date when="1656-03-04"/><p><persName>C</persName></p></div>
    <div><date when="1565-03-04"/><p><persName>D</persName></p></div>
    <div><date when="1566-03-04"/><date when="1566-05-04"/><date when="1567-01-01"/></div></body></text></TEI>'''
    root = C.parse_xml(xml.encode())
    rd = V.RunningDate(root, 1500, 1700)
    els = {C.surface_text(e): e for e in root.iter() if C.ln(e.tag) == 'persName'}
    assert [rd.year_at(els[k]) for k in 'ABCD'] == [1563, 1564, 1564, 1565]       # 1656 is a typo: ignored, the 1564 date stays


# ---- registers ---------------------------------------------------------------------------------------------------------
def test_register_ptr_note_and_hash_authority_tokens():
    xml = f'''<listPlace {NS}><place xml:id="amsterdam"><placeName>Amsterdam</placeName>
    <note type="references"><ptr type="GettyThesaurus" target="#TNG_7006952"/></note></place>
    <place xml:id="agypten" type="country"><country>Ägypten</country><location><geo>27.0 30.0</geo></location>
    <idno type="url">http://geonames.org/ 357994</idno></place>
    <place xml:id="x"><placeName>X</placeName><listBibl><bibl><ref target="http://www.bbkl.de/x">BBKL</ref></bibl></listBibl></place></listPlace>'''
    ctx = _ctx()
    ents, _ = C.parse_register(C.parse_xml(xml.encode()), ctx, 'x')
    by = {e.name: e for e, _a in ents}
    assert by['Amsterdam'].auth['tgn'] == '7006952'                    # '#TNG_' typo of the HAB registers
    assert by['Ägypten'].auth['geonames'] == '357994' and by['Ägypten'].lat == 27.0
    assert C.classify_token('PND_100728340') == ('gnd', '100728340') and C.classify_token('#TGN_7001242') == ('tgn', '7001242')


# ---- real data ---------------------------------------------------------------------------------------------------------
real = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()), reason='raw data / scan not available')


@real
def test_scan_extra_makes_the_reharvest_visible():
    from ratlas.collections import get
    for slug, n in (('cavriana', 100), ('dracor_eng', 700), ('kaiserin_eleonora_magdalena', 1600), ('abacus', 5)):
        col = get(slug)
        files = col.iter_files(col.make_context())
        assert len(files) >= n, slug
        if V.use_new():
            assert all('originals_2026-10-17' in f.path for f in files), slug


@real
def test_dracor_cast_is_excluded_authors_are_not():
    from ratlas import helpers_batch0 as H0
    ms, ents, _ = H0.extract_paths('dracor_eng', [V.NEW + '/dracor_eng/alexander-croesus.xml'])
    for m in ms:
        classify.finalize(m)
    cast = [m for m in ms if m.role == 'cast']
    assert cast and all(m.excluded and m.person_class == 'character' for m in cast)
    authors = [m for m in ms if m.role == 'author']
    assert authors and all(not m.excluded and m.auth['wikidata'] for m in authors)


@real
def test_hainhofer_reisen_lod_register_names_the_slugs():
    from ratlas import helpers_batch0 as H0
    col_files = None
    from ratlas.collections import get
    col = get('hainhofer_reisen')
    ctx = col.make_context()
    ents = col.prepare(ctx)
    assert len(ents) > 4400 and Counter(e.kind for e in ents) == {'person': 3502, 'place': 1036, 'org': 55}
    e = ctx.shared.lookup('psn:hans_von_aach')
    assert e is not None and e.name == 'Hans von Aachen' and e.auth['gnd'] == '118643525' and e.auth['ulan'] == '500127905'
    assert ctx.shared.lookup('plc:aachen').auth['geonames'] == '3247449'


@real
def test_betamasaheft_authority_files():
    from ratlas.collections import get
    col = get('betamasaheft_geez')
    ctx = col.make_context()
    ctx.all_files = col.iter_files(ctx)
    ents = col.prepare(ctx)
    k = {e.local_key: e for e in ents}
    assert 'betamasaheft_geez:PRS9038' in k and k['betamasaheft_geez:PRS9038'].auth['wikidata'] == 'Q738321'
    assert any(e.kind == 'place' and e.lat is not None for e in ents)
    assert k['betamasaheft_geez:ETH1973'].person_class == 'people' and k['betamasaheft_geez:ETH1973'].name == 'Oromo'


@real
def test_bullinger_register_names_and_cert():
    from ratlas.collections import get
    col = get('bullinger')
    ctx = col.make_context()
    ents = col.prepare(ctx)
    k = {e.local_key: e for e in ents}
    assert len(ents) == 6043 + 3077 + 337
    assert k['bullinger:p1283'].name == 'Konrad Gessner' and k['bullinger:p1283'].auth['gnd']
    assert k['bullinger:l803'].lat is not None and k['bullinger:l803'].auth['geonames']
    assert k['bullinger:p18562'].note_type == 'biblical'


def test_generic_terms_pronouns_kinship_titles_and_places():
    from ratlas import generic_terms as G
    for s in ('er', 'Sie', 'wir', 'ich', 'il', 'he', 'Ihr', 'ſie'):
        assert G.person_generic(s) == 'pronoun', s
    for s in ('Vatter', 'sein Vater', 'der Kayser', 'Gemahlin', 'my brother', 'la reine', 'Madame', 'Kayserliche', 'Ihre Majestät', 'maiestas regia'):
        assert G.person_generic(s) == 'common_noun', s
    for s in ('Kaiser Ferdinand', 'Fürst Ludwig', 'Johann Georg', 'Richard Napier', 'De Groot', 'Le Roy', 'Venus', 'Madame de Sévigné'):
        assert G.person_generic(s) is None, s
    for s in ('Stadt', 'die Stadt', 'Schloß', 'Kirche', 'alhier', 'of this towne', 'in der Stadt', 'the same town', 'anhero'):
        assert G.place_generic(s) == 'common_noun', s
    for s in ('Berg', 'Zerbst', 'Neustadt', 'Hall', 'Stadtlohn', 'Kirchberg', 'Castle Hedingham', 'La Haye'):
        assert G.place_generic(s) is None, s
    assert G.generic_kind('org', 'er') is None


def test_finalize_marks_generic_mentions_unless_an_authority_id_names_them():
    from ratlas.classify import finalize
    from ratlas.model import Mention
    m = finalize(Mention(coll_slug='x', source_path='p', doc_id='d', kind='person', tag='persName', surface='er'))
    assert m.excluded and 'pronoun_or_common_noun' in m.excluded_reason
    m = finalize(Mention(coll_slug='x', source_path='p', doc_id='d', kind='place', tag='placeName', surface='die Stadt'))
    assert m.excluded
    m = finalize(Mention(coll_slug='x', source_path='p', doc_id='d', kind='place', tag='placeName', surface='die Stadt', auth={'geonames': '1', 'other': []}))
    assert not m.excluded
    m = finalize(Mention(coll_slug='x', source_path='p', doc_id='d', kind='place', tag='placeName', surface='Zerbst'))
    assert not m.excluded
