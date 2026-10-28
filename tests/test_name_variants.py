"""Tests of the name modules: folded key, variants, reference form, extra.json of the globe page (name_variants.py)."""
import json

import pandas as pd
import pytest

from ratlas import name_variants as N


# ---------------------------------------------------------------------------------------------------------------- folded key
@pytest.mark.parametrize('a,b', [
    ('Zürich', 'Zuerich'), ('Zürich', 'ZURICH'), ('Aegypten', 'Ägypten'), ('Jena', 'Iena'), ('Vienna', 'Uienna'), ('Viennæ', 'Viennae'),
    ('Straße', 'Strasse'), ('Wien', 'Wien,'), ('St. Gallen', 'St Gallen'), ('Yorck', 'Iorck'), ('Hamburg', 'hamburg'), ('Köln', 'Koeln'),
])
def test_fold_key_merges_spelling_variants(a, b):
    assert N.fold_key(a) == N.fold_key(b)


@pytest.mark.parametrize('a,b', [('Paris', 'Parma'), ('Neustadt', 'Neustadt an der Aisch'), ('Orleans', 'Orlando'), ('Bern', 'Berne')])
def test_fold_key_keeps_different_names_apart(a, b):
    assert N.fold_key(a) != N.fold_key(b)


def test_fold_key_edge_cases():
    assert N.fold_key(None) == '' and N.fold_key('') == '' and N.fold_key(12) == ''
    assert N.fold_key('  Saint-Denis ') == 'saint denis'
    assert N.fold_key('Froͤdnoͧw') == N.fold_key('Frodnouw')                    # small superscript letters of print are read as letters


def test_clean_form_and_usable_form():
    assert N.clean_form('of Boughton, ') == 'Boughton'
    assert N.clean_form('  Zürich \n') == 'Zürich'
    assert N.clean_form(None) == ''
    assert N.usable_form('Wien')
    assert not N.usable_form('x') and not N.usable_form('1234') and not N.usable_form('alhier') and not N.usable_form(None)
    assert not N.usable_form('A' * 61)


# ---------------------------------------------------------------------------------------------------------------- languages
@pytest.mark.parametrize('code,expected', [('de', 'de'), ('deu', 'de'), ('ger', 'de'), ('de-gml', 'nds'), ('frm', 'fr'), ('lat', 'la'), ('sco', 'en'), ('grc', 'grc'),
                                          ('?', ''), (None, ''), ('', ''), ('und', ''), ('DE', 'de')])
def test_lang_code(code, expected):
    assert N.lang_code(code) == expected


def test_edition_languages_needs_volume_and_agreement():
    counts = [('ed1', 'de', 90), ('ed1', 'la', 5), ('ed1', '?', 400), ('ed2', 'de', 60), ('ed2', 'fr', 60), ('ed3', 'en', 10)]
    assert N.edition_languages(counts) == {'ed1': 'de'}


# ---------------------------------------------------------------------------------------------------------------- variants
def test_group_variants_merges_the_same_name_only():
    forms = [('Zürich', 'de', 'ed1', 5), ('Zuerich', 'de', 'ed2', 2), ('ZÜRICH', 'de', 'ed1', 1), ('Tigurum', 'la', 'ed3', 4), ('Zurych', 'pl', 'ed4', 1)]
    vs = N.group_variants(forms)
    assert [(v.form, v.n) for v in vs] == [('Zürich', 9), ('Tigurum', 4)]          # Zurych is the same name (y/i); Tigurum is another name of the same place
    z = vs[0]
    assert z.lang == 'de' and z.editions == {'ed1': 6, 'ed2': 2, 'ed4': 1}
    assert z.spellings == {'Zürich': 5, 'Zuerich': 2, 'ZÜRICH': 1, 'Zurych': 1}


def test_group_variants_never_returns_two_variants_with_one_key():
    forms = [(f, 'de', 'e', 1) for f in ('Köln', 'Koeln', 'Cöln', 'Coeln', 'Colonia', 'Cologne')]
    vs = N.group_variants(forms)
    keys = [v.key for v in vs]
    assert len(keys) == len(set(keys))
    assert sum(v.n for v in vs) == 6


def test_group_variants_drops_unusable_and_cleans():
    vs = N.group_variants([('alhier', 'de', 'e', 9), ('of Boughton', 'en', 'e', 2), ('x', 'en', 'e', 3), ('', 'en', 'e', 1), ('Boughton', 'en', 'e', 1)])
    assert [(v.form, v.n) for v in vs] == [('Boughton', 3)]
    kept = N.group_variants([('alhier', 'de', 'e', 9)], drop_unusable=False)
    assert kept[0].n == 9


def test_group_variants_language_from_edition_when_missing():
    vs = N.group_variants([('Gefell', None, 'ed', 3), ('Gefell', '?', 'ed', 1)], edition_lang={'ed': 'de'})
    assert vs[0].lang == 'de' and vs[0].n_inferred_lang == 4
    vs = N.group_variants([('Gefell', None, 'ed', 3)])
    assert vs[0].lang == ''


def test_representative_spelling_is_the_most_frequent_then_the_accented():
    assert N.group_variants([('Zuerich', 'de', 'e', 2), ('Zürich', 'de', 'e', 2)])[0].form == 'Zürich'
    assert N.group_variants([('Zuerich', 'de', 'e', 3), ('Zürich', 'de', 'e', 2)])[0].form == 'Zuerich'


def test_top_variants_form():
    vs = N.group_variants([(f'Name{c}', 'de', 'e', 20 - i) for i, c in enumerate('abcdefghklmnopq')])
    t = N.top_variants(vs)
    assert len(t) == N.TOP_VARIANTS == 10
    assert t[0] == ['Namea', 'de', 20] and len(vs) == 15
    assert [x[2] for x in t] == sorted((x[2] for x in t), reverse=True)


# ---------------------------------------------------------------------------------------------------------------- reference form
def test_reference_form_order():
    vs = N.group_variants([('Tiguri', 'la', 'e', 3)])
    assert N.reference_form('Zurich', ['Zürich'], 'Zurich (exo)', vs) == ('Zurich', 'wikidata')
    assert N.reference_form(None, ['Zürich'], 'Zurich (exo)', vs) == ('Zürich', 'register')
    assert N.reference_form(None, [], 'Zurich (exo)', vs) == ('Zurich (exo)', 'exonym')
    assert N.reference_form(None, [], None, vs) == ('Tiguri', 'surface')
    assert N.reference_form(None, [], None, []) == (None, None)


def test_reference_form_skips_bot_labels_and_common_nouns():
    vs = N.group_variants([('Duisburg', 'de', 'e', 3)])
    assert N.reference_form('lsjbot Duisburg', [], None, vs) == ('Duisburg', 'surface')
    assert N.reference_form('this towne', ['Duisburg'], None, vs) == ('Duisburg', 'register')


def test_reference_form_custom_order_and_error():
    vs = N.group_variants([('Tiguri', 'la', 'e', 3)])
    assert N.reference_form('Zurich', ['Zürich'], None, vs, order=('register', 'wikidata')) == ('Zürich', 'register')
    with pytest.raises(ValueError):
        N.reference_form('Zurich', order=('nonsense',))


def test_register_name():
    assert N.register_name([('surf:xyz', 'Wienn'), ('abacus:vien', 'Wien')]) == 'Wien'
    assert N.register_name([('abacus:krak', 'Cracow (Kraków)')]) == 'Cracow'
    assert N.register_name([('surf:a', 'Wien')]) is None
    assert N.register_name([]) is None


def test_name_based_count():
    assert N.name_based_count(['name_exonym', 'register', 'name_latin', None, 'wikidata', 'name_propagation', 'name_geonames']) == 4


# ---------------------------------------------------------------------------------------------------------------- extra.json from toy tables
@pytest.fixture
def toy_build(tmp_path):
    fd = tmp_path / 'final'
    fd.mkdir()
    pl = pd.DataFrame([
        dict(coll_slug='a', entity_key='a:zur', name='Zürich', n_in_period=6, n_mentions=6, global_id='g1', coord_source='register', place_kind='city', qid='Q72', note_type='city',
             tgn_type=None, gn_fcode='PPLA', gn_pop=400000.0, geonames_id=None, gc_evidence=None, gc_cand=None, gc_source=None, cb_place_id=None),
        dict(coll_slug='b', entity_key='b:tig', name='Tigurum', n_in_period=3, n_mentions=3, global_id='g1', coord_source='name_latin', place_kind='city', qid=None, note_type=None,
             tgn_type=None, gn_fcode=None, gn_pop=None, geonames_id=None, gc_evidence=None, gc_cand=None, gc_source='name_latin', cb_place_id=None),
        dict(coll_slug='a', entity_key='a:boh', name='Böhmen, Königreich', n_in_period=2, n_mentions=2, global_id='g2', coord_source='name_exonym', place_kind='area', qid=None,
             note_type=None, tgn_type=None, gn_fcode=None, gn_pop=None, geonames_id=None, gc_evidence=None, gc_cand=None, gc_source='name_exonym', cb_place_id=None),
    ])
    pl.to_parquet(fd / 'places_local_geo.parquet')
    pd.DataFrame([
        dict(id='g1', qid='Q72', label_en='Zurich', name='Zürich', classes='Q515:city', globe_id='pl:Q72'),
        dict(id='g2', qid=None, label_en=None, name='Böhmen', classes=None, globe_id='pl:boh'),
    ]).to_parquet(fd / 'places_global.parquet')
    (fd / 'globe_places.json').write_text(json.dumps({'places': [{'id': 'pl:Q72', 'k': 'c'}, {'id': 'pl:boh', 'k': 'a'}]}), encoding='utf-8')
    base = dict(kind='place', canonical=True, excluded=False, in_period=True)
    ms = [dict(base, coll_slug='a', local_key='a:zur', surface='Zürich', lang='de')] * 4 + [dict(base, coll_slug='a', local_key='a:zur', surface='Zuerich', lang='de')] * 2 \
        + [dict(base, coll_slug='b', local_key='b:tig', surface='Tigurum', lang='la')] * 3 \
        + [dict(base, coll_slug='a', local_key='a:boh', surface='Böhmen, Königreich', lang='de')] * 2 \
        + [dict(base, coll_slug='a', local_key='a:zur', surface='Zürich', lang='de', excluded=True)]          # excluded mentions are not counted
    pd.DataFrame(ms).to_parquet(tmp_path / 'mentions.parquet')
    core = {'rows': [['Zurich', 47.4, 8.5, 'x', 1, [], 'Q72', 'w', 3, 9, 'Zürich', 'c', {}, 0], ['Bohemia', 49.8, 15.5, 'x', 1, [], '', 'e', 3, 2, 'Böhmen', 'a', {}, 0]]}
    (tmp_path / 'core.json').write_text(json.dumps(core), encoding='utf-8')
    return tmp_path


def test_build_extra_shape_and_content(toy_build):
    out = toy_build / 'extra.json'
    glob = toy_build / 'globals.tsv'
    e = N.build_extra(toy_build / 'final', toy_build / 'mentions.parquet', out, toy_build / 'core.json', globals_out=glob)
    assert json.loads(out.read_text('utf-8')) == e
    assert set(e) == {'types', 'rows'}
    assert all(len(v) == 3 for v in e['types'].values())
    assert set(e['rows']) == {'0', '1'}                                   # keyed by the row index of core.json
    typ, variants, nb = e['rows']['0']
    assert typ == 'city'
    assert variants[0] == ['Zürich', 'de', 6]                              # Zürich + Zuerich merged
    assert ['Tigurum', 'la', 3] in variants
    assert nb == 3                                                         # the Latin mentions only
    typ, variants, nb = e['rows']['1']
    assert typ == 'kingdom'
    assert variants == [['Böhmen, Königreich', 'de', 2]] and nb == 2
    for v in e['rows'].values():
        assert e['types'][v[0]][2] in ('c', 'a')                           # the coarse kind is derived from the fine type
    g = pd.read_csv(glob, sep='\t')
    r = g.set_index('global_id').loc['g1']
    assert r['reference_form'] == 'Zurich' and r['reference_source'] == 'wikidata' and r['folded_key'] == 'zurich' and r['type'] == 'city'
    r = g.set_index('global_id').loc['g2']
    assert r['reference_source'] in ('register', 'surface')


def test_build_extra_types_agree_with_the_page_kinds(toy_build):
    core = json.loads((toy_build / 'core.json').read_text('utf-8'))
    core['rows'][0][11] = 'n'                                             # the page says 'natural feature': the fine type must follow
    (toy_build / 'core.json').write_text(json.dumps(core), encoding='utf-8')
    e = N.build_extra(toy_build / 'final', toy_build / 'mentions.parquet', core_path=toy_build / 'core.json')
    assert e['types'][e['rows']['0'][0]][2] == 'n'


def test_build_extra_rejects_a_core_of_another_size(toy_build):
    core = json.loads((toy_build / 'core.json').read_text('utf-8'))
    core['rows'].pop()
    (toy_build / 'core.json').write_text(json.dumps(core), encoding='utf-8')
    with pytest.raises(ValueError):
        N.build_extra(toy_build / 'final', toy_build / 'mentions.parquet', core_path=toy_build / 'core.json')
