import pandas as pd
import pyarrow as pa

from ratlas import merge as M
from ratlas.model import AUTH_KEYS


def _df(rows):
    base = dict(coll_slug='x', renascor_id='x', source_path='a.xml', doc_id='a', doc_year=1600, in_period=True,
                kind='place', role='mention', zone='txt', surface='', local_key=None, in_register=None, canonical=True,
                a_other_n=0, _other=None, licence='L', tier='T1')
    for k in AUTH_KEYS:
        base['a_' + k] = None
    return pd.DataFrame([{**base, **r} for r in rows])


def test_norm():
    assert M._norm('  Évora,  [Portugal] ') == 'evora portugal'
    assert M._norm('of Stonystratford') == 'stonystratford' and M._norm('in Southwork') == 'southwork' and M._norm('at bishops gat') == 'bishops gat'
    assert M._norm('In') == 'in' and M._norm('to be') == 'to be'
    assert M._norm(None) == '' and M._norm(float('nan')) == ''


def test_local_table_keys_and_bins():
    df = _df([
        dict(surface='Paris', local_key='x:paris', in_register=True, geonames='2988507', a_geonames='2988507'),
        dict(surface='Parigi', local_key='x:paris', in_register=True, doc_year=1575, source_path='b.xml'),
        dict(surface='Lyon'), dict(surface='lyon ', doc_year=None, in_period=None),
        dict(surface='Rouen', local_key='x:rouen', in_register=False),
        dict(surface='', local_key=None),                       # nothing to identify it by: dropped
        dict(surface='Dupe', canonical=False),
    ])
    ents = pd.DataFrame([dict(local_key='x:paris', kind='place', name='Paris', lat=48.85, lon=2.35,
                              coord_source='tei:geo', birth=None, death=None, note_type='place',
                              variants=['Parisius'], **{'a_' + k: None for k in AUTH_KEYS})])
    lt = M.local_table(df, ents, 'place').set_index('entity_key')
    assert set(lt.index) == {'x:paris', 'surf:lyon', 'x:rouen'}
    p = lt.loc['x:paris']
    assert p['n_mentions'] == 2 and p['name'] == 'Paris' and p['has_coords'] and p['key_type'] == 'register'
    assert p['q1600'] == 1 and p['q1575'] == 1 and p['geonames'] == '2988507'
    assert 'Parisius' in p['variants']
    ly = lt.loc['surf:lyon']
    assert ly['n_mentions'] == 2 and ly['n_undated'] == 1 and ly['undated_year'] == 1 and ly['key_type'] == 'surface'
    assert lt.loc['x:rouen']['key_type'] == 'opaque'
    M._to_table(lt.reset_index(), M.LOCAL_SCHEMA)


def test_exp_of_scales_typed_rs_by_zone():
    r = {'c': {'txt|rs': 90, 'reg|rs': 10, 'hdr|name': 5, 'txt|persName': 3},
         'types': {'rs|person': 50, 'name|person': 5, 'persName|display': 3}}
    e = M._exp_of(r)
    assert round(e[('txt', 'person')]) == 3 + 45          # 50 * 90/100 + persName; header <name> is not counted
    assert ('hdr', 'person') not in e or e[('hdr', 'person')] == 0


def test_local_table_v2_default_counts_and_class():
    df = _df([dict(surface='Sol', local_key='x:sol', in_register=True, part='txt', excluded=True, person_class='celestial'),
              dict(surface='Sol', local_key='x:sol', in_register=True, part='txt', excluded=False, person_class='celestial'),
              dict(surface='Sol', local_key='x:sol', in_register=True, part='hdr', excluded=True, person_class=None)])
    lt = M.local_table(df, pd.DataFrame(), 'place').set_index('entity_key')
    r = lt.loc['x:sol']
    assert r['n_mentions'] == 3 and r['n_default'] == 1 and r['n_in_period_default'] == 1
    assert r['person_class'] == 'celestial' and r['n_txt'] == 2 and r['n_hdr'] == 1
    M._to_table(lt.reset_index(), M.LOCAL_SCHEMA)


def test_deviations_vs_v1_columns(tmp_path):
    ct = pd.DataFrame([dict(slug='a', mentions_canonical=90, person=50, place=30, org=10, mentions_default=80, person_default=45,
                            place_default=26, org_default=9, mentions_excluded=10, excluded_reasons='{"character": 10}',
                            **{f'zone_{z}': 0 for z in M.ZONE_CLASSES})])
    v1 = tmp_path / 'v1.tsv'
    pd.DataFrame([dict(slug='a', mentions_canonical=100, person=60, place=30, org=10)]).to_csv(v1, sep='\t', index=False)
    out = M.deviations_vs_v1(ct, v1, tmp_path)
    r = out.iloc[0]
    assert r['delta_mentions_canonical'] == -10 and r['delta_person'] == -10 and r['v2_default'] == 80 and r['reason']
    assert (tmp_path / 'deviations_vs_v1.tsv').exists()


def test_a_character_is_a_character_in_every_mention_of_its_key():
    """v2 audit, check 6: Panurge (key persh_panurge) was excluded 417 times and counted 55 times."""
    import pyarrow as pa
    from ratlas.merge import unify_character_keys
    t = pa.table({'kind': ['person'] * 5 + ['place'],
                  'local_key': ['e:panurge', 'e:panurge', 'e:panurge', None, None, 'e:panurge'],
                  'surface': ['Panurge', 'Panurge', 'Panurge', 'Roy', 'Roy', 'Panurge'],
                  'person_class': ['character', 'human', 'mythological', 'character', 'human', None],
                  'excluded': [True, False, False, True, False, False],
                  'excluded_reason': ['character', None, 'footnote', 'character', None, None]})
    out, n = unify_character_keys(t)
    d = out.to_pandas()
    assert n == 3
    assert list(d['person_class'][:5]) == ['character'] * 5 and pd.isna(d['person_class'][5])   # the place of the same key is untouched
    assert list(d['excluded']) == [True] * 5 + [False]
    assert d.loc[2, 'excluded_reason'] == 'footnote;character'
    assert unify_character_keys(pa.table({'kind': ['person'], 'local_key': ['a'], 'surface': ['A'], 'person_class': ['human'],
                                          'excluded': [False], 'excluded_reason': [None]}))[1] == 0


def test_pronouns_and_common_nouns_leave_the_default_counts():
    """v2.1 (owner default): tagged pronouns and common nouns are flagged and excluded, the rows stay; a common noun with an authority id is kept."""
    import pyarrow as pa
    from ratlas.merge import exclude_generic
    from ratlas.model import AUTH_TYPE, auth_to_arrow, empty_auth
    ids = empty_auth()
    ids['gnd'] = '118'
    t = pa.table({'kind': ['person', 'person', 'person', 'person', 'place', 'place', 'place', 'org'],
                  'surface': ['er', 'sein Vater', 'Kayser', 'Kaiser Ferdinand', 'die Stadt', 'Schloß', 'Zerbst', 'Sie'],
                  'excluded': [False] * 5 + [True] + [False] * 2, 'excluded_reason': [None, None, None, None, None, 'footnote', None, None],
                  'auth': pa.array([auth_to_arrow(empty_auth()), auth_to_arrow(empty_auth()), auth_to_arrow(ids), auth_to_arrow(empty_auth()),
                                    auth_to_arrow(empty_auth()), auth_to_arrow(empty_auth()), auth_to_arrow(empty_auth()),
                                    auth_to_arrow(empty_auth())], AUTH_TYPE)})
    out, n, rep = exclude_generic(t)
    d = out.to_pandas()
    assert list(d['excluded']) == [True, True, False, False, True, True, False, False]
    assert d.loc[0, 'excluded_reason'] == 'pronoun_or_common_noun' and d.loc[5, 'excluded_reason'] == 'footnote'
    assert n == 3 and set(rep['class']) == {'pronoun', 'common_noun'}
    assert exclude_generic(out)[1] == 0                                  # idempotent


def _gen_table():
    import pyarrow as pa
    from ratlas.model import AUTH_TYPE, auth_to_arrow, empty_auth
    ids = empty_auth()
    ids['wikidata'] = 'Q1'
    return pa.table({
        'kind': ['person', 'person', 'place', 'place', 'person', 'person', 'place', 'person'],
        'surface': ['Vatter', 'Vatter', 'alhier', 'alhier', 'Schwestern', 'Kaiser Ferdinand', 'Provintzien', 'er'],
        'ref_raw': ['#a_vater', None, '#ort_x', None, '#s_1 #s_2 #s_3', '#ferd', '#p_1 #p_2', '#a_vater'],
        'excluded': [False] * 8, 'excluded_reason': pa.array([None] * 8, pa.string()),
        'auth': pa.array([auth_to_arrow(ids), auth_to_arrow(empty_auth()), auth_to_arrow(ids), auth_to_arrow(empty_auth()), auth_to_arrow(ids),
                          auth_to_arrow(ids), auth_to_arrow(ids), auth_to_arrow(ids)], AUTH_TYPE)})


def test_common_nouns_with_an_id_are_flagged_generic_with_id_and_stay_in_the_counts():
    """v2.2: a mention of 'Vater' / 'alhier' that carries an id refers to that entity: it stays linked and counted, but is flagged (never a display name)."""
    from ratlas.merge import exclude_generic, flag_generic_with_id
    t, _n, _r = exclude_generic(_gen_table())
    out, k = flag_generic_with_id(t)
    d = out.to_pandas()
    assert list(d['generic_with_id']) == [True, False, True, False, True, False, True, False]     # a pronoun is never kept; a name is not generic
    assert k == 4 and list(d['excluded'])[:3] == [False, True, False]                             # the one without an id is excluded (v2.1), the others stay
    assert flag_generic_with_id(out)[1] == 4 and out.column_names.count('generic_with_id') == 1       # idempotent


def test_mentions_of_several_entities_leave_the_per_entity_counts():
    """v2.2: 'Schwestern' with three register refs counted for the first sister only (Anhalt diary 6,097 mentions, Eleonora Magdalena 407); they are excluded
    (reason multi_referent) from the default counts, the rows stay."""
    from ratlas.merge import exclude_generic, exclude_multi_referent
    t, _n, _r = exclude_generic(_gen_table())
    out, n, rep = exclude_multi_referent(t)
    d = out.to_pandas()
    assert list(d['excluded']) == [False, True, False, True, True, False, True, True] and n == 2
    assert d.loc[4, 'excluded_reason'] == 'multi_referent' and d.loc[6, 'excluded_reason'] == 'multi_referent'
    assert d.loc[7, 'excluded_reason'] == 'pronoun_or_common_noun'                 # the first reason stays first
    assert set(rep['surface_key']) == {'schwestern', 'provintzien'} and set(rep['n_refs']) == {3, 2}
    assert exclude_multi_referent(out)[1] == 0                                      # idempotent
    from ratlas import generic_terms as G
    assert G.multi_referent('#a #b') and not G.multi_referent('#a #a') and not G.multi_referent('gnd:1 viaf:2') and G.multi_referent('eats:1 eats:2')
