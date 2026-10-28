"""Tests of the place-type vocabulary and the mappings from the sources' own terms (placetypes.py, data/placetypes.tsv)."""
import csv

import pytest

from ratlas import placetypes as P


# ---------------------------------------------------------------------------------------------------------------- vocabulary
def test_vocabulary_is_a_tree_of_five_groups():
    assert 25 <= len(P.LEAVES) <= 50
    assert set(P.GROUPS) <= set(P.TYPES)
    for c, t in P.TYPES.items():
        assert t.code == c
        assert t.kind in P.KINDS
        assert t.label
        if t.parent is None:
            assert c in P.GROUPS                                # only the five groups are roots
        else:
            assert t.parent in P.GROUPS
            assert P.TYPES[t.parent].kind == t.kind             # a leaf has the coarse kind of its group
    assert {P.TYPES[g].kind for g in P.GROUPS} == set(P.KINDS)


def test_every_type_the_owner_listed_exists():
    wanted = '''continent country kingdom empire republic duchy county landgraviate margraviate principality electorate bishopric canton lordship bailiwick district shire
    parish province region city town market_town village hamlet city_quarter river lake sea bay island mountain pass valley forest cape marsh church monastery castle palace inn
    street square gate bridge fort building site other'''.split()
    assert [w for w in wanted if w not in P.TYPES] == []


def test_wikidata_class_qids_are_unique_and_well_formed():
    qids = [t.qid for t in P.TYPES.values() if t.qid]
    assert len(qids) == len(set(qids))
    assert all(q.startswith('Q') and q[1:].isdigit() for q in qids)


def test_rank_covers_exactly_the_types():
    assert set(P._RANK) == set(P.TYPES)


def test_kind_group_and_json_form():
    assert P.kind_of('city') == 'c' and P.kind_of('duchy') == 'a' and P.kind_of('river') == 'n' and P.kind_of('castle') == 'b'
    assert P.kind_of(None) == 'u' and P.kind_of('nonsense') == 'u' and P.kind_of('other') == 'u'
    assert P.group_of('city') == 'settlement' and P.group_of('area') == 'area' and P.group_of('nonsense') == 'other'
    assert P.is_generic('settlement') and not P.is_generic('city')
    j = P.types_json()
    assert set(j) == set(P.TYPES)
    assert j['city'] == ['city', 'settlement', 'c']
    assert j['settlement'][1] == ''                                # a group has no parent
    assert all(len(v) == 3 for v in j.values())


# ---------------------------------------------------------------------------------------------------------------- folding
def test_fold():
    assert P.fold('Kurfürstentum') == 'kurfurstentum'
    assert P.fold("St. Paul's") == 'st paul s'
    assert P.fold('Großherzogtum') == 'grossherzogtum'
    assert P.fold(None) == ''


# ---------------------------------------------------------------------------------------------------------------- Wikidata
@pytest.mark.parametrize('label,expected', [
    ('free imperial city', 'city'), ('big city', 'city'), ('city of Switzerland', 'city'), ('town in Germany', 'town'), ('market town', 'market_town'),
    ('municipality of Switzerland', 'settlement'), ('village in Poland', 'village'), ('hamlet', 'hamlet'), ('Ortsteil', 'city_quarter'),
    ('canton of Switzerland', 'canton'), ('federated state of Germany', 'province'), ('landgraviate', 'landgraviate'), ('prince-bishopric', 'bishopric'),
    ('historical region', 'region'), ('sovereign state', 'country'), ('island country', 'country'), ('river island', 'island'), ('mountain pass', 'pass'),
    ('river', 'river'), ('human settlement', 'settlement'), ('urban district of Bavaria', 'city'), ('Residenzstadt', 'city'),
    ('tourist destination', None), ('financial center', None), ('', None), (None, None), ('entity', None),
])
def test_wd_label_type(label, expected):
    assert P.wd_label_type(label) == expected


def test_wd_type_reviewed_qid_beats_label():
    assert P.wd_type('Q6256', 'whatever') == 'country'
    assert P.wd_type(None, 'abbey') == 'monastery'
    assert P.wd_type('Q99999999', None) is None


# ---------------------------------------------------------------------------------------------------------------- GeoNames, TGN, Orbis, registers
@pytest.mark.parametrize('code,pop,expected', [
    ('PPLC', None, 'city'), ('PPLA', None, 'city'), ('PPLA2', None, 'city'), ('PPLA3', None, 'settlement'), ('PPLA3', 60000, 'city'), ('PPLA3', 6000, 'town'),
    ('PPLA3', 300, 'village'), ('PPL', None, 'settlement'), ('PPLX', None, 'city_quarter'), ('PPLL', None, 'hamlet'), ('PPLQ', None, 'site'),
    ('CSTL', None, 'castle'), ('ADM1', None, 'province'), ('ADM2', None, 'district'), ('STM', None, 'river'), ('LK', None, 'lake'), ('MT', None, 'mountain'),
    ('PCLI', None, 'country'), ('PCLX', None, 'country'), ('CONT', None, 'continent'), ('PASS', None, 'pass'), ('MSTY', None, 'monastery'),
    ('', None, None), (None, None, None), ('nan', None, None),
])
def test_gn_type(code, pop, expected):
    assert P.gn_type(code, pop) == expected


def test_gn_population_nan_is_ignored():
    assert P.gn_type('PPLA3', float('nan')) == 'settlement'


def test_gn_first_letter_fallback_is_generic():
    assert P.gn_type('PLZZ') == 'settlement'
    assert P.kind_of(P.gn_type('HLLZ')) == 'n'


@pytest.mark.parametrize('t,expected', [
    ('inhabited place', 'settlement'), ('former nation/state/empire', 'country'), ('river; stream', 'river'), ('Inhabited Place', 'settlement'),
    ('general region', 'region'), ('weird', None), ('', None), (None, None),
])
def test_tgn_type(t, expected):
    assert P.tgn_type(t) == expected


@pytest.mark.parametrize('t,expected', [
    ('St.', 'city'), ('Mfl.', 'market_town'), ('Gfsch.', 'county'), ('Fl.', 'river'), ('D.', 'village'), ('market_town', 'market_town'), ('country_or_region', 'region'),
    ('Xyz', None), ('', None), (None, None), ('people', None),
])
def test_orbis_type(t, expected):
    assert P.orbis_type(t) == expected


@pytest.mark.parametrize('t,expected', [
    ('herrschaft, kastlanei', 'lordship'), ('place_f_church', 'church'), ('flurname', 'site'), ('place', None), ('klarissenkloster', 'monastery'), ('region', 'region'),
    ('Place_F_Street', 'street'), ('place_pa_parish', 'parish'), ('amt', 'bailiwick'), ('', None), (None, None), ('something unheard of', None),
])
def test_register_type(t, expected):
    assert P.register_type(t) == expected


def test_register_type_casebooks_entry_type():
    assert P.register_type(None, 'parish') == 'parish'
    assert P.register_type('place', 'village') == 'village'


# ---------------------------------------------------------------------------------------------------------------- names
@pytest.mark.parametrize('name,expected', [
    ('Böhmen, Königreich', 'kingdom'), ('Pommern, Herzogtum', 'duchy'), ('Askanien (Aschersleben), Grafschaft', 'county'), ('Hessen, Landgrafschaft', 'landgraviate'),
    ('Gaster, Landvogtei', 'bailiwick'), ('Ballenstedt, Amt', 'bailiwick'), ('Rheda, Herrschaft', 'lordship'), ('Trient, Hochstift', 'bishopric'),
    ('Weston, Hertfordshire', None), ('Zürich', None), ('Stolberg', None),
])
def test_qualifier_type(name, expected):
    assert P.qualifier_type(name) == expected


def test_qualifier_needs_a_comma_and_a_head():
    assert P.qualifier_word('Königreich') == (None, None)
    assert P.qualifier_word(None) == (None, None)
    assert P.qualifier_word('Böhmen, Königreich') == ('konigreich', 'kingdom')


@pytest.mark.parametrize('name,expected', [
    ('Kloster Zinna', 'monastery'), ("St. Paul's Cathedral", 'church'), ('The Thames', 'river'), ('Monte Rosa', 'mountain'), ('Whitehall Palace', 'palace'),
    ('Thames Street', 'street'), ('London Bridge', 'bridge'), ('Parish of St. Dunstan', 'church'),
    ('Offenbach', None), ('Hamburg', None), ('Rattenberg am Inn', None), ('Berg', None), ('Bad', None),
])
def test_name_type(name, expected):
    assert P.name_type(name) == expected


def test_names_that_merely_contain_a_type_word_are_not_typed():
    # whole words only; 'Burg' / 'Berg' / 'Bach' inside a name are the name, not a type
    for n in ('Magdeburg', 'Offenbach', 'Strasburg', 'Bergen', 'Marburg an der Lahn'):
        assert P.name_type(n) is None, n


def test_label_qualifier():
    assert P.qualifier_type_of_label('Duchy of Savoy') == 'duchy'
    assert P.qualifier_type_of_label('Prince-Bishopric of Liège') == 'bishopric'
    assert P.qualifier_type_of_label('Free Imperial City of Nuremberg') == 'city'
    assert P.qualifier_type_of_label('University of Padua') is None
    assert P.qualifier_type_of_label('Savoy') is None
    assert P.qualifier_type_of_label(None) is None


# ---------------------------------------------------------------------------------------------------------------- the reviewed table
def test_reviewed_table_loads_and_every_type_is_known():
    tab = P.load_reviewed()
    assert len(tab) > 500
    assert all(t is None or t in P.TYPES for t in tab.values())
    assert {s for s, _ in tab} <= set(P.SOURCES)


def test_reviewed_table_has_no_unreviewed_row_of_20_mentions_or_more():
    with open(P.REVIEWED_PATH, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    assert rows
    seen = set()
    for r in rows:
        assert tuple(r) == P.REVIEWED_COLUMNS
        k = (r['source'], r['key'])
        assert k not in seen, k
        seen.add(k)
        assert r['status'] in ('reviewed', 'auto')
        if int(r['n_mentions'] or 0) >= P.MIN_MENTIONS:
            assert r['status'] == 'reviewed', k
            assert r['type'] == P.NEUTRAL or r['type'] in P.TYPES, k
            assert r['type'], k


def test_reviewed_table_agrees_with_the_code_on_the_big_classes():
    assert P.wd_type('Q515') == 'city'
    assert P.wd_type('Q23058') == 'canton'
    assert P.wd_type('Q6256') == 'country'
    assert P.wd_type('Q1221156') == 'province'


def test_reviewed_row_overrides_rules(tmp_path):
    p = tmp_path / 'pt.tsv'
    p.write_text('source\tkey\tlabel\ttype\tn_mentions\tn_places\tstatus\tnote\n'
                 'wd\tQ515\tcity\tvillage\t1\t1\treviewed\t\n'
                 'wd\tQ777\ttourist destination\t-\t1\t1\treviewed\t\n'
                 'qualifier\tamt\t\tdistrict\t1\t1\treviewed\t\n'
                 'name\tkloster\t\tbuilding\t1\t1\treviewed\t\n'
                 'gn\tZZZ\t\t\t1\t1\tauto\t\n', encoding='utf-8')
    assert P.wd_type('Q515', 'city', p) == 'village'
    assert P.wd_type('Q777', 'city', p) is None                      # '-' = reviewed, no type; no fall back to the label
    assert P.qualifier_type('Ballenstedt, Amt', p) == 'district'
    assert P.qualifier_type('Ballenstedt, Amt') == 'bailiwick'       # the real table is untouched
    assert P.name_type('Kloster Zinna', p) == 'building'
    assert ('gn', 'ZZZ') not in P.load_reviewed(str(p))                # a row without a type is ignored


@pytest.mark.parametrize('row,msg', [
    ('wd\tQ1\tx\tnot_a_type\t1\t1\treviewed\t', 'unknown type'),
    ('bogus\tQ1\tx\tcity\t1\t1\treviewed\t', 'unknown source'),
])
def test_bad_reviewed_row_is_an_error(tmp_path, row, msg):
    p = tmp_path / 'pt.tsv'
    p.write_text('source\tkey\tlabel\ttype\tn_mentions\tn_places\tstatus\tnote\n' + row + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=msg):
        P.load_reviewed(str(p))


# ---------------------------------------------------------------------------------------------------------------- classification
def test_classify_agrees_sources():
    r = P.classify(P.Evidence(wd_classes=['Q515:city'], gn_fcodes=['PPLA3'], names=['Zürich'], kind='c'))
    assert r.code == 'city' and r.source == 'wd' and r.consistent


def test_classify_specific_beats_generic():
    r = P.classify(P.Evidence(gn_fcodes=['PPL'], notes=['dorf'], kind='c'))
    assert r.code == 'village'


def test_classify_without_evidence_gives_the_group_of_the_kind():
    assert P.classify(P.Evidence(kind='c')).code == 'settlement'
    assert P.classify(P.Evidence(kind='a')).code == 'area'
    assert P.classify(P.Evidence(kind='n')).code == 'natural'
    assert P.classify(P.Evidence(kind='b')).code == 'built'
    assert P.classify(P.Evidence(kind='u')).code == 'other'
    assert P.classify(P.Evidence()).code == 'other'


def test_classify_never_contradicts_the_atlas_kind():
    r = P.classify(P.Evidence(wd_classes=[('Q4022', 'river')], kind='c'))
    assert P.kind_of(r.code) == 'c'
    assert r.code == 'settlement' and not r.consistent
    r = P.classify(P.Evidence(wd_classes=[('Q4022', 'river')], kind='n'))
    assert r.code == 'river' and r.consistent


def test_classify_polity_qualifier_in_the_name():
    assert P.classify(P.Evidence(gn_fcodes=['PPL'], names=['Hoym, Amt'], kind='a')).code == 'bailiwick'
    assert P.classify(P.Evidence(names=['Pommern, Herzogtum'], kind='a')).code == 'duchy'


def test_classify_label_of_the_item():
    assert P.classify(P.Evidence(wd_label_en='Duchy of Savoy', kind='a')).code == 'duchy'


def test_classify_many_votes_by_weight():
    evs = [P.Evidence(gn_fcodes=['PPLA2']), P.Evidence(wd_classes=['Q515:city']), P.Evidence(gn_fcodes=['PPLX'])]
    assert P.classify_many(evs).code == 'city'
    assert P.classify_many(evs, weights=[1, 1, 10]).code == 'city_quarter'
    assert P.classify_many([]).code == 'other'


def test_parse_classes():
    assert P.parse_classes('Q4022:river|Q573344:main stem') == [('Q4022', 'river'), ('Q573344', 'main stem')]
    assert P.parse_classes(None) == []
    assert P.parse_classes([('Q1', 'x'), 'Q2:y']) == [('Q1', 'x'), ('Q2', 'y')]


def test_pool_evidence_deduplicates():
    a = P.Evidence(wd_classes=['Q515:city'], gn_fcodes=['PPLA'], names=['Wien'])
    b = P.Evidence(wd_classes=['Q515:city'], gn_fcodes=['PPLA'], names=['Wien', 'Wienn'])
    e = P.pool_evidence([a, b], 'c')
    assert e.wd_classes == [('Q515', 'city')]
    assert len(e.gn_fcodes) == 1
    assert e.names == ['Wien', 'Wienn'] and e.kind == 'c'


# ---------------------------------------------------------------------------------------------------------------- coverage on a toy frame
def _toy_frame():
    import pandas as pd
    rows = [
        dict(coll_slug='a', entity_key='a:1', name='Zürich', n_in_period=10, coord_source='wikidata', classes=[('Q515', 'city')], fcode='PPLA', gn_pop=None, tgn_type=None,
             note_type=None, cb_type=None, orbis=[], exo_kind=None, cand_key=None, kind='c', point='p1', global_id='g1', wd_label_en=None),
        dict(coll_slug='a', entity_key='a:2', name='Bach, Amt', n_in_period=5, coord_source='register', classes=[], fcode=None, gn_pop=None, tgn_type=None,
             note_type=None, cb_type=None, orbis=[], exo_kind=None, cand_key=None, kind='a', point='p2', global_id='g2', wd_label_en=None),
        dict(coll_slug='a', entity_key='a:3', name='Nowhere', n_in_period=3, coord_source='register', classes=[], fcode=None, gn_pop=None, tgn_type=None,
             note_type=None, cb_type=None, orbis=[], exo_kind=None, cand_key=None, kind='c', point='p3', global_id='g3', wd_label_en=None),
        dict(coll_slug='a', entity_key='a:4', name='Unplaced', n_in_period=7, coord_source='none', classes=[], fcode=None, gn_pop=None, tgn_type=None,
             note_type=None, cb_type=None, orbis=[], exo_kind=None, cand_key=None, kind='u', point=None, global_id=None, wd_label_en=None),
    ]
    return pd.DataFrame(rows)


def test_classify_frame_and_coverage():
    df = P.classify_frame(_toy_frame())
    t = dict(zip(df['entity_key'], df['type']))
    assert t == {'a:1': 'city', 'a:2': 'bailiwick', 'a:3': 'settlement', 'a:4': 'other'}
    cov = P.coverage(df)
    assert cov['placed_mentions'] == 18                       # the unplaced one is not counted
    assert cov['with_type'] == 18                             # 'settlement' is a type (generic); 'other' is only the unplaced one
    assert cov['with_specific_type'] == 15 and cov['generic_only'] == 3
    assert cov['mentions_by_type'] == {'city': 10, 'bailiwick': 5, 'settlement': 3}


def test_term_inventory_and_uncovered():
    df = P.classify_frame(_toy_frame())
    inv = P.term_inventory(df)
    d = {(r.source, r.key): r.n_mentions for r in inv.itertuples()}
    assert d[('wd', 'Q515')] == 10 and d[('gn', 'PPLA')] == 10 and d[('qualifier', 'amt')] == 5
    assert P.uncovered(inv, min_mentions=1) == []              # everything in the toy frame is covered by the table or the rules
    ct = P.coverage_table(df, inv)
    assert set(ct['section']) == {'summary', 'type', 'term'}
    assert not ct[ct['section'] == 'term']['note'].str.contains('NOT COVERED').any()


def test_merge_inventory_keeps_existing_rows(tmp_path):
    df = P.classify_frame(_toy_frame())
    inv = P.term_inventory(df)
    p = tmp_path / 'pt.tsv'
    out = P.merge_inventory(inv, p)
    assert out['new'] == len(inv)
    rows = {(r['source'], r['key']): r for r in P.read_reviewed_rows(p)}
    rows[('wd', 'Q515')]['type'] = 'town'
    rows[('wd', 'Q515')]['status'] = 'reviewed'
    P.write_reviewed_rows(list(rows.values()), p)
    out = P.merge_inventory(inv, p)
    assert out['new'] == 0
    assert {(r['source'], r['key']): r for r in P.read_reviewed_rows(p)}[('wd', 'Q515')]['type'] == 'town'
