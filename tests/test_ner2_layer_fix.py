import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('ner2_layer_fix', Path(__file__).resolve().parent.parent / 'scripts/ner2_layer_fix.py')
FX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FX)


def ctx(rule, text, surface):
    i = text.index(surface)
    return FX.apply_context_rule(rule, text[:i], text[i + len(surface):])


def test_council_before():
    assert ctx('council', 'as it was decreed in the Councell of Nice and Constantinople', 'Nice')
    assert ctx('council', 'les canons du concile de Nice', 'Nice')
    assert ctx('council', 'the coūcell holden at Nice: go ony further', 'Nice')
    assert ctx('council', 'the .318. Bisshops at Nice, by the 200', 'Nice')


def test_council_after_and_negatives():
    assert ctx('council', 'the creed of Nice Creed is said', 'Nice')
    assert not ctx('council', 'a Town on the Mediterranean, five Miles from Nice to the West', 'Nice')
    assert not ctx('council', 'The Council was over. The Duke went to Nice and rested', 'Nice')     # a sentence break cuts the cue
    assert not ctx('council', 'in the County of Nice. It was built in 1231', 'Nice')


def test_idf_and_jordan():
    assert ctx('idf', 'l\'Isle de France, aux Indes, pres de Maurice', 'Isle de France')
    assert not ctx('idf', 'le Gouvernement de Paris et de l\'Isle de France, et de Normandie', 'Isle de France')
    assert ctx('jordan', 'and they passed Iordan into the land of Gad', 'Iordan')
    assert not ctx('jordan', 'Of Iordan and of Moab the king spake', 'Iordan')


def test_context_rule_of():
    assert FX.context_rule_of('nice') == 'council'
    assert FX.context_rule_of('isle de france') == 'idf'
    assert FX.context_rule_of('rome') is None


def test_capitalise_and_labels():
    assert FX.capitalise_label('the bath') == 'Bath'
    assert FX.capitalise_label('isle of man') == 'Isle of Man'
    assert FX.capitalise_label('stratford upon avon') == 'Stratford upon Avon'
    assert FX.label_ok('Paris') and not FX.label_ok('Paris, France') and not FX.label_ok('Arles (Provence)') and not FX.label_ok('Republic of France')
    assert FX.pick_label([('Republic of France', 9, 'en'), ('Fraunce', 3, 'en'), ('France', 4, 'fr')]) == 'France'
    assert FX.pick_label([('the bath', 5, 'en'), ('Bath', 4, 'en')]) == 'Bath'          # 'the bath' and 'Bath' are one label: 9 mentions


def test_label_for_prefers_english():
    forms = [('España', 3913, 'es'), ('Spaine', 1500, 'en'), ('Spain', 900, 'en'), ('Spagna', 300, 'it')]
    assert FX.label_for(forms, exonym_label='Spain') == 'Spain'
    assert FX.label_for(forms) == 'Spaine'                                       # no table label: the most frequent English form
    assert FX.label_for([('Roma', 900, 'it'), ('Rome', 5, 'en')]) == 'Roma'     # English texts carry too little of the point
    assert FX.label_for([('Constantinople', 100, 'en'), ('Istanbul', 5, 'tr')], exonym_label='Istanbul') == 'Constantinople'
    assert FX.label_for([('Rom', 400, 'de'), ('Rome', 200, 'en')], exonym_label='Rome', ed_name='Rom') == 'Rome'


def test_radius_tier_thresholds():
    assert FX.radius_floor('a', 5) == 40 and FX.radius_floor('a', 150) == 150 and FX.radius_floor('c', 20) == 0 and FX.radius_floor('n', 99) == 0
    assert FX.tier_of(11) == 'main' and FX.tier_of(10) == 'rare'
    assert FX.pair_thresholds(0, 0) == (3.0, 10.0)
    assert FX.pair_thresholds(5, 350) == (105.0, 105.0)
    assert FX.pair_thresholds(40, 40)[0] == 12.0


def test_split_move_never_exceeds_the_form():
    assert FX.split_move({'a': 3, 'b': 1}, {'a': 5, 'b': 1, 'c': 2, 'd': 0}) == {'a': 3, 'b': 1}


def test_pkey_and_equiv_and_same_name():
    assert FX.pkey_of('Q90', 'gn:1', 1, 2) == 'Q90'
    assert FX.pkey_of(None, 'gn:12', 1, 2) == 'gn:12'
    assert FX.pkey_of(None, 'x', 1.00049, 2.0) == 'xy:1.0,2.0'
    assert FX.equiv_class('Napoli') == FX.equiv_class('Naples') is not None
    assert FX.equiv_class('Fraunce') == FX.equiv_class('Republic of France')
    assert FX.equiv_class('Rome') is None
    assert FX.same_name('milan', 'milan', '', '') and not FX.same_name('rome', 'paris', 'rm', 'prs')


def test_union_find():
    u = FX.UnionFind()
    u.union('a', 'b')
    u.union('c', 'b')
    assert u.find('a') == u.find('c') != u.find('d')


def test_hav():
    assert abs(float(FX.hav_km(43.70313, 7.26608, 40.429, 29.721)) - 1883) < 15


# ---- a small layer end to end ----------------------------------------------------------------------------------------
def unit(run, lang, key, form, grp, lat, lon, kind='c', rad=0.0, src='name_geonames', placed=True, cname=None, qid=None, cand=''):
    cname = cname or form
    return dict(run=run, lang=lang, key=key, variant='', form=form, placed=placed, grp=grp, lat=lat, lon=lon, kind=kind, rad=rad, src=src, conf='medium', cand=cand, qid=qid,
                cname=cname, fate='placed' if placed else 'unplaced', fx='', pkey=FX.pkey_of(qid, cand, lat, lon), mfold=FX.fold(cname), grp0=grp)


def tiny_state():
    rows = [
        unit('v1', 'en', 'rome', 'Rome', 'v1:0', 41.89, 12.48),
        unit('v1', 'en', 'nice', 'Nice', 'v1:1', 43.70, 7.27, cname='Nizza (Nice)'),
        unit('v1', 'en', 'nicea', 'Nicea', 'v1:2', 40.43, 29.72, src='name_exonym', cname='İznik'),
        unit('v1', 'fr', 'tage', 'Tage', 'v1:3', 44.73, 0.90),
        unit('v1', 'la', 'tagus', 'Tagus', 'v1:4', 38.85, -9.01),
        unit('v1', 'it', 'prioli', 'Prioli', 'v1:5', 45.0, 12.0),
        unit('v1', 'es', 'toledo', 'Toledo', 'v1:6', 39.86, -4.02),
        unit('es', 'es', 'toledo', 'Toledo', 'es:0', 39.87, -4.04, cname='Toledo'),
        unit('es', 'es', 'roma', 'Roma', 'es:1', 41.903, 12.496, cname='Rome'),
    ]
    U = pd.DataFrame(rows)
    D = pd.DataFrame([(0, 'd1', 10), (1, 'd1', 3), (1, 'd2', 4), (2, 'd3', 2), (3, 'd1', 1), (4, 'd1', 2), (5, 'd2', 2), (6, 'd1', 5), (7, 'd5', 7), (8, 'd5', 5), (8, 'dN', 1)], columns=['uid', 'doc_id', 'n'])
    docs = pd.DataFrame({'collection': ['A', 'A', 'A', 'B', 'Nakala'], 'year': [1500, 1600, 1550, 1520, 1580], 'lang': ['en'] * 5}, index=['d1', 'd2', 'd3', 'd5', 'dN'])
    return FX.State(U, D, docs)


def test_snapshot_and_baseline():
    S = tiny_state()
    FX.stage_baseline(S)
    snap = S.snapshot()
    assert snap['points'] == 7 and snap['mentions'] == 10 + 7 + 2 + 1 + 2 + 2 + 5      # the two es-run units are set aside


def test_nakala_and_spanish():
    S = tiny_state()
    FX.stage_baseline(S)
    FX.stage_nakala(S)
    assert int(S.D['n'].sum()) == 10 + 3 + 4 + 2 + 1 + 2 + 2 + 5 + 7 + 5                  # the Nakala mention is gone
    FX.stage_spanish(S)
    U = S.U
    assert not U.loc[(U['run'] == 'v1') & (U['lang'] == 'es'), 'placed'].any()
    assert U.loc[(U['run'] == 'es') & (U['key'] == 'roma'), 'placed'].all()
    assert U.at[8, 'grp'] == 'v1:0'                                                       # Roma (es) joins Rome by name and distance
    assert U.at[7, 'grp'] == 'es:0'                                                       # no v1 Spanish point left to join
    assert S.snapshot()['mentions'] == 10 + 7 + 2 + 1 + 2 + 2 + 7 + 5


def test_move_mentions_and_nice():
    S = tiny_state()
    FX.stage_baseline(S)
    nu, k = S.move_mentions(1, {'d1': 2, 'd2': 9}, 'ctx:council')
    assert k == 2 + 4 and int(S.D[S.D['uid'] == 1]['n'].sum()) == 1
    S.retarget(nu, 2, 'nice->iznik')
    assert S.U.at[nu, 'grp'] == 'v1:2' and S.U.at[nu, 'lat'] == 40.43
    ctxdf = pd.DataFrame([('v1', 'd1', 'en', 'nice', 'council', 3, 2, 0)], columns=['run', 'doc_id', 'lang', 'key', 'rule', 'n_total', 'n_match', 'n_notext'])
    S2 = tiny_state()
    FX.stage_baseline(S2)
    FX.stage_nice(S2, ctxdf)
    g, _ = S2.group_table()
    assert g.at['v1:2', 'n'] == 2 + 2 and g.at['v1:1', 'n'] == 5


def test_pper_spares_exonym_forms():
    S = tiny_state()
    FX.stage_baseline(S)
    pper = pd.DataFrame([('v1', 'it', 'prioli', 0.95, 2), ('v1', 'en', 'nicea', 0.99, 2), ('v1', 'en', 'rome', 0.1, 10)], columns=['run', 'lang', 'key', 'p_per', 'n'])
    FX.stage_pper(S, pper)
    assert not S.U.at[5, 'placed'] and S.U.at[2, 'placed'] and S.U.at[0, 'placed']        # Prioli out; Nicea comes from the exonym table; Rome is not a person


def test_blacklist():
    S = tiny_state()
    S.U.loc[len(S.U)] = unit('v1', 'fr', 'uti', 'Uti', 'v1:9', 44.2, 1.5)
    S.D.loc[len(S.D)] = (len(S.U) - 1, 'd1', 1)
    FX.stage_baseline(S)
    before = S.snapshot()['mentions']
    FX.stage_blacklist(S)
    assert S.snapshot()['mentions'] == before - 1


def test_table_edit_and_floor_and_merge():
    S = tiny_state()
    FX.stage_baseline(S)
    # the table sends Tage (fr) to the point of Tagus (la) and makes it a river
    ruid = S.uid_of('la', 'tagus', run='v1')
    S.retarget(S.uid_of('fr', 'tage', run='v1'), ruid, 'table:tage', {'name': 'Tagus', 'kind': 'n', 'r': None})
    g, _ = S.group_table()
    assert g.at['v1:4', 'kind'] == 'n' and g.at['v1:4', 'n'] == 3 and 'v1:3' not in g.index
    # area floor
    S.U.loc[len(S.U)] = unit('v1', 'en', 'flanders', 'Flanders', 'v1:10', 51.0, 3.5, kind='a', rad=5.0)
    S.D.loc[len(S.D)] = (len(S.U) - 1, 'd1', 30)
    g2, _ = S.group_table(floor=True)
    assert g2.at['v1:10', 'rad'] == 40
    # merge: Napoli and Naples 0.6 km apart under known equivalent names
    S.U.loc[len(S.U)] = unit('v1', 'it', 'napoli', 'Napoli', 'v1:11', 40.8520, 14.2681, cname='Napoli')
    S.D.loc[len(S.D)] = (len(S.U) - 1, 'd1', 5)
    S.U.loc[len(S.U)] = unit('v1', 'en', 'naples', 'Naples', 'v1:12', 40.8518, 14.2681, cname='Naples')
    S.D.loc[len(S.D)] = (len(S.U) - 1, 'd1', 9)
    before = S.snapshot()['points']
    _, merged = FX.stage_merge(S, lambda s: {FX.fold(s)})
    assert S.snapshot()['points'] == before - 1 and any({'v1:11', 'v1:12'} <= set(m) for m in merged)
    g3, _ = S.group_table()
    assert g3.at['v1:12', 'n'] == 14 if 'v1:12' in g3.index else g3.at['v1:11', 'n'] == 14


def test_label_audit_on_assembled_points():
    S = tiny_state()
    FX.stage_baseline(S)
    pts, docs_by = FX.assemble(S, {})
    names = {p['id']: p for p in pts}
    assert names['v1:0']['n'] == 10 and names['v1:0']['nd'] == 1 and sum(names['v1:1']['q']) == 7
    assert [d for d, _ in docs_by['v1:1']] == ['d2', 'd1']
    assert FX.doc_link('EEBO_TCP_fill/A06134', 'EEBO_TCP_fill').endswith('A06134.0001.001')
    assert FX.doc_link('BibIT/bibit000033', 'BibIT') is None


# ---- pass 2 (23 Oct): the sweep verdicts ---------------------------------------------------------------------------------
def sweep_state():
    S = tiny_state()
    rows = [unit('v1', 'en', 'rojas', 'Rojas', 'v1:20', 40.0, -3.0),
            unit('v1', 'en', 'candie', 'Candie', 'v1:21', 35.2, 24.9, cname='Candie'),
            unit('v1', 'en', 'crete', 'Crete', 'v1:22', 35.25, 24.9, kind='a', rad=130, cname='Crete'),
            unit('v1', 'la', 'leodium', 'Leodium', 'v1:23', 50.64, 5.57, cname='Lüttich'),
            unit('v1', 'en', 'liege', 'Liège', 'v1:24', 50.63, 5.57, cname='Liège'),
            unit('v1', 'en', 'lints', 'Lints', 'v1:25', 53.8, -1.5),
            unit('v1', 'en', 'leeds', 'Leeds', 'v1:26', 53.8, -1.55),
            unit('v1', 'es', 'yndias', 'Yndias', 'v1:27', 22.0, 79.0, kind='a', rad=350, cname='Indien'),
            unit('v1', 'la', 'india', 'India', 'v1:28', 22.5, 78.0, kind='a', rad=1500, cname='India')]
    for r in rows:
        S.U.loc[len(S.U)] = r
        S.D.loc[len(S.D)] = (len(S.U) - 1, 'd1', 20 if r['key'] in ('crete', 'liege', 'leeds', 'india') else 4)
    S.U['grp0'] = S.U['grp']
    FX.stage_baseline(S)
    return S


def entry(S, lang, key, action, **kw):
    return dict(id_v2='ml2:0', name_v2=key, anchor=['v1', lang, key, ''], action=action, **kw)


def test_sweep_drop_move_merge_clean_relabel():
    S = sweep_state()
    before = S.snapshot()
    ents = [entry(S, 'en', 'rojas', 'drop'),
            entry(S, 'en', 'candie', 'move', lat=35.24, lon=24.81, kind='a', radius_km=130, label='Crete'),
            entry(S, 'la', 'leodium', 'merge', into_anchor=['v1', 'en', 'liege', ''], into_id_v2='ml2:1', into_name_v2='Liège'),
            entry(S, 'en', 'lints', 'clean', remove_forms=['Lints']),
            entry(S, 'la', 'india', 'relabel', label='India (test)'),
            entry(S, 'es', 'yndias', 'merge', into_anchor=['v1', 'la', 'india', ''], into_id_v2='ml2:2', into_name_v2='India', exclude_forms=['Yndias'])]
    FX.stage_sweep(S, ents)
    U = S.U
    uid = lambda lg, k: S.uid_of(lg, k, run='v1')  # noqa: E731
    assert not U.at[uid('en', 'rojas'), 'placed']                                                    # not a place: no point
    assert not U.at[uid('en', 'lints'), 'placed'] and U.at[uid('en', 'leeds'), 'placed']             # a bad form leaves, the point stays
    assert U.at[uid('la', 'leodium'), 'grp'] == U.at[uid('en', 'liege'), 'grp']                      # merge: same point as the target
    assert U.at[uid('la', 'leodium'), 'lat'] == U.at[uid('en', 'liege'), 'lat']
    assert not U.at[uid('es', 'yndias'), 'placed']                                                   # the form that must not follow is unplaced
    c = uid('en', 'candie')
    assert (U.at[c, 'lat'], U.at[c, 'lon'], U.at[c, 'kind'], U.at[c, 'qid']) == (35.24, 24.81, 'a', None)
    G, _ = S.group_table(floor=True)
    assert G.at['v1:21', 'rad'] == 130
    assert S.G_set['v1:28']['name'] == 'India (test)'
    after = S.snapshot()
    assert after['mentions'] == before['mentions'] - 4 - 4 - 4                                       # Rojas, Lints, Yndias
    assert S.sweep_rec['merge'][0]['mentions'] == 4 and S.sweep_rec['drop'][0]['mentions'] == 4


def test_sweep_split_by_collection():
    # homonym audit: a point that mixes an English town (collection A) and a New England town (collection B) keeps only the A mentions
    S = sweep_state()
    leeds = S.uid_of('en', 'leeds', run='v1')
    S.D.loc[len(S.D)] = (leeds, 'd5', 6)                                                             # d5 is a document of collection B
    before = S.snapshot()
    FX.stage_sweep(S, [entry(S, 'en', 'leeds', 'split', collections=['B'])])
    assert S.doc_counts(leeds) == {'d1': 20}                                                         # the A mentions stay on the point
    new = [u for u in S.U.index if S.U.at[u, 'variant'] == 'sweep3:split']
    assert len(new) == 1 and not S.U.at[new[0], 'placed'] and S.doc_counts(new[0]) == {'d5': 6}      # the B mentions lose it
    assert S.sweep_rec['split'][0]['mentions'] == 6
    assert S.snapshot()['mentions'] == before['mentions'] - 6


def test_sweep_exact_radius_skips_the_floor():
    S = sweep_state()
    FX.stage_sweep(S, [entry(S, 'en', 'candie', 'move', lat=43.97, lon=11.38, kind='a', radius_km=15)])
    G, _ = S.group_table(floor=True)
    assert G.at['v1:21', 'rad'] == 15                                                                # Mugello: a valley of 15 km, not 40
    G0, _ = S.group_table(floor=False)
    assert G0.at['v1:21', 'kind'] == 'a'


def test_sweep_point_absorbed_by_a_bigger_one_keeps_the_bigger_one():
    S = sweep_state()
    FX.stage_sweep(S, [entry(S, 'en', 'candie', 'move', lat=35.24, lon=24.81, kind='a', radius_km=130, label='Crete')])
    FX.stage_merge(S, lambda s: {FX.fold(s)}, step_name='again')                                     # Candie and Crete are the same name here -> one point
    G, _ = S.group_table(floor=True)
    g = S.U.at[S.uid_of('en', 'crete', run='v1'), 'grp']
    assert S.U.at[S.uid_of('en', 'candie', run='v1'), 'grp'] == g
    assert 'name' not in S.G_set.get(g, {}) and (G.at[g, 'lat'], G.at[g, 'lon']) == (35.25, 24.9)


def test_merge_exonym_key_and_wikidata_distance():
    S = sweep_state()
    # Leodium (cname Lüttich) and Liège: 1 km apart. The keys give 'ex:liege' to the first and 'liege' to the second: they meet only through the plain exonym key.
    keys = lambda s: {'luttich', 'ex:liege'} if FX.fold(s) == 'luttich' else {FX.fold(s)}  # noqa: E731
    _, merged = FX.stage_merge(S, keys)
    assert any({'v1:23', 'v1:24'} <= set(m) for m in merged)
    # same Wikidata id, 40 km apart: one point; different ids: not
    S2 = sweep_state()
    S2.U.at[S2.uid_of('en', 'crete', run='v1'), 'qid'] = 'Q1'
    S2.U.at[S2.uid_of('en', 'candie', run='v1'), 'qid'] = 'Q1'
    S2.U.at[S2.uid_of('en', 'candie', run='v1'), 'lat'] = 35.55
    _, merged2 = FX.stage_merge(S2, lambda s: set())
    assert any({'v1:21', 'v1:22'} <= set(m) for m in merged2)


def test_overrides_file_is_well_formed():
    ents = FX.load_overrides()
    assert len(ents) == 171                     # 109 of the sweep + 62 of the homonym audit (23 Oct)
    kinds = {}
    for e in ents:
        kinds[e['action']] = kinds.get(e['action'], 0) + 1
        assert len(e['anchor']) == 4
        if e['action'] == 'move':
            assert -90 <= e['lat'] <= 90 and -180 <= e['lon'] <= 180 and e['kind'] in 'can'
            assert (e['radius_km'] is None) == (e['kind'] != 'a') or e['kind'] == 'n'
        if e['action'] == 'merge':
            assert len(e['into_anchor']) == 4 and e['into_anchor'] != e['anchor']
        if e['action'] == 'split':
            assert e['collections'] and all(isinstance(c, str) for c in e['collections'])
    assert kinds == {'drop': 27, 'unplace': 36, 'move': 56, 'merge': 16, 'relabel': 14, 'clean': 5, 'split': 17}
    assert len({tuple(e['anchor']) for e in ents}) == len(ents)


def test_patch_report_pass2_replaces_the_section(tmp_path):
    f = tmp_path / 'r.md'
    f.write_text('# titre\n\ntexte\n', 'utf-8')
    FX.patch_report_pass2(f, FX.PASS2_TITLE + '\n\nun\n')
    FX.patch_report_pass2(f, FX.PASS2_TITLE + '\n\ndeux\n')
    t = f.read_text('utf-8')
    assert t.count(FX.PASS2_TITLE) == 1 and 'deux' in t and 'un\n' not in t.split(FX.PASS2_TITLE)[1] and t.startswith('# titre')
