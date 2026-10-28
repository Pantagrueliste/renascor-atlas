from ratlas import namegate as NG


def test_small_letters_and_folds():
    assert NG.small_letters('Froͤdnoͧw') == 'Froedn' + 'ouw'
    assert NG.nrm('Froͤdnoͧw') == 'froednouw'
    assert NG.fold('Æneas') == 'aeneas'
    assert NG.pfold('magdenburgk') == NG.pfold('magdeburg') or NG.pfold('magdenburgk').startswith('magden')


def test_person_compare_ok_and_mismatch():
    weak, _ = NG.build_weak([['Johann Georg'], ['Johann Meyer'], ['Johann Schmidt']] * 20, 40)
    assert 'johann' in weak
    st, m = NG.compare_person(['Johann Casimir von der Pfalz', 'Duca di Duponte'], ['Georg der Bärtige', 'Georges de Saxe'], weak)
    assert st == 'mismatch'                                    # cavriana: a wrong VIAF
    st, _ = NG.compare_person(['Martin Luther', 'Lutherus'], ['Martin Luther', 'Martinus Lutherus'], weak)
    assert st == 'ok'
    st, _ = NG.compare_person(['Johann Wolff'], ['Johann Georg Meyer'], weak)
    assert st == 'given_only'
    st, _ = NG.compare_person(['Kirsten'], [], weak)
    assert st == 'no_label'
    st, _ = NG.compare_person(['Ovidius'], ['Ovid', 'Publius Ovidius Naso'], weak)
    assert st == 'ok'                                          # Latin form, ending-insensitive


def test_place_relation_levels():
    ek = NG.name_keys('Florenz', ['Fiorenza', 'Florence'])
    lvl, _, _ = NG.place_relation(ek, NG.auth_keys(['Florence', 'Firenze']))
    assert lvl == 'exact' and NG.place_name_strong(lvl)
    lvl, _, _ = NG.place_relation(NG.name_keys('Spagna', ['Spagna']), NG.auth_keys(['Famagusta', 'Gazimağusa']))
    assert not NG.place_name_ok(lvl)                           # Spain -> Famagusta
    lvl, _, _ = NG.place_relation(NG.name_keys('Magdenburgk', []), NG.auth_keys(['Magdeburg']))
    assert NG.place_name_ok(lvl)
    lvl, _, _ = NG.place_relation(NG.name_keys('Nicaea', []), NG.auth_keys(['Nicea', 'İznik']))
    assert NG.place_name_ok(lvl)
    lvl, _, _ = NG.place_relation(NG.name_keys('Argos', []), NG.auth_keys(['Zadar', 'Zara', 'Jadera']))
    assert not NG.place_name_ok(lvl)                           # croala: Q3370 is Zadar
    assert NG.place_relation(set(), {'x'})[0] == 'noname'
    assert NG.place_relation({'x1'}, set())[0] == 'noauthname'


def test_person_category_and_class_ok():
    lab = {'Q41710': 'ethnic group', 'Q5': 'human', 'Q4167410': 'Wikimedia disambiguation page', 'Q202444': 'given name', 'Q164950': 'dynasty',
           'Q4271324': 'mythical character', 'Q4164871': 'position', 'Q515': 'city', 'Q43229': 'organization'}
    assert NG.person_category(['Q5'], lab) == 'human'
    assert NG.person_category(['Q4167410'], lab) == 'err_disambiguation'
    assert NG.person_category(['Q202444'], lab) == 'err_name'
    assert NG.person_category(['Q164950'], lab) == 'legit_dynasty'
    assert NG.person_category(['Q4271324'], lab) == 'legit_myth'
    assert NG.person_category(['Q4164871'], lab) == 'err_position'
    assert NG.person_category(['Q515'], lab) == 'err_place'
    assert NG.person_category(['Q41710'], lab) == 'legit_people'
    assert NG.person_category([], lab) == 'unknown'
    # human persons need Q5; dynasties and myth characters have their own classes
    assert NG.person_class_ok('human', 'human') == (True, '')
    assert NG.person_class_ok(None, 'legit_myth')[0] is True            # a god or a biblical figure on an unmarked entry: kept, marked
    assert NG.person_class_ok('mythological', 'legit_myth')[0] is True
    assert NG.person_class_ok('dynasty', 'legit_dynasty')[0] is True
    assert NG.person_class_ok('human', 'legit_dynasty') == (False, 'class_nonhuman_dynasty')   # a family for an individual
    assert NG.person_class_ok('human', 'legit_people')[0] is False and NG.person_class_ok('people', 'legit_people')[0] is True
    assert NG.person_class_ok('human', 'err_disambiguation') == (False, 'class_disambiguation')
    assert NG.person_class_ok('human', 'unknown')[0] is None


def test_place_and_gnd_class():
    assert NG.place_class_ok(['Q515'])[0] is True
    assert NG.place_class_ok(['Q5'])[0] is False                # a person is not a place
    assert NG.place_class_ok(['Q4830453'])[0] is False         # Thomson Reuters (Switzerland)
    assert NG.place_class_ok(['Q4167410']) == (False, 'class_disambiguation')
    assert NG.place_class_ok(['Q21286738']) == (False, 'duplicate_item')
    assert NG.place_class_ok([])[0] is True
    assert NG.gnd_class_ok(['Person', 'DifferentiatedPerson'], 'place') == (False, 'class_gnd_person')
    assert NG.gnd_class_ok(['PlaceOrGeographicName'], 'place')[0] is True
    assert NG.gnd_class_ok(['PlaceOrGeographicName'], 'person') == (False, 'class_gnd_place')


def test_dates():
    assert NG.wd_year('+1483-11-10T00:00:00Z') == 1483 and NG.wd_year('-0065-01-01T00:00:00Z') == -65
    assert NG.reg_year('um 1520') == 1520 and NG.reg_year('16. Jh.') is None and NG.reg_year('1520?') is None
    assert NG.chronology_flag(1794, 9, 1572) is True           # Strozzi: b. 1794 in a 1572 letter
    assert NG.chronology_flag(1560, 9, 1572) is False
    assert NG.chronology_flag(1600, 7, 1650) is False
    assert NG.register_dates_flag('1555', '1584', 1462, 11, 1515, 11) is True
    assert NG.register_dates_flag('1555', '1584', 1556, 11, 1584, 11) is False
    assert NG.register_dates_flag('65', None, -65, 11, None, None) is False   # BC without sign: skipped


def test_homonym_gate_kalkar_like():
    # Kalkar (Eifel hamlet, pop 0) vs Kalkar near Kleve (pop 13000): same country, 131 km apart
    auth = {'gid': '2893838', 'lat': 50.60, 'lon': 6.77, 'pop': 0, 'cc': 'DE', 'fcode': 'PPL'}
    town = {'gid': '2893837', 'lat': 51.74, 'lon': 6.29, 'pop': 13000, 'cc': 'DE', 'fcode': 'PPL'}
    lower_rhine = ([51.79, 51.65, 51.85, 51.55, 51.70], [6.14, 6.60, 6.40, 6.30, 6.05])         # the edition's other places, 100+ km from the Eifel
    eifel = ([50.62, 50.55, 50.70, 50.50, 50.65], [6.75, 6.80, 6.70, 6.90, 6.60])
    v = NG.homonym_check(auth, [auth, town], 'Kalkar', lower_rhine)
    assert v.reject and v.town_gid == '2893837' and v.town_km > 100 and v.context_km > 30
    # v2.1: a candidate in the edition's own territory is kept (the v2 gate rejected 79 right ids against 20 wrong ones)
    v = NG.homonym_check(auth, [auth, town], 'Kalkar', eifel)
    assert not v.reject and not v.suspect and v.context_km < 30
    # an edition with too few placed places cannot tell: suspect (suggested), never rejected
    v = NG.homonym_check(auth, [auth, town], 'Kalkar', None)
    assert not v.reject and v.suspect
    assert NG.homonym_check(auth, [auth, town], 'Kalkar', ([51.79], [6.14])).suspect
    # an editor's qualifier says which one it is
    assert not NG.homonym_check(auth, [town], 'Altenburg (bei Nienburg/Saale)', lower_rhine).reject
    assert not NG.homonym_check(auth, [town], 'Neukirchen, heute Bergisch Neukirchen', lower_rhine).reject
    # the town lies in another country, or next door, or is a river / region: no homonym
    assert not NG.homonym_check(auth, [dict(town, cc='BE')], 'Kalkar', lower_rhine).reject
    assert not NG.homonym_check(auth, [dict(town, lat=50.7, lon=6.8)], 'Kalkar', lower_rhine).reject
    assert not NG.homonym_check(auth, [dict(town, fcode='STM')], 'Kalkar', lower_rhine).reject
    assert not NG.homonym_check(dict(auth, fcode='ADM1'), [town], 'Kalkar', lower_rhine).reject
    # a populated authority is never rejected
    assert not NG.homonym_check(dict(auth, pop=5000), [town], 'Kalkar', lower_rhine).reject
    # no town with the name
    assert not NG.homonym_check(auth, [auth], 'Kalkar', lower_rhine).reject
    # Montdidier: pop 77 in the Moselle against a town of 6000 in the Somme
    m = {'gid': '6437726', 'lat': 49.1, 'lon': 6.5, 'pop': 77, 'cc': 'FR', 'fcode': 'PPL'}
    t2 = {'gid': '6556', 'lat': 49.65, 'lon': 2.57, 'pop': 6000, 'cc': 'FR', 'fcode': 'PPL'}
    somme = ([49.9, 49.6, 50.0, 49.8, 49.7], [2.3, 2.6, 2.5, 2.1, 2.9])
    assert NG.homonym_check(m, [m, t2], 'Montdidier', somme).reject
    assert NG.has_qualifier('Orsoy (Rheinberg)') and not NG.has_qualifier('Jüchen')


def test_homonym_gate_larger_namesake_in_the_same_region_is_only_suggested():
    """v2.2: Kalkar of the Eifel is not accepted for a Lower Rhine edition just because that edition has places within 30 km of it (13 of the 34
    pairs the v2.1 gate re-accepted were wrong): the larger Kalkar of the region (131 km) is as close to the edition's places, so the pair is suggested."""
    auth = {'gid': '2893838', 'lat': 50.60, 'lon': 6.77, 'pop': 0, 'cc': 'DE', 'fcode': 'PPL'}
    town = {'gid': '2893837', 'lat': 51.74, 'lon': 6.29, 'pop': 13000, 'cc': 'DE', 'fcode': 'PPL'}
    both = ([51.79, 51.65, 50.62, 50.55, 51.70], [6.14, 6.60, 6.75, 6.80, 6.05])      # places near the town AND near the hamlet
    v = NG.homonym_check(auth, [auth, town], 'Kalkar', both)
    assert v.suspect and not v.reject and v.town_gid == '2893837' and v.town_context_km <= NG.HOMONYM_ISOLATED_KM and v.context_km <= NG.HOMONYM_ISOLATED_KM
    # only the hamlet's neighbourhood is in the edition: the village is favoured (the town is isolated from the edition)
    eifel = ([50.62, 50.55, 50.70, 50.50, 50.65], [6.75, 6.80, 6.70, 6.90, 6.60])
    v = NG.homonym_check(auth, [auth, town], 'Kalkar', eifel)
    assert not v.suspect and not v.reject and v.town_context_km > NG.HOMONYM_ISOLATED_KM
    # the larger namesake is far away (not the same region): the v2.1 rule stands (accepted when the edition is around the village)
    far = dict(town, lat=53.7, lon=10.0)
    v = NG.homonym_check(auth, [auth, far], 'Kalkar', both)
    assert not v.suspect and not v.reject and (NG.hav(auth['lat'], auth['lon'], far['lat'], far['lon']) > NG.HOMONYM_REGION_KM)
    # an isolated village is still rejected whatever the region
    assert NG.homonym_check(auth, [auth, town], 'Kalkar', ([51.79, 51.65, 51.85, 51.55, 51.70], [6.14, 6.60, 6.40, 6.30, 6.05])).reject
    assert NG.HOMONYM_REGION_KM == 150.0 and [t['gid'] for t, _ in NG.homonym_towns(auth, [auth, town], 'Kalkar')] == ['2893837']


def test_wikipedia_title_gate():
    weak, _ = NG.build_weak([['Ralph']] * 50, 40)
    assert NG.wikipedia_title_ok('Little Dunmow', ['Ralph Baynard'], 'person', weak) is False
    assert NG.wikipedia_title_ok('Ralph Baynard', ['Ralph Baynard'], 'person', weak) is True
    assert NG.wikipedia_title_ok('Peasants\' Revolt', ['Sir Ralph Standish'], 'person', weak) is False
    assert NG.wikipedia_title_ok('Stralsund', ['Stralsund'], 'place') is True
    assert NG.wikipedia_title_ok('Cambridge (Massachusetts)', ['Cambridge'], 'place') is True
    assert NG.wikipedia_title_ok('Bermondsey Abbey', ['Alwinus Child'], 'person', weak) is False


def test_two_weak_tokens_and_family_names():
    weak, _ = NG.build_weak([['Anne of Denmark'], ['Christian of Denmark'], ['Anne Boleyn'], ['Anne Hyde']] * 15, 30)
    assert 'anne' in weak and 'denmark' in weak
    st, _ = NG.compare_person(['Anne of Denmark, queen of England'], ['Anne of Denmark'], weak)
    assert st == 'ok_weak'
    assert NG.compare_person(['Anne Seymour'], ['Anne Boleyn'], weak)[0] == 'given_only'
    assert NG.infer_person_class('Bellasis family', None) == 'dynasty' and NG.infer_person_class('(Familie) Kambli', 'human') == 'dynasty'
    assert NG.infer_person_class('Haus Österreich', None) == 'dynasty' and NG.infer_person_class('Martin Luther', None) is None
    assert NG.infer_person_class('Bellasis family', 'mythological') == 'mythological'


def test_rulers_match_by_name_and_regnal_number():
    weak, _ = NG.build_weak([['Ferdinand II'], ['Ferdinand III'], ['Ferdinand I']] * 20, 40)
    assert 'ferdinand' in weak
    st, m = NG.compare_person(['Ferdinand II., Heiliges Römisches Reich, Kaiser'], ['Ferdinand II, Holy Roman Emperor'], weak)
    assert st == 'ok' and m == ['ferdinand ii']
    assert NG.compare_person(['Ferdinand II., Kaiser'], ['Ferdinand III, Holy Roman Emperor'], weak)[0] == 'mismatch'   # another Ferdinand
    assert NG.ruler_keys('Karl V.') == [('karl', 'v')] and NG.ruler_keys('Heinrich') == []


def test_calibrated_constants_are_the_ones_the_calibration_fitted():
    """v2.1: the thresholds of the homonym gate and of the verify rule are what ``scripts/calibrate_rules.py`` fitted on 70 % of the labels."""
    import json
    from pathlib import Path
    from ratlas import verify_geocode as V
    cal = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'calibration_v2_1.json').read_text())
    assert NG.HOMONYM_ISOLATED_KM == cal['homonym']['fitted_km']
    assert V.GN_MIN_P == cal['verify']['fitted_p']
    assert NG.HOMONYM_MIN_CONTEXT == cal['homonym']['min_context']
    for k in ('homonym', 'verify'):          # the held-out part exists and was not used to choose anything
        assert cal[k]['split']['test'] > 20
    assert cal['homonym']['test_held_out']['new_rule']['accuracy'] > cal['homonym']['test_held_out']['old_rule_reject_every_candidate']['accuracy']


def test_tgn_fuzzy_relation_counts_from_090_only():
    """v2.2: Kythnos was accepted for Sifnos at 0.83; a fuzzy relation is a name match for a TGN id from 0.90."""
    assert NG.TGN_FUZZY_MIN == 0.90
    assert NG.fuzzy_ratio('fuzzy0.83') == 0.83 and NG.fuzzy_ratio('exact') is None
    assert NG.place_name_ok('fuzzy0.83') and not NG.place_name_ok('fuzzy0.83', NG.TGN_FUZZY_MIN)       # the default is unchanged for the other gates
    assert NG.place_name_ok('fuzzy0.90', NG.TGN_FUZZY_MIN) and NG.place_name_ok('exact', NG.TGN_FUZZY_MIN) and NG.place_name_ok('substring', NG.TGN_FUZZY_MIN)
    lvl = NG.place_relation(NG.name_keys('Cythnos', []), NG.auth_keys(['Sífnos, Nísos', 'Sifnos', 'Siphnos']))[0]
    assert lvl.startswith('fuzzy') and NG.fuzzy_ratio(lvl) < 0.90
    assert NG.head_key('Bad Düben, Alaunwerk') == 'bad duben' and NG.head_key('Ehrenberg (Tirol)') == 'ehrenberg'
