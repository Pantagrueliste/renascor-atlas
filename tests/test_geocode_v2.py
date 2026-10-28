import numpy as np
import pandas as pd

from ratlas import exonyms as EX
from ratlas import geotext as GT
from ratlas import placeclass as PCL
from ratlas import reconcile_geocode as GC


def _gn(rows):
    df = pd.DataFrame(rows, columns=['key', 'gid', 'prim', 'name', 'lat', 'lon', 'fclass', 'fcode', 'cc', 'adm1', 'pop'])
    df['skel'] = [GT.skeleton(k) for k in df['key']]
    df['exact'] = True
    return GC.GN(df)


def _places(rows):
    P = pd.DataFrame(rows, columns=['coll_slug', 'entity_key', 'name', 'n_mentions', 'lat', 'lon', 'variants', 'note_type', 'person_class'])
    P['n_in_period'] = P['n_mentions']
    P['key_type'] = 'surface'
    P['src0'] = np.where(P['lat'].notna(), 'register', 'none')
    P['family'] = P['coll_slug'].map(GC.FAMILY).fillna('other')
    P['place_class'] = [PCL.classify(n, t, c) for n, t, c in zip(P['name'], P['note_type'], P['person_class'])]
    return P


def _run(rows, gn, pairs=None):
    P = _places(rows)
    pairs = pairs if pairs is not None else pd.DataFrame({'doc': [0], 'row': [0]})
    G = GC.Geocoder(P, _gn(gn), pairs, pd.DataFrame(columns=['h', 't', 'l']))
    out = GC.run_pipeline(G, np.array([], dtype=int), log=lambda m: None)
    return G, out


def _row(coll, key, name, note=None, pclass=None, lat=np.nan, lon=np.nan, variants=None, n=5):
    return (coll, key, name, n, lat, lon, variants or [name], note, pclass)


def test_exonym_table_beats_alternate_names():
    # GeoNames lists Batavis as an alternate name of Passau and Nicaea as one of Nice
    gn = [('batavis', '1', False, 'Passau', 48.57, 13.46, 'P', 'PPL', 'DE', '02', 50000),
          ('nicaea', '2', False, 'Nice', 43.7, 7.27, 'P', 'PPLA', 'FR', '93', 340000),
          ('perse', '3', True, 'Percey', 47.8, 3.6, 'P', 'PPL', 'FR', 'A1', 90),
          ('prussia', '4', True, 'Prussia', 44.9, 7.9, 'P', 'PPL', 'IT', '12', 40)]
    rows = [_row('ckcc_epistolarium', 'a', 'Batavis'), _row('controversia_confessio', 'b', 'Nicaea'),
            _row('epistemon_bvh', 'c', 'Perse'), _row('bullinger', 'd', 'Prussia'), _row('bullinger', 'e', 'Galli')]
    G, out = _run(rows, gn)
    a, b, c, d, e = (out[i]['final'] for i in range(5))
    assert a['method'] == 'name_exonym' and a['conf'] == 'medium' and abs(a['lat'] - 52.1) < 1 and a['radius'] >= 100   # the Dutch, an area
    assert b['method'] == 'name_exonym' and abs(b['lat'] - 40.43) < 0.1 and abs(b['lon'] - 29.72) < 0.1               # Iznik, not Nice
    assert c['method'] == 'name_exonym' and c['lat'] < 40                                                            # Persia
    assert d['method'] == 'name_exonym' and d['lat'] > 50                                                            # Prussia, not Piedmont
    assert e['method'] == 'name_exonym' and e['conf'] == 'medium'                                                     # France as an area
    assert EX.lookup('tiguri')[3] == 'Zürich' and EX.lookup('batavia')[4] == 'area'


def test_classes_never_geocoded():
    gn = [('cielo', '1', True, 'Cielo', 22.1, -100.9, 'P', 'PPL', 'MX', 'SL', 300),
          ('jupiter', '2', True, 'Jupiter', 26.9, -80.1, 'P', 'PPL', 'US', 'FL', 55000),
          ('welt', '3', True, 'Welt', 54.4, 8.9, 'P', 'PPL', 'DE', 'SH', 400),
          ('hier', '4', True, 'Hier', 48.2, 13.5, 'P', 'PPL', 'AT', '04', 100),
          ('galli', '5', True, 'Galli', 38.0, 15.0, 'P', 'PPL', 'IT', '19', 50),
          ('flur', '6', True, 'Flur', 47.0, 8.0, 'P', 'PPL', 'CH', 'LU', 80),
          ('hundred', '7', True, 'Macclesfield', 53.26, -2.12, 'P', 'PPLA2', 'GB', 'ENG', 52000)]
    rows = [_row('bdpn', 'a', 'Cielo', note='celestial', pclass='celestial'), _row('bdpn', 'b', 'Jupiter', pclass='celestial'),
            _row('germanc', 'c', 'Welt'), _row('germanc', 'd', 'hier'),
            _row('ckcc_epistolarium', 'e', 'Galli', note='nation', pclass='people'),
            _row('ssrq_swisslaw', 'f', 'Flur', note='flurname'), _row('hearth_tax', 'g', 'Macclesfield Hundred')]
    G, out = _run(rows, gn)
    for i in range(7):
        assert out[i]['final'] is None, (i, out[i])
    assert G.why[0] == 'class_celestial' and G.why[2] == 'class_common_noun' and G.why[3] == 'class_deictic'
    assert G.why[4] == 'class_ethnonym' and G.why[5] == 'micro_toponym' and G.why[6] == 'administrative_area'
    res = GC.result_rows(G, out)
    assert res.empty
    ur = GC.unresolved_rows(G, out).set_index('entity_key')['gc_reason']
    assert ur['a'] == 'class_celestial' and ur['g'] == 'administrative_area'


def test_register_type_is_passed_to_the_geocoder():
    # Tabor: a mountain in the register, a Bohemian town of the same name in GeoNames
    gn = [('tabor', '1', True, 'Tábor', 49.41, 14.66, 'P', 'PPLA2', 'CZ', '52', 34000),
          ('tabor', '2', True, 'Mount Tabor', 32.69, 35.39, 'T', 'MT', 'IL', '03', 0)]
    G, out = _run([_row('abacus', 'a', 'Tabor', note='moun'), _row('abacus', 'b', 'Tabor')], gn)
    a, b = out[0]['final'], out[1]['final']
    assert a and abs(a['lat'] - 32.69) < 0.1                    # the mountain
    assert b is None or abs(b['lat'] - 49.41) < 0.1             # without the type the town (or abstention)
    # no mountain in GeoNames: a village is not accepted for a mountain
    G, out = _run([_row('abacus', 'a', 'Tabor', note='moun')], gn[:1])
    assert out[0]['final'] is None


def test_hard_regional_prior():
    gn = [('krempe', '1', True, 'Krempe', 53.83, 9.48, 'P', 'PPL', 'DE', '10', 2300),
          ('jarocin', '2', True, 'Jarocin', 51.97, 17.5, 'P', 'PPLA2', 'PL', '78', 25000)]
    rows = [_row('rejestr_poznan', 'a', 'Krempe'), _row('rejestr_poznan', 'b', 'Jarocin'), _row('luene_nonnenbriefe', 'c', 'Krempe')]
    G, out = _run(rows, gn)
    assert out[0]['final'] is None and G.why[0] == 'outside_region'       # 530 km from Poznań
    assert out[1]['final'] and out[1]['final']['conf'] in ('high', 'medium')
    c = out[2]['final']
    assert c is None or c['lat'] > 53                                      # Lüneburg's region contains it
    # a major feature is exempt
    assert GC.region_keep('rejestr_poznan', [48.85], [2.35], [2_100_000], ['PPLC'])[0]
    assert not GC.region_keep('rejestr_poznan', [48.85], [2.35], [90_000], ['PPL'])[0]
    assert GC.region_keep('hearth_tax', [53.3], [-2.1], [0], ['PPL'])[0]


def test_qualifier_locates_the_head_name():
    gn = [('neukloster', '1', True, 'Neukloster', 53.87, 11.69, 'P', 'PPL', 'DE', '12', 3300),
          ('buxtehude', '2', True, 'Buxtehude', 53.47, 9.69, 'P', 'PPLA3', 'DE', '06', 40000),
          ('altkloster', '3', True, 'Altkloster', 53.46, 9.70, 'P', 'PPL', 'DE', '06', 800)]
    rows = [_row('luene_nonnenbriefe', 'a', 'Neukloster (Buxtehude)'), _row('luene_nonnenbriefe', 'b', 'Altkloster (Buxtehude)')]
    G, out = _run(rows, gn)
    assert out[0]['final'] is None or out[0]['final']['conf'] == 'low'      # not Neukloster in Mecklenburg
    b = out[1]['final']
    assert b and abs(b['lat'] - 53.46) < 0.2


def test_bracketed_modern_name_first_for_cidtc():
    gn = [('schoneck', '1', True, 'Schöneck', 52.88, 17.59, 'P', 'PPL', 'PL', '78', 600),
          ('skarszewy', '2', True, 'Skarszewy', 54.07, 18.43, 'P', 'PPLA3', 'PL', '82', 6500)]
    G, out = _run([_row('cidtc', 'a', 'Schoneck (Skarszewy)'), _row('bullinger', 'b', 'Schoneck (Skarszewy)')], gn)
    a = out[0]['final']
    assert a and abs(a['lat'] - 54.07) < 0.1


def test_forms_small_letters_and_admin():
    f = GT.forms('Froͤdnoͧw')
    assert f.keys(('t0',)) == ['froednouw']
    assert GT.forms('Macclesfield Hundred').admin and GT.forms('Humber Ward').admin and GT.forms('honour of Tutbury').admin
    assert not GT.forms('Woodcote tithing').admin
    assert GT.skeleton('wittemberg') == GT.skeleton('wittenberg') and GT.skeleton('jhena') == GT.skeleton('jena')
    assert GT.skeleton('magdeburgk') == GT.skeleton('magdeburg')
    assert GT.forms('Welt').junk and GT.forms('hier').junk and GT.forms('Herberg').junk


def test_placeclass():
    assert PCL.classify('Sol', None, 'celestial') == 'celestial'
    assert PCL.classify('Zeus', 'mythological') == 'mythical'
    assert PCL.classify('Galli', 'nation') == 'ethnonym'
    assert PCL.classify('Wien') is None
    assert PCL.classify('fluminis', 'epithet_only') == 'common_noun'
    assert PCL.classify('Hai') == 'biblical' and PCL.classify('Wien') is None
    assert PCL.is_ethnonym_surface('Gallis') and PCL.is_ethnonym_surface('Hispanos') and not PCL.is_ethnonym_surface('Gallia')


def test_exonym_table_is_consistent():
    assert len(EX.TABLE) > 700
    for k, (lat, lon, rad, label, kind) in EX.TABLE.items():
        assert -90 <= lat <= 90 and -180 <= lon <= 180 and rad > 0 and kind in ('city', 'area'), k
        assert k == k.lower() and k == k.strip(), k


def test_exonym_table_uses_the_head_name_only():
    gn = [('pamplona', '1', True, 'Pamplona', 42.82, -1.64, 'P', 'PPLA', 'ES', '32', 195000),
          ('ehrenberg', '2', True, 'Ehrenberg', 47.4, 10.7, 'P', 'PPL', 'AT', '07', 300)]
    rows = [_row('postscriptum', 'a', 'España, Navarra, Pamplona'), _row('bullinger', 'b', 'Ehrenberg (Tirol)'),
            _row('bullinger', 'c', 'Ludham, England')]
    G, out = _run(rows, gn)
    a = out[0]['final']
    assert a is None or (a['method'] != 'name_exonym' and abs(a['lat'] - 42.82) < 0.2)       # Pamplona, not Spain
    b = out[1]['final']
    assert b is None or b['method'] != 'name_exonym'                                        # not Tyrol
    c = out[2]['final']
    assert c is None or c['method'] != 'name_exonym'                                        # not England


def test_exonym_variants_and_leading_city_parts():
    gn = []
    rows = [_row('cidtc', 'a', 'Neuenburg', variants=['Neuenburg', 'Nowe', 'castri Novensis']), _row('bdpn', 'b', 'Lima, Perú'),
            _row('postscriptum', 'c', 'España, Navarra, Pamplona')]
    G, out = _run(rows, gn)
    a, b = out[0]['final'], out[1]['final']
    assert a and a['method'] == 'name_exonym' and abs(a['lat'] - 53.14) < 0.1          # Nowe on the Vistula, by a variant
    assert b and b['method'] == 'name_exonym' and abs(b['lat'] + 12.05) < 0.1          # Lima, not Peru
    assert out[2]['final'] is None                                                     # not Spain


def test_exonym_variants_vote():
    # cidtc pl147: Nowe by two variants against one stray Latin name of Nuremberg
    rows = [_row('cidtc', 'a', 'Neuenburg', variants=['Neuenburg', 'Norimbergae', 'Nowe', 'castri Novensis']),
            _row('cidtc', 'b', 'Zzyzx', variants=['Zzyzx', 'Norimbergae', 'Nowe']),                 # a tie: no exonym
            _row('cidtc', 'c', 'Wherever', variants=['Wherever', 'Nowe'])]                          # a single variant hit: taken
    G, out = _run(rows, [])
    a, b, c = (out[i]['final'] for i in range(3))
    assert a['method'] == 'name_exonym' and abs(a['lat'] - 53.14) < 0.1
    assert b is None
    assert c and abs(c['lat'] - 53.14) < 0.1


def test_english_country_names_are_in_the_exonym_table():
    """v2 audit, per-edition sample: the CKCC register names its countries in English; Sweden (2,440 mentions) and Poland (583) stayed unplaced
    because GeoNames' 'Kingdom of Sweden' is a low-confidence candidate for a Dutch collection."""
    for name, country in (('sweden', 'Sweden'), ('poland', 'Poland'), ('hungary', 'Hungary'), ('greece', 'Greece'), ('spain', 'Spain'),
                          ('germany', 'Germany'), ('the netherlands', 'Low Countries / Netherlands')):
        hit = EX.lookup(name)
        assert hit is not None and hit[3] == country and hit[4] == 'area', name


def test_an_article_before_a_common_noun_or_a_deictic():
    assert PCL.classify('die Stadt') == 'common_noun' and PCL.classify('the town') == 'common_noun'
    assert PCL.classify('dem Haag') is None and PCL.classify('La Rochelle') is None            # a toponym with an article stays a toponym


def test_tgn_names_join_the_name_dictionary(tmp_path):
    """v2.1: a Latin or historical name that only the Getty TGN knows becomes a name of the GeoNames feature of the same place (Reginopolis = Regensburg), or a TGN-only candidate; a result that rests on it says so (gc_tgn)."""
    import zipfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    def gnline(gid, name, alt, lat, lon, cls, code, cc, pop):
        return '\t'.join([gid, name, name, alt, str(lat), str(lon), cls, code, cc, '', '05', '', '', '', str(pop), '', '', 'Europe/Berlin', '2020-01-01'])
    z = tmp_path / 'allCountries.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr('allCountries.txt', '\n'.join([gnline('2849483', 'Regensburg', 'Ratisbonne', 49.0195, 12.0975, 'P', 'PPLA3', 'DE', 150000),
                                                   gnline('2849484', 'Regensdorf', '', 47.43, 8.47, 'P', 'PPL', 'CH', 17000)]))
    local = tmp_path / 'places_local.parquet'
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({'name': ['Reginopolis', 'Tigurum', 'Ratisbonne'], 'variants': [['Reginopolis'], ['Tigurum'], ['Ratisbonne']]})), local)
    tg = tmp_path / 'tgn'
    tg.mkdir()
    pd.DataFrame([dict(subject_id='7000874', name='Regensburg', place_type='inhabited place', place_types='inhabited place', lat=49.02, lon=12.10, merged_stat='M'),
                  dict(subject_id='7004140', name='Zürich', place_type='inhabited place', place_types='inhabited place', lat=47.37, lon=8.54, merged_stat='M')]
                 ).to_parquet(tg / 'tgn_places.parquet', index=False)
    pd.DataFrame([dict(subject_id='7000874', name='Regensburg', is_preferred=True), dict(subject_id='7000874', name='Reginopolis', is_preferred=False),
                  dict(subject_id='7000874', name='Ratisbonne', is_preferred=False), dict(subject_id='7004140', name='Zürich', is_preferred=True),
                  dict(subject_id='7004140', name='Tigurum', is_preferred=False)]).to_parquet(tg / 'tgn_names.parquet', index=False)
    out = tmp_path / 'gn_index.parquet'
    df = GC.build_index(local, z, out, tg)
    r = df[df.key == 'reginopolis']
    assert len(r) == 1 and r.iloc[0]['gid'] == '2849483' and r.iloc[0]['src'] == 'tgn' and r.iloc[0]['pop'] == 150000 and not r.iloc[0]['prim']
    assert (df[df.key == 'ratisbonne'].src == 'geonames').all()           # GeoNames knew this one: not added twice
    t = df[df.key == 'tigurum']
    assert len(t) == 1 and t.iloc[0]['gid'] == 'tgn7004140' and t.iloc[0]['fcode'] == 'PPL' and t.iloc[0]['src'] == 'tgn'
    P = _places([_row('bullinger', 'a', 'Reginopolis')])
    P = pd.concat([P, _places([_row('bullinger', 'b', 'Tigurum')])], ignore_index=True)
    Gc = GC.Geocoder(P, GC.GN(pd.read_parquet(out)), pd.DataFrame({'doc': [0], 'row': [0]}), pd.DataFrame(columns=['h', 't', 'l']))
    run = GC.run_pipeline(Gc, np.array([], dtype=int), log=lambda m: None)
    f = run[0]['final']
    assert f is not None and f['method'] == 'name_geonames' and abs(f['lat'] - 49.0195) < 0.01 and f['tgn'] and 'Getty TGN name' in f['evidence']
    rows = GC.result_rows(Gc, run)
    assert bool(rows.set_index('entity_key').loc['a', 'gc_tgn'])


def test_a_tgn_name_is_only_a_fallback_where_geonames_has_candidates():
    """v2.1: TGN variants never outrank GeoNames candidates of the same name (a TGN variant 'Evesham' of Epsom against the real Evesham)."""
    rows = [('evesham', '1', True, 'Evesham', 52.09, -1.95, 'P', 'PPL', 'GB', '', 20000, 'geonames'),
            ('evesham', '2', False, 'Epsom', 51.33, -0.27, 'P', 'PPL', 'GB', '', 31000, 'tgn'),
            ('reginopolis', '3', False, 'Regensburg', 49.02, 12.1, 'P', 'PPLA3', 'DE', '', 150000, 'tgn')]
    df = pd.DataFrame(rows, columns=['key', 'gid', 'prim', 'name', 'lat', 'lon', 'fclass', 'fcode', 'cc', 'adm1', 'pop', 'src'])
    df['skel'] = [GT.skeleton(k) for k in df['key']]
    df['exact'] = True
    gn = GC.GN(df)
    P = _places([_row('old_bailey', 'a', 'Evesham'), _row('bullinger', 'b', 'Reginopolis')])
    G = GC.Geocoder(P, gn, pd.DataFrame({'doc': [0], 'row': [0]}), pd.DataFrame(columns=['h', 't', 'l']))
    t, k, idx = G.gn_rows(0, G.name_forms[0])
    assert list(gn.gid[idx]) == ['1']                                   # the TGN candidate is dropped
    t, k, idx = G.gn_rows(1, G.name_forms[1])
    assert list(gn.gid[idx]) == ['3']                                   # no GeoNames candidate: the TGN name is used


def test_v22_exonym_additions_phoenicia_chur_pfaltz_braband_and_no_bruggen():
    assert EX.lookup('phoenicien')[3] == 'Phoenicia' and EX.lookup('chur pfaltz')[3] == 'Palatinate' and EX.lookup('braband')[3] == 'Brabant'
    assert EX.lookup('bruggen') is None              # Brüggen on the Lower Rhine is not Bruges
    assert EX.lookup('brugae')[3] == 'Bruges'
