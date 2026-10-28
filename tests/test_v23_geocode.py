"""v2.3 geocoder: the Old Bailey jurisdiction prior, London parishes, compound strings, polities and qualifiers, head / alternative names, English small homonyms,
and the Latin path."""
import numpy as np
import pandas as pd

from ratlas import exonyms as EX
from ratlas import latin_geo as LG
from ratlas import polities as POL
from ratlas import reconcile_geocode as GC
from ratlas import regions as RGN
from test_geocode_v2 import _gn, _places, _row


def _geocoder(rows, gn, pairs=None, doc_regions=None, resources=None, hint=None, extra_cols=None):
    P = _places(rows)
    for c, v in (extra_cols or {}).items():
        P[c] = v
    pairs = pairs if pairs is not None else pd.DataFrame({'doc': list(range(len(rows))), 'row': list(range(len(rows)))})
    return GC.Geocoder(P, _gn(gn), pairs, hint if hint is not None else pd.DataFrame(columns=['h', 't', 'l']), doc_regions, resources)


def _run(rows, gn, pairs=None, doc_regions=None, resources=None, hint=None, extra_cols=None):
    G = _geocoder(rows, gn, pairs, doc_regions, resources, hint, extra_cols)
    out = GC.run_pipeline(G, np.array([], dtype=int), log=lambda m: None)
    return G, out


def _f(out, i):
    return out[i]['final']


# ---- 1. Old Bailey: London plus Middlesex for every name ----------------------------------------------------------------------------------
def test_london_keep_rules():
    ok = RGN.london_keep
    # Greater London without the Hertfordshire corner
    assert ok('old_bailey', [51.51], [-0.10], [0], ['PPLX'])[0]
    assert not ok('old_bailey', [51.676], [-0.274], [40], ['PPL'])[0]                    # Green Street, Hertfordshire, 25 km north of the City
    assert not ok('map_of_early_modern_london', [51.69], [-0.17], [0], ['PPL'])[0]       # Potters Bar
    assert ok('map_of_early_modern_london', [51.62], [-0.30], [0], ['PPL'])[0]           # Stanmore, Middlesex
    # a country, a capital and a million-inhabitant settlement pass; an ADMINISTRATIVE unit of a million does not (the Département du Nord)
    assert ok('map_of_early_modern_london', [48.85], [2.35], [2_100_000], ['PPLC'])[0]
    assert ok('map_of_early_modern_london', [46.0], [2.0], [0], ['PCLI'])[0]
    assert not ok('map_of_early_modern_london', [50.33], [3.36], [2_600_000], ['ADM2'])[0]
    # Old Bailey: names that are not British, and provincial towns of 100,000
    assert ok('old_bailey', [51.44], [3.57], [45_000], ['PPL'], ['NL'])[0]                 # Flushing = Vlissingen, 'a Town in Holland'
    assert not ok('map_of_early_modern_london', [51.44], [3.57], [45_000], ['PPL'], ['NL'])[0]   # MoEML: everything is in London
    assert not ok('old_bailey', [50.33], [3.36], [2_600_000], ['ADM2'], ['FR'])[0]         # ... but not an administrative unit
    assert ok('old_bailey', [51.45], [-2.59], [430_000], ['PPLA'], ['GB'])[0]                # Bristol
    assert not ok('old_bailey', [55.1], [-2.2], [0], ['PPL'], ['GB'])[0]                     # Holburn, Northumberland
    assert RGN.london_applies('old_bailey', 'Oxford') and RGN.london_applies('map_of_early_modern_london') and not RGN.london_applies('hearth_tax')


OB = [('holburn', '1', True, 'Holburn', 55.65, -1.95, 'P', 'PPL', 'GB', 'ENG', 0),                         # Northumberland
      ('petty france', '2', True, 'Petty France', 51.567, -2.31, 'P', 'PPL', 'GB', 'ENG', 40),             # Gloucestershire
      ('gatehouse', '3', True, 'Gatehouse of Fleet', 54.88, -4.18, 'P', 'PPL', 'GB', 'SCT', 900),
      ('kingsland', '4', True, 'Kingsland', 52.24, -2.88, 'P', 'PPL', 'GB', 'ENG', 400),                    # Herefordshire
      ('kingsland', '5', True, 'Kingsland', 51.5476, -0.0775, 'P', 'PPLX', 'GB', 'ENG', 0),                 # Hackney
      ('hermitage', '6', True, 'Hermitage', 51.47, -1.26, 'P', 'PPL', 'GB', 'ENG', 3000),                   # Berkshire
      ('hamstead', '7', True, 'Hamstead', 50.70, -1.40, 'P', 'PPL', 'GB', 'ENG', 0),                        # Isle of Wight
      ('flushing', '8', True, 'Flushing', 50.16, -5.07, 'P', 'PPL', 'GB', 'ENG', 722),                      # Cornwall
      ('bristol', '9', True, 'Bristol', 51.4545, -2.5879, 'P', 'PPLA', 'GB', 'ENG', 430000),
      ('green street', '10', True, 'Green Street', 51.676, -0.274, 'P', 'PPL', 'GB', 'ENG', 40),           # Hertfordshire
      ('green street', '11', True, 'Green Street', 51.5310, -0.0460, 'P', 'PPLX', 'GB', 'ENG', 0)]          # Bethnal Green


def test_old_bailey_names_are_london_or_middlesex_unless_they_say_otherwise():
    names = ['Holburn', 'Petty-France', 'Gatehouse', 'Kingsland', 'Hermitage', 'Hamstead', 'Flushing', 'Bristol', 'Green-street']
    rows = [_row('old_bailey', f'k{i}', n) for i, n in enumerate(names)]
    G, out = _run(rows, OB)
    f = {n: _f(out, i) for i, n in enumerate(names)}
    for n in ('Holburn', 'Petty-France', 'Gatehouse', 'Hermitage', 'Hamstead'):
        assert f[n] is None or f[n]['conf'] == 'low', (n, f[n])                                  # no namesake in London: nothing, not a provincial hamlet
    assert f['Kingsland'] and abs(f['Kingsland']['lat'] - 51.5476) < 0.01                       # the Hackney one, not Herefordshire
    assert f['Flushing'] and f['Flushing']['method'] == 'name_exonym' and abs(f['Flushing']['lat'] - 51.4425) < 0.01      # Vlissingen, not Cornwall
    assert f['Bristol'] and abs(f['Bristol']['lat'] - 51.4545) < 0.01                           # a provincial town of 430,000 is no homonym of a hamlet
    assert f['Green-street'] and abs(f['Green-street']['lat'] - 51.5310) < 0.01                 # London's, not the Hertfordshire hamlet (25 km north)


# ---- 2. MoEML: a London church needs its qualifier -----------------------------------------------------------------------------------------
def test_london_churches_do_not_copy_the_point_of_another_parish_of_the_same_saint():
    rows = [_row('map_of_early_modern_london', 'a', 'St. Margaret (Lothbury)', lat=51.5148, lon=-0.0910),
            _row('map_of_early_modern_london', 'b', 'Parish of St Margaret (Westminster)'),
            _row('map_of_early_modern_london', 'c', 'St. Margaret Lothbury'),
            _row('map_of_early_modern_london', 'd', 'St. Margaret')]
    G, out = _run(rows, [])
    assert _f(out, 1) is None and G.why.get(1) == 'parish_qualifier'                           # Westminster is not Lothbury (3.8 km off)
    c = _f(out, 2)
    assert c and abs(c['lat'] - 51.5148) < 0.001                                              # the same saint with the same qualifier
    assert _f(out, 3) is None                                                                  # the bare saint's name is ambiguous
    assert GC.PCL.classify('North') == 'deictic' and GC.PCL.classify('the South') == 'deictic'  # POOL1's 'North' was the Département du Nord


# ---- 3. compound strings ------------------------------------------------------------------------------------------------------------------
CMP = [('espana', '1', True, 'España', 40.0, -4.0, 'A', 'PCLI', 'ES', '00', 47_000_000),
       ('guipuzcoa', '2', True, 'Gipuzkoa', 43.10, -2.30, 'A', 'ADM2', 'ES', '59', 700_000),
       ('irun', '3', True, 'Irun', 43.3378, -1.7888, 'P', 'PPL', 'ES', '59', 61_000),
       ('zaragoza', '4', True, 'Zaragoza', 41.65, -0.888, 'P', 'PPLA', 'ES', '02', 680_000),
       ('zaragoza', '5', True, 'Provincia de Zaragoza', 41.60, -1.20, 'A', 'ADM2', 'ES', '02', 960_000),
       ('bubierca', '6', True, 'Bubierca', 41.2, -1.97, 'P', 'PPL', 'ES', '02', 150),
       ('amsterdam', '7', True, 'Amsterdam', 52.374, 4.89, 'P', 'PPLC', 'NL', '07', 800_000)]


def test_compound_strings_resolve_on_the_leaf_and_never_fall_back_to_the_container():
    rows = [_row('postscriptum', 'a', 'Guipúzcoa, Irún'), _row('postscriptum', 'b', 'Guipúzcoa, Behobia'), _row('postscriptum', 'c', 'España, Zaragoza, Bubierca'),
            _row('postscriptum', 'd', 'Europa, Holanda, Amesterdão'), _row('postscriptum', 'e', 'Europa, Amsterdam'), _row('postscriptum', 'f', 'España, Zaragoza, Calatorao')]
    G, out = _run(rows, CMP)
    a = _f(out, 0)
    assert a and abs(a['lat'] - 43.3378) < 0.01 and a['conf'] in ('high', 'medium')            # Irún, not the centroid of Gipuzkoa (45 km off)
    assert _f(out, 1) is None or _f(out, 1)['conf'] == 'low'                                    # Behobia is in no gazetteer here: nothing, never the province
    c = _f(out, 2)
    assert c and abs(c['lat'] - 41.2) < 0.01                                                    # Bubierca near the PROVINCE of Zaragoza, not on the city
    assert _f(out, 3) is None                                                                   # never the Europe point
    e = _f(out, 4)
    assert e and abs(e['lat'] - 52.374) < 0.01
    assert _f(out, 5) is None                                                                   # an unknown leaf stays unplaced
    # the same collection's register place 'España, Irún' (the TEI's own point) is what a sibling string copies, not the province centroid
    rows2 = [_row('h15corpus', 'x', 'Guipúzcoa, Irún'), _row('h15corpus', 'y', 'España, Irún', lat=43.3378, lon=-1.7888)]
    G2, out2 = _run(rows2, CMP, pd.DataFrame({'doc': [0, 0], 'row': [0, 1]}))
    x = _f(out2, 0)
    assert x and abs(x['lat'] - 43.3378) < 0.001 and x['conf'] in ('high', 'medium')
    f = G.name_forms[2]
    assert f.leaf_only and f.t1 == [] and f.keys(('t0',))[0] == 'bubierca' and f.container == 'espana'


# ---- 4. polities and qualifiers --------------------------------------------------------------------------------------------------------------
def test_parse_qualified_names():
    q = POL.parse_qualified('Böhmen, Königreich')
    assert q.head == 'bohmen' and q.qtype == 'kingdom' and q.is_area and q.alts == []
    q = POL.parse_qualified('Askanien (Aschersleben), Grafschaft')
    assert q.head == 'askanien' and q.alts == ['aschersleben'] and q.qtype == 'county'
    q = POL.parse_qualified('Niederlande, Vereinigte (Republik der Sieben Vereinigten Provinzen)')
    assert q.head == 'niederlande' and q.qtype == 'union' and q.alts == ['republik der sieben vereinigten provinzen'] and not q.is_area
    q = POL.parse_qualified('Löbau (Lubawa)')
    assert q.head == 'lobau' and q.alts == ['lubawa'] and q.qtype is None
    q = POL.parse_qualified('Weston, Hertfordshire')
    assert q.head == 'weston' and q.qtype is None
    assert POL.parse_qualified('Saporoger Sitsch, Hetmanat').qtype == 'hetmanate' and POL.parse_qualified('Hessen-Kassel, Landgrafschaft').qtype == 'landgraviate'
    assert POL.parse_qualified('Nassau-Siegen, Grafschaft bzw. Fürstentum (seit 1652)').qtype == 'county'
    assert POL.parse_qualified('Obersächsischer Reichskreis').full == 'obersachsischer reichskreis'
    assert POL.parse_qualified(None).head == '' and POL.parse_qualified('').qtype is None


def test_polity_table_and_qualifier_kind():
    rows = [_row('religionsfrieden_quellenedition', 'a', 'Böhmen, Königreich'), _row('tagebuecher_christian_ii_anhalt', 'b', 'Niederlande, Vereinigte (Republik der Sieben Vereinigten Provinzen)'),
            _row('tagebuecher_christian_ii_anhalt', 'c', 'Hessen-Kassel, Landgrafschaft'), _row('tagebuecher_christian_ii_anhalt', 'd', 'Pommern, Herzogtum'),
            _row('tagebuecher_christian_ii_anhalt', 'e', 'Mecklenburg, Herzogtum'), _row('tagebuecher_christian_ii_anhalt', 'f', 'Moldau, Fürstentum'),
            _row('tagebuecher_christian_ii_anhalt', 'g', 'Moldau'), _row('bullinger', 'h', 'Fünf Orte'), _row('tagebuecher_christian_ii_anhalt', 'i', 'Moskauer Reich (Großrussland)'),
            _row('cidtc', 'j', 'Ermelandt'), _row('ckcc_epistolarium', 'k', 'Weston, Hertfordshire'), _row('tagebuecher_christian_ii_anhalt', 'l', 'Saporoger Sitsch, Hetmanat')]
    gn = [('pommern', '1', True, 'Pommern', 50.2, 7.3, 'P', 'PPL', 'DE', '08', 700),            # the village on the Moselle
          ('moldau', '2', True, 'Moldau', 50.1, 14.4, 'H', 'STM', 'CZ', '52', 0)]
    G, out = _run(rows, gn)
    f = {i: _f(out, i) for i in range(len(rows))}
    assert f[0]['method'] == 'name_exonym' and f[0]['cand'] == 'exonym:Kingdom of Bohemia' and f[0]['qid'] == 'Q42585' and f[0]['conf'] == 'medium' and f[0]['radius'] >= 100
    assert f[1]['cand_name'] == 'Dutch Republic' and f[1]['qid'] == 'Q170072' and abs(f[1]['lat'] - 52.08) < 0.1
    assert f[2]['cand_name'] == 'Landgraviate of Hesse-Kassel' and abs(f[2]['lat'] - 51.3127) < 0.01 and f[2]['radius'] == 70.0
    assert f[3]['cand_name'] == 'Duchy of Pomerania' and f[3]['lat'] > 53                    # the duchy, not the village on the Moselle
    assert f[4]['method'] == 'name_exonym' and abs(f[4]['lat'] - 53.7) < 0.1 and f[4]['radius'] >= 60 and f[4]['tier'] == 'polity_exonym'    # head in the exonym table, area by qualifier
    assert f[5]['cand_name'] == 'Principality of Moldavia' and abs(f[5]['lat'] - 47.16) < 0.1
    assert f[6] is None or f[6]['cand_name'] != 'Principality of Moldavia'                      # the bare Moldau is the river (or nothing)
    assert f[7]['cand_name'] == 'Five Catholic Cantons' and f[7]['radius'] == 30.0 and abs(f[7]['lon'] - 8.498) < 0.01
    assert f[8]['cand_name'] == 'Muscovy' and abs(f[8]['lon'] - 37.6) < 0.2
    assert f[9]['cand_name'] == 'Warmia' and f[9]['qid'] == 'Q322620'
    assert f[10] is None                                                                        # a county qualifier of English style is no polity
    assert f[11]['cand_name'] == 'Zaporozhian Sich'
    assert all(p.basis for p in POL.POLITIES) and all(p.radius >= 25 for p in POL.POLITIES)       # every entry says where its point comes from


def test_head_and_alternative_name_are_resolved_separately():
    """'Löbau (Lubawa)' of the Dantiscus letters: the Saxon Löbau (placed elsewhere in the atlas) drowned Lubawa in Prussia."""
    gn = [('lobau', '1', True, 'Löbau', 51.095, 14.667, 'P', 'PPLA3', 'DE', '13', 15000),
          ('lubawa', '2', True, 'Lubawa', 53.50, 19.75, 'P', 'PPLA3', 'PL', '85', 10000),
          ('gdansk', '3', True, 'Gdańsk', 54.35, 18.65, 'P', 'PPLA', 'PL', '82', 470000),
          ('torun', '4', True, 'Toruń', 53.01, 18.60, 'P', 'PPLA', 'PL', '73', 200000),
          ('elblag', '5', True, 'Elbląg', 54.16, 19.40, 'P', 'PPLA2', 'PL', '85', 120000)]
    rows = [_row('cidtc', 'a', 'Löbau (Lubawa)'), _row('cidtc', 'g', 'Gdańsk', lat=54.35, lon=18.65), _row('cidtc', 't', 'Toruń', lat=53.01, lon=18.60),
            _row('cidtc', 'e', 'Elbląg', lat=54.16, lon=19.40), _row('dta_1600', 'l', 'Löbau', lat=51.095, lon=14.667)]
    pairs = pd.DataFrame({'doc': [0, 0, 0, 0, 1], 'row': [0, 1, 2, 3, 4]})
    G, out = _run(rows, gn, pairs)
    a = _f(out, 0)
    assert a and abs(a['lat'] - 53.50) < 0.1 and a['conf'] in ('medium', 'high'), a            # Lubawa: the edition's places are in Prussia
    assert 'alternative' in a['evidence'] or 'alt' in a['tier'] or 'agree' in a['tier'] or a['method'] in ('name_geonames', 'name_propagation')


def test_a_county_named_after_its_seat_is_an_area_at_the_seat():
    gn = [('aschersleben', '1', True, 'Aschersleben', 51.755, 11.46, 'P', 'PPLA3', 'DE', '14', 26000)]
    rows = [_row('tagebuecher_christian_ii_anhalt', 'a', 'Askanien (Aschersleben), Grafschaft'), _row('tagebuecher_christian_ii_anhalt', 'b', 'Askanien, Grafschaft')]
    G, out = _run(rows, gn)
    a = _f(out, 0)
    assert a and a['conf'] == 'medium' and abs(a['lat'] - 51.755) < 0.01 and a['radius'] == 40.0 and a['tier'] == 'polity_alt'
    assert _f(out, 1) is None


# ---- 5. English small homonyms --------------------------------------------------------------------------------------------------------------
WESTONS = [('weston', '1', True, 'Weston', 51.85, -0.20, 'P', 'PPL', 'GB', 'ENG', 0),            # Hertfordshire
           ('weston', '2', True, 'Weston', 50.70, -3.30, 'P', 'PPL', 'GB', 'ENG', 600),          # Devon
           ('weston', '3', True, 'Weston', 53.10, -2.40, 'P', 'PPL', 'GB', 'ENG', 900),          # Cheshire
           ('milton', '4', True, 'Milton', 51.80, -0.30, 'P', 'PPL', 'GB', 'ENG', 50),
           ('milton', '5', True, 'Milton', 51.90, -0.10, 'P', 'PPL', 'GB', 'ENG', 80)]            # two in the county


def test_hearth_tax_village_is_placed_when_exactly_one_namesake_lies_in_the_county():
    rows = [_row('hearth_tax', 'a', 'Weston'), _row('hearth_tax', 'b', 'Milton'), _row('hearth_tax', 'c', 'Weston')]
    pairs = pd.DataFrame({'doc': [0, 0, 1], 'row': [0, 1, 2]})
    herts = (51.80, -0.20, 40.0)
    cheshire = (53.20, -2.60, 60.0)
    G, out = _run(rows, WESTONS, pairs, doc_regions={0: herts, 1: cheshire})
    a, b, c = _f(out, 0), _f(out, 1), _f(out, 2)
    assert a and abs(a['lat'] - 51.85) < 0.01 and a['conf'] in ('medium', 'high')              # exactly one Weston in Hertfordshire (a hamlet with no inhabitants)
    assert b is None or b['conf'] == 'low'                                                       # two Miltons in the county: nothing
    assert c and abs(c['lat'] - 53.10) < 0.01
    assert G.why.get(1) in ('regional_ambiguous', None)


def test_casebooks_village_needs_one_namesake_next_to_the_directorys_suggestion():
    rows = [_row('casebooks', 'a', 'Weston'), _row('casebooks', 'b', 'Weston'), _row('casebooks', 'c', 'Milton')]
    anchors = [_row('casebooks', f'p{i}', f'Placed {i}', lat=51.8 + 0.01 * i, lon=-0.2) for i in range(25)]       # the cloud of the collection's placed places
    rows += anchors
    n = len(rows)
    extra = {'cb_lat': [51.84, 51.0, np.nan] + [np.nan] * 25, 'cb_lon': [-0.21, -0.20, np.nan] + [np.nan] * 25, 'cb_geo_conf': ['high', 'high', None] + [None] * 25,
             'cb_match': ['low', 'low', 'none'] + [None] * 25}
    G, out = _run(rows, WESTONS, extra_cols=extra)
    a, b, c = _f(out, 0), _f(out, 1), _f(out, 2)
    assert a and abs(a['lat'] - 51.85) < 0.01 and a['conf'] in ('medium', 'high')              # exactly one Weston next to the suggestion
    assert b is None or b['conf'] == 'low'                                                       # the suggestion is 80 km from any Weston
    assert c is None or c['conf'] == 'low'                                                       # no county, no suggestion, two Miltons round the cloud: nothing
    assert G.why.get(2) == 'regional_ambiguous'


# ---- 6. the Latin path ---------------------------------------------------------------------------------------------------------------------
def _wdlatin():
    w = LG.WikidataLatin()
    w.add('Q43631', {'la': 'Lugdunum Batavorum', 'en': 'Leiden'}, {'la': ['Leida', 'Lugdunum Batavorum']})
    w.add('Q1731', {'la': 'Rhenus'}, {})
    w.add('Q99', {'la': 'Isara'}, {})
    w.add('Q98', {'la': 'Isara'}, {})
    return w


LATIN_GN = [('colmar', '10', True, 'Colmar', 48.079, 7.358, 'P', 'PPLA2', 'FR', 'A1', 68000),
            ('kolmar', '10', False, 'Colmar', 48.079, 7.358, 'P', 'PPLA2', 'FR', 'A1', 68000)]


def test_latin_name_is_deinflected_and_found_in_the_wikidata_latin_labels():
    rows = [_row('corpus_corporum', 'a', 'Leiden', lat=52.16, lon=4.49), _row('corpus_corporum', 'b', 'Leidam'), _row('epistemon_bvh', 'c', 'Leidam'),
            _row('corpus_corporum', 'd', 'Rhenum'), _row('corpus_corporum', 'e', 'Isaram'), _row('corpus_corporum', 'f', 'Rhine', lat=51.0, lon=6.9)]
    qid = [None] * len(rows)
    qid[0], qid[5] = 'Q43631', 'Q1731'
    qid2 = {'Q99': (45.7, 5.9), 'Q98': (49.5, 2.9)}          # two different places named Isara: a homonym
    rows += [_row('corpus_corporum', 'g', 'Isara1', lat=45.7, lon=5.9), _row('corpus_corporum', 'h', 'Isara2', lat=49.5, lon=2.9)]
    qid += ['Q99', 'Q98']
    n = len(rows)
    P_extra = {'qid': qid, 'coord_kind': [None] * n, 'coord_radius_km': [np.nan] * n}
    G, out = _run(rows, [], resources={'wdlatin': _wdlatin()}, extra_cols=P_extra)
    b = _f(out, 1)
    assert b and b['method'] == 'name_latin' and b['conf'] == 'medium' and abs(b['lat'] - 52.16) < 0.01 and b['key'] == 'leida' and b['tier'] == 'latin_atlas'
    assert _f(out, 2) is None or _f(out, 2)['method'] != 'name_latin'                          # a French edition: Leidam is not read as Latin
    d = _f(out, 3)
    assert d and d['method'] == 'name_latin' and abs(d['lon'] - 6.9) < 0.01                      # Rhenum -> Rhenus (Rhine)
    assert _f(out, 4) is None and G.why.get(4) == 'latin_ambiguous'                               # Isara: two different places
    assert 'Latin' in b['evidence'] and G.latin_level[1] == 'latin' and G.latin_level[2] == 'non_latin'


def test_latin_path_orbis_latinus_gives_the_modern_name_and_the_country():
    orbis = LG.OrbisLatinus([('Colmaria', 'Kolmar', 'city', 'Elsaß', 'A', '1'), ('Argentoratum', 'Straßburg', 'city', 'Elsaß', 'A', '1'),
                             ('Tigurum', 'Zürich', 'city', 'Schweiz', 'A', '2')])
    assert orbis.get('Colmariam') == [] and len(orbis.get('Colmaria')) == 1 and orbis.get('Tigurum') == []         # the ambiguous key gives nothing
    cloud = [_row('corpus_corporum', f'c{i}', f'Ort {i}', lat=48.0 + 0.05 * i, lon=7.2 + 0.05 * i, n=3) for i in range(12)]           # the edition's places are in Alsace
    rows = [_row('corpus_corporum', 'a', 'Colmariam'), _row('ckcc_epistolarium', 'b', 'Colmariam')] + cloud
    # v2.4: the Orbis alone places a name only when the places of the same documents support it (the letter names other places of Alsace)
    pairs = pd.DataFrame({'doc': [0, 1] + [0] * 12, 'row': list(range(len(rows)))})
    G, out = _run(rows, LATIN_GN, pairs=pairs, resources={'orbis': orbis})
    a = _f(out, 0)
    assert a and a['method'] == 'name_latin' and abs(a['lat'] - 48.079) < 0.01 and a['conf'] in ('medium', 'high') and a['tier'] == 'latin_orbis'
    G0, out0 = _run(rows, LATIN_GN, resources={'orbis': orbis})                               # no document names a neighbour: a suggestion only
    assert _f(out0, 0)['conf'] == 'low' and G0.why.get(0) == 'latin_orbis_alone'
    assert LG.region_countries('Frankr. (Nievre)') == {'FR'} and LG.region_countries('Preußen (Posen)') >= {'PL', 'DE'} and LG.region_countries('Niederöeterr') >= {'AT'}
    assert LG.region_countries('Mfl.') is None and LG.region_countries('') is None


def test_latin_combine_gates():
    h = lambda s, sc, la, lo, r=None, q=None: LG.Hit(s, sc, la, lo, r, q)           # noqa: E731
    assert LG.combine([], 'latin')[:2] == (None, 'latin_no_match')
    assert LG.combine([h('wikidata', 0.9, 52.0, 4.5), h('exonym', 0.9, 52.1, 4.4)], 'latin')[0] == 'high'          # two families agree, strong rule
    assert LG.combine([h('wikidata', 0.9, 52.0, 4.5), h('atlas', 0.9, 52.0, 4.5)], 'latin')[0] == 'medium'         # the same family twice is one source
    assert LG.combine([h('wikidata', 0.9, 52.0, 4.5)], 'latin')[0] == 'medium'
    assert LG.combine([h('wikidata', 0.9, 52.0, 4.5)], 'mixed')[0] == 'low'                                         # one source in a mixed context: a suggestion
    assert LG.combine([h('wikidata', 0.5, 52.0, 4.5)], 'latin')[0] == 'low'                                         # a weak rule
    assert LG.combine([h('wikidata', 0.9, 52.0, 4.5), h('orbis', 0.8, 48.0, 7.3)], 'latin')[:2] == (None, 'latin_ambiguous')   # two places
    assert LG.combine([h('wikidata', 0.9, 46.0, 2.0, 350.0), h('exonym', 0.9, 47.5, 2.5, 350.0)], 'latin')[0] == 'high'          # an area covers its own centroid
    # a river has several points: the same Wikidata item is one place
    assert LG.combine([h('wikidata', 0.8, 45.2, 29.7, None, 'Q1653'), h('atlas', 0.8, 48.5, 12.0, None, 'Q1653')], 'latin')[0] == 'medium'
    # a weak rule (Delfos -> Delfi) is a suggestion, whatever the sources say
    assert LG.combine([h('wikidata', 0.45, 52.0, 4.3), h('orbis', 0.45, 52.0, 4.3)], 'latin')[0] == 'low'
    # two places carry the name: nothing, unless the edition (or a large prominence) decides -- then at most medium
    two = [h('wikidata', 0.9, 51.9, 5.9, None, 'Q584'), h('wikidata', 0.9, 44.5, 11.3, None, 'Q658594')]
    assert LG.combine(two, 'latin')[:2] == (None, 'latin_ambiguous')
    c, why, ch = LG.combine(two, 'latin', lambda cl: 0)
    assert c == 'medium' and why == 'ok_tiebreak' and ch.qid == 'Q584'
    assert LG.combine([h('wikidata', 0.9, 51.9, 5.9, None, 'Q584'), h('exonym', 0.9, 51.8, 5.8)], 'latin', lambda cl: 0)[0] == 'high'
    assert LG.METHOD == 'name_latin' and LG.SOURCE_CODE == 'l'


def test_latin_holdout_hides_the_place_and_its_item():
    rows = [_row('corpus_corporum', 'a', 'Leida', lat=52.16, lon=4.49), _row('corpus_corporum', 'b', 'Colmaria', lat=48.079, lon=7.358)]
    rows += [_row('corpus_corporum', f'c{i}', f'Ort {i}', lat=48.0 + 0.05 * i, lon=7.2 + 0.05 * i, n=3) for i in range(12)]
    P_extra = {'qid': ['Q43631', None] + [None] * 12, 'coord_kind': [None] * 14, 'coord_radius_km': [np.nan] * 14}
    pairs = pd.DataFrame({'doc': [0, 1] + [1] * 12, 'row': list(range(14))})                   # Colmaria's document names the Alsatian places of the cloud
    G = _geocoder(rows, LATIN_GN, pairs=pairs, resources={'wdlatin': _wdlatin(), 'orbis': LG.OrbisLatinus([('Colmaria', 'Kolmar', 'city', 'Elsaß', 'A', '1')])}, extra_cols=P_extra)
    df, summ = GC.evaluate_latin(G, 10, 1)
    assert summ['n_hidden'] == 10 or summ['n_hidden'] == 14
    # Leida: its own item is hidden with it (no other place carries Q43631), so nothing vouches for it
    assert df.set_index('name').loc['Leida', 'conf'] is None or pd.isna(df.set_index('name').loc['Leida', 'conf'])
    c = df.set_index('name').loc['Colmaria']
    assert c['conf'] in ('medium', 'high') and c['km'] < 1.0
    assert summ['high+medium']['within50'] >= 0.99


def test_two_places_with_the_same_latin_name_are_decided_by_the_edition_or_stay_unplaced():
    """Wikidata calls both the Rhine and the Reno 'Rhenus': the Latin letters of a Dutch cloud mean the Rhine; without such a cloud nothing is placed."""
    w = LG.WikidataLatin()
    w.add('Q584', {'la': 'Rhenus'}, {})
    w.add('Q658594', {'la': 'Rhenus'}, {})
    anchors = [_row('corpus_corporum', f'n{i}', f'Stad {i}', lat=52.0 + 0.05 * i, lon=5.0 + 0.05 * i, n=3) for i in range(12)]
    rows = [_row('corpus_corporum', 'r', 'Rhein', lat=51.97, lon=5.9, n=30), _row('corpus_corporum', 'x', 'Reno', lat=44.5, lon=11.3, n=1), _row('corpus_corporum', 'u', 'Rhenum')] + anchors
    qid = ['Q584', 'Q658594', None] + [None] * 12
    n = len(rows)
    G, out = _run(rows, [], resources={'wdlatin': w}, extra_cols={'qid': qid, 'coord_kind': [None] * n, 'coord_radius_km': [np.nan] * n})
    u = _f(out, 2)
    assert u and u['method'] == 'name_latin' and abs(u['lat'] - 51.97) < 0.01 and u['conf'] == 'medium' and u['tier'].endswith('+tiebreak')
    # with no cloud and no prominence (equal weights) nothing is placed
    rows2 = [_row('corpus_corporum', 'r', 'Rhein', lat=51.97, lon=5.9, n=2), _row('corpus_corporum', 'x', 'Reno', lat=44.5, lon=11.3, n=2), _row('corpus_corporum', 'u', 'Rhenum')]
    G2, out2 = _run(rows2, [], resources={'wdlatin': w}, extra_cols={'qid': ['Q584', 'Q658594', None], 'coord_kind': [None] * 3, 'coord_radius_km': [np.nan] * 3})
    assert _f(out2, 2) is None and G2.why.get(2) == 'latin_ambiguous'
    # a weak rule (Greek accusative plural read as Latin: Delfos -> Delfi) is never more than a suggestion
    w2 = LG.WikidataLatin()
    w2.add('Q690', {'la': 'Delfi'}, {})
    orb = LG.OrbisLatinus([('Delfi', 'Delft', 'city', 'Niederl.', 'A', '1')])
    rows3 = [_row('corpus_corporum', 'd', 'Delft', lat=52.01, lon=4.36), _row('corpus_corporum', 'u', 'Delfos')]
    G3, out3 = _run(rows3, [('delft', '1', True, 'Delft', 52.01, 4.36, 'P', 'PPL', 'NL', '11', 100000)], resources={'wdlatin': w2, 'orbis': orb},
                    extra_cols={'qid': ['Q690', None], 'coord_kind': [None] * 2, 'coord_radius_km': [np.nan] * 2})
    f3 = _f(out3, 1)
    assert f3 is None or f3['conf'] == 'low'


def test_language_of_the_mentions_decides_the_latin_context_of_a_place(tmp_path):
    """A place whose own mentions say Latin (lang la / lat) is read as Latin in an edition that is mostly French, and the reverse: ``load_pairs`` carries the counts."""
    rows = []
    for key, langs in (('surf:a', ['la', 'lat', 'la', None]), ('surf:b', ['fr', 'fra', 'fre']), ('surf:c', [None, None]), ('surf:d', ['la', 'fr', 'de'])):
        for i, lg in enumerate(langs):
            rows.append(dict(coll_slug='ckcc_epistolarium', doc_id=f'{key}{i}', source_path='x.xml', kind='place', tag='placeName', zone='text', local_key=None,
                             surface=key[5:], canonical=True, excluded=False, lang=lg))
    p = tmp_path / 'm.parquet'
    pd.DataFrame(rows).to_parquet(p)
    P = pd.DataFrame({'coll_slug': 'ckcc_epistolarium', 'entity_key': ['surf:a', 'surf:b', 'surf:c', 'surf:d'], 'name': ['a', 'b', 'c', 'd']})
    pairs, hint, _ = GC.load_pairs(p, P)
    assert hint.loc[0, 'n_known'] == 3 and hint.loc[0, 'n_latin'] == 3 and hint.loc[1, 'n_latin'] == 0 and hint.loc[3, 'n_known'] == 3 and hint.loc[3, 'n_latin'] == 1
    levels = GC.Geocoder._latin_levels(object.__new__(GC.Geocoder), P, hint.reindex(range(4)).fillna(0.0))
    assert levels[0] == 'latin' and levels[1] == 'non_latin'
    assert levels[2] == 'mixed' and levels[3] == 'mixed'          # nothing known: the edition's level (ckcc is mixed); one Latin in three: the edition's level


def test_latin_guards_orbis_village_known_homonym_and_far_single_source():
    cloud_nl = [_row('corpus_corporum', f'n{i}', f'Stad {i}', lat=52.0 + 0.05 * i, lon=5.0 + 0.05 * i, n=3) for i in range(12)]
    # (a) the Orbis Latinus alone names a village: a suggestion
    orb = LG.OrbisLatinus([('Viana', 'Weinstetten', 'village', 'Württemberg', 'A', '1')])
    gn = [('weinstetten', '1', True, 'Weinstetten', 48.29, 9.97, 'P', 'PPL', 'DE', '01', 120)]
    near = [_row('corpus_corporum', f's{i}', f'Ort {i}', lat=48.2 + 0.05 * i, lon=9.9 + 0.05 * i, n=3) for i in range(12)]
    G, out = _run([_row('corpus_corporum', 'u', 'Vianam')] + near, gn, resources={'orbis': orb})
    assert _f(out, 0) is None or _f(out, 0)['conf'] == 'low'
    # (b) Lugdunum is Lyon or Leiden (v2.4, ``latin_homonyms``): the Dutch cloud says Leiden, a French cloud alone says nothing (Lyon needs a cue of its own)
    w = LG.WikidataLatin()
    w.add('Q456', {'la': 'Lugdunum'}, {})
    rows = [_row('corpus_corporum', 'ly', 'Lyon', lat=45.76, lon=4.84, n=40), _row('corpus_corporum', 'u', 'Lugduno')] + cloud_nl
    n = len(rows)
    extra = {'qid': ['Q456'] + [None] * (n - 1), 'coord_kind': [None] * n, 'coord_radius_km': [np.nan] * n}
    dn = len(rows)
    pairs = pd.DataFrame({'doc': [0, 1] + [1] * 12, 'row': list(range(dn))})                                     # the letter with 'Lugduno' names Dutch places
    G, out = _run(rows, [], pairs=pairs, resources={'wdlatin': w}, extra_cols=extra)
    assert _f(out, 1) and abs(_f(out, 1)['lat'] - 52.16) < 0.01 and _f(out, 1)['tier'] == 'latin_homonym'          # Leiden, not Lyon
    cloud_fr = [_row('corpus_corporum', f'f{i}', f'Lieu {i}', lat=45.5 + 0.05 * i, lon=4.6 + 0.05 * i, n=3) for i in range(12)]
    rows = [_row('corpus_corporum', 'ly', 'Lyon', lat=45.76, lon=4.84, n=40), _row('corpus_corporum', 'u', 'Lugduno')] + cloud_fr
    pairs = pd.DataFrame({'doc': [0, 1] + [1] * 12, 'row': list(range(len(rows)))})
    G, out = _run(rows, [], pairs=pairs, resources={'wdlatin': w}, extra_cols={'qid': ['Q456'] + [None] * (len(rows) - 1), 'coord_kind': [None] * len(rows), 'coord_radius_km': [np.nan] * len(rows)})
    assert _f(out, 1) is None and G.why.get(1) == 'latin_homonym_needs_cue'                                       # a French cloud is no explicit support for Lyon
    aux = {1: {'windows': ['viso Brisaco, Geneva, Lugduno ivit visurus regem apud Rhodanum']}}
    G, out = _run(rows, [], pairs=pairs, resources={'wdlatin': w, 'mention_aux': aux}, extra_cols={'qid': ['Q456'] + [None] * (len(rows) - 1), 'coord_kind': [None] * len(rows), 'coord_radius_km': [np.nan] * len(rows)})
    assert _f(out, 1) and _f(out, 1)['conf'] == 'medium' and abs(_f(out, 1)['lat'] - 45.764) < 0.01                  # 'Rhodanum' in the passage: explicit support
    # (c) one family, a place far from everything the edition talks about and mentioned twice in the atlas (Vallesia -> Wales): a suggestion
    w2 = LG.WikidataLatin()
    w2.add('Q25', {'la': 'Vallesia'}, {})
    rows = [_row('corpus_corporum', 'wa', 'Wales', lat=52.5, lon=-3.5, n=2), _row('corpus_corporum', 'u', 'Vallesiam')] + cloud_nl
    G, out = _run(rows, [], pairs=pd.DataFrame({'doc': [0, 1] + [1] * 12, 'row': list(range(len(rows)))}), resources={'wdlatin': w2}, extra_cols={'qid': ['Q25'] + [None] * (len(rows) - 1), 'coord_kind': [None] * len(rows), 'coord_radius_km': [np.nan] * len(rows)})
    assert _f(out, 1) is None or _f(out, 1)['conf'] == 'low'


def test_latin_reading_does_not_override_a_placed_vernacular_name():
    """'Monaco' is the principality (placed elsewhere in the atlas) and the ablative of Monacum = Munich: two readings that disagree give a suggestion, not a placement."""
    w = LG.WikidataLatin()
    w.add('Q1726', {'la': 'Monacum'}, {})
    cloud = [_row('cidtc', f'c{i}', f'Ort {i}', lat=48.0 + 0.05 * i, lon=11.4 + 0.05 * i, n=3) for i in range(12)]
    rows = [_row('cidtc', 'mu', 'München', lat=48.14, lon=11.58, n=40), _row('ckcc_epistolarium', 'mc', 'Monaco', lat=43.73, lon=7.42, n=5), _row('cidtc', 'u', 'Monaco')] + cloud
    n = len(rows)
    pairs = pd.DataFrame({'doc': [0, 1, 2] + [2] * 12, 'row': list(range(n))})
    G, out = _run(rows, [], pairs=pairs, resources={'wdlatin': w}, extra_cols={'qid': ['Q1726'] + [None] * (n - 1), 'coord_kind': [None] * n, 'coord_radius_km': [np.nan] * n})
    f = _f(out, 2)
    assert f is None or f['conf'] == 'low' or abs(f['lat'] - 43.73) < 0.1


def test_alternative_name_that_is_a_country_or_region_is_a_qualifier_not_the_place():
    gn = [('frankreich', '1', True, 'France', 46.6, 2.5, 'A', 'PCLI', 'FR', '00', 65_000_000),
          ('lobau', '2', True, 'Löbau', 51.095, 14.667, 'P', 'PPLA3', 'DE', '13', 15000)]
    rows = [_row('urfehdebuecher_basel_x', 'a', 'Molsheim (Frankreich)'), _row('urfehdebuecher_basel_x', 'b', 'Löbau (Frankreich)')]
    G, out = _run(rows, gn)
    assert _f(out, 0) is None or _f(out, 0)['conf'] == 'low'                                   # not the centroid of France
    f = _f(out, 1)
    assert f is None or f['conf'] == 'low' or abs(f['lat'] - 51.095) < 0.1                     # the head, if anything


def test_head_of_a_county_is_a_region_not_a_commune():
    gn = [('hohnstein', '1', True, 'Hohnstein', 50.983, 14.117, 'A', 'ADM4', 'DE', '13', 3150),
          ('aschersleben', '2', True, 'Aschersleben', 51.755, 11.46, 'P', 'PPLA3', 'DE', '14', 26000)]
    G, out = _run([_row('tagebuecher_christian_ii_anhalt', 'a', 'Hohnstein, Grafschaft'), _row('tagebuecher_christian_ii_anhalt', 'b', 'Askanien (Aschersleben), Grafschaft')], gn)
    assert _f(out, 0) is None or _f(out, 0)['conf'] == 'low'        # the commune of Saxony is not the county of the Harz
    assert _f(out, 1) and _f(out, 1)['tier'] == 'polity_alt'        # a bracketed seat is


def test_strongly_inflected_form_of_a_prominent_place_is_medium_in_a_mixed_edition():
    h = LG.Hit('wikidata', 0.8, 52.0, 4.5, None, 'Q584', 'Rhenus', 1, 'Rhein', 50.0)
    assert LG.combine([h], 'mixed')[0] == 'low' and LG.combine([h], 'mixed', None, True)[0] == 'medium' and LG.combine([h], 'non_latin', None, False)[0] == 'low'
