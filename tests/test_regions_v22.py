"""v2.2 geocoder priors from the document itself: county of a Hearth Tax file, London prior (MoEML, Old Bailey), file consistency, exonym fixes,
edition-qualified names (Bergen / Bergen op Zoom)."""
import numpy as np
import pandas as pd

from ratlas import exonyms as EX
from ratlas import reconcile_geocode as GC
from ratlas import regions as RGN
from test_geocode_v2 import _gn, _places, _row


def _geocoder(rows, gn, pairs=None, doc_regions=None, anchors=True):
    P = _places(rows)
    pairs = pairs if pairs is not None else pd.DataFrame({'doc': [0], 'row': [0]})
    return GC.Geocoder(P, _gn(gn), pairs, pd.DataFrame(columns=['h', 't', 'l']), doc_regions)


def _run(rows, gn, pairs=None, doc_regions=None):
    G = _geocoder(rows, gn, pairs, doc_regions)
    out = GC.run_pipeline(G, np.array([], dtype=int), log=lambda m: None)
    return G, out


# ---- pure functions -------------------------------------------------------------------------------------------------------------
def test_county_of_a_file_from_its_own_headings():
    assert RGN.county_region('West Riding of Yorkshire - Claro and Ainsty wapentakes')[0] == 'West Riding of Yorkshire'
    assert RGN.county_region('East Ridd Com Ebor')[0] == 'East Riding of Yorkshire'
    assert RGN.county_region('Yorkshire, North Riding')[0] == 'North Riding of Yorkshire'
    assert RGN.county_region('York, City')[0] == 'York' and RGN.county_region('Norwich, City')[0] == 'Norwich'
    assert RGN.county_region('Kingston-upon-Hull')[0] == 'Kingston upon Hull' and RGN.county_region('Westmerland')[0] == 'Westmorland'
    assert RGN.county_region('Bottom of Westmerland')[0] == 'Westmorland' and RGN.county_region('Nowhere') is None
    # the county tag first, then the header, then the cities
    assert RGN.doc_region(['Norfolk'], [], ["King's Lynn", 'Norwich'])[0] == 'Norfolk'
    assert RGN.doc_region([], ['Warwickshire - Kineton Hundred (Brailes Division) -'])[0] == 'Warwickshire'
    assert RGN.doc_region([], [], ['Chichester city']) is None


def test_london_like_names_and_the_box():
    for n in ('Rosemary-Lane', 'Newgate-street', 'St. Stephen (Coleman Street)', 'Lyon’s Inn', 'Bedford House', 'The Lion (Shoreditch)', 'Temple', 'Marshalsea'):
        assert RGN.london_like(n), n
    for n in ('Oxford', 'Lancashire', 'Eling', 'The Hague', 'Bristol'):
        assert not RGN.london_like(n), n
    assert RGN.in_london(51.5, -0.1) and not RGN.in_london(50.8, -2.87) and not RGN.in_london(45.76, 4.84)
    # v2.3: the London prior is a jurisdiction prior for EVERY Old Bailey name (a provincial town of 100,000 or a name outside the British Isles is let through
    # by ``london_keep``), not only for the names that look like London streets
    assert RGN.london_applies('map_of_early_modern_london', 'Pimlico') and RGN.london_applies('old_bailey', 'Oxford') and RGN.london_applies('old_bailey', 'Rosemary-Lane')
    assert not RGN.london_applies('hearth_tax', 'Temple')


def test_regional_file_needs_eight_places_and_a_tight_cluster():
    la = np.array([53.8, 53.9, 53.7, 53.85, 53.95, 53.75, 53.8, 53.9, 50.4])
    lo = np.array([-1.6, -1.5, -1.7, -1.4, -1.8, -1.5, -1.6, -1.55, -4.9])
    m = RGN.is_regional(la, lo)
    assert m is not None and abs(m[0] - 53.8) < 0.2
    assert RGN.is_regional(la[:5], lo[:5]) is None                                       # too few places
    assert RGN.is_regional(np.array([50.0, 55.0] * 6), np.array([-3.0, 0.0] * 6)) is None   # not a cluster


# ---- the geocoder -------------------------------------------------------------------------------------------------------------
NEWTON = [('newton', '1', True, 'Newton', 50.52, -4.9, 'P', 'PPL', 'GB', 'ENG', 900),                  # Cornwall, the more populous
          ('newton', '2', True, 'Newton', 53.70, -1.40, 'P', 'PPL', 'GB', 'ENG', 300)]               # West Riding


def test_newton_of_a_west_riding_file_is_not_in_cornwall():
    """v2.1: 169 Hearth Tax places were in another county than their file. The file's own county (West Riding) is the prior."""
    rows = [_row('hearth_tax', 'a', 'Newton'), _row('hearth_tax', 'b', 'Newton')]
    pairs = pd.DataFrame({'doc': [0, 1], 'row': [0, 1]})
    G, out = _run(rows, NEWTON, pairs, doc_regions=None)
    wr = (53.85, -1.65, 85.0)
    G2, out2 = _run(rows, NEWTON, pairs, doc_regions={0: wr, 1: (50.4, -4.9, 55.0)})
    a, b = out2[0]['final'], out2[1]['final']
    assert a and abs(a['lat'] - 53.70) < 0.1                                    # the West Riding file gets the West Riding Newton
    assert b and abs(b['lat'] - 50.52) < 0.1                                    # a Cornwall file gets the Cornish one
    # a place that occurs in a file with a region and one without is not constrained
    G3, out3 = _run(rows[:1], NEWTON, pd.DataFrame({'doc': [0, 5], 'row': [0, 0]}), doc_regions={0: wr})
    assert 0 not in G3.row_regions
    assert G2.row_regions[0] == [wr]


def test_doc_codes_separate_files_that_share_a_doc_id():
    m = pd.DataFrame({'coll_slug': ['hearth_tax'] * 4 + ['bullinger'] * 2, 'doc_id': ['d2e1'] * 4 + ['x', 'x'],
                      'source_path': ['a/WR1.xml', 'a/WR1.xml', 'a/Kent.xml', 'a/Kent.xml', 'b/1.xml', 'b/1.xml']})
    c = GC.doc_codes(m).tolist()
    assert c[0] == c[1] and c[2] == c[3] and c[0] != c[2] and c[4] == c[5] and len({c[0], c[2], c[4]}) == 3
    m2 = pd.DataFrame({'coll_slug': ['q', 'q'], 'doc_id': ['t1', 't1'], 'source_path': ['a.xml', 'b.xml']})       # one doc_id in two files: two documents
    assert GC.doc_codes(m2).nunique() == 2


def test_file_regions_come_from_the_headings_of_the_file():
    m = pd.DataFrame({
        'coll_slug': ['hearth_tax'] * 4 + ['bullinger'], 'doc': [0, 0, 0, 1, 2],
        'tag': ['placeName', 'placeName@county', 'placeName@township', 'placeName@city', 'placeName'],
        'zone': ['header', 'text', 'text', 'text', 'text'],
        'surface': ['West Riding of Yorkshire - Osgoldcross wapentake', 'West Riding', 'Airmyn', 'Norwich Citty', 'Zürich']})
    r = GC.file_regions(m)
    assert set(r) == {0, 1} and r[0][2] == 85.0 and abs(r[0][0] - 53.85) < 1e-9 and abs(r[1][0] - 52.63) < 1e-9


LONDON = [('marshalsea', '1', True, 'Marshalsea', 50.80, -2.87, 'P', 'PPL', 'GB', 'ENG', 40),          # Dorset
          ('marshalsea', '2', True, 'The Marshalsea', 51.50, -0.09, 'S', 'PRN', 'GB', 'ENG', 0),      # not a PPL: ignored by the P/A classes
          ('petty france', '1', True, 'Petty France', 51.567, -2.31, 'P', 'PPL', 'GB', 'ENG', 40),
          ('oxford', '1', True, 'Oxford', 51.75, -1.26, 'P', 'PPLA2', 'GB', 'ENG', 160000),
          ('rosemary lane', '1', True, 'Rosemary Lane', 50.92, -3.20, 'P', 'PPL', 'GB', 'ENG', 30),
          ('rosemary lane', '2', True, 'Rosemary Lane', 51.51, -0.07, 'P', 'PPLX', 'GB', 'ENG', 0),
          ('pimlico', '1', True, 'Pimlico', 51.74, -0.42, 'P', 'PPL', 'GB', 'ENG', 200),
          ('pimlico', '2', True, 'Pimlico', 51.489, -0.139, 'P', 'PPLX', 'GB', 'ENG', 0),
          ('lyon', '1', True, 'Lyon', 45.764, 4.836, 'P', 'PPLA', 'FR', 'A', 500000)]


def test_london_prior_moeml_and_old_bailey():
    rows = [_row('map_of_early_modern_london', 'a', 'Marshalsea'), _row('map_of_early_modern_london', 'b', 'Petty France'),
            _row('map_of_early_modern_london', 'c', 'Pimlico'), _row('map_of_early_modern_london', 'd', 'Oxford'),
            _row('old_bailey', 'e', 'Rosemary-Lane'), _row('old_bailey', 'f', 'Oxford'), _row('map_of_early_modern_london', 'g', 'Lyon’s Inn'),
            _row('map_of_early_modern_london', 'h', 'Savoy Manor'), _row('old_bailey', 'i', 'Savoy')]
    pairs = pd.DataFrame({'doc': list(range(len(rows))), 'row': list(range(len(rows)))})
    G, out = _run(rows, LONDON, pairs)
    f = {i: out[i]['final'] for i in range(len(rows))}
    assert f[0] is None or f[0]['lat'] > 51.2                                   # Marshalsea: not Dorset
    assert f[1] is None and G.why[1] == 'outside_region'                        # Petty France of Somerset is out
    assert f[2] and abs(f[2]['lat'] - 51.489) < 0.05                            # Pimlico: the London one, though the Hertfordshire one is bigger
    assert f[3] is None                                                         # MoEML: Oxford is not London (not in the box, not a major feature)
    assert f[4] and abs(f[4]['lat'] - 51.51) < 0.05                             # Old Bailey street names are London's
    assert f[5] and abs(f[5]['lat'] - 51.75) < 0.05                             # ... a town name is not constrained to London
    assert f[6] is None                                                         # Lyon's Inn is not Lyon
    assert f[7] and f[7]['method'] == 'name_exonym' and abs(f[7]['lat'] - 51.5104) < 0.01        # Savoy Manor: the Strand, not Savoie
    assert f[8] and abs(f[8]['lat'] - 51.5104) < 0.01 and abs(f[8]['lon'] + 0.1205) < 0.01
    assert EX.lookup('savoy')[3] == 'Savoy' and EX.lookup('savoy', 'bullinger')[3] == 'Savoy'               # the duchy elsewhere
    assert EX.lookup('savoy', 'map_of_early_modern_london')[3] == 'Savoy (London)'


def test_exonym_table_hanse_alps_thermopylae_windsor():
    gn = [('hanse', '1', True, 'Hänse', 51.0, 10.0, 'P', 'PPL', 'DE', '16', 0),
          ('alpen', '2', True, 'Alpen', 51.58, 6.52, 'P', 'PPL', 'DE', '07', 12000),
          ('thermopyles', '3', True, 'Saint-Marcellin', 45.15, 5.32, 'P', 'PPL', 'FR', 'A', 8000),
          ('windesore', '4', True, 'Broadwindsor', 50.82, -2.80, 'P', 'PPL', 'GB', 'ENG', 1200)]
    rows = [_row('bullinger', 'a', 'Hanse (Hansestädte)'), _row('ckcc_epistolarium', 'b', 'Alpen'), _row('epistemon_bvh', 'c', 'Thermopyles'),
            _row('ckcc_epistolarium', 'd', 'Windesore'), _row('corpus_corporum', 'e', 'Thermopylis'), _row('bullinger', 'f', 'Hansestädte')]
    pairs = pd.DataFrame({'doc': list(range(6)), 'row': list(range(6))})
    G, out = _run(rows, gn, pairs)
    a, b, c, d, e, f = (out[i]['final'] for i in range(6))
    assert a['method'] == 'name_exonym' and a['cand_name'] == 'Hanseatic League' and a['radius'] >= 200 and a['lat'] > 52      # not the hamlet Hänse
    assert f['cand_name'] == 'Hanseatic League'
    assert b['cand_name'] == 'Alps' and b['radius'] >= 300 and abs(b['lat'] - 46.6) < 1                                         # the mountains, not Alpen on the Rhine
    assert c['cand_name'] == 'Thermopylae' and abs(c['lat'] - 38.797) < 0.01 and e['cand_name'] == 'Thermopylae'
    assert d['cand_name'] == 'Windsor' and abs(d['lat'] - 51.482) < 0.01
    assert EX.FEATURE_KIND['Alps'] == 'n' and EX.FEATURE_KIND['Thermopylae'] == 'n' and EX.FEATURE_KIND['Hanseatic League'] == 'a'


def test_bare_name_takes_the_qualified_place_of_the_same_edition():
    """Bergen in the Descartes letters is Bergen op Zoom, which the register has; GeoNames' Bergen is the Norwegian city."""
    gn = [('bergen', '1', True, 'Bergen', 60.39, 5.32, 'P', 'PPLA', 'NO', '12', 250000),
          ('bergen', '2', True, 'Bergen', 52.8, 9.97, 'P', 'PPL', 'DE', '06', 15000)]
    rows = [_row('ckcc_epistolarium', 'op', 'Bergen op Zoom', lat=51.494, lon=4.288), _row('ckcc_epistolarium', 'bare', 'Bergen'),
            _row('sozinianische_briefwechsel', 'no', 'Bergen'), _row('ckcc_epistolarium', 'f1', 'Frankfurt an der Oder', lat=52.35, lon=14.55),
            _row('ckcc_epistolarium', 'f2', 'Frankfurt am Main', lat=50.11, lon=8.68), _row('ckcc_epistolarium', 'f3', 'Frankfurt')]
    pairs = pd.DataFrame({'doc': list(range(6)), 'row': list(range(6))})
    G, out = _run(rows, gn, pairs)
    b = out[1]['final']
    assert b and b['tier'] == 'edition_qualified' and abs(b['lat'] - 51.494) < 0.01 and b['conf'] == 'medium'
    n = out[2]['final']
    assert n is None or abs(n['lat'] - 51.494) > 1                              # another edition without the qualified form: GeoNames decides
    assert out[5]['final'] is None or out[5]['final']['tier'] != 'edition_qualified'        # two qualified Frankfurts, 1000 km apart: no reading


def test_file_consistency_takes_back_the_far_name_based_point():
    gn = [('newton', '1', True, 'Newton', 50.52, -4.9, 'P', 'PPL', 'GB', 'ENG', 900)]
    base = [_row('hearth_tax', f'r{i}', f'Place{i}', lat=53.8 + 0.02 * i, lon=-1.6 + 0.02 * i) for i in range(9)]
    rows = base + [_row('hearth_tax', 'n', 'Newton'), _row('hearth_tax', 'ok', 'Newton')]
    ndoc = len(rows)
    pairs = pd.DataFrame({'doc': [0] * 10 + [1], 'row': list(range(10)) + [10]})
    G, out = _run(rows, gn, pairs)                        # no document regions: the Cornish Newton is placed, it is 500 km from the file's median
    assert out[9]['final'] and out[9]['final']['lat'] < 51
    dropped = GC.file_outliers(G, out, log=lambda m: None)
    assert dropped == {9: 'file_outlier'} and out[9]['final']['conf'] == 'low' and GC.reason_of(G, 9, out[9]) == 'file_outlier'
    # row 10 is alone in its file: nothing to compare with, kept
    assert out[10]['final']['conf'] in ('high', 'medium')
    # a collection that is not regional by nature (letters, legal sources) is left alone
    rows3 = [(('bullinger',) + r[1:]) for r in rows]
    G3, out3 = _run(rows3, gn, pairs)
    assert GC.file_outliers(G3, out3, log=lambda m: None) == {}


def test_london_prior_does_not_let_a_first_order_region_through():
    """The parishes of Jersey are ADM1 in GeoNames: 'St. Mary (Abchurch)' of MoEML must not become Saint Mary of Jersey."""
    gn = [('saint mary', '1', True, 'Saint Mary', 49.236, -2.172, 'A', 'ADM1', 'JE', '01', 4000),
          ('saint mary', '2', True, 'St Mary', 51.51, -0.09, 'P', 'PPLX', 'GB', 'ENG', 0)]
    rows = [_row('map_of_early_modern_london', 'a', 'St. Mary')]
    G, out = _run(rows, gn)
    f = out[0]['final']
    assert f is None or abs(f['lat'] - 51.51) < 0.1
    keep = G.row_keep(0, [49.236, 51.51], [-2.172, -0.09], [4000, 0], ['ADM1', 'PPLX'])
    assert list(keep) == [False, True]


def test_exonym_cities_obey_london_and_file_regions_but_not_the_collections_hard_box():
    """Rome is Rome in the Lüneburg nuns' letters (hard box of Lower Saxony), Lyon's Inn is not Lyon in MoEML (London prior)."""
    gn = [('rom', '1', True, 'Rom', 46.4, 0.0, 'P', 'PPL', 'FR', 'A', 300)]
    G, out = _run([_row('luene_nonnenbriefe', 'a', 'Rom'), _row('map_of_early_modern_london', 'b', 'Rom')], gn, pd.DataFrame({'doc': [0, 1], 'row': [0, 1]}))
    a, b = out[0]['final'], out[1]['final']
    assert a and a['method'] == 'name_exonym' and abs(a['lat'] - 41.903) < 0.1          # outside the hard box, but the exonym table's Rome
    assert b is None or b['method'] != 'name_exonym'                                       # MoEML: Rome is not London
