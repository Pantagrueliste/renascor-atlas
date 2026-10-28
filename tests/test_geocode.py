import numpy as np
import pandas as pd

from ratlas import geotext as GT
from ratlas import reconcile_geocode as GC


def test_forms_clean_surfaces():
    f = GT.forms('of Newp')
    assert f.keys(('t0',)) == ['newp']
    assert GT.forms('of this towne').junk
    assert GT.forms('denha004').junk
    assert GT.forms('UNKNOWN').junk
    assert GT.forms('s.l.').junk
    f = GT.forms('Cracow (Kraków)')
    assert 'cracow' in f.keys(('t0',)) and 'krakow' in f.keys(('paren',))
    assert 'einsiedeln' in GT.forms('Kloster Einsiedeln').keys(('t1',))
    assert 'woodcote' in GT.forms('Woodcote tithing').keys(('t1',))
    assert 'zurich' in GT.forms('von Zürich').keys(('t1',))
    assert 'venedig' in GT.forms('Venedig, Republik').keys(('t1',))
    assert 'london' in GT.forms("London's").keys(('t0',))


def test_morph_and_skeleton():
    assert 'roma' in GT.morph_keys('romae')
    assert 'argentoratum' in GT.morph_keys('argentorati')
    assert GT.skeleton('freyburg') == GT.skeleton('freiburg')
    assert GT.skeleton('stonystratford') == GT.skeleton('stony stratford')
    assert GT.skeleton('hanslop') == GT.skeleton('hanslope')
    assert GT.skeleton('sankt gallen') == GT.skeleton('st gallen')


def _gn(rows):
    df = pd.DataFrame(rows, columns=['key', 'gid', 'prim', 'name', 'lat', 'lon', 'fclass', 'fcode', 'cc', 'adm1', 'pop'])
    df['skel'] = [GT.skeleton(k) for k in df['key']]
    df['exact'] = True
    return GC.GN(df)


def _places(rows):
    P = pd.DataFrame(rows, columns=['coll_slug', 'entity_key', 'name', 'n_mentions', 'lat', 'lon', 'variants'])
    P['n_in_period'] = P['n_mentions']
    P['key_type'] = 'surface'
    P['note_type'] = None
    P['src0'] = np.where(P['lat'].notna(), 'register', 'none')
    P['family'] = P['coll_slug'].map(GC.FAMILY).fillna('other')
    return P


def test_propagation_and_geonames_and_abstention():
    P = _places([
        ('ckcc_epistolarium', 'a', 'Zürich', 50, 47.37, 8.54, ['Zürich']),
        ('sozinianische_briefwechsel', 'b', 'Zürich', 5, 47.38, 8.55, ['Zürich']),
        ('bullinger', 'c', 'Zürich', 100, np.nan, np.nan, ['Zürich']),
        ('bullinger', 'd', 'von Basel', 20, np.nan, np.nan, ['von Basel']),
        ('bullinger', 'e', 'Newport', 3, np.nan, np.nan, ['Newport']),
        ('bullinger', 'f', 'Romae', 3, np.nan, np.nan, ['Romae']),
        ('ckcc_epistolarium', 'g', 'denha004', 30, np.nan, np.nan, ['denha004', 'The Hague']),
    ])
    gn = _gn([
        ('basel', '1', True, 'Basel', 47.56, 7.59, 'P', 'PPLA', 'CH', '06', 170000),
        ('newport', '2', True, 'Newport', 51.58, -3.0, 'P', 'PPL', 'GB', 'WLS', 150000),
        ('newport', '3', True, 'Newport', 50.7, -1.29, 'P', 'PPL', 'GB', 'ENG', 25000),
        ('newport', '4', True, 'Newport', 41.49, -71.3, 'P', 'PPL', 'US', 'RI', 24000),
        ('roma', '5', False, 'Roma', 41.89, 12.48, 'P', 'PPLC', 'IT', '07', 2800000),
        ('the hague', '6', True, 'The Hague', 52.08, 4.3, 'P', 'PPLA', 'NL', '11', 500000),
    ])
    pairs = pd.DataFrame({'doc': [0, 0, 1], 'row': [0, 3, 4]})
    G = GC.Geocoder(P, gn, pairs, pd.DataFrame(columns=['h', 't', 'l']))
    out = GC.run_pipeline(G, np.array([], dtype=int), log=lambda m: None)
    z = out[2]['final']
    assert z['method'] == 'name_propagation' and z['conf'] == 'high' and abs(z['lat'] - 47.37) < 0.1
    b = out[3]['final']
    assert b['method'] == 'name_geonames' and b['conf'] in ('high', 'medium') and abs(b['lon'] - 7.59) < 0.1
    n = out[4]['final']
    assert n is None or n['conf'] == 'low'                       # Newport: ambiguous homonym, abstain
    r = out[5]['final']
    assert r and abs(r['lat'] - 41.89) < 0.1                       # Latin ending
    h = out[6]['final']
    assert h and abs(h['lat'] - 52.08) < 0.1                       # id-like name, variant used
    res = GC.result_rows(G, out)
    assert set(res['gc_source']) <= {'name_propagation', 'name_geonames', 'name_exonym'}


def test_holdout_summary():
    P = _places([('ckcc_epistolarium', str(i), 'Zürich', 5, 47.37, 8.54, ['Zürich']) for i in range(4)])
    gn = _gn([('zurich', '1', True, 'Zürich', 47.37, 8.54, 'P', 'PPLA', 'CH', '06', 340000)])
    G = GC.Geocoder(P, gn, pd.DataFrame({'doc': [0], 'row': [0]}), pd.DataFrame(columns=['h', 't', 'l']))
    hid = np.array([0])
    out = GC.run_pipeline(G, hid, log=lambda m: None)
    df, s = GC.evaluate(G, out, hid)
    assert s['final']['high+medium']['within10'] == 1.0


def test_apply_geocode_never_overwrites_and_only_high_medium_fill(tmp_path):
    from ratlas import reconcile_build as B
    pl = pd.DataFrame({
        'coll_slug': ['a', 'a', 'a', 'a'], 'entity_key': ['1', '2', '3', '4'],
        'coord_source_geo': ['register', 'none', 'none', 'none'],
        'geo_lat': [10.0, np.nan, np.nan, np.nan], 'geo_lon': [20.0, np.nan, np.nan, np.nan]})
    gc = pd.DataFrame({
        'coll_slug': ['a', 'a', 'a'], 'entity_key': ['1', '2', '3'],
        'gc_lat': [1.0, 47.0, 48.0], 'gc_lon': [1.0, 8.0, 9.0], 'gc_source': ['name_geonames', 'name_propagation', 'name_geonames'],
        'gc_conf': ['high', 'high', 'low'], 'gc_radius_km': [5.0, 5.0, 5.0]})
    f = tmp_path / 'gc.parquet'
    gc.to_parquet(f)
    out = B.apply_geocode(pl, f)
    assert out.loc[0, 'coord_source_geo'] == 'register' and out.loc[0, 'geo_lat'] == 10.0     # never overwritten
    assert out.loc[1, 'coord_source_geo'] == 'name_propagation' and out.loc[1, 'geo_lat'] == 47.0
    assert out.loc[1, 'coord_conf'] == 'high'
    assert out.loc[2, 'coord_source_geo'] == 'none' and np.isnan(out.loc[2, 'geo_lat'])      # low stays off the table
    assert out.loc[2, 'gc_conf'] == 'low'
    assert out.loc[3, 'coord_source_geo'] == 'none'
    assert out.loc[0, 'coord_conf'] == 'authority'
