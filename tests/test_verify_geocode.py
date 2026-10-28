import numpy as np
import pandas as pd

from ratlas import verify_geocode as V
from ratlas.reconcile_build import apply_geocode


def _row(coll, key, name, src, conf, lat, lon, ev, n=3, cand='', p=1.0):
    return dict(coll_slug=coll, entity_key=key, name=name, n_in_period=n, coord_source=src, coord_conf=conf, lat=lat, lon=lon,
                gc_evidence=ev, gc_cand_name=cand, gc_p=p)


GN = "GeoNames 1 X ({f}, {cc}, pop {p}); key 'x' (t0, name); p=1.00, 1 candidate places; prior coll=0.70/0.00 doc=0.00"


def test_parse_and_major():
    e = V.parse_evidence(GN.format(f='PPL', cc='FR', p=814) + ' | corroborated by propagation (0 km apart; high)')
    assert (e['fcode'], e['cc'], e['pop'], e['corroborated']) == ('PPL', 'FR', 814, True)
    assert V.parse_evidence('3 placed entities named x')['fcode'] is None
    assert V.is_major('PPLA2', 10, 'DE') and V.is_major('PPL', 60000, 'DE') and not V.is_major('PPL', 800, 'FR')
    assert not V.is_major('ADM1', 5000, 'JE') and not V.is_major('ADM4', 60000, 'FR')


def test_rules():
    g = pd.DataFrame([
        _row('ssrq_swisslaw', 'a', 'Mignieres', 'name_geonames', 'high', 48.36, 1.43, GN.format(f='PPL', cc='FR', p=814), p=0.7),  # taken back (p < 0.8)
        _row('ssrq_swisslaw', 'b', 'Delley', 'name_geonames', 'high', 46.91, 6.97,
             GN.format(f='PPL', cc='CH', p=0) + ' | corroborated by propagation (0 km apart; high)'),                              # corroborated
        _row('bullinger', 'c', 'Hallau', 'name_geonames', 'high', 47.7, 8.46, GN.format(f='PPL', cc='CH', p=2000)),              # certain (p = 1): kept
        _row('bullinger', 'd', 'Aichach', 'name_geonames', 'medium', 48.46, 11.13, GN.format(f='PPLA3', cc='DE', p=21000), p=0.75),  # uncertain, not major
        _row('bullinger', 'e', 'Frankfurt', 'name_geonames', 'medium', 50.1, 8.7, GN.format(f='PPLA3', cc='DE', p=650000), p=0.5),     # major
        _row('casebooks', 'f', 'of lowton', 'name_geonames', 'high', 50.96, -3.14, GN.format(f='PPL', cc='GB', p=0)),            # far
        _row('casebooks', 'g', 'of Moulsoe', 'name_geonames', 'high', 52.07, -0.67, GN.format(f='PPL', cc='GB', p=0)),           # near
        _row('map_of_early_modern_london', 'h', 'St. Mary', 'name_geonames', 'medium', 49.24, -2.17,
             GN.format(f='ADM1', cc='JE', p=1818)),                                                                              # Jersey
        _row('hearth_tax', 'i', 'Wales', 'name_propagation', 'medium', 52.35, -3.63, '2 placed entities named wales'),
        _row('cidtc', 'j', 'Löbau (Lubawa)', 'name_propagation', 'high', 51.09, 14.67, '2 placed entities'),
        _row('selbstzeugnisse_hab_diarium', 'k', 'Bruga', 'name_propagation', 'medium', 47.48, 8.21, 'x'),
        _row('map_of_early_modern_london', 'm', 'The Wall', 'name_geonames', 'high', 52.66, -1.85, GN.format(f='PPL', cc='GB', p=0)),
        _row('map_of_early_modern_london', 'n', 'Hatfield', 'name_propagation', 'medium', 51.76, -0.22, 'x'),
        _row('casebooks', 'o', 'of Ringsteede', 'name_propagation', 'medium', 55.44, 11.79, 'x'),
        _row('casebooks', 'p', 'Bremen', 'name_propagation', 'medium', 53.08, 8.81, 'x'),
        _row('bullinger', 'l', 'Zürich', 'name_propagation', 'high', 47.37, 8.54, '13 placed entities named zurich'),
    ])
    r = V.rules(g).set_index('entity_key')['rule'].to_dict()
    assert r == {'a': 'gn_uncorroborated', 'd': 'gn_uncorroborated', 'f': 'casebooks_far', 'h': 'channel_islands',
                 'i': 'prop_stratum', 'j': 'audit_manual', 'k': 'block_collection', 'm': 'london_microtoponym', 'o': 'english_parish_abroad'}


def test_apply_verify(tmp_path):
    pl = pd.DataFrame({'coll_slug': ['c', 'c', 'c'], 'entity_key': ['a', 'b', 'd'], 'coord_source_geo': ['none'] * 3,
                       'geo_lat': [np.nan] * 3, 'geo_lon': [np.nan] * 3, 'n_in_period': [1, 2, 3]})
    gc = pd.DataFrame({'coll_slug': ['c', 'c'], 'entity_key': ['a', 'b'], 'gc_source': ['name_geonames'] * 2, 'gc_conf': ['high', 'medium'],
                       'gc_lat': [1.0, 2.0], 'gc_lon': [1.0, 2.0], 'gc_radius_km': [5.0, 5.0], 'gc_reason': [None, None]})
    gp = tmp_path / 'gc.parquet'
    gc.to_parquet(gp)
    vp = tmp_path / 'v.tsv'
    pd.DataFrame({'coll_slug': ['c'], 'entity_key': ['a'], 'rule': ['gn_uncorroborated']}).to_csv(vp, sep='\t', index=False)
    out = apply_geocode(pl.copy(), gp, vp).set_index('entity_key')
    assert out.loc['a', 'coord_source_geo'] == 'none' and out.loc['a', 'gc_conf'] == 'low' and out.loc['a', 'verify_rule'] == 'gn_uncorroborated'
    assert out.loc['b', 'coord_source_geo'] == 'name_geonames' and out.loc['b', 'coord_conf'] == 'medium'


def test_manual_rule_takes_back_the_saxon_lobau_not_lubawa():
    """v2.3: the audit rule for 'Löbau (Lubawa)' (cidtc) was written against the point in Saxony; the head / alternative-name path now answers Lubawa (19.7 E), which stays."""
    g = pd.DataFrame([_row('cidtc', 'j', 'Löbau (Lubawa)', 'name_propagation', 'high', 51.09, 14.67, '2 placed entities'),
                      _row('cidtc', 'k', 'Löbau (Lubawa)', 'name_geonames', 'medium', 53.50, 19.75, GN.format(f='PPLA3', cc='PL', p=9302))])
    r = V.rules(g).set_index('entity_key')['rule'].to_dict()
    assert r == {'j': 'audit_manual'}
