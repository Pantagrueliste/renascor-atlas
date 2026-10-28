import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ratlas import merge as M
from ratlas import reconcile_v2 as V
from ratlas.model import AUTH_KEYS, auth_to_arrow
from test_reconcile_gates import _auth, _wd


def _mention(coll, kind, surface, year=1600, key=None, reg=None, auth=None, role='mention', excluded=False, person_class=None,
             cert=None, tag=None, zone='text'):
    return dict(coll_slug=coll, renascor_id=coll, source_path=f'{coll}.xml', doc_id=coll, doc_year=year,
                doc_year_from=year, doc_year_to=year, date_source='header', in_period=1450 <= year <= 1700, kind=kind,
                tag=tag or ('placeName' if kind == 'place' else 'persName'), role=role, zone=zone, part='txt',
                person_class=person_class, excluded=excluded, excluded_reason='x' if excluded else None, cert=cert,
                surface=surface, ref_raw=None, local_key=key, in_register=reg, auth=auth_to_arrow(auth), context=None,
                lang=None, licence='L', tier='T1', canonical=True, overlap_note=None)


def _entity(coll, kind, key, name, lat=None, lon=None, auth=None, note_type=None, person_class=None, birth=None, death=None):
    return dict(local_key=key, coll_slug=coll, renascor_id=coll, kind=kind, name=name, variants=[], lat=lat, lon=lon,
                coord_source='tei:geo' if lat is not None else None, birth=birth, death=death, auth=auth_to_arrow(auth),
                note_type=note_type, source_path=f'{coll}.xml', licence='L', tier='T1', n_mentions=1, n_mentions_canonical=1,
                person_class=person_class)


def _locals(rows, ents, kind):
    tbl = pa.Table.from_pylist(rows, schema=M.MENTION_OUT)
    df = M._flat(tbl)
    et = pa.Table.from_pylist(ents, schema=M.ENTITY_OUT)
    edf = et.select([c for c in et.column_names if c != 'auth']).to_pandas()
    ea = et.column('auth').combine_chunks()
    for k in AUTH_KEYS:
        edf['a_' + k] = ea.field(k).to_pandas()
    parts = [M.local_table(df[df['coll_slug'] == c], edf[(edf['kind'] == kind) & (edf['coll_slug'] == c)], kind)
             for c in sorted(df['coll_slug'].unique())]
    return pd.concat(parts, ignore_index=True)


def _make(tmp_path, extra_rows=(), extra_ents=(), extra_wd=(), extra_gn=(), extra_classes=None, extra_mapping=None):
    v2, rd, out = tmp_path / 'v2', tmp_path / 'rd', tmp_path / 'out'
    v2.mkdir()
    A, Bc = 'colla', 'collb'
    rows = [
        # Rome: register point + explicit QID
        _mention(A, 'place', 'Roma', key='colla:rome', reg=True, auth={'wikidata': 'Q220'}),
        _mention(Bc, 'place', 'Rome', auth={'wikidata': 'Q220'}),
        _mention(Bc, 'place', 'Rome', auth={'wikidata': 'Q220'}, year=1650),
        # Spain: a TGN id that leads to Famagusta
        _mention(A, 'place', 'Spagna', key='colla:spain', reg=True, auth={'tgn': '7002327'}),
        # Italy: a country whose register point is the capital's; the GeoNames id gives the centroid
        _mention(A, 'place', 'Italia', key='colla:italy', reg=True, auth={'geonames': '3175395'}),
        _mention(A, 'place', 'Italia', key='colla:italy', reg=True, auth={'geonames': '3175395'}, year=1500),
        # a repository that is not a place of the text
        _mention(A, 'place', 'Madrid', role='repository', excluded=True, tag='settlement'),
        # Staretschwil: register 600 km from the authority, no same-name feature for either
        _mention(A, 'place', 'Staretschwil', key='colla:star', reg=True, auth={'wikidata': 'Q9999'}),
        # persons
        _mention(A, 'person', 'Martin Luther', key='colla:luther', reg=True, auth={'gnd': '118575449'}),
        _mention(Bc, 'person', 'Johann Casimir', key='collb:jc', reg=True, auth={'viaf': '1'}),
        _mention(A, 'place', 'Galli'), _mention(A, 'place', 'Galli'), _mention(Bc, 'place', 'France'),
        _mention(Bc, 'place', 'Italie'),               # an unregistered name copied from Italia (an area): an area, not a point
        _mention(A, 'person', 'Servant', key='colla:servant', reg=True, excluded=True, person_class='character', role='cast'),
        _mention(Bc, 'person', 'Somebody', cert='low', auth={'gnd': '118575449'}),
    ] + list(extra_rows)
    ents = [
        _entity(A, 'place', 'colla:rome', 'Roma', 41.9, 12.5, auth={'wikidata': 'Q220'}),
        _entity(A, 'place', 'colla:spain', 'Spagna', 40.0, -4.0, auth={'tgn': '7002327'}),
        _entity(A, 'place', 'colla:italy', 'Italia', 41.9, 12.5, auth={'geonames': '3175395'}, note_type='country'),
        _entity(A, 'place', 'colla:star', 'Staretschwil', 48.18, 16.30, auth={'wikidata': 'Q9999'}),
        _entity(A, 'person', 'colla:luther', 'Martin Luther', auth={'gnd': '118575449'}),
        _entity(Bc, 'person', 'collb:jc', 'Johann Casimir von der Pfalz', auth={'viaf': '1'}),
        _entity(A, 'person', 'colla:servant', 'Servant', person_class='character', note_type='character'),
    ] + list(extra_ents)
    tbl = pa.Table.from_pylist(rows, schema=M.MENTION_OUT)
    pq.write_table(tbl, v2 / 'mentions.parquet')
    pq.write_table(pa.Table.from_pylist(ents, schema=M.ENTITY_OUT), v2 / 'entities.parquet')
    pq.write_table(pa.Table.from_pandas(_locals(rows, ents, 'place'), schema=M.LOCAL_SCHEMA, preserve_index=False, safe=False), v2 / 'places_local.parquet')
    pq.write_table(pa.Table.from_pandas(_locals(rows, ents, 'person'), schema=M.LOCAL_SCHEMA, preserve_index=False, safe=False), v2 / 'persons_local.parquet')
    pd.DataFrame({'slug': [A, Bc], 'licence': ['L', 'L'], 'tier': ['T1', 'T1'], 'renascor_ids': [A, Bc]}).to_csv(v2 / 'collections.tsv', sep='\t', index=False)
    wd = [_wd('Q220', {'en': 'Rome', 'it': 'Roma', 'la': 'Roma'}, classes=['Q515'], lat=41.89, lon=12.48),
          _wd('Q193190', {'en': 'Famagusta'}, classes=['Q515'], lat=35.12, lon=33.95),
          _wd('Q9999', {'en': 'Staretschwil', 'de': 'Staretschwil'}, classes=['Q515'], lat=47.45, lon=8.1),
          _wd('Q9554', {'en': 'Martin Luther'}, birth='+1483-11-10T00:00:00Z', occupations=['Q1']),
          _wd('Q719626', {'en': 'George, Duke of Saxony'}, birth='+1471-08-27T00:00:00Z')] + list(extra_wd)
    mapping = {('gnd', '118575449'): ['Q9554'], ('viaf', '1'): ['Q719626'], ('tgn', '7002327'): ['Q193190']}
    mapping.update(extra_mapping or {})
    gn = [dict(gid='3175395', name='Italia', names='italia|italy|italie|italien', lat=42.83, lon=12.83, pop=60000000, cc='IT', fcode='PCLI', fclass='A')] + list(extra_gn)
    auth = _auth(tmp_path, wd, mapping, gn=gn, classes={'Q1': 'theologian', **(extra_classes or {})})
    del auth
    return v2, rd


def test_build_v2_end_to_end(tmp_path):
    v2, rd = _make(tmp_path)
    out = tmp_path / 'out'
    gc = pd.DataFrame({'coll_slug': ['colla', 'collb', 'collb'], 'entity_key': ['surf:galli', 'surf:france', 'surf:italie'],
                       'name': ['Galli', 'France', 'Italie'],
                       'gc_lat': [46.6, 46.6, 42.83], 'gc_lon': [2.5, 2.5, 12.83],
                       'gc_source': ['name_exonym'] * 2 + ['name_propagation'], 'gc_conf': ['medium'] * 2 + ['high'],
                       'gc_radius_km': [350.0, 350.0, 5.0], 'gc_cand': ['exonym:France'] * 2 + ['colla:colla:italy'],
                       'gc_cand_name': ['France'] * 2 + ['Italia'],
                       'gc_evidence': ["exonym table: 'galli' = France (area, radius 350 km)"] * 2 + ['1 placed entity named italia']})
    gcp = tmp_path / 'geocode_places.parquet'
    gc.to_parquet(gcp)
    rep = V.build(v2, rd, out, geocode=gcp)
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index('entity_key')
    # Rome keeps the register point and its QID
    assert geo.loc['colla:rome', 'qid'] == 'Q220' and geo.loc['colla:rome', 'coord_source'] == 'register'
    # Spain: the TGN id leads to Famagusta, the names differ: suggested, no QID, the register point stays
    assert pd.isna(geo.loc['colla:spain', 'qid']) and geo.loc['colla:spain', 'qid_suggested'] == 'Q193190'
    assert geo.loc['colla:spain', 'coord_source'] == 'register' and 'tgn:7002327' in geo.loc['colla:spain', 'ids_suggested']
    # Italy: the capital's register point is dropped; the GeoNames centroid, flagged as an area
    it = geo.loc['colla:italy']
    assert it['reg_area_dropped'] and it['coord_source'] == 'geonames' and it['coord_kind'] == 'area' and it['coord_radius_km'] == 350
    assert abs(it['lat'] - 42.83) < 0.01 and it['place_kind'] == 'area'
    # a name copied from an area is an area with the same radius (v2 audit, check 5: 'Polen, Königreich' was a point on Krakow)
    ie = geo.loc['surf:italie']
    assert ie['coord_source'] == 'name_propagation' and ie['coord_kind'] == 'area' and ie['coord_radius_km'] == 350
    # the excluded repository counts for nothing
    mad = geo[geo['name'] == 'Madrid'].iloc[0]
    assert mad['n_mentions'] == 0 and mad['scope'] == 'excluded'
    # Staretschwil: register in Lower Austria, authority in Aargau (> 100 km): flagged, register kept without a namecheck
    st = geo.loc['colla:star']
    assert st['coord_conflict'] and st['flag_100km'] and st['arbitration'] == 'none_no_namecheck'
    # gate log
    lg = pd.read_csv(out / 'gate_log.tsv', sep='\t')
    r = lg[(lg.entity_key == 'colla:spain')].iloc[0]
    assert r['status'] == 'suggested' and r['reason'] == 'name_mismatch'
    assert ((lg.entity_key == 'collb:jc') & (lg.reason == 'name_mismatch')).any()
    assert rep['gates']['by_status_reason']['place:suggested:name_mismatch'] >= 1
    # persons: Luther accepted, Johann Casimir suggested, the character is not in the global table
    pl = pd.read_parquet(out / 'persons_local_global.parquet').set_index('entity_key')
    assert pl.loc['colla:luther', 'qid'] == 'Q9554'
    assert pd.isna(pl.loc['collb:jc', 'qid']) and pl.loc['collb:jc', 'qid_suggested'] == 'Q719626'
    assert pl.loc['colla:servant', 'scope'] == 'excluded'
    pg = pd.read_parquet(out / 'persons_global.parquet')
    assert 'pe:Q9554' in set(pg['id']) and not any('servant' in str(i) for i in pg['id'])
    assert pg.set_index('id').loc['pe:Q9554', 'occupations'][0] == 'theologian'
    # globe: kinds and per-collection quarters
    g = json.loads((out / 'globe_places.json').read_text())
    pts = {p['id']: p for p in g['places']}
    rome = pts['pl:Q220']
    assert rome['k'] == 'c' and rome['n'] == 3 and sum(rome['q']) == 3
    assert set(rome['cq']) == {'colla', 'collb'} and sum(rome['cq']['collb']) == 2
    ita = [p for p in g['places'] if p['name'] == 'Italia'][0]
    assert ita['k'] == 'a' and ita['r'] == 350
    assert all(p['name'] != 'Madrid' for p in g['places'])
    # the exonym table: Galli and France are one area point called France, source e
    fr = [p for p in g['places'] if p['name'] == 'France']
    assert len(fr) == 1 and fr[0]['m'] == 2 and fr[0]['n'] == 3 and fr[0]['k'] == 'a' and fr[0]['s'] == 'e' and fr[0]['r'] == 350


def test_arbitration_by_names(tmp_path):
    import pandas as pd
    pl = pd.DataFrame({
        'coll_slug': ['a'] * 5, 'entity_key': ['rhine', 'wrong', 'star', 'tie', 'ok'],
        'coord_conflict': [True, True, True, True, False], 'coord_conflict_km': [300.0, 300.0, 600.0, 60.0, 5.0],
        'auth_name_level': ['exact', 'substring', 'exact', 'exact', 'exact'],
        'has_reg': [True] * 5, 'has_wd': [True] * 5, 'has_gn': [False] * 5,
        'qid': ['Q584', 'Q1', 'Q9', 'Q2', 'Q3'], 'qid_via': ['wikidata'] * 5, 'id_geonames': [None] * 5})
    nc = pd.DataFrame({'coll_slug': ['a'] * 4, 'entity_key': ['rhine', 'wrong', 'star', 'tie'],
                       'd_reg': [2.0, 3.0, -1.0, 10.0], 'd_auth': [250.0, 400.0, -1.0, 10.0]}).set_index(['coll_slug', 'entity_key'])
    out = V.arbitrate_v2(pl.copy(), nc)
    o = out.set_index('entity_key')
    # the river: register nearer a same-name feature, but the names agree: coordinate unused, identity kept
    assert o.loc['rhine', 'arbitration'] == 'register' and o.loc['rhine', 'auth_rejected'] and not o.loc['rhine', 'identity_dropped']
    assert o.loc['rhine', 'qid'] == 'Q584'
    # a weak name match loses its identity too
    assert o.loc['wrong', 'identity_dropped'] and pd.isna(o.loc['wrong', 'qid'])
    # both unknown, > 100 km, strong names: the authority wins
    assert o.loc['star', 'arbitration'] == 'authority_tie_100km' and o.loc['star', 'reg_rejected'] and not o.loc['star', 'has_reg']
    # a plain tie keeps the register
    assert o.loc['tie', 'arbitration'] == 'tie_register' and o.loc['tie', 'has_reg']
    assert o.loc['ok', 'arbitration'] is None


def test_kinds():
    assert V.kind_of_classes(['city', 'big city']) == 'city'
    assert V.kind_of_classes(['sovereign state']) == 'area'
    assert V.kind_of_classes(['historical region']) == 'area'
    assert V.kind_of_classes(['river']) == 'other'
    assert V.kind_of_classes(['human settlement', 'municipality of Germany']) == 'city'
    assert V.kind_of_classes([]) is None
    # v2 audit, check 5: a federal state is an area even when Wikidata also calls it a 'ward'; a city-state stays a city
    assert V.kind_of_classes(['federated state of Germany', 'federated state', 'free state', 'ward']) == 'area'
    assert V.kind_of_classes(['voivodeship of Poland']) == 'area'
    assert V.kind_of_classes(['chartered community']) == 'area'
    assert V.kind_of_classes(['federated state of Germany', 'big city', 'city']) == 'city'
    assert V.kind_of_classes(['municipality of Switzerland', 'cantonal capital of Switzerland', 'city of Switzerland']) == 'city'
    assert V.kind_of_fcode('PCLI') == 'area' and V.kind_of_fcode('PPLA') == 'city' and V.kind_of_fcode('STM') == 'other'
    assert V.kind_of_regtype('moun') == 'other' and V.kind_of_regtype('country') == 'area' and V.kind_of_regtype('stadt') == 'city'


def test_default_counts_carry_the_end_of_the_date_range(tmp_path):
    """The chronology gate reads ``last_year``: the latest year an entity can be cited in, from default mentions only (excluded commentary of
    1950 does not count; a wide range has no doc_year but an end)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [_mention('a', 'person', 'X', year=1600, key='a:x'),
            _mention('a', 'person', 'X', year=1650, key='a:x'),
            _mention('a', 'person', 'X', year=1950, key='a:x', excluded=True)]
    wide = _mention('a', 'person', 'Y', year=1600, key='a:y')
    wide.update(doc_year=None, doc_year_from=1501, doc_year_to=1700, in_period=None)
    rows.append(wide)
    d = tmp_path / 'v2'
    d.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=M.MENTION_OUT), d / 'mentions.parquet')
    dc = V.default_counts(d, ('person',)).set_index('entity_key')
    assert dc.loc['a:x', 'last_year'] == 1650 and dc.loc['a:y', 'last_year'] == 1700


def test_free_imperial_cities_and_city_states_are_points():
    """v2 audit, per-edition sample: Nuremberg (a free imperial city) and Hamburg (a city-state) were drawn as 150 km areas."""
    assert V.kind_of_classes(['free imperial city', 'big city', 'city']) == 'city'
    assert V.kind_of_classes(['city-state', 'federated state of Germany', 'big city']) == 'city'
    assert V.kind_of_classes(['free imperial city']) == 'city'
    assert V.kind_of_classes(['historical country', 'free imperial city']) == 'city'     # a city first (Metz, Strasbourg), whatever else it was


def test_tgn_gives_a_point_and_the_casebooks_directory_places_its_parishes(tmp_path):
    """v2.1: an accepted TGN id is a coordinate source of its own (after register, Wikidata and GeoNames); the Casebooks directory places the PLACE ids
    it matched with confidence high / medium (low stays a suggestion) and replaces the name-based result, never a register coordinate."""
    from test_reconcile_gates import _tgn_dir
    A, CB = 'colla', 'casebooks'
    rows = [_mention(A, 'place', 'Ratisbona', key='colla:rat', reg=True, auth={'tgn': '7000874'}),
            _mention(A, 'place', 'Rimini', key='colla:rim', reg=True, auth={'tgn': '7004444'}),
            _mention(A, 'place', 'Ancona', key='colla:anc', reg=True, auth={'tgn': '7003980'}),
            _mention(CB, 'place', 'of Hanslop', key='casebooks:PLACE1', reg=False), _mention(CB, 'place', 'of Hanslop', key='casebooks:PLACE1', reg=False),
            _mention(CB, 'place', 'of Wicken', key='casebooks:PLACE2', reg=False), _mention(CB, 'place', 'of Preston', key='casebooks:PLACE3', reg=False),
            _mention(CB, 'place', 'of Nowhere', key='casebooks:PLACE4', reg=False), _mention(CB, 'place', 'in East Smithfield', key='casebooks:PLACE5', reg=False)]
    ents = [_entity(A, 'place', 'colla:rat', 'Ratisbona', auth={'tgn': '7000874'}), _entity(A, 'place', 'colla:rim', 'Rimini', auth={'tgn': '7004444'}),
            _entity(A, 'place', 'colla:anc', 'Ancona', auth={'tgn': '7003980'}, note_type='country')]
    v2, rd = _make(tmp_path, rows, ents)
    # the casebooks places have no register entries (opaque keys): the local table is built from the mentions
    cbm = tmp_path / 'casebooks_place_map.tsv'
    hdr = ['place_id', 'surfaces', 'directory_entry', 'entry_type', 'county', 'parish', 'country', 'lat', 'lon', 'confidence', 'method', 'n_mentions',
           'score', 'spatial_p', 'geo_method', 'geo_confidence', 'geo_radius_km', 'geonames_id', 'geonames_name', 'directory_path', 'alt_candidates']

    def row(pid, entry, lat, lon, conf, geo_conf='high', rad=2.0, method='gn_exact_single'):
        d = dict.fromkeys(hdr, '')
        d.update(place_id=pid, directory_entry=entry, county='Buckinghamshire', lat=str(lat), lon=str(lon), confidence=conf, geo_confidence=geo_conf,
                 geo_radius_km=str(rad), geo_method=method, n_mentions='1')
        return d
    pd.DataFrame([row('PLACE1', 'Hanslope', 52.1, -0.8, 'high'), row('PLACE2', 'Wicken', 52.05, -0.9, 'medium'),
                  row('PLACE3', 'Preston Bissett', 51.97, -1.0, 'low'), row('PLACE4', '', '', '', 'none'),
                  row('PLACE5', 'East Smithfield', 51.51, -0.07, 'high', geo_conf='low', rad=4.0, method='london_default')], columns=hdr).to_csv(cbm, sep='\t', index=False)
    gc = pd.DataFrame({'coll_slug': [CB], 'entity_key': ['casebooks:PLACE1'], 'name': ['of Hanslop'], 'gc_lat': [53.0], 'gc_lon': [-1.0],
                       'gc_source': ['name_geonames'], 'gc_conf': ['medium'], 'gc_radius_km': [5.0], 'gc_cand': ['gn:1'], 'gc_cand_name': ['Hanslope'],
                       'gc_evidence': ['GeoNames 1 Hanslope (PPL, GB, pop 10); key x']})
    gcp = tmp_path / 'geocode_places.parquet'
    gc.to_parquet(gcp)
    out = tmp_path / 'out'
    rep = V.build(v2, rd, out, geocode=gcp, tgn_dir=_tgn_dir(tmp_path), casebooks=cbm)
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index('entity_key')
    assert geo.loc['colla:rat', 'coord_source'] == 'tgn' and abs(geo.loc['colla:rat', 'lat'] - 49.0) < 1e-6 and geo.loc['colla:rat', 'tgn_id'] == '7000874'
    assert geo.loc['colla:rat', 'place_kind'] == 'city'
    assert geo.loc['colla:anc', 'coord_source'] == 'none'          # the register says country, the TGN subject is a town: no point (audit check 5)
    assert geo.loc['colla:rim', 'coord_source'] == 'none' and 'tgn_name_mismatch' in geo.loc['colla:rim', 'ids_suggested']   # Rimini carries Ravenna's id
    h = geo.loc['casebooks:PLACE1']
    assert h['coord_source'] == 'casebooks_directory' and abs(h['lat'] - 52.1) < 1e-6 and h['coord_conf'] == 'high' and h['coord_kind'] == 'point'
    assert h['cb_entry'] == 'Hanslope' and abs(h['cb_gc_km'] - 100) < 60           # the name-based result it replaces is kept in gc_*
    assert geo.loc['casebooks:PLACE2', 'coord_source'] == 'casebooks_directory' and geo.loc['casebooks:PLACE2', 'coord_conf'] == 'medium'
    low = geo.loc['casebooks:PLACE3']
    assert low['coord_source'] == 'none' and low['cb_match'] == 'low' and not low['cb_used'] and abs(low['cb_lat'] - 51.97) < 1e-6   # a suggestion only
    assert geo.loc['casebooks:PLACE4', 'coord_source'] == 'none'
    # v2.3: the directory's own geocoding is low (a guess from a street name): the match is right but the point is not a point; it stays a suggestion
    assert geo.loc['casebooks:PLACE5', 'coord_source'] == 'none' and not geo.loc['casebooks:PLACE5', 'cb_used'] and abs(geo.loc['casebooks:PLACE5', 'cb_lat'] - 51.51) < 1e-6
    assert rep['casebooks_directory']['placed_distinct'] == 2 and rep['casebooks_directory']['suggested_only_low'] == 1 and rep['casebooks_directory']['geo_low_point_not_used'] == 1
    assert rep['places_local']['source_casebooks_directory_distinct_in_period'] == 2 and rep['places_local']['source_tgn_distinct_in_period'] == 1
    g = json.loads((out / 'globe_places.json').read_text())
    assert {p['s'] for p in g['places']} >= {'c', 't'}


def test_towns_of_a_republic_or_a_kingdom_are_not_areas_and_generic_classes_say_nothing():
    """v2.2: 275 towns (7,602 mentions) were areas in v2.1 because the label 'municipality with town privileges in the Czech Republic' contains
    'republic' and 'city of United Kingdom' contains 'kingdom'; Ballenstedt (class 'geographic location') was an area."""
    assert V.class_head('municipality with town privileges in the Czech Republic') == 'municipality with town privileges'
    assert V.class_head('city of United Kingdom') == 'city' and V.class_head('big city') == 'big city'
    prague = ['municipality with town privileges in the Czech Republic', 'district of the Czech Republic', 'capital of region', 'region of the Czech Republic',
              'big city', 'municipality of the Czech Republic', 'national capital']
    assert V.kind_of_classes(prague) == 'city'
    assert V.kind_of_classes(['college town', 'city', 'county town', 'big city', 'city of United Kingdom', 'civil parish']) == 'city'
    assert V.kind_of_classes(['unparished area', 'market town', 'city of United Kingdom']) == 'city'
    assert V.kind_of_classes(['municipality of the Czech Republic', 'district town']) == 'city'
    assert V.kind_of_classes(['geographic location']) is None and V.kind_of_classes(['geographic location', 'human settlement']) == 'city'
    # real polities stay areas, also with a country in the label
    assert V.kind_of_classes(['kingdom of Italy']) == 'area' and V.kind_of_classes(['historical country']) == 'area'
    assert V.kind_of_classes(['region of the Czech Republic']) == 'area' and V.kind_of_classes(['province of Italy']) == 'area'
    assert V.polity_classes(['duchy', 'historical region']) and not V.polity_classes(['province of Italy', 'city of United Kingdom'])


def test_wikidata_other_needs_a_natural_or_built_class_and_new_polity_words():
    """v2.2: 'other' is a river, a castle, a bridge ...; 'municipal part of the Czech Republic' says nothing (None: GeoNames decides); satrapies and viceroyalties are areas."""
    assert V.kind_of_classes(['municipal part of the Czech Republic', 'cadastral area in the Czech Republic']) is None
    assert V.kind_of_classes(['river']) == 'other' and V.kind_of_classes(['castle', 'cultural heritage monument']) == 'other'
    assert V.kind_of_classes(['satrapy of the Sasanian Empire']) == 'area' and V.kind_of_classes(['viceroyalty of the Spanish Empire']) == 'area'
    assert V.kind_of_classes(['regional unit of Greece']) == 'area'


def test_plural_class_labels_are_read_in_the_singular():
    """v2.2: Wolhynien's class is 'Ukrainian historical regions' (an area on the audit's check 5)."""
    assert V._singular('Ukrainian historical regions | Duchies of Silesia | counties') == 'Ukrainian historical region | Duchy of Silesia | county'
    assert V.kind_of_classes(['Ukrainian historical regions']) == 'area' and V.kind_of_classes(['Duchies of Silesia']) == 'area'
    assert V.kind_of_classes(['municipalities']) == 'city'
