import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ratlas import merge as M
from ratlas import reconcile_build as B
from ratlas.model import AUTH_KEYS, auth_to_arrow


def test_haversine_and_valid():
    assert abs(B.haversine_km(48.85, 2.35, 51.5, -0.12)[()] - 344) < 5
    ok = B.valid_coord(pd.Series([0, 91, 10]), pd.Series([0, 0, 10]))
    assert list(ok) == [False, False, True]


def test_assign_clusters_attaches_unmapped_ids_to_a_unique_qid_cluster():
    loc = pd.DataFrame({
        'qid': ['Q1', None, None, None, 'Q2', 'Q3'],
        'id_gnd': [None, None, None, None, None, None],
        'id_viaf': ['V1', 'V1', 'V9', None, 'V7', 'V7'],
    })
    cl = B.assign_clusters(loc, ('gnd', 'viaf'))
    assert list(cl['gid'][:2]) == ['Q1', 'Q1']             # viaf V1 only carried by Q1 -> attached
    assert cl['gid'][2] == 'viaf:V9' and cl['basis'][2] == 'viaf'
    assert cl['gid'][3] is None
    # V7 is carried by two QID clusters: an unmapped entity with V7 would stay on its own id
    loc2 = pd.concat([loc, pd.DataFrame({'qid': [None], 'id_gnd': [None], 'id_viaf': ['V7']})], ignore_index=True)
    assert B.assign_clusters(loc2, ('gnd', 'viaf'))['gid'].iloc[-1] == 'viaf:V7'


def _mention(coll, kind, surface, year=1600, key=None, reg=None, auth=None, role='mention', tier='T1'):
    return dict(coll_slug=coll, renascor_id=coll, source_path=f'{coll}.xml', doc_id=coll, doc_year=year,
                doc_year_from=year, doc_year_to=year, date_source='header', in_period=1450 <= year <= 1700, kind=kind,
                tag='placeName', role=role, zone='txt', surface=surface, ref_raw=None, local_key=key, in_register=reg,
                auth=auth_to_arrow(auth), context=None, lang=None, licence='L', tier=tier, canonical=True, overlap_note=None)


def _entity(coll, kind, key, name, lat=None, lon=None, auth=None):
    return dict(local_key=key, coll_slug=coll, renascor_id=coll, kind=kind, name=name, variants=[], lat=lat, lon=lon,
                coord_source='tei:geo' if lat is not None else None, birth=None, death=None, auth=auth_to_arrow(auth),
                note_type=None, source_path=f'{coll}.xml', licence='L', tier='T1', n_mentions=1, n_mentions_canonical=1)


def _flat_local(rows, ents, kind):
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


def test_build_end_to_end(tmp_path):
    v1, rd, out = tmp_path / 'v1', tmp_path / 'rd', tmp_path / 'out'
    v1.mkdir()
    rd.mkdir()
    A, Bc = 'colla', 'collb'
    rows = [
        # Rome: register coordinates in A (close to Wikidata), plain name + explicit QID in B
        _mention(A, 'place', 'Roma', key='colla:rome', reg=True, auth={'wikidata': 'Q220'}),
        _mention(Bc, 'place', 'Rome', auth={'wikidata': 'Q220'}),
        _mention(Bc, 'place', 'Rome', auth={'wikidata': 'Q220'}, year=1650),
        # Paris through a GND id only, no register coordinates
        _mention(Bc, 'place', 'Paris', auth={'gnd': '4044660-8'}),
        # a register place 600 km from Wikidata (conflict) and one with (0,0)
        _mention(A, 'place', 'Wien', key='colla:wien', reg=True, auth={'geonames': '2761369'}),
        _mention(A, 'place', 'Nowhere', key='colla:nowhere', reg=True),
        # GeoNames only
        _mention(Bc, 'place', 'Graz', auth={'geonames': '2778067'}),
        # no id at all
        _mention(Bc, 'place', 'Dorf'),
        # a place whose GND maps to a human: dropped
        _mention(Bc, 'place', 'Hans', auth={'gnd': '11111-1'}),
        # persons
        _mention(A, 'person', 'Luther', key='colla:luther', reg=True, auth={'wikidata': 'Q9554', 'viaf': '1234'}),
        _mention(Bc, 'person', 'M. Luther', auth={'viaf': '1234'}),
        _mention(Bc, 'person', 'Someone', auth={'gnd': '22222-2'}),
        _mention(A, 'person', 'Someone', auth={'gnd': '22222-2'}, tier='T3'),
        _mention(A, 'place', 'Old', year=1300, key='colla:old', reg=False),
    ]
    ents = [
        _entity(A, 'place', 'colla:rome', 'Roma', 41.9, 12.5, {'wikidata': 'Q220'}),
        _entity(A, 'place', 'colla:wien', 'Wien', 52.0, 13.4, {'geonames': '2761369'}),          # Wien in Berlin
        _entity(A, 'place', 'colla:nowhere', 'Nowhere', 0.0, 0.0),
        _entity(A, 'person', 'colla:luther', 'Martin Luther', None, None, {'wikidata': 'Q9554', 'viaf': '1234'}),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=M.MENTION_OUT), v1 / 'mentions.parquet')
    pq.write_table(pa.Table.from_pylist(ents, schema=M.ENTITY_OUT), v1 / 'entities.parquet')
    for kind, name in (('place', 'places'), ('person', 'persons')):
        lt = _flat_local(rows, ents, kind)
        pq.write_table(M._to_table(lt, M.LOCAL_SCHEMA), v1 / f'{name}_local.parquet')
    pd.DataFrame([dict(slug=A, licence='CC0-1.0', tier='T1', renascor_ids='a'),
                  dict(slug=Bc, licence='CC-BY-4.0', tier='T2', renascor_ids='b')]).to_csv(v1 / 'collections.tsv', sep='\t', index=False)
    pd.DataFrame([('gnd', '4044660-8', 'Q90'), ('geonames', '2778067', 'Q13298'), ('gnd', '11111-1', 'Q42'),
                  ('gnd', '22222-2', 'Q77')], columns=['scheme', 'id', 'qids']).to_csv(rd / 'wd_map.tsv', sep='\t', index=False)
    pd.DataFrame([
        dict(qid='Q220', lat=41.89, lon=12.48, n_coords=1, label_en='Rome', label_fr='Rome', label_de='Rom', label_it='Roma', classes='Q515:city', country_qid='Q38', country='Italy', country_iso='IT'),
        dict(qid='Q90', lat=48.85, lon=2.35, n_coords=1, label_en='Paris', label_fr='Paris', label_de='Paris', label_it='Parigi', classes='Q515:city', country_qid='Q142', country='France', country_iso='FR'),
        dict(qid='Q13298', lat=47.07, lon=15.44, n_coords=1, label_en='Graz', label_fr='', label_de='Graz', label_it='', classes='', country_qid='', country='', country_iso=''),
        dict(qid='Q42', lat=0, lon=0, n_coords=0, label_en='Douglas Adams', label_fr='', label_de='', label_it='', classes='Q5:human', country_qid='', country='', country_iso=''),
    ]).assign(lat=lambda d: d['lat'].where(d['n_coords'] > 0, '')).to_csv(rd / 'wd_places.tsv', sep='\t', index=False)
    pd.DataFrame([
        dict(qid='Q9554', label_en='Martin Luther', label_fr='', label_de='Martin Luther', label_it='', birth='1483-11-10T00:00:00Z', birth_prec='11', death='1546-02-18T00:00:00Z', death_prec='11', occupations='Q1234713:theologian|Q36180:writer', n_occupations=2, classes='Q5:human'),
        dict(qid='Q77', label_en='Five', label_fr='', label_de='', label_it='', birth='', birth_prec='', death='', death_prec='', occupations='', n_occupations=0, classes='Q5:human'),
    ]).to_csv(rd / 'wd_persons.tsv', sep='\t', index=False)
    gn = tmp_path / 'gn.tsv'
    pd.DataFrame([('2761369', 'Wien', 48.2082, 16.3738, 'P', 'PPLC', 'AT', 1900000)],
                 columns=['geonameid', 'name', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country_code', 'population']).to_csv(gn, sep='\t', index=False)

    rep = B.build(v1, rd, out, gn)
    loc = pq.read_table(out / 'places_local_geo.parquet').to_pandas().set_index('entity_key')
    assert loc.loc['colla:rome', 'coord_source'] == 'register' and not loc.loc['colla:rome', 'coord_conflict']
    wien = loc.loc['colla:wien']
    assert wien['coord_source'] == 'register' and wien['coord_conflict'] and wien['coord_conflict_km'] > 400
    assert wien['auth_source'] == 'geonames'
    assert loc.loc['colla:nowhere', 'coord_source'] == 'none' and loc.loc['colla:nowhere', 'reg_invalid']
    rome_b = loc.loc['surf:rome']
    assert rome_b['coord_source'] == 'wikidata' and rome_b['qid'] == 'Q220' and rome_b['global_id'] == 'pl:Q220'
    paris = loc.loc['surf:paris']
    assert paris['qid'] == 'Q90' and paris['coord_source'] == 'wikidata' and paris['qid_via'] == 'gnd'
    graz = loc.loc['surf:graz']
    assert graz['qid'] == 'Q13298' and graz['coord_source'] == 'wikidata'
    assert pd.isna(loc.loc['surf:hans', 'qid'])                 # human QID dropped for a place
    dorf = loc.loc['surf:dorf']
    assert dorf['scope'] == 'edition' and dorf['coord_source'] == 'none' and dorf['global_id'].startswith('pl:e:collb:')
    pg = pq.read_table(out / 'places_global.parquet').to_pandas().set_index('id')
    rome = pg.loc['pl:Q220']
    assert rome['n_collections'] == 2 and rome['n_mentions'] == 3 and rome['coord_source'] == 'wikidata'
    assert rome['n_in_period'] == 3 and rome['q1600'] == 2 and rome['q1650'] == 1 and rome['q1575'] == 0
    assert rome['label'] == 'Rome' and rome['label_it'] == 'Roma' and set(rome['top_collections']) == {A, Bc}
    assert pg.loc['pl:geonames:2761369', 'coord_source'] == 'geonames' if 'pl:geonames:2761369' in pg.index else True
    g = json.load(open(out / 'globe_places.json'))
    names = {p['name']: p for p in g['places']}
    assert 'Rome' in names or 'Roma' in names
    assert 'Dorf' not in names and 'Old' not in names           # no coordinates / out of period
    r = next(p for p in g['places'] if p['qid'] == 'Q220')
    assert r['n'] == 3 and len(r['q']) == 10 and r['q'][6] == 2 and r['q'][8] == 1 and r['lat'] == 41.89
    assert {c['slug'] for c in g['header']['collections']} == {A, Bc}
    pp = pq.read_table(out / 'persons_global.parquet').to_pandas().set_index('id')
    lu = pp.loc['pe:Q9554']
    assert lu['n_locals'] == 2 and lu['n_collections'] == 2          # B's viaf-only Luther attached through viaf 1234
    assert lu['birth'].startswith('1483') and int(lu['birth_precision']) == 11 and list(lu['occupations'])[0] == 'theologian'
    assert pp.loc['pe:Q77', 'n_collections'] == 2 and pp.loc['pe:Q77', 'tiers'] == 'T1,T3'
    assert rep['places_local']['coord_conflicts_50km'] == 1
