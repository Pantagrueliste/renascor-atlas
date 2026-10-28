import zipfile

import pandas as pd

from ratlas import reconcile_gnscan as S


def _line(gid, name, ascii_, alts, lat, lon, fc, fcode, cc, pop):
    return '\t'.join([gid, name, ascii_, alts, str(lat), str(lon), fc, fcode, cc, '', '01', '', '', '', str(pop), '', '', 'Europe/Berlin', '2020-01-01'])


def test_gnscan_names_of_ids_and_same_name_features(tmp_path):
    z = tmp_path / 'allCountries.zip'
    rows = [_line('2893837', 'Kalkar', 'Kalkar', 'Calcar,Kalkar', 51.74, 6.29, 'P', 'PPL', 'DE', 13000),
            _line('2893838', 'Kalkar', 'Kalkar', '', 50.60, 6.77, 'P', 'PPL', 'DE', 0),
            _line('1', 'Rhein', 'Rhein', 'Rhine,Rhenus', 51.0, 6.0, 'H', 'STM', 'DE', 0),
            _line('2', 'Nowhere', 'Nowhere', '', 10.0, 10.0, 'S', 'HSE', 'DE', 0)]
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr('allCountries.txt', '\n'.join(rows) + '\n')
    ids = tmp_path / 'ids.tsv'
    pd.DataFrame({'kind': ['place', 'person'], 'scheme': ['geonames', 'geonames'], 'id': ['2893838', '1'], 'n_mentions': [1, 1],
                  'n_in_period': [1, 1], 'n_register': [0, 0]}).to_csv(ids, sep='\t', index=False)
    loc = pd.DataFrame({'name': ['Kalkar', 'Rhenus'], 'variants': [['Calcar'], []], 'other_auth': [[], []],
                        'gnd': [None, None], 'viaf': [None, None], 'wikidata': [None, None], 'geonames': ['2893838', None],
                        'tgn': [None, None], 'pleiades': [None, None], 'hls': [None, 'x']})
    lp = tmp_path / 'places_local.parquet'
    loc.to_parquet(lp)
    out = tmp_path / 'out'
    S.run(ids, lp, z, out)
    g = pd.read_parquet(out / 'gn_ids.parquet')
    assert list(g['gid']) == ['2893838'] and g.iloc[0]['pop'] == 0 and g.iloc[0]['fcode'] == 'PPL'     # persons' place ids are not wanted
    s = pd.read_parquet(out / 'gn_samename.parquet')
    kal = s[s['key'] == 'kalkar']
    assert set(kal['gid']) == {'2893837', '2893838'} and kal.set_index('gid').loc['2893837', 'pop'] == 13000
    assert 'calcar' in set(s['key']) and 'rhenus' in set(s['key'])                 # alternate names of the classes P A T H L
    assert '2' not in set(s['gid'])                                                  # class S is not indexed
    # every row carries all the names of its feature (the gate asks whether the entry name and the item's label are names of ONE feature)
    names = s.set_index('gid').loc['2893837', 'names']
    nm = set((names.iloc[0] if hasattr(names, 'iloc') else names).split('|'))
    assert {'kalkar', 'calcar'} <= nm
