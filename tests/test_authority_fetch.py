import json

import pandas as pd

from ratlas import authority_fetch as A


def _ent(qid, labels, aliases=None, p31=('Q5',), claims=None, redirects=None):
    cl = {'P31': [{'rank': 'normal', 'mainsnak': {'snaktype': 'value', 'datavalue': {'value': {'id': c}}}} for c in p31]}
    for prop, vals in (claims or {}).items():
        cl[prop] = [{'rank': 'normal', 'mainsnak': {'snaktype': 'value', 'datavalue': {'value': v}}} for v in vals]
    return {'id': qid, 'labels': {l: {'language': l, 'value': v} for l, v in labels.items()},
            'aliases': {l: [{'language': l, 'value': a} for a in v] for l, v in (aliases or {}).items()}, 'claims': cl}


def test_reduce_gate_keeps_labels_aliases_classes_identifiers_and_duplicates():
    resp = {'entities': {
        'Q9554': _ent('Q9554', {'en': 'Martin Luther', 'la': 'Martinus Lutherus'}, {'de': ['Luther']},
                      claims={'P227': ['118575449'], 'P569': [{'time': '+1483-11-10T00:00:00Z', 'precision': 11}], 'P106': [{'id': 'Q1'}]}),
        'Q32': _ent('Q32', {'en': 'Duplicate'}, p31=('Q21286738',), claims={'P2959': [{'id': 'Q2090'}]}),
        'Q2': _ent('Q777', {'en': 'Redirected'}),
        'Q404': {'id': 'Q404', 'missing': ''}}}
    # P106 / P2959 are item values, P227 a string
    resp['entities']['Q9554']['claims']['P106'] = [{'rank': 'normal', 'mainsnak': {'snaktype': 'value', 'datavalue': {'value': {'id': 'Q1'}}}}]
    resp['entities']['Q32']['claims']['P2959'] = [{'rank': 'normal', 'mainsnak': {'snaktype': 'value', 'datavalue': {'value': {'id': 'Q2090'}}}}]
    r = A.reduce_gate(resp)
    assert r['Q9554']['labels']['la'] == 'Martinus Lutherus' and r['Q9554']['aliases']['de'] == ['Luther']
    assert r['Q9554']['p31'] == ['Q5'] and r['Q9554']['idents'] == {'P227': ['118575449']} and r['Q9554']['p106'] == ['Q1']
    assert r['Q9554']['birth'] == ['1483-11-10T00:00:00Z', 11]
    assert r['Q32']['dup'] == ['Q2090']
    assert r['Q2']['id'] == 'Q777'                                   # a redirect is visible: requested Q2, real item Q777
    assert r['Q404'] == {'missing': True}
    assert 'en' in A.LANGS10 and 'la' in A.LANGS10 and len(A.LANGS10) == 10


def test_search_queries_batch_and_syntax():
    qs = A.search_queries({'gnd': {str(100000000 + i) for i in range(60)}, 'hls': {'61'}})
    assert [s for s, _, _ in qs].count('gnd') == 3                  # 25 ids per call
    assert qs[0][2].startswith('haswbstatement:P227=100000000|P227=')
    hls = [q for s, _, q in qs if s == 'hls'][0]
    assert 'P902=61' in hls and 'P902=000061' in hls               # both forms of an HLS id


def test_idents_to_map_normalises_and_keeps_only_wanted_ids():
    ents = {'Q9554': {'id': 'Q9554', 'idents': {'P227': ['118575449'], 'P214': ['999']}},
            'Q5': {'id': 'Q5', 'idents': {'P902': ['000061']}}, 'Q404': {'missing': True}}
    m = A.idents_to_map(ents, {'gnd': {'118575449'}, 'hls': {'61'}})
    assert m == {('gnd', '118575449'): {'Q9554'}, ('hls', '61'): {'Q5'}}


class FakeNet:
    """No network: canned answers, and a record of what was asked."""
    n_net = n_cached = 0

    def __init__(self, *a, **k):
        self.calls = []

    def api(self, params, reduce=None):
        self.calls.append(dict(params))
        if params.get('list') == 'search':
            return reduce({'query': {'search': [{'title': 'Q9554'}], 'searchinfo': {'totalhits': 1}}})
        if params.get('action') == 'wbgetentities' and params.get('props') == 'labels':
            return {q: 'human' if q == 'Q5' else 'theologian' for q in params['ids'].split('|')}
        ids = params['ids'].split('|')
        db = {'Q9554': _ent('Q9554', {'en': 'Martin Luther'}, claims={'P227': ['118575449']}),
              'Q777': _ent('Q777', {'en': 'Other'})}
        return reduce({'entities': {q: db[q] for q in ids if q in db}})

    def get_json(self, url, kind='lobid'):
        self.calls.append({'lobid': url})
        return {'member': [{'gndIdentifier': '4044660-8', 'preferredName': 'Paris', 'variantName': ['Lutetia'],
                            'type': ['AuthorityResource', 'PlaceOrGeographicName'],
                            'sameAs': [{'id': 'http://www.wikidata.org/entity/Q777'}]}]}


def test_run_end_to_end_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'Net', FakeNet)
    old = tmp_path / 'old'
    old.mkdir()
    pd.DataFrame({'kind': ['person'], 'scheme': ['gnd'], 'id': ['1'], 'n_mentions': [1], 'n_in_period': [1], 'n_register': [0]}).to_csv(old / 'ids.tsv', sep='\t', index=False)
    (old / 'wd_map.tsv').write_text('scheme\tid\tqids\n')
    ids = tmp_path / 'ids.tsv'
    pd.DataFrame({'kind': ['person', 'person', 'place'], 'scheme': ['gnd', 'gnd', 'gnd'], 'id': ['1', '118575449', '4044660-8'],
                  'n_mentions': [1, 1, 1], 'n_in_period': [1, 1, 1], 'n_register': [0, 0, 0]}).to_csv(ids, sep='\t', index=False)
    dest = tmp_path / 'dest'
    A.run(ids, old, dest, tmp_path / 'cache', log=lambda m: None)
    m = pd.read_csv(dest / 'wd_map.tsv', sep='\t', dtype=str)
    got = {(r.scheme, r.id): r.qids for r in m.itertuples()}
    assert got[('gnd', '118575449')] == 'Q9554'                     # found by the search, confirmed by the identifier statement
    assert got[('gnd', '4044660-8')] == 'Q777'                      # lobid: the GND record of a place says sameAs Q777
    e = pd.read_parquet(dest / 'wd_entities.parquet').set_index('qid')
    assert json.loads(e.loc['Q9554', 'labels'])['en'] == 'Martin Luther' and list(e.loc['Q9554', 'occupations']) == []
    g = pd.read_parquet(dest / 'gnd_records.parquet').set_index('gnd')
    assert list(g.loc['4044660-8', 'types']) == ['PlaceOrGeographicName'] and g.loc['4044660-8', 'variants'][0] == 'Lutetia'
    assert (dest / 'wd_class_labels.tsv').exists()
