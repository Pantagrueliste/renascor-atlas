"""Merged points and label fixes (ratlas.placemerge, data/curation/place_merges.json): the rules on toy rows and the shape of the data."""
import collections

import pytest

from ratlas import dicotopo as DT
from ratlas import placemerge as PM


def row(name, lat, lon, n, members=None, gone=None, shard=0):
    # [name, lat, lon, q10, editions, top, qid, code, tier, mentions, original, kind, per-edition counts, shard, members, gone]
    return [name, lat, lon, [0] * 10, 1, [], None, 'w', 2, n, name, 'c', [], shard, members, gone]


def entry(keep, absorb, rows=None, models=None, source='duplicates', **kw):
    info = {}
    for i in [keep] + absorb:
        if i.startswith('e:'):
            r = rows[int(i[2:])]
            info[i] = {'name': r[0], 'lat': r[1], 'lon': r[2]}
        else:
            info[i] = {'name': 'm', 'lat': models[i][0], 'lon': models[i][1]}
    return {'keep': keep, 'absorb': absorb, 'reading': 'both', 'reason': 'test', 'source': source, 'casebooks': False, 'new_label': None, 'info': info, **kw}


def test_a_merge_joins_members_and_marks_the_absorbed_rows_gone():
    rows = [row('Navarre', 42.8, -1.6, 10), row('Navarra', 42.8, -1.65, 4), row('Elsewhere', 10, 10, 1)]
    done, left = PM.plan([entry('e:0', ['e:1'], rows)], rows, {})
    assert left == [] and done[0]['keep'] == 0 and done[0]['absorb'] == [1]
    PM.merge_rows(rows, 0, [1])
    assert rows[0][14] == [[0, 0], [1, 0]] and rows[1][15] == 1 and rows[1][14] is None and rows[2][15] is None
    assert PM.lead_map(rows) == {0: 0, 1: 0}


def test_an_absorbed_point_brings_its_own_members():
    rows = [row('A', 0, 0, 5), row('B', 0, 0.01, 3, members=[[1, 0], [2, 3]]), row('B member', 0, 0.01, 1, gone=1)]
    PM.merge_rows(rows, 0, [1])
    assert [j for j, _ in rows[0][14]] == [0, 1, 2] and rows[1][15] == 1


def test_two_entries_about_the_same_point_do_not_clash():
    rows = [row('A', 0, 0, 5), row('B', 0, 0, 3), row('C', 0, 0, 2)]
    es = [entry('e:0', ['e:1'], rows), entry('e:2', ['e:1'], rows)]
    done, left = PM.plan(es, rows, {})
    PM.merge_rows(rows, done[0]['keep'], done[0]['absorb'])
    assert len(done) == 2
    PM.merge_rows(rows, done[1]['keep'], done[1]['absorb'])         # the second names B, which is in A's group now: the group joins C, the keep of the entry
    assert done[1]['keep'] == 2 and done[1]['absorb'] == [0]
    assert sorted(j for j, _ in rows[2][14]) == [0, 1, 2] and rows[0][15] == 1 and rows[2][15] is None


def test_an_entry_is_left_out_when_the_row_is_not_the_one_it_was_written_for():
    rows = [row('A', 0, 0, 5), row('B', 0, 0, 3)]
    e = entry('e:0', ['e:1'], rows)
    rows[1][0] = 'Renamed'
    done, left = PM.plan([e], rows, {})
    assert not done and 'Renamed' in left[0]['reason']
    rows[1][0] = 'B'
    rows[1][1] = 5.0
    assert not PM.plan([e], rows, {})[0]


def test_a_pair_on_the_keep_apart_list_is_not_merged():
    rows = [row('Nice', 43.7, 7.26, 5), row('Iznik', 40.4, 29.7, 3)]
    done, left = PM.plan([entry('e:0', ['e:1'], rows)], rows, {}, keep_apart={frozenset(('e:0', 'e:1'))})
    assert not done and 'keep-apart' in left[0]['reason']


def test_model_points_are_followed_to_their_new_id():
    rows = [row('Arras', 50.29, 2.78, 20)]
    models = {'ml2:239': (50.29, 2.78)}
    e = entry('e:0', ['ml2:239'], rows, models)
    assert PM.plan([e], rows, {'ml2:239': None})[1][0]['reason'].endswith('not in the layer any more')
    done, left = PM.plan([e], rows, {'ml2:239': 'ml2:17'}, {'ml2:17': (50.29, 2.78)})
    assert done[0]['models'] == ['ml2:17']
    assert PM.plan([e], rows, {'ml2:239': 'ml2:17'}, {'ml2:17': (45.0, 2.78)})[1]       # 500+ km from the point of the entry: refused


def test_match_model_points_finds_a_point_by_its_units_when_ids_move():
    old_keys = {'ml2:5': ['ml:10', 'ml:11'], 'ml2:6': ['ml3:7'], 'ml2:7': ['ml:99']}
    new = [{'id': 'ml2:3', 'prev': ['ml:10', 'ml:11', 'mles:2'], 'n': 50}, {'id': 'ml2:4', 'prev': ['ml3:7'], 'n': 9}, {'id': 'ml2:8', 'prev': ['ml:12'], 'n': 4}]
    assert PM.match_model_points(old_keys, new) == {'ml2:5': 'ml2:3', 'ml2:6': 'ml2:4', 'ml2:7': None}
    split = [{'id': 'a', 'prev': ['ml:10'], 'n': 3}, {'id': 'b', 'prev': ['ml:11'], 'n': 8}]
    assert PM.match_model_points({'x': ['ml:10', 'ml:11']}, split)['x'] == 'b'          # a tie goes to the point with more mentions


def test_labels_and_qids_are_applied_only_to_the_point_they_were_written_for():
    rows = [row('Arragon', 41, -1, 5), row('Gone', 0, 0, 1, gone=1), row('Already fixed', 0, 0, 1)]
    names = {'ml2:3': 'Blackwalle'}
    labels = [{'point': 'e:0', 'old': 'Arragon', 'new': 'Aragon'}, {'point': 'e:1', 'old': 'Gone', 'new': 'x'}, {'point': 'e:2', 'old': 'Other', 'new': 'y'},
              {'point': 'ml2:9', 'old': 'Blackwalle', 'new': 'Blackwall'}]
    rep = PM.apply_labels(rows, labels, names, {'ml2:9': 'ml2:3'})
    assert rows[0][0] == 'Aragon' and rows[1][0] == 'Gone' and rows[2][0] == 'Already fixed' and names['ml2:3'] == 'Blackwall'
    assert (rep['rows'], rep['model'], rep['skipped_gone'], rep['skipped_changed']) == (1, 1, 1, 1)
    rows[0][6] = 'Q31'
    assert PM.apply_qid_fixes(rows, [{'point': 'e:0', 'old': 'Q31', 'new': 'Q239'}, {'point': 'e:2', 'old': 'Q1', 'new': 'Q2'}]) == 1 and rows[0][6] == 'Q239'


def test_the_data_file_is_well_formed():
    d = PM.load()
    assert d['counts'] == {'merges': 272, 'labels': 371} and len(d['merges']) == 272
    assert collections.Counter(m['source'] for m in d['merges']) == {'duplicates': 259, 'dicotopo': 7, 'integration': 6}
    ids = set()
    for m in d['merges']:
        assert m['reading'] in ('both', 'augmented') and m['keep'] not in m['absorb']
        for i in [m['keep']] + m['absorb']:
            assert i.startswith(('e:', 'ml2:')) and i in m['info']
            ids.add(i)
    ids |= {x['point'] for x in d['labels']} | set(d['unlink'])
    ids |= {i for i in DT.load()['point_id'] if i.startswith('ml2:')}
    assert {i for i in ids if i.startswith('ml2:')} <= set(d['model_keys']) | {i for i in DT.load()['point_id']}
    # the Belgium of the editors is Brussels
    assert d['qid_fixes'] == [{'point': 'e:41', 'old': 'Q31', 'new': 'Q239', 'reason': 'Brussels, not Belgium'}]
    assert {'e:41'} <= {x['point'] for x in d['labels']} and PM.keep_apart_pairs()
    # the four known duplicates are merged: Navarra, Transylvania, Cheshire, Macedonia; Arragon is relabelled
    labels = {x['old']: x['new'] for x in d['labels']}
    assert labels['Arragon'] == 'Aragon' and labels['Belgium'] == 'Brussels'


def test_every_model_point_of_the_data_has_a_key_that_can_be_matched():
    d = PM.load()
    assert all(isinstance(v, list) and v for v in d['model_keys'].values())
    assert 'ml2:450' in d['unlink']


def test_the_layer_overrides_of_version_3_keep_their_counts():
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'model_layer_overrides_v3.json').read_text('utf-8'))
    assert d['counts'] == dict(collections.Counter(e['action'] for e in d['entries'])) == {'move': 3, 'merge': 6, 'unplace': 6, 'forms_move': 1}
    by = {tuple(e['anchor']): e for e in d['entries']}
    assert by[('v3', 'de', 'neutra', '')]['label'] == 'Nitra' and by[('v3', 'de', 'neutra', '')]['no_merge'] is True
    assert by[('v3', 'de', 'ruhr', '')]['action'] == 'unplace'
    assert by[('v1', 'en', 'blackheath', '')]['action'] == 'forms_move' and by[('v1', 'en', 'blackheath', '')]['forms'] == ['Blackheath']
