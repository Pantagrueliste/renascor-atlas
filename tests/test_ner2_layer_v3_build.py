import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ner2_layer_v3_build as B  # noqa: E402


def _flags(rows):
    return pd.DataFrame(rows, columns=['doc_id', 'set', 'words', 'sampled_shingles', 'layer_union_contain', 'layer_best_contain', 'layer_best_doc', 'new_best_contain',
                                       'new_best_doc', 'exact_layer_doc', 'exact_new_docs'])


def _intra(rows):
    return pd.DataFrame(rows, columns=['cand', 'cand_set', 'other', 'other_set', 'matched', 'cand_shingles', 'other_shingles', 'contain_cand'])


def test_layer_duplicate_and_short_documents():
    f = _flags([('a', 'tei', 500, 100, 0.8, 0.8, 'L1', 0, '', '', ''), ('b', 'tei', 500, 100, 0.3, 0.3, 'L2', 0, '', '', ''), ('c', 'tei', 20, 5, 1.0, 1.0, 'L3', 0, '', '', '')])
    drop = B.decide_duplicates(f, _intra([]), pd.DataFrame({'n_tags': [0, 0, 0]}, index=['a', 'b', 'c']))
    assert set(drop) == {'a'}                    # b under 50 %, c under 16 sampled shingles


def test_mutual_containment_keeps_the_more_tagged_then_the_longer():
    f = _flags([('a', 'tei', 100, 40, 0, 0, '', 1.0, 'b', '', ''), ('b', 'seen', 100, 40, 0, 0, '', 1.0, 'a', '', '')])
    i = _intra([('a', 'tei', 'b', 'seen', 40, 40, 40, 1.0), ('b', 'seen', 'a', 'tei', 40, 40, 40, 1.0)])
    d = pd.DataFrame({'n_tags': [0, 12]}, index=['a', 'b'])
    assert set(B.decide_duplicates(f, i, d)) == {'a'}        # b is tagged: it stays


def test_contained_untagged_document_goes_but_a_tagged_one_stays_when_the_big_one_is_untagged():
    f = _flags([('s', 'seen', 50, 20, 0, 0, '', 0.9, 'big', '', ''), ('big', 'tei', 500, 200, 0, 0, '', 0.1, 's', '', '')])
    i = _intra([('s', 'seen', 'big', 'tei', 18, 20, 200, 0.9), ('big', 'tei', 's', 'seen', 18, 200, 20, 0.09)])
    assert B.decide_duplicates(f, i, pd.DataFrame({'n_tags': [5, 0]}, index=['s', 'big'])) == {}
    assert set(B.decide_duplicates(f, i, pd.DataFrame({'n_tags': [0, 0]}, index=['s', 'big']))) == {'s'}


def test_review_overrides_v3_are_well_formed():
    import json
    d = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'model_layer_overrides_v3.json').read_text('utf-8'))
    assert d['entries'] and all(e['anchor'][0] in ('v3', 'v1') and e['anchor'][3] == '' for e in d['entries'])
    assert all(e['anchor'][0] == 'v3' for e in d['entries'] if e['action'] != 'forms_move')       # v29: forms_move names a form of the v1 run (Blackheath)
    assert {e['action'] for e in d['entries']} <= {'move', 'merge', 'unplace', 'forms_move'}
    for e in d['entries']:
        if e['action'] == 'merge':
            assert e['into_anchor'][0] in ('v1', 'es', 'v3') and e['into_id_v2'] and e['into_name_v2']
        if e['action'] == 'move':
            assert e['kind'] in ('c', 'a', 'n') and -90 <= e['lat'] <= 90 and -180 <= e['lon'] <= 180
