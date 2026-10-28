"""Pure logic of the NER layer v3: the recommended rule and the overlap of model spans with editor tags."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve().parent.parent / 'scripts' / 'ner2_layer_v3_merge.py'
_s = importlib.util.spec_from_file_location('ner2_layer_v3_merge', _p)
M = importlib.util.module_from_spec(_s)
sys.modules['ner2_layer_v3_merge'] = M
_s.loader.exec_module(M)


def test_recommended_rule():
    conf = np.array([0.99, 0.97, 0.6, 0.4, 0.99, 0.99])
    ag = np.array([2, 4, 2, 3, 1, np.nan])
    lang = np.array(['la', 'la', 'es', 'es', 'de', 'de'])
    assert M.is_recommended(conf, ag, lang).tolist() == [True, False, True, False, False, False]


def _tags():
    return pd.DataFrame({'doc_id': ['a', 'a', 'b'], 'char_start': [10, 30, 5], 'char_end': [20, 40, 9],
                         'tag': ['placeName', 'persName', 'name'], 'type': [None, '', 'place'], 'tag_kind': ['place', 'person', 'place']})


def test_overlap_by_one_character_and_adjacent():
    sp = pd.DataFrame({'doc_id': ['a', 'a', 'a', 'a', 'b', 'c'], 'char_start': [19, 20, 0, 28, 8, 0], 'char_end': [25, 30, 10, 31, 12, 50]})
    f = M.flag_overlaps(sp, _tags())
    # [19,25) shares one character with [10,20); [20,30) touches both tags without sharing one; [0,10) touches the first; [28,31) overlaps the person tag by one character
    assert f['ed_overlap'].tolist() == [True, False, False, True, True, False]
    assert f['ed_kind'].tolist()[0] == 'place' and f['ed_kind'].tolist()[3] == 'person'
    assert f['ed_len'].tolist()[0] == 1 and f['ed_len'].tolist()[3] == 1


def test_longest_overlap_wins_and_empty():
    tg = pd.DataFrame({'doc_id': ['a', 'a'], 'char_start': [0, 5], 'char_end': [6, 30], 'tag': ['persName', 'placeName'], 'type': ['', ''], 'tag_kind': ['person', 'place']})
    sp = pd.DataFrame({'doc_id': ['a'], 'char_start': [4], 'char_end': [12]})
    f = M.flag_overlaps(sp, tg)
    assert f['ed_tag'].iloc[0] == 'placeName' and f['ed_len'].iloc[0] == 7
    assert not M.flag_overlaps(sp, tg.iloc[0:0])['ed_overlap'].any()
    assert len(M.flag_overlaps(sp.iloc[0:0], tg)) == 0
