import pytest
import pyarrow.parquet as pq

from ratlas import common as C

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')


def test_grand_tour_extract(tmp_path):
    from ratlas.runner import run
    st = run('grand_tour', out=tmp_path, verbose=False)
    assert st['file_errors'] == 0
    # colls.json: 2,076 person + 2,337 place + 122 org tags in the three in-period journals
    assert st['mentions_by_kind']['person'] == 2076 and st['mentions_by_kind']['org'] == 122
    assert abs(st['mentions_by_kind']['place'] - 2337) <= 2
    assert st['entities_by_kind'] == {'person': 397, 'place': 323, 'org': 37}
    assert 0.32 < st['resolved_share'] < 0.35          # 33% of tags resolve to a register entry
    assert st['in_period_share'] == 1.0
    m = pq.read_table(tmp_path / 'grand_tour' / 'mentions.parquet').to_pandas()
    assert set(m.licence) == {'CC-BY-SA-4.0'} and set(m.tier) == {'T1'}
    assert m.context.str.len().max() <= 120
    e = pq.read_table(tmp_path / 'grand_tour' / 'entities.parquet').to_pandas()
    assert e.lat.notna().sum() > 40 and (e.n_mentions > 0).sum() > 700


def test_registry():
    from ratlas.collections import discover
    d = discover()
    assert 'grand_tour' in d and d['grand_tour'].tier == 'T1'
