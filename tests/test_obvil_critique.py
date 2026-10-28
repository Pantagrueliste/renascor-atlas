import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')


def test_two_files_only():
    col = get('obvil_critique')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert sorted(r.name for r in files) == ['la-bruyere_ouvrages-esprit_1694.xml', 'la-grange_preface.xml']
    ms = []
    for rec in files:
        m, e = col.extract(rec, ctx)
        ms += m
        assert e == []
    assert len(ms) == 12 and {m.surface for m in ms} >= {'Ariste', 'Zoïle'}
    assert {m.doc_year for m in ms} == {1694} and all(m.in_period for m in ms)
    assert all(m.local_key is None and m.tier == 'T2' for m in ms)
