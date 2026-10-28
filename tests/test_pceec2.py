import pytest

from ratlas import common as C
from ratlas.collections import get

BASE = 'eng_dh/pceec2-main/data/'
real = pytest.mark.skipif(not (C.raw_root() / BASE / 'aif_2022.csv').exists(), reason='raw data not available')

PSD = b'''( (CODE <L_ALLEN_001>))
( (METADATA (AUTHOR WILLIAM_ALLEN:MALE:_:1532:47)
	    (RECIPIENT RICHARD_HOPKINS:MALE:_:1546?:33?)
	    (LETTER ALLEN_001:E2:1579:AUTOGRAPH:OTHER))
  (IP-MAT (NP-VOC (NPR Mr.))) (ID ALLEN,8.16))
( (METADATA (AUTHOR WILLIAM_ALLEN:MALE:_:1532:47)
	    (RECIPIENT RICHARD_HOPKINS:MALE:_:1546?:33?)
	    (LETTER ALLEN_001:E2:1579:AUTOGRAPH:OTHER))
  (IP-MAT (CONJ and)) (ID ALLEN,8.17))
( (METADATA (AUTHOR WILLIAM_ALLEN:MALE:_:1532:47) (RECIPIENT _:_:_:_:_) (LETTER ALLEN_002:E2:_:AUTOGRAPH:OTHER)) (IP-MAT (X y)))'''


def test_synthetic():
    import ratlas.collections.pceec2 as M
    col = get('pceec2')
    ctx = col.make_context()
    rec = C.FileRecord(path=BASE + 'parsed/allen.psd', coll='eng_dh', data=PSD)
    ms, es = M.extract(rec, ctx)
    assert es == [] and [(m.doc_id, m.role, m.surface, m.doc_year) for m in ms] == [
        ('ALLEN_001', 'sender', 'WILLIAM ALLEN', 1579), ('ALLEN_001', 'addressee', 'RICHARD HOPKINS', 1579),
        ('ALLEN_002', 'sender', 'WILLIAM ALLEN', None)]
    assert ms[0].in_period is True and ms[2].in_period is None and ms[2].date_source is None
    assert all(m.context is None and m.tier == 'T3' for m in ms)


@real
def test_real():
    col = get('pceec2')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 84
    ents = col.prepare(ctx)
    assert len(ents) >= 1090
    rec = next(r for r in files if r.path.endswith('allen.psd'))
    ms, _ = col.extract(rec, ctx)
    assert ms[0].role == 'sender' and ms[0].local_key == 'pceec2:WILLIAM_ALLEN' and ms[0].in_register
    assert ms[0].doc_id == 'ALLEN_001' and ms[0].doc_year == 1579
    ms[0].to_row()
