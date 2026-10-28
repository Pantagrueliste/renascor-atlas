import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('s', Path(__file__).resolve().parent.parent / 'scripts/ner2_es_setup_score.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def rec():
    m = lambda *xs: [{'s': s, 'e': e, 'conf': c} for s, e, c in xs]
    return {'models': {'r3_xlmr_large_H': m((0, 1, 0.99)), 'r3_mdeberta_HM': m((0, 1, 0.99), (5, 6, 0.99), (8, 9, 0.5)), 'r3_mmbert_base_HM': m((5, 6, 0.7)), 'r3_hmteams_HM': []}}


def test_recommended_needs_agreement_and_conf():
    r = rec()
    assert S.recommended(r, 'r3_mdeberta_HM') == {(0, 1), (5, 6)}
    assert S.recommended(r, 'r3_xlmr_large_H') == {(0, 1)}
    assert S.recommended(r, 'r3_mdeberta_HM', 0.98, 3) == set()


def test_score():
    assert S.score({'a': {(0, 1), (2, 3)}}, {'a': {(0, 1), (4, 5)}}) == (1, 1, 1)
    p, r, f = S.prf(1, 1, 1)
    assert p == r == f == 0.5


def test_union():
    assert S.setup_spans(rec(), 'union xlmr+mdeberta') == {(0, 1), (5, 6), (8, 9)}
