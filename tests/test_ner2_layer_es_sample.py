import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('s', Path(__file__).resolve().parent.parent / 'scripts/ner2_layer_es_sample.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_wilson():
    lo, hi = S.wilson(40, 40)
    assert hi == 1.0 and 0.91 < lo < 0.92
    lo, hi = S.wilson(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = S.wilson(20, 40)
    assert abs((lo + hi) / 2 - 0.5) < 1e-6
