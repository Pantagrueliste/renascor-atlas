"""Tests of the independent audit scripts (scripts/ner_audit_*.py): decoding, strict span notation and agreement with the pipeline's own functions."""
import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decode_matches_pipeline_strict_iob2():
    pytest = __import__('pytest')
    np = pytest.importorskip('numpy')
    from ratlas.ner.spans import tags_to_spans
    pred = load('ner_audit_predict')
    score = load('ner_audit_score')
    rnd = random.Random(5)
    for _ in range(300):
        tags = [rnd.choice([0, 0, 1, 2]) for _ in range(rnd.randint(0, 25))]
        a = [(s, e) for s, e, _ in pred.decode(np.array(tags), np.ones(len(tags)))]
        assert a == tags_to_spans(tags)
        assert score.spans_of(tags) == tags_to_spans(tags)


def test_stray_inside_tag_is_ignored():
    pytest = __import__('pytest')
    np = pytest.importorskip('numpy')
    pred = load('ner_audit_predict')
    assert [(s, e) for s, e, _ in pred.decode(np.array([2, 2, 0, 1, 2]), np.ones(5))] == [(3, 5)]


def test_prf_and_overlap():
    score = load('ner_audit_score')
    r = score.prf(8, 2, 4)
    assert (r['p'], r['r']) == (0.8, 0.6667)
    assert score.overlaps((0, 3), (2, 5)) and not score.overlaps((0, 2), (2, 5))
    assert score.norm('Roſa') == 'rosa'


def test_seed_passage_picker_uses_sentence_boundaries():
    pytest = __import__('pytest')
    pytest.importorskip('lxml')
    seed = load('ner_audit_seed_passages')
    text = ' '.join(('Alpha beta gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu Nu Xi Omicron ' * 2).strip() + '.' for _ in range(40))
    got = seed.pick_passage(text, random.Random(1))
    assert got is not None
    assert 100 <= len(got[1].split()) <= 170 and got[1].endswith('.')
