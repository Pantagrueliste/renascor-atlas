"""Span metrics, subword alignment and windowing of the NER fine-tuning code (no torch, no model: a fake tokenizer)."""
import numpy as np

from ratlas.ner import nerdata as nd
from ratlas.ner.spans import Counts, char_to_token_spans, prf, spans_to_tags, tags_to_spans


class FakeTok:
    """Splits every word into pieces of 3 characters; offsets as a fast tokenizer would give them (a leading space belongs to no piece)."""

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=True, truncation=False):
        ids, offs = [], []
        for t in texts:
            i, o, p = [], [], 0
            for w in t.split(' '):
                for k in range(0, len(w), 3):
                    i.append(10 + k)
                    o.append((p + k, min(p + k + 3, p + len(w))))
                p += len(w) + 1
            ids.append(i)
            offs.append(o)
        return {'input_ids': ids, 'offset_mapping': offs}


def rec(i, words, loc=None, per=None, **kw):
    return {'id': i, 'tokens': words, 'ner_tags_loc': loc or [0] * len(words), 'ner_tags_per': per or [0] * len(words), 'loc_valid': True, 'per_valid': True,
            'lang': 'fr', 'edition': 'ed', **kw}


def test_tags_to_spans_strict():
    assert tags_to_spans([0, 1, 2, 2, 0, 1, 0, 1, 1]) == [(1, 4), (5, 6), (7, 8), (8, 9)]
    assert tags_to_spans([2, 2, 0, 1, 2]) == [(3, 5)]            # a stray I opens nothing
    assert tags_to_spans([1, 0, 2, 2]) == [(0, 1)]               # I after O is ignored
    assert tags_to_spans([]) == []
    assert spans_to_tags(5, [(1, 3), (4, 5)]) == [0, 1, 2, 0, 1]


def test_matches_seqeval_if_installed():
    seqeval = __import__('pytest').importorskip('seqeval.metrics')
    import random
    rng = random.Random(3)
    for _ in range(50):
        g = [rng.choice([0, 1, 2]) for _ in range(30)]
        p = [rng.choice([0, 1, 2]) for _ in range(30)]
        name = ['O', 'B-LOC', 'I-LOC']
        from seqeval.scheme import IOB2
        f_ref = seqeval.f1_score([[name[x] for x in g]], [[name[x] for x in p]], mode='strict', scheme=IOB2, zero_division=0)
        c = Counts()
        c.add('LOC', ['all'], set(tags_to_spans(g)), set(tags_to_spans(p)))
        assert abs(prf(*c.c['LOC']['all'])['f1'] - round(f_ref, 4)) < 1e-4


def test_counts_groups_and_macro():
    c = Counts()
    c.add('LOC', ['all', 'lang=fr', 'edition=a'], {(0, 1), (3, 4)}, {(0, 1), (5, 6)})
    c.add('LOC', ['all', 'lang=la', 'edition=b'], {(0, 2)}, {(0, 2)})
    r = c.report()['LOC']
    assert r['overall']['tp'] == 2 and r['overall']['fp'] == 1 and r['overall']['fn'] == 1
    assert r['by_edition']['a']['f1'] == 0.5 and r['by_edition']['b']['f1'] == 1.0
    assert r['macro_edition_f1'] == 0.75


def test_char_to_token_spans():
    toks = ['Je', 'vais', 'a', 'Paris', ',']
    starts, p = [], 0
    for t in toks:
        starts.append(p)
        p += len(t) + 1
    assert char_to_token_spans(starts, [len(t) for t in toks], [(10, 15)]) == [(3, 4)]
    assert char_to_token_spans(starts, [len(t) for t in toks], [(8, 15)]) == [(2, 4)]
    assert char_to_token_spans(starts, [len(t) for t in toks], [(100, 105)]) == []


def test_word_of_offsets():
    starts, ends = [0, 6, 12], [5, 11, 17]
    assert nd.word_of_offsets(starts, ends, [(0, 3), (3, 5), (5, 11), (12, 15), (0, 0)]) == [0, 0, 1, 2, -1]


def test_encode_first_subword_labels_and_windows():
    words = ['Parisiensis', 'in', 'Roma', 'et', 'Lugduni']
    r = rec('x', words, loc=[1, 0, 1, 0, 1])
    enc = nd.encode_records(FakeTok(), [r])
    assert enc.n_unaligned == 0
    ws = enc.wstart(0)
    assert ws.tolist() == [0, 4, 5, 7, 8, 11]                 # 4 + 1 + 2 + 1 + 3 pieces
    rows = nd.build_windows(enc, 100, 0)
    assert rows.tolist() == [[0, 0, 5, 0, 11]]
    b = nd.batch_windows(enc, rows, [1], [2], 0, True)
    assert b['input_ids'].shape == (1, 13)
    lab = b['lab_loc'][0]
    assert [int(x) for x in lab[[1, 5, 6, 8, 9]]] == [1, 0, 1, 0, 1]
    assert (lab != nd.IGNORE).sum() == 5


def test_windows_cover_every_word_and_overlap():
    ws = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])           # 8 words of 3 subwords
    nover = nd.make_windows(ws, 9, 0)
    assert nover == [(0, 3), (3, 6), (6, 8)]
    over = nd.make_windows(ws, 9, 3)
    covered = set()
    for a, b in over:
        assert ws[b] - ws[a] <= 9
        covered |= set(range(a, b))
    assert covered == set(range(8))
    assert over[0][1] > over[1][0]                            # real overlap
    assert nd.make_windows(np.array([0, 50, 52]), 9, 0) == [(0, 1), (1, 2)]   # a word longer than the limit still gets a window


def test_merge_window_keeps_the_centre_prediction():
    from ratlas.ner.train import merge_window
    words = ['aaa'] * 6
    enc = nd.encode_records(FakeTok(), [rec('x', words)])
    n = 6
    pl, pp = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    best = np.full(n, -1, dtype=np.int64)
    nwords = enc.word_off[1:] - enc.word_off[:-1]
    # window 1 covers words 0-3 and says 1 everywhere, window 2 covers words 2-5 and says 2 everywhere
    merge_window(enc, [0, 0, 4, 0, 4], np.arange(4), np.arange(4), np.ones(4, dtype=int), np.zeros(4, dtype=int), pl, pp, best, nwords)
    merge_window(enc, [0, 2, 6, 2, 6], np.arange(4), np.arange(2, 6), np.full(4, 2), np.zeros(4, dtype=int), pl, pp, best, nwords)
    assert pl.tolist() == [1, 1, 1, 2, 2, 2]


def test_length_batches_respect_budget_and_cover_all():
    rows = np.array([[0, 0, 1, 0, l] for l in [10, 250, 30, 250, 20, 240, 5]])
    for rng in (None, np.random.default_rng(0)):
        bs = nd.length_batches(rows, 600, 4, rng)
        flat = sorted(int(i) for b in bs for i in b)
        assert flat == list(range(len(rows)))
        for b in bs:
            assert len(b) <= 4 and (max(rows[b, 4]) + 4) * len(b) <= 600 or len(b) == 1


def test_llm_answer_parsing_and_marking():
    from ratlas.ner.llm import mark, parse_answer
    p, q, ok = parse_answer('Sure: {"places": ["Paris", "Rome"], "persons": ["Jean de Troyes"]}')
    assert (p, q, ok) == (['Paris', 'Rome'], ['Jean de Troyes'], True)
    p, q, ok = parse_answer('{"places": ["Lyon", "Mar')               # cut off
    assert p == ['Lyon'] and ok is False
    toks = ['Jean', 'de', 'Troyes', 'va', 'a', 'Paris', ',', 'puis', 'Paris', 'Nord']
    assert mark(toks, ['Jean de Troyes', 'Paris']) == {(0, 3), (5, 6), (8, 9)}
    assert mark(toks, ['Paris', 'Paris Nord']) == {(5, 6), (8, 10)}   # the longer match wins over the name it contains
    assert mark(toks, []) == set()
