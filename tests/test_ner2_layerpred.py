import numpy as np

from ratlas.ner import layerpred as LP


class ToyTok:
    """Whitespace tokenizer with offsets, like a fast tokenizer without subword splitting; CLS = 0, SEP = 2."""
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2

    def __call__(self, texts, add_special_tokens=True, return_offsets_mapping=False):
        single = isinstance(texts, str)
        ts = [texts] if single else texts
        ids, offs = [], []
        import re
        for t in ts:
            ms = list(re.finditer(r'\S+', t))
            i = [10 + (hash(m.group()) % 1000) for m in ms]
            o = [(m.start(), m.end()) for m in ms]
            if add_special_tokens:
                i = [0] + i + [2]
            ids.append(i)
            offs.append(o)
        out = {'input_ids': ids[0] if single else ids}
        if return_offsets_mapping:
            out['offset_mapping'] = offs[0] if single else offs
        return out


def test_passage_plan_covers_and_owns_once():
    for n in (1, 5, 199, 200, 201, 380, 381, 1000, 1234):
        w0, w1, o0, o1 = LP.passage_plan(n, 200, 180)
        assert w0[0] == 0 and w1[-1] == n
        own = np.zeros(n, dtype=int)
        for a, b in zip(o0, o1):
            assert 0 <= a < b <= n
            own[a:b] += 1
        assert (own == 1).all(), n
        assert (o0 >= w0).all() and (o1 <= w1).all()
        assert (np.diff(w0) == 180).all()


def test_token_windows_like_v1():
    assert LP.token_windows(100, 254, 190) == [(0, 100)]
    w = LP.token_windows(300, 254, 190)
    assert w == [(0, 254), (46, 300)]
    w = LP.token_windows(600, 254, 190)
    assert w[0] == (0, 254) and w[-1][1] == 600
    assert all(e - s <= 254 for s, e in w)


def test_first_subwords():
    words = ['London', ',', 'and', 'Yorke']
    text = ' '.join(words)
    lens = np.array([len(x) for x in words])
    starts = np.cumsum(lens + 1) - (lens + 1)
    ends = starts + lens
    # subwords: Lon|don|,|and|Yorke ; plus a stray space-only token at the start
    offs = np.array([[0, 3], [3, 6], [7, 8], [9, 12], [13, 18]])
    ft = LP.first_subwords(offs, starts, ends)
    assert ft.tolist() == [0, 2, 3, 4]


def test_decode_and_merge():
    # two passages of 6 words: B I O B O I(stray) | O B I I O B
    lab = np.array([1, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 1], dtype=np.int8)
    pm = np.full(12, 0.9, dtype=np.float32)
    pm[1] = 0.7
    pp = np.zeros(12, dtype=np.float32)
    off = np.array([0, 6, 12])
    pid, s, e, c, p = LP.decode_spans(lab, pm, pp, off)
    assert list(zip(pid, s, e)) == [(0, 0, 2), (0, 3, 4), (1, 7, 10), (1, 11, 12)]
    assert abs(c[0] - 0.8) < 1e-6
    # a span ending exactly at a passage end is cut there
    lab2 = np.array([0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0], dtype=np.int8)
    pid, s, e, c, p = LP.decode_spans(lab2, pm, pp, off)
    assert list(zip(pid, s, e)) == [(0, 5, 6)]
    # merge: the window farthest from its edge wins
    gw = np.array([0, 1, 0, 1], dtype=np.int32)
    mg = np.array([0, 5, 3, 1], dtype=np.int32)
    lb = np.array([1, 0, 2, 1], dtype=np.int8)
    labs, pm_, pp_ = LP.merge_windows(3, gw, mg, lb, np.array([.5, .6, .7, .8]), np.zeros(4))
    assert labs.tolist() == [2, 0, 0] and abs(pm_[0] - 0.7) < 1e-6 and pm_[2] == 1.0


def test_agreement():
    h = {'doc': np.array([0, 0, 1]), 'w_start': np.array([3, 10, 3]), 'w_end': np.array([5, 12, 4])}
    o = {'doc': np.array([0, 0, 1]), 'w_start': np.array([3, 11, 9]), 'w_end': np.array([5, 13, 10])}
    ex, ov = LP.agreement(h, o)
    assert ex.tolist() == [True, False, False]
    assert ov.tolist() == [True, True, False]


def test_prepare_pack_and_spans_roundtrip():
    ti = LP.tok_info(ToyTok(), max_len=32, overlap=8)
    words = ' '.join(f'w{i}' for i in range(30)) + ' London , and Yorke . ' + ' '.join(f'x{i}' for i in range(300))
    pack = LP.prepare_pack(ti, [7], [words], [None], size=200, stride=180)
    assert pack['pass_doc'].tolist() == [7, 7] and pack['pass_off'].tolist()[0] == 0
    assert pack['cs'][30] == words.index('London')
    # every word of every passage gets exactly one output position in some window
    got = np.concatenate([b['gword'] for b in pack['batches']])
    assert set(got.tolist()) == set(range(int(pack['pass_off'][-1])))
    # a fake model: B on the word 'London', I on the next word, everything else O
    lab = np.zeros(int(pack['pass_off'][-1]), dtype=np.int8)
    lab[30] = 1
    lab[31] = 2
    pm = np.full(len(lab), 0.99, dtype=np.float32)
    sp = LP.pack_spans(pack, lab, pm, np.zeros(len(lab), dtype=np.float32))
    assert sp['doc'].tolist() == [7] and sp['w_start'].tolist() == [30] and sp['w_end'].tolist() == [32]
    assert words[sp['char_start'][0]:sp['char_end'][0]] == 'London ,'


def test_selected_passages_only():
    ti = LP.tok_info(ToyTok(), max_len=32, overlap=8)
    text = ' '.join(f'w{i}' for i in range(500))
    pack = LP.prepare_pack(ti, [3], [text], [np.array([1])], size=200, stride=180)
    assert pack['pass_idx'].tolist() == [1] and pack['pass_w0'].tolist() == [180]
    assert int(pack['pass_off'][-1]) == 200
