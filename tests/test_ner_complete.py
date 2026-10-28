"""Completeness measure and splits of the NER data on toy passages (ratlas.ner.complete, ratlas.ner.splits)."""
import json

from ratlas.ner import complete as K
from ratlas.ner import splits as S


def passage(path, text, tagged, i=0, slug='ed', lang='la', **kw):
    spans = []
    for label, s in tagged:
        a = text.index(s)
        spans.append({'start': a, 'end': a + len(s), 'label': label, 'tag': 'placeName' if label == 'LOC' else 'persName'})
    p = {'id': f'{slug}/{path}/{i}', 'slug': slug, 'path': path, 'i': i, 'text': text, 'spans': spans, 'nested': [], 'ignored': [], 'orgs': [],
         'lang': lang, 'lang_src': 'doc', 'words': len(text.split()), 'flags': [], 'dropped_span': False, 'licence': 'CC-BY-4.0', 'tier': 'T1',
         'year': 1600}
    p.update(kw)
    return p


def test_untagged_share_exhaustive_and_partial():
    ps = []
    for k in range(40):          # Basel is always tagged
        ps.append(passage(f'd{k}', f'Er reiste nach Basel und kam heim. Nummer {k}', [('LOC', 'Basel')]))
    rep, out = K.analyse(ps)
    assert rep['loc_untagged_share_docs'] == 0.0 and rep['loc_verdict'] == 'exhaustive'
    # add 20 occurrences of Basel left untagged in tagged documents -> 20/60
    for k in range(20):
        ps.append(passage(f'd{k}', f'Wieder Basel hier {k}', [], i=1))
    rep, out = K.analyse(ps)
    assert abs(rep['loc_untagged_share_docs'] - 20 / 60) < 1e-3 and rep['loc_verdict'] == 'partial'
    assert sum(1 for q in out if q['u_loc']) == 20
    assert all(q['untagged_examples'] == [['LOC', 'Basel']] for q in out if q['u_loc'])


def test_candidates_exclude_common_words_ambiguous_strings_and_short_ones():
    ps = []
    for k in range(40):
        ps.append(passage(f'd{k}', f'Der Lincoln kam nach Rom und sah den Fluss {k} und den fluss', [('PER', 'Lincoln'), ('LOC', 'Rom')]))
    ps.append(passage('x', 'Lincoln', [('LOC', 'Lincoln')]))              # Lincoln tagged as place too: candidate of neither label
    ps.append(passage('y', 'Ab Rom Fluss Fluss Fluss', [('LOC', 'Fluss'), ('LOC', 'Ab')]))   # 'Fluss' is frequent in lower case, 'Ab' too short
    for k in range(5):
        ps.append(passage(f'z{k}', 'Lincoln blieb in Rom und am Fluss', []))
    rep, out = K.analyse(ps)
    assert rep['loc_candidate_types'] == 1                 # only Rom
    assert rep['loc_untagged_docs'] == 1                   # Rom untagged in y (a document with tags)
    assert rep['loc_untagged_all'] == 5 + 1                # and in z0..z4 (documents without any tag)
    assert rep['per_untagged_docs'] == 0 and rep['per_candidate_types'] == 0


def test_spans_nested_ignored_and_org_cover_matches():
    ps = [passage(f'd{k}', f'Hans von Basel in der Universität Basel {k}', [('PER', 'Hans von Basel')], nested=[{'start': 9, 'end': 14, 'label': 'LOC', 'tag': 'placeName'}],
                  orgs=[[27, 41]]) for k in range(35)]
    ps += [passage(f'e{k}', f'Reise nach Basel {k}', [('LOC', 'Basel')]) for k in range(35)]
    rep, out = K.analyse(ps)
    assert rep['loc_untagged_docs'] == 0          # the two Basel inside the PER span and the organisation are covered


def test_verdicts():
    assert K.verdict(None, 0) == 'none' and K.verdict(0.0, 10) == 'too_few' and K.verdict(0.05, 100) == 'exhaustive'
    assert K.verdict(0.3, 100) == 'partial' and K.verdict(0.7, 100) == 'sparse'


def test_tokens_and_bio_tags_split_a_straddling_token():
    toks, tags, loc, per, offs = S.tokens_and_tags("d'Anvers et Saint-Denis; Rom", [
        {'start': 2, 'end': 8, 'label': 'LOC'}, {'start': 12, 'end': 23, 'label': 'LOC'}, {'start': 25, 'end': 28, 'label': 'PER'}])
    assert toks == ['d', "'", 'Anvers', 'et', 'Saint', '-', 'Denis', ';', 'Rom']
    assert tags == [0, 0, 1, 0, 1, 2, 2, 0, 3]
    assert loc == [0, 0, 1, 0, 1, 2, 2, 0, 0] and per == [0, 0, 0, 0, 0, 0, 0, 0, 1]
    # adjacent spans keep two B tags; a span boundary inside a token splits it
    toks, tags, *_ = S.tokens_and_tags('Roma Napoli Rom', [{'start': 0, 'end': 4, 'label': 'LOC'}, {'start': 5, 'end': 11, 'label': 'LOC'},
                                                       {'start': 12, 'end': 14, 'label': 'LOC'}])
    assert toks == ['Roma', 'Napoli', 'Ro', 'm'] and tags == [1, 1, 1, 0]


def test_cap_tokens_waterfilling():
    a = S.cap_tokens({'big': 1000, 'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100}, 0.15)
    assert a['a'] == 100 and a['big'] < 1000
    assert a['big'] <= 0.15 * sum(a.values()) + 1
    assert S.cap_tokens({}, 0.15) == {}
    assert S.cap_tokens({'x': 10, 'y': 5}, 0.15) == {'x': 10, 'y': 5}      # fewer than 1/cap editions: no cut


def test_choose_test_holds_out_whole_editions_per_language():
    info = {'a': {'lang': 'la', 'lang_share': 0.9, 'loc_spans': 5000, 'group': 'a'},
            'b': {'lang': 'la', 'lang_share': 0.9, 'loc_spans': 600, 'group': 'b'},
            'c': {'lang': 'la', 'lang_share': 0.9, 'loc_spans': 1200, 'group': 'c'},
            'd': {'lang': 'it', 'lang_share': 0.95, 'loc_spans': 800, 'group': 'd'},
            'e': {'lang': 'de', 'lang_share': 0.5, 'loc_spans': 900, 'group': 'e'},
            'f': {'lang': 'de', 'lang_share': 0.9, 'loc_spans': 100, 'group': 'f'},
            'g': {'lang': 'de', 'lang_share': 0.9, 'loc_spans': 4000, 'group': 'g'}}
    test, notes = S.choose_test(info)
    assert set(test) == {'b'} or set(test) == {'c'}        # not the largest, the closest to 10% of the language total
    assert 'a' not in test and 'd' not in test and 'g' not in test
    assert any(n['lang'] == 'it' and 'no edition held out' in n['note'] for n in notes)


def test_validity_rules():
    exh = {'loc_verdict': 'exhaustive', 'per_verdict': 'none'}
    par = {'loc_verdict': 'partial', 'per_verdict': 'partial'}
    assert S.validity({'u_loc': 3, 'u_per': 0}, exh) == (True, False)
    assert S.validity({'u_loc': 3, 'u_per': 0}, par) == (False, True)
    assert S.validity({'u_loc': 0, 'u_per': 1}, {'loc_verdict': 'sparse', 'per_verdict': 'too_few'}) == (False, False)


def test_build_end_to_end(tmp_path):
    """Two training editions, two latin-test candidates: files are written, test editions are whole editions, no id in two splits."""
    data = tmp_path / 'd'
    for sub in ('scored', 'completeness', 'raw/summary'):
        (data / sub).mkdir(parents=True)
    import gzip
    for slug, lang, n in (('big', 'la', 400), ('mid', 'la', 150), ('small', 'la', 60), ('other', 'de', 80), ('other2', 'de', 90), *[(f'a{i}', 'fr', 60) for i in range(6)]):
        ps = []
        for k in range(n):
            filler = ' '.join(f'{slug[:2]}{(k * 7 + j * 13) % 997}x' for j in range(12))
            ps.append(passage(f'{slug}{k // 4}', f'Er kam nach Wien{slug} und {filler} blieb dort', [('LOC', f'Wien{slug}')], i=k % 4, slug=slug, lang=lang,
                              u_loc=0, u_per=0))
        with gzip.open(data / 'scored' / f'{slug}.jsonl.gz', 'wt') as fh:
            for p in ps:
                fh.write(json.dumps(p) + '\n')
        (data / 'completeness' / f'{slug}.json').write_text(json.dumps({'slug': slug, 'loc_verdict': 'exhaustive', 'per_verdict': 'none', 'passages': n, 'docs': n // 4,
                                                                      'docs_with_spans': n // 4, 'loc_spans': n, 'per_spans': 0, 'loc_candidate_types': 1, 'per_candidate_types': 0,
                                                                      'loc_untagged_share_docs': 0.0, 'loc_untagged_share_all': 0.0, 'per_untagged_share_docs': None,
                                                                      'per_untagged_share_all': None, 'words': n * 8}))
    import ratlas.ner.splits as SP
    old = SP.MIN_TEST_LOC
    SP.MIN_TEST_LOC = 50
    try:
        res = SP.build(data, None)
        stats = SP.write_all(res, data)
        tabs = SP.write_tables(res, stats, data)
        sheet = SP.datasheet(res, tabs, stats, data, 'ner-v1')
    finally:
        SP.MIN_TEST_LOC = old
    sp = res['split']
    assert sp['test'] and sp['train'] and sp['dev'] is not None
    test_eds = {p['slug'] for p in sp['test']}
    assert test_eds and not (test_eds & {p['slug'] for p in sp['train']})
    assert not (test_eds & {p['slug'] for p in sp['dev']})
    ids = [p['id'] for s in sp.values() for p in s]
    assert len(ids) == len(set(ids))
    line = json.loads((data / 'train.jsonl').open().readline())
    assert len(line['tokens']) == len(line['ner_tags']) and line['loc_valid'] is True
    assert (data / 'loc' / 'train.jsonl').exists() and (data / 'tables' / 'completeness.tsv').exists()
    assert '# DATASHEET' in sheet and 'test' in sheet


def test_shingles_are_stable_across_processes():
    """The leakage check must not depend on the salted hash() of Python."""
    import subprocess
    import sys
    from ratlas.ner.splits import shingles
    text = ' '.join(f'mot{i}' for i in range(200))
    code = ("from ratlas.ner.splits import shingles; t=' '.join(f'mot{i}' for i in range(200)); "
            "print(sum(sorted(shingles(t))))")
    outs = {subprocess.run([sys.executable, '-c', code], env={'PYTHONHASHSEED': str(s), 'PATH': ''}, capture_output=True, text=True,
                           cwd=str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src')).stdout.strip() for s in (1, 2)}
    assert len(outs) == 1 and outs != {''}
    assert str(sum(sorted(shingles(text)))) in outs


def test_choose_test_never_holds_out_a_not_test_edition():
    info = {'big': {'lang': 'en', 'lang_share': 0.9, 'loc_spans': 5000, 'group': 'big'},
            'hearth_tax': {'lang': 'en', 'lang_share': 0.9, 'loc_spans': 700, 'group': 'hearth_tax'},
            'mid': {'lang': 'en', 'lang_share': 0.9, 'loc_spans': 900, 'group': 'mid'}}
    test, _ = S.choose_test(info)
    assert set(test) == {'mid'}
