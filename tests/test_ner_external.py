"""Tests of ratlas.ner.external (BIO helpers of the external NER datasets) and of the converters' readers."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from ratlas.ner import external as E

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'ner_external_convert.py'


def load_convert():
    spec = importlib.util.spec_from_file_location('ner_external_convert', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules['ner_external_convert'] = m
    spec.loader.exec_module(m)
    return m


def test_split_tag_styles():
    assert E.split_tag('B-LOC') == ('B', 'LOC')
    assert E.split_tag('I-per') == ('I', 'per')
    assert E.split_tag('PRS-B') == ('B', 'PRS')
    assert E.split_tag('GEO-I') == ('I', 'GEO')
    assert E.split_tag('LOC') == ('I', 'LOC')          # IO scheme
    assert E.split_tag('B-PER_NAME') == ('B', 'PER_NAME')
    assert E.split_tag('I-SHIP_TYPE') == ('I', 'SHIP_TYPE')
    assert E.split_tag('O')[0] == 'O' and E.split_tag('_')[0] == 'O' and E.split_tag('0')[0] == 'O'


def test_map_label_prefix_and_case():
    assert E.map_label('loc.adm.town', ['loc'], ['pers']) == 'LOC'
    assert E.map_label('PERS', ['loc'], ['pers']) == 'PER'
    assert E.map_label('pers.ind.articleauthor', ['loc'], ['pers']) == 'PER'
    assert E.map_label('location', ['loc'], ['pers']) is None      # no prefix match without a dot: 'location' is not 'loc'
    assert E.map_label('ORG', ['loc'], ['pers']) is None


def test_bio_spans_iob1_and_dropped():
    d = Counter()
    tags = ['I-LOC', 'I-LOC', 'O', 'I-PER', 'B-PER', 'I-ORG', 'I-ORG', 'B-LOC']
    assert E.bio_spans(tags, ['LOC'], ['PER'], d) == [(0, 2, 'LOC'), (3, 4, 'PER'), (4, 5, 'PER'), (7, 8, 'LOC')]
    assert d == {'ORG': 1}


def test_ids_and_overlap():
    joint, loc, per, acc = E.ids_from_spans(5, [(0, 2, 'LOC'), (1, 3, 'PER'), (3, 5, 'PER')])
    assert joint == [1, 2, 0, 3, 4]         # the overlapping second span is dropped
    assert loc == [1, 2, 0, 0, 0] and per == [0, 0, 0, 1, 2]
    assert acc == [(0, 2, 'LOC'), (3, 5, 'PER')]


def test_make_record_text_and_char_spans():
    r = E.make_record(rid='x', source='s', subset='a', split='train', lang='fr', tokens=['Le', 'port', 'de', 'La', 'Rochelle', '.'],
                      bio=['O', 'O', 'O', 'B-LOC', 'I-LOC', 'O'], no_space_after=[False, False, False, False, True, True])
    assert r['text'] == 'Le port de La Rochelle.'
    assert r['spans'] == [{'start': 11, 'end': 22, 'label': 'LOC'}]
    assert r['text'][11:22] == 'La Rochelle'
    assert r['ner_tags'] == [0, 0, 0, 1, 2, 0] and r['n_loc'] == 1 and r['n_per'] == 0
    assert r['loc_valid'] and r['per_valid']


def test_invalid_label_view_is_blanked():
    r = E.make_record(rid='x', source='s', subset='a', split='train', lang='en', tokens=['John', 'in', 'Paris'], bio=['B-PER', 'O', 'B-LOC'], per_valid=False)
    assert r['ner_tags_per'] == [0, 0, 0] and r['per_valid'] is False and r['ner_tags_loc'] == [0, 0, 1]


def test_chunk_never_cuts_inside_an_entity():
    tags = ['O'] * 3 + ['B-LOC', 'I-LOC', 'I-LOC'] + ['O'] * 6
    for a, b in E.chunk(12, tags, None, limit=4):
        assert b == 12 or E.split_tag(tags[b])[0] != 'I'
    assert E.chunk(5, None, None, limit=10) == [(0, 5)]


def test_suggested_split_is_deterministic_and_balanced():
    ks = [f'k{i}' for i in range(2000)]
    s = Counter(E.suggested_split(k) for k in ks)
    assert all(E.suggested_split(k) == E.suggested_split(k) for k in ks[:50])
    assert 1500 < s['train'] < 1700 and 100 < s['dev'] < 300 and 100 < s['test'] < 300


def test_out_writes_and_counts(tmp_path):
    o = E.Out(tmp_path, 'demo', 'CC0', 'open')
    r = E.make_record(rid='1', source='demo', subset='x', split='train', lang='de', tokens=['Wien'], bio=['B-LOC'])
    o.write(r)
    rows = o.close()
    assert rows[0]['LOC'] == 1 and rows[0]['licence_flag'] == 'open' and rows[0]['quality'] == 'gold'
    line = json.loads((tmp_path / 'demo' / 'train.jsonl').read_text())
    assert line['licence'] == 'CC0' and line['ner_tags'] == [1]


def test_read_hipe_sentences_and_flags(tmp_path):
    p = tmp_path / 'x.tsv'
    p.write_text('TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n'
                 '# hipe2022:document_id = d1\n# hipe2022:date = 1923-01-15\n# hipe2022:language = fr\n'
                 'Paris\tB-loc\t_\tB-loc.adm.town\t_\t_\tO\t_\t_\tNoSpaceAfter\n'
                 '.\tO\t_\tO\t_\t_\tO\t_\t_\tEndOfSentence\n'
                 'Jean\tB-pers\t_\t_\t_\t_\tO\t_\t_\t_\n'
                 'Dupont\tI-pers\t_\t_\t_\t_\tO\t_\t_\t_\n', encoding='utf-8')
    c = load_convert()
    sents = list(c.read_hipe(p))
    assert [s['tokens'] for s in sents] == [['Paris', '.'], ['Jean', 'Dupont']]
    assert sents[0]['nsa'] == [True, False] and sents[0]['doc'] == 'd1' and c.hipe_year(sents[0]['meta']) == 1923
    spans = E.bio_spans(sents[1]['tags'], ['loc'], ['pers'])
    assert spans == [(0, 2, 'PER')]


def test_char_to_tok_spans_and_pile_style_matching():
    c = load_convert()
    toks, st, en = c.tokenize("Jean d'Arc went to Rouen, France.")
    assert toks[:3] == ['Jean', "d'Arc", 'went']
    ts, cut = c.char_to_tok_spans(st, en, [(0, 10, 'PER'), (19, 24, 'LOC')])
    assert ts == [(0, 2, 'PER'), (4, 5, 'LOC')] and cut == 0
    assert c.spans_to_bio(len(toks), ts)[:2] == ['B-PER', 'I-PER']


def test_nested_layers_keep_their_own_views():
    # a place inside a person name: the LOC view and the PER view are independent, the joint view keeps the longer span
    r = E.make_record(rid='x', source='s', subset='a', split='train', lang='la', tokens=['Guido', 'de', 'Levies', 'dedit'],
                      tok_spans=[(0, 3, 'PER'), (2, 3, 'LOC')])
    assert r['ner_tags'] == [3, 4, 4, 0]
    assert r['ner_tags_per'] == [1, 2, 2, 0] and r['ner_tags_loc'] == [0, 0, 1, 0]
    assert r['n_loc'] == 1 and r['n_per'] == 1 and len(r['spans']) == 2


def test_medieval_charters_reader(tmp_path):
    import zipfile
    zp = tmp_path / 'medieval_charters' / 'raw'
    zp.mkdir(parents=True)
    rows = 'Ego\tO\tO\tlat\tCBMA_1\nIohannes\tB-PERS\tO\tlat\tCBMA_1\ncomes\tO\tO\tlat\tCBMA_1\nCabilonensis\tO\tB-LOC\tlat\tCBMA_1\n.\tO\tO\tlat\tCBMA_1\n'
    with zipfile.ZipFile(zp / 'Multilingual_NER_medieval_charters.zip', 'w') as z:
        for n in ('train_set_all_17_03_2022.txt', 'val_set_all_17_03_2022.txt', 'test_set_all_17_03_2022.txt', 'Nesle_Eslonza.txt'):
            z.writestr('datasets/' + n, rows)
    spec = importlib.util.spec_from_file_location('ner_external_medieval', SCRIPT.parent / 'ner_external_medieval.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = E.Out(tmp_path, 'medieval_charters', 'CC BY 4.0', 'open')
    m.convert(tmp_path, out)
    rows_ = out.close()
    assert {r['split'] for r in rows_} == {'train', 'dev', 'test', 'test_ood'}
    rec = json.loads((tmp_path / 'medieval_charters' / 'train.jsonl').read_text().splitlines()[0])
    assert rec['lang'] == 'la' and rec['subset'] == 'CBMA' and rec['n_per'] == 1 and rec['n_loc'] == 1
