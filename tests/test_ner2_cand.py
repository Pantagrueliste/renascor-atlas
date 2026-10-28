"""NER v2 candidates: clusters of predicted spans, editor spans excluded, groups by (edition, surface)."""
from __future__ import annotations

import gzip
import json

from ratlas.ner import ner2_cand as C
from ratlas.ner.splits import tokens_and_tags


def rec(i, text, gold, ed='ed1', doc='d1'):
    spans = [{'start': a, 'end': b, 'label': 'LOC', 'tag': 'placeName'} for a, b in gold]
    toks, tags, loc, per, offs = tokens_and_tags(text, spans)
    return {'id': i, 'v1_id': i, 'edition': ed, 'doc': doc, 'lang': 'en', 'text': text, 'tokens': toks, 'token_starts': offs, 'spans': spans,
            'ner_tags_loc': loc, 'loc_valid': True}


def write(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')


def test_norm_surface():
    assert C.norm_surface('Straßbur- g') == C.norm_surface('Straßburg')
    assert C.norm_surface(' Eſſex, ') == 'essex'
    assert C.family('r3_xlmr_large_H') == 'ours' and C.family('zs_flair_ner_en_large') == 'offshelf' and C.family('zs_llm_qwen35_9b') == 'llm'


def test_groups(tmp_path):
    t1 = 'We went to London and then to York by the sea .'
    t2 = 'In London he met York Minster priests .'
    r1, r2 = rec('a/1', t1, [(t1.index('London'), t1.index('London') + 6)]), rec('a/2', t2, [])
    data = tmp_path / 'd.jsonl'
    data.write_text(''.join(json.dumps(r) + '\n' for r in (r1, r2)))

    def tok_idx(r, w):
        return r['tokens'].index(w)

    def pred(r, spans):
        return {'id': r['id'], 'LOC': [{'s': s, 'e': e, 'conf': c} for s, e, c in spans], 'PER': []}

    i1, y1, l2, y2 = tok_idx(r1, 'London'), tok_idx(r1, 'York'), tok_idx(r2, 'London'), tok_idx(r2, 'York')
    write(tmp_path / 'r3_mdeberta_HM.jsonl.gz', [pred(r1, [(i1, i1 + 1, .99), (y1, y1 + 1, .99)]), pred(r2, [(l2, l2 + 1, .99), (y2, y2 + 2, .95)])])
    write(tmp_path / 'r3_mmbert_base_HM.jsonl.gz', [pred(r1, [(i1, i1 + 1, .9), (y1, y1 + 1, .8)]), pred(r2, [(l2, l2 + 1, .9), (y2, y2 + 1, .8)])])
    write(tmp_path / 'zs_flair.jsonl.gz', [pred(r1, [(y1, y1 + 1, .9)]), pred(r2, [(y2, y2 + 1, .9)])])
    files = {p.name[:-9]: p for p in tmp_path.glob('*.jsonl.gz')}
    out, summ = C.build(data, files)
    groups = {(g['kind'], g['surface']): g for g in out['ed1']}
    # York: found in both passages by 3 systems, never tagged; London in passage 1 is the editors' (not a candidate), in passage 2 it is a candidate
    y = groups[('pred_not_gold', 'york')]
    assert y['n'] == 2 and y['n_passages'] == 2 and y['n_systems'] == 3 and y['n_ours'] == 2 and y['n_offshelf'] == 1
    assert y['n_top4_max'] == 2 and y['n_rule_v1'] == 1           # passage 1: two top models, conf .99 ; passage 2: two top models but best conf .95
    assert len(y['contexts']) == 2 and y['contexts'][0]['span'] == 'York'
    lo = groups[('pred_not_gold', 'london')]
    assert lo['n'] == 1 and lo['n_gold_same_surface'] == 1 and lo['contexts'][0]['id'] == 'a/2'
    assert summ['ed1']['clusters_touching_gold'] == 1 and summ['ed1']['gold_touched'] == 1
    assert not any(k[0] == 'gold_not_found' for k in groups)


def test_gold_not_found(tmp_path):
    t = 'The Kingdome of France and Paris .'
    r = rec('b/1', t, [(4, 12), (t.index('Paris'), t.index('Paris') + 5)])
    data = tmp_path / 'd.jsonl'
    data.write_text(json.dumps(r) + '\n')
    p = t.split().index('Paris')
    write(tmp_path / 'zs_x.jsonl.gz', [{'id': 'b/1', 'LOC': [{'s': r['tokens'].index('Paris'), 'e': r['tokens'].index('Paris') + 1, 'conf': 1.0}], 'PER': []}])
    out, _ = C.build(data, {'zs_x': tmp_path / 'zs_x.jsonl.gz'})
    (g,) = [g for g in out['ed1'] if g['kind'] == 'gold_not_found']
    assert g['surface'] == 'kingdome' and g['n'] == 1 and g['n_gold_same_surface'] == 1 and g['tags'] == {'placeName': 1}
