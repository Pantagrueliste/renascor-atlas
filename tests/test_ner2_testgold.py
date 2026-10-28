"""NER v2 test_v2 gold: judgement layouts of the six judges, decisions per occurrence, application to a passage."""
from ratlas.ner import ner2_testgold as G


def row(**kw):
    base = {'kind': 'pred_not_gold', 'edition': 'ed', 'surface': 'london', 'n': 3}
    base.update(kw)
    return base


def test_canon_row_layouts():
    # elan: group_decision + per_context
    c = G.canon_row(row(group_decision='mixed', decision_unseen_occurrences='error',
                        per_context=[{'id': 'a', 's': 1, 'e': 2, 'span': 'x', 'decision': 'add'}], contexts=[{'id': 'a', 's': 1, 'e': 2}]))
    assert c['default'] == 'error' and c['mixed'] and [(o['id'], o['decision']) for o in c['occ']] == [('a', 'add')]
    # koenigsfelden: verdict, contexts with verdict
    c = G.canon_row(row(verdict='add', mixed=True, contexts=[{'id': 'a', 's': 1, 'e': 2, 'verdict': 'error', 'override': True}]))
    assert c['default'] == 'add' and c['occ'][0]['decision'] == 'error'
    # kaiserin: mixed without a default for the unseen occurrences -> ambiguous
    c = G.canon_row(row(decision='mixed', contexts=[{'id': 'a', 's': 1, 'e': 2, 'decision': 'add'}]))
    assert c['default'] == 'ambiguous'
    # reed: unseen_decision in a mixed group, and a fix span
    c = G.canon_row(row(decision='mixed', occurrences=[{'id': 'a', 's': 1, 'e': 2, 'span': 'x', 'decision': 'error'}], unseen_decision='add'))
    assert c['default'] == 'add'
    c = G.canon_row(row(decision='error', fix_span=['Essex'], fix_verdict='add'))
    assert c['default'] == 'error' and c['fixes'] == ['Essex']
    # sozinianische: correct_span per occurrence
    c = G.canon_row(row(decision='error', occurrences=[{'id': 'a', 's': 1, 'e': 2, 'span': 'Acro', 'decision': 'error', 'correct_span': 'Acro ceraunia'}]))
    assert c['occ'][0]['fix'] == ['Acro ceraunia']
    # unknown decision word -> ambiguous, never a silent add
    assert G.canon_row(row(decision='maybe'))['default'] == 'ambiguous'


def occ(i, s, e, text='London'):
    return {'id': i, 's': s, 'e': e, 'cs': s, 'ce': e, 'systems': ['a', 'b'], 'text': text}


def test_iter_decisions_default_explicit_and_text_match():
    j = {('ed', 'pred_not_gold', 'london'): G.canon_row(row(decision='add', mixed=True, contexts=[
        {'id': 'p1', 'span': 'London', 'decision': 'ambiguous'}]))}                  # hainhofer layout: id + span, no word indices
    cands = {('ed', 'london'): [occ('p1', 3, 4), occ('p2', 5, 6)]}
    got = {o['id']: (dec, explicit) for _, _, o, dec, _, explicit in G.iter_decisions(cands, j)}
    assert got == {'p1': ('ambiguous', True), 'p2': ('add', False)}
    # a group nobody judged yields None (kept as a false positive, never masked)
    got = list(G.iter_decisions({('ed', 'paris'): [occ('p1', 0, 1)]}, j))
    assert got[0][3] is None


def test_apply_changes_add_mask_remove():
    editor = [(0, 2), (10, 11)]
    ch = {'add': [{'s': 5, 'e': 6}, {'s': 1, 'e': 3}], 'mask': [{'s': 7, 'e': 9, 'what': 'candidate'}], 'remove': [{'s': 10, 'e': 11}]}
    res = G.apply_changes(editor, ch, 20)
    assert res['gold'] == [(0, 2), (5, 6)]          # (1, 3) touches an editor span and is skipped, (10, 11) is removed
    assert res['mask'] == [7, 8] and len(res['skipped']) == 1 and res['removed'] == [(10, 11)]


def test_find_fix_closest_span():
    rec = {'tokens': ['the', 'Countye', 'of', 'Essex', 'and', 'Essex']}
    assert G._find_fix(rec, 'Essex', 1, 4) == (3, 4)
    assert G._find_fix(rec, 'Countye of Essex', 3, 4) == (1, 4)
    assert G._find_fix(rec, 'Kent', 1, 4) is None
