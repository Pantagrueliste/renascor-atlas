"""Places inside titles (ratlas.titles): the detector, the lists of removed mentions and the ecclesiastical switch (decision 1 of 2026-10-28)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from ratlas import titles as T

ATLAS = Path(__file__).resolve().parents[2] / 'RQ' / 'derived' / 'atlas' / 'v2.4'
needs_atlas = pytest.mark.skipif(not (ATLAS / 'mentions.parquet').exists(), reason='atlas v2.4 tables not here')


def cat(left, surface, right='', lang=None):
    r = T.detect(left, surface, right, lang)
    return r and r['cat']


@pytest.mark.parametrize('left,surface,lang,expected', [
    ('the death of the King of ', 'France', 'en', 'royal'),
    ('le roy de ', 'France', 'fr', 'royal'),
    ('rex ', 'Franciae', 'la', 'royal'),
    ('il re di ', 'Francia', 'it', 'royal'),
    ('el rey de ', 'Francia', 'es', 'royal'),
    ('den coninck van ', 'Spanjen', 'nl', 'royal'),
    ('the Sultan of ', 'Turkey', 'en', 'royal'),
    ('the Duke of ', 'Anjou', 'en', 'noble'),
    ('my Lord of ', 'Salisbury', 'en', 'noble'),
    ('the Prince of ', 'Orange', 'en', 'noble'),
    ('monseigneur le comte de ', 'Flandre', 'fr', 'noble'),
    ('der Herzog von ', 'Sachsen', 'de', 'noble'),
    ('dux ', 'Mediolani', 'la', 'noble'),
    ('il duca di ', 'Milano', 'it', 'noble'),
    ('Archbishop of ', 'York', 'en', 'ecclesiastical'),
    ("l'evesque de ", 'Paris', 'fr', 'ecclesiastical'),
    ('episcopus ', 'Wintoniensis', 'la', 'ecclesiastical'),
    ('Erzbischoff zu ', 'Cöln', 'de', 'ecclesiastical'),
])
def test_places_inside_titles_are_found(left, surface, lang, expected):
    assert cat(left, surface, lang=lang) == expected


@pytest.mark.parametrize('left,surface,lang', [
    ('the kingdom of ', 'France', 'en'),
    ('in the duchy of ', 'Milan', 'en'),
    ('he went to ', 'York', 'en'),
    ('his journey to ', 'Salisbury', 'en'),
    ('the city of ', 'Rome', 'en'),
    ('le royaume de ', 'France', 'fr'),
    ('in the kingdom of ', 'Naples', 'en'),
])
def test_domains_and_journeys_stay_places(left, surface, lang):
    assert cat(left, surface, lang=lang) is None


def test_the_three_categories_are_separate_and_the_lexicon_matches_the_data_file():
    assert T.removed_categories(True) == ('royal', 'noble', 'ecclesiastical')
    assert T.removed_categories(False) == ('royal', 'noble')
    f = json.loads((T.DATA / 'title_rules.json').read_text('utf-8'))
    r = T.rules_json()
    assert f['categories'] == r['categories'] == ['royal', 'noble', 'ecclesiastical']
    assert f['titles'] == r['titles'] and f['weak_titles'] == r['weak_titles']
    assert sum(len(v) for byl in r['titles'].values() for v in byl.values()) > 500


def test_editor_list_and_the_ecclesiastical_switch():
    all_ = T.load_editor_exclusions(ecclesiastical=True)
    no_ecc = T.load_editor_exclusions(ecclesiastical=False)
    assert len(all_) == 7735 and all_['row'].is_unique
    assert set(all_['category']) == {'royal', 'noble', 'ecclesiastical'}
    assert set(no_ecc['category']) == {'royal', 'noble'}
    assert len(all_) - len(no_ecc) == int((all_['category'] == 'ecclesiastical').sum()) > 2000
    assert T.ECCLESIASTICAL_REMOVED is False and len(T.load_editor_exclusions()) == len(no_ecc)


def test_model_list_counts_are_consistent_and_the_switch_keeps_the_see():
    on = T.load_model_exclusions(ecclesiastical=True)
    off = T.load_model_exclusions(ecclesiastical=False)
    assert (on['n_excluded'] <= on['n_total']).all() and not on.duplicated(['src', 'doc_id', 'lang', 'key']).any()
    assert set(on['src']) == {'ml', 'mles', 'ml3'}
    assert on['n_excluded'].sum() == 21218 and len(on) == 10016
    assert off['n_excluded'].sum() == int((on['n_royal'] + on['n_noble']).sum()) < on['n_excluded'].sum()


def test_keys_are_checked_against_the_mentions_table():
    excl = pd.DataFrame({'row': [1, 3], 'doc_id': ['b', 'd'], 'surface': ['Y', 'W']})
    assert T.check_editor_keys(excl, ['a', 'b', 'c', 'd'], ['X', 'Y', 'Z', 'W']) == 2
    with pytest.raises(ValueError, match='do not match'):
        T.check_editor_keys(excl, ['a', 'b', 'c', 'd'], ['X', 'Y', 'Z', 'other'])
    with pytest.raises(ValueError, match='outside'):
        T.check_editor_keys(excl, ['a', 'b'], ['X', 'Y'])


@needs_atlas
def test_editor_keys_match_the_real_table():
    import pyarrow.parquet as pq
    t = pq.read_table(ATLAS / 'mentions.parquet', columns=['doc_id', 'surface'])
    assert T.check_editor_keys(T.load_editor_exclusions(ecclesiastical=True), t['doc_id'].to_pandas(), t['surface'].to_pandas()) == 7735


def _load_script(name):
    import importlib.util
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_titles_are_subtracted_from_the_document_counts_of_the_layer():
    B = _load_script('ner2_layer_v3_build')
    U = pd.DataFrame({'run': ['v1', 'es', 'v3'], 'lang': ['en', 'es', 'de'], 'key': ['rome', 'roma', 'rom']}, index=[0, 1, 2])
    D = pd.DataFrame({'uid': [0, 0, 1, 2], 'doc_id': ['d1', 'd2', 'd1', 'd3'], 'n': [5, 2, 1, 4]})
    ex = pd.DataFrame({'src': ['ml', 'ml', 'mles', 'ml3'], 'doc_id': ['d1', 'd2', 'd1', 'd3'], 'lang': ['en', 'en', 'es', 'de'], 'key': ['rome', 'rome', 'roma', 'rom'],
                       'n_excluded': [2, 2, 1, 1]})
    out, rep = B.subtract_titles(D, U, ex)
    assert out.to_dict('records') == [{'uid': 0, 'doc_id': 'd1', 'n': 3}, {'uid': 2, 'doc_id': 'd3', 'n': 3}]      # a form with nothing left leaves the document
    assert rep == {'lines_matched': 4, 'mentions_removed': 6}
