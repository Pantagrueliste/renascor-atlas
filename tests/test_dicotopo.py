"""DicoTopo place names (ratlas.dicotopo): the display-name rule and the table of matched points (decision 4 of 2026-10-28)."""
import pytest

from ratlas import dicotopo as D


def test_case_is_restored_on_capital_headwords():
    assert D.restore_case('SAINT-GERMAIN-EN-LAYE') == 'Saint-Germain-en-Laye'
    assert D.restore_case('BAR-SUR-SEINE') == 'Bar-sur-Seine'
    assert D.restore_case("L'ISLE-ADAM") == "L'Isle-Adam"
    assert D.restore_case('Saint-Dizier') == 'Saint-Dizier'          # already typographic: untouched


def test_a_real_english_exonym_is_kept():
    assert D.choose_label('Bretagne', 'Bretagne', en_label='Brittany', fr_label='Bretagne') == ('Brittany', 'english_exonym')
    assert D.choose_label('BOURGOGNE', None, en_label='Burgundy', fr_label='Bourgogne') == ('Burgundy', 'english_exonym')
    # the English label is the French one (Rouen), or carries a qualifier: no exonym
    assert D.choose_label('ROUEN', 'Rouen', en_label='Rouen', fr_label='Rouen')[1] != 'english_exonym'
    assert D.english_exonym('Lyon, France', 'Lyon') is None


def test_headword_is_the_display_name_otherwise():
    assert D.choose_label('SAINT-GERMAIN-EN-LAYE') == ('Saint-Germain-en-Laye', 'dicotopo_headword')
    assert D.choose_label('Perpignan', 'Perpignan') == ('Perpignan', 'dicotopo_headword')


def test_official_name_replaces_a_headword_that_lost_accents_or_is_older():
    assert D.choose_label('Nimes', 'Nîmes') == ('Nîmes', 'official_name')
    assert D.choose_label('Le Câtelet', 'Le Catelet')[0] == 'Le Catelet'
    assert D.choose_label('Châlons-sur-Marne', 'Châlons-en-Champagne', headword_older=True) == ('Châlons-en-Champagne', 'official_name')
    # today's name is a merged commune: the historical headword stays
    assert D.choose_label('Moret', 'Moret-Loing-et-Orvanne', headword_older=True, keep_headword=True)[0] == 'Moret'
    # the headword as printed, whatever the official name
    assert D.choose_label('Nimes', 'Nîmes', policy='headword')[0] == 'Nimes'


def test_old_forms_are_parsed():
    assert D.parse_forms('Roem (1280) | Rotomagus (350) | Rothome') == [('Roem', 1280), ('Rotomagus', 350), ('Rothome', None)]
    assert D.parse_forms('') == []


def test_the_table_of_matched_points():
    d = D.load()
    assert len(d) == 832 and d['point_id'].is_unique
    assert set(d['reading']) == {'editor', 'model'}
    assert (d['article_id'] != '').all() and (d['name'] != '').all()
    assert d.loc[d['point_id'] == 'row:245', 'article_id'].iloc[0] == 'P13520839' and d.loc[d['point_id'] == 'row:245', 'insee'].iloc[0] == '76540'
    # a model point is found again through the units it holds, not through its id
    m = d[d['reading'] == 'model']
    assert m['prev'].map(lambda x: isinstance(x, list) and len(x) > 0).all()
    assert (d[d['reading'] == 'editor']['prev'].isna()).all()
    # the 49 headwords older than today's name are shown under the official name by default, and as printed under the other policy
    old = d[d['notes'].str.contains('headword_differs_from_official_name')]
    assert len(old) >= 40
    assert D.load(policy='headword').set_index('point_id').loc['row:777', 'label'] == 'Nimes'
    assert d.set_index('point_id').loc['row:777', 'label'] == 'Nîmes'
