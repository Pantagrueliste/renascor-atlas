"""v2.2: the 'wrong' verdicts of the manual review are hard overrides; the editor's uncertain identifications are suggestions."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from ratlas import manual_overrides as MO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def _script():
    spec = importlib.util.spec_from_file_location('make_hard_overrides', ROOT / 'scripts' / 'make_hard_overrides.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# the 13 entities the v2.1 homonym gate re-accepted although the manual review had judged the id wrong (sample check of v2.1)
RE_ACCEPTED = {
    ('bullinger', 'bullinger:l983', 'geonames', '8981663'): 'Marignano',
    ('bullinger', 'bullinger:l1577', 'geonames', '2946699'): 'Boll',
    ('bullinger', 'bullinger:l1881', 'geonames', '2922518'): 'Gärnitz',
    ('bullinger', 'bullinger:l2140', 'geonames', '2956049'): 'Anhausen',
    ('bullinger', 'bullinger:l3098', 'geonames', '2867614'): 'Münden',
    ('bullinger', 'bullinger:l3119', 'geonames', '2997652'): 'Longueville',
    ('bullinger', 'bullinger:l3355', 'geonames', '2866073'): 'Neuburg',
    ('religionsfrieden_quellenedition', 'religionsfrieden_quellenedition:e000001_religionsvergleich_kleve_mark_ravensberg_juelich_berg_1672_73#kalkar',
     'geonames', '2893838'): 'Kalkar',
    ('religionsfrieden_quellenedition', 'religionsfrieden_quellenedition:e000001_religionsvergleich_kleve_mark_ravensberg_juelich_berg_1672_73#muelheim_am_rhein',
     'geonames', '2867842'): 'Mülheim am Rhein',
    ('religionsfrieden_quellenedition', 'religionsfrieden_quellenedition:e000001_patente_di_gratia_di_pinerolo_1655_ital#fenile',
     'geonames', '8962768'): 'Fenile',
    ('religionsfrieden_quellenedition', 'religionsfrieden_quellenedition:e000001_religionsvergleich_kleve_mark_ravensberg_juelich_berg_1672_73#brueggen',
     'geonames', '2943337'): 'Brüggen',
    ('religionsfrieden_quellenedition', 'religionsfrieden_quellenedition:e000001_religionsvergleich_kleve_mark_ravensberg_juelich_berg_1672_73#juechen',
     'geonames', '2894636'): 'Jüchen',
    ('hainhofer_reisen', 'hainhofer_reisen:landsberg', 'geonames', '2880509'): 'Landsberg am Lech',
}
UNSURE = {  # 'undecidable' with the editor's "Identifizierung unsicher": suggested
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:kloster_zinna', 'geonames', '2887390'),
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:boekel', 'geonames', '2758758'),
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:langenberg', 'geonames', '2881177'),
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:luedersdorf', 'geonames', '2875434'),
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:neuhaus_2', 'geonames', '2865154'),
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:rohrbach', 'geonames', '2845711'),
    ('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:steinheim_main', 'geonames', '2827987'),
}


def _rows():
    ov = MO.load_id_overrides(DATA)
    return {(r.coll_slug, r.entity_key, r.scheme, r.id): r.action for r in ov.itertuples(index=False)}


def test_the_thirteen_re_accepted_wrong_entities_are_rejected_by_hard_overrides():
    rows = _rows()
    for k, name in RE_ACCEPTED.items():
        assert rows.get(k) in ('reject', 'suggest'), name
    assert sum(1 for k in RE_ACCEPTED if rows.get(k) == 'reject') >= 11          # the two the gate had only suggested may stay suggestions


def test_uncertain_identifications_of_the_editor_are_suggestions():
    rows = _rows()
    for k in UNSURE:
        assert rows.get(k) == 'suggest', k


def test_rohrbach_of_the_elector_is_in_heidelberg():
    t = MO.read_tsv(DATA / 'coord_overrides.tsv')
    r = t[t.entity_key == 'tagebuecher_christian_ii_anhalt:rohrbach'].iloc[0]
    assert abs(float(r.lat) - 49.38) < 0.05 and abs(float(r.lon) - 8.69) < 0.05


def test_builder_turns_wrong_into_reject_or_suggest_and_unsure_into_suggest_and_is_idempotent():
    m = _script()
    review = pd.DataFrame([
        dict(case_id='A1', group='gate_homonym', kind='place', coll_slug='c', entity_key='c:k1', scheme='geonames', id='1', verdict='wrong',
             evidence='x', outcome='true_reject'),
        dict(case_id='A2', group='disco_cert', kind='place', coll_slug='c', entity_key='c:k2', scheme='wikidata', id='Q2', verdict='wrong',
             evidence='x', outcome='true_suggest'),
        dict(case_id='A3', group='gate_homonym', kind='place', coll_slug='c', entity_key='c:k3', scheme='geonames', id='3', verdict='undecidable',
             evidence='Not decidable (editor marks "Identifizierung unsicher")', outcome='undecidable'),
        dict(case_id='A4', group='gate_homonym', kind='place', coll_slug='c', entity_key='c:k4', scheme='geonames', id='4', verdict='undecidable',
             evidence='thin context', outcome='undecidable'),
        dict(case_id='A5', group='gate_homonym', kind='place', coll_slug='c', entity_key='c:gone', scheme='geonames', id='5', verdict='wrong',
             evidence='x', outcome='true_reject'),
        dict(case_id='A6', group='bdpn_placed', kind='place', coll_slug='c', entity_key='c:k6', scheme='', id='', verdict='wrong', evidence='x', outcome=''),
    ])
    pairs = pd.DataFrame([dict(coll_slug='c', entity_key='c:k1', scheme='geonames', id='1', status='accepted'),
                          dict(coll_slug='c', entity_key='c:k2', scheme='wikidata', id='Q2', status='suggested'),
                          dict(coll_slug='c', entity_key='c:k3', scheme='geonames', id='3', status='rejected'),
                          dict(coll_slug='c', entity_key='c:k4', scheme='geonames', id='4', status='rejected')])
    empty = pd.DataFrame(columns=MO.COLS)
    new, skipped = m.build(review, pairs, empty)
    got = {r.case_id: r.action for r in new.itertuples(index=False)}
    assert got == {'A1': 'reject', 'A2': 'suggest', 'A3': 'suggest'} and skipped == ['A5']
    MO.validate(new)
    new2, _ = m.build(review, pairs, new)
    assert new2.empty


def test_coordinate_overrides_of_a_key_with_a_hash_are_read():
    """The Religionsfrieden keys are 'file#name': a '#' inside a line is not a comment."""
    from ratlas.reconcile_build import load_overrides
    o = load_overrides(DATA).set_index('entity_key')
    k = [x for x in o.index if x.endswith('#brueggen')]
    assert len(k) == 1 and abs(o.loc[k[0], 'ov_lat'] - 51.2405) < 1e-6
    assert o['ov_drop'].sum() == 3
