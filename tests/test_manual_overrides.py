"""Manual-review overrides: the loader, the application to the gate pairs, the build, and the shipped tables."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratlas import manual_overrides as MO
from ratlas import reconcile_v2 as V
from test_reconcile_v2 import _make

DATA = Path(__file__).resolve().parents[1] / 'data'


def _pair(coll, kind, ek, scheme, pid, qid, status='suggested', reason='name_mismatch', name='x', n=1):
    return dict(coll_slug=coll, kind=kind, entity_key=ek, name=name, scheme=scheme, id=pid, n_mentions=n, n_in_period=n, qid=qid,
                qid_raw=qid, src='wd', level='none', label=None, second_id='', k=1.0, authority_lat=np.nan, authority_lon=np.nan,
                status=status, reason=reason, detail='')


def _ov(*rows):
    cols = MO.COLS
    return MO.validate(pd.DataFrame([dict(zip(cols, r)) for r in rows]))


def test_validate_rejects_malformed_rows():
    good = ('c', 'place', 'c:a', 'geonames', '1', '', 'accept', 'MR1', 'n')
    MO.validate(pd.DataFrame([dict(zip(MO.COLS, good))]))
    for bad in (('c', 'place', 'c:a', 'geonames', '1', '', 'delete', 'MR1', 'n'),          # unknown action
                ('c', 'thing', 'c:a', 'geonames', '1', '', 'accept', 'MR1', 'n'),            # unknown kind
                ('c', 'place', 'c:a', '*', '', '', 'reject', 'MR1', 'n'),                      # * needs a QID
                ('c', 'place', 'c:a', 'geonames', '', '', 'reject', 'MR1', 'n'),               # empty id
                ('c', 'place', 'c:a', 'geonames', '1', 'Qx', 'reject', 'MR1', 'n'),            # malformed qid
                ('c', 'place', 'c:a', 'geonames', '1', '-', 'reject', 'MR1', 'n')):            # '-' only with accept
        with pytest.raises(ValueError):
            MO.validate(pd.DataFrame([dict(zip(MO.COLS, bad))]))
    with pytest.raises(ValueError):                                                            # contradictory rows
        MO.validate(pd.DataFrame([dict(zip(MO.COLS, good)), dict(zip(MO.COLS, ('c', 'place', 'c:a', 'geonames', '1', '', 'reject', 'MR2', 'n')))]))


def test_apply_accept_reject_suggest_and_dash():
    pairs = pd.DataFrame([
        _pair('c', 'place', 'c:a', 'geonames', '1', 'Q10'),                         # suggested -> accepted
        _pair('c', 'place', 'c:b', 'geonames', '2', 'Q20', status='accepted', reason='exact'),   # accepted -> rejected
        _pair('c', 'place', 'c:c', 'geonames', '3', 'Q30', status='rejected', reason='duplicate_item'),   # rejected -> accepted, item dropped
        _pair('c', 'place', 'c:d', 'wikipedia', 'en:X', 'Q40', status='accepted', reason='title'),        # accepted -> suggested
        _pair('c', 'place', 'c:e', 'geonames', '5', 'Q50'),                         # untouched
        _pair('c', 'person', 'c:p', 'gnd', '9', 'Q60'),                             # another kind: untouched by place rows
    ])
    ents = pd.DataFrame({'coll_slug': ['c'] * 5, 'entity_key': ['c:a', 'c:b', 'c:c', 'c:d', 'c:e'], 'name': list('abcde'), 'n_mentions': [1] * 5})
    ov = _ov(('c', 'place', 'c:a', 'geonames', '1', '', 'accept', 'MR1', 'ok'),
             ('c', 'place', 'c:b', 'geonames', '2', '', 'reject', 'MR2', 'wrong'),
             ('c', 'place', 'c:c', 'geonames', '3', '-', 'accept', 'MR3', 'dup item'),
             ('c', 'place', 'c:d', 'wikipedia', 'en:X', '', 'suggest', 'MR4', 'doubt'))
    out, rep = MO.apply_id_overrides(pairs, ov, ents, 'place')
    g = out.set_index('entity_key')
    assert g.loc['c:a', 'status'] == 'accepted' and g.loc['c:a', 'reason'] == 'manual_review' and 'MR1' in g.loc['c:a', 'detail']
    assert g.loc['c:b', 'status'] == 'rejected' and g.loc['c:d', 'status'] == 'suggested'
    assert g.loc['c:c', 'status'] == 'accepted' and g.loc['c:c', 'qid'] is None and g.loc['c:c', 'qid_raw'] == 'Q30'
    assert g.loc['c:e', 'status'] == 'suggested' and g.loc['c:e', 'reason'] == 'name_mismatch'
    assert out[out['kind'] == 'person'].iloc[0]['status'] == 'suggested'
    assert rep['accepted'] == 2 and rep['rejected'] == 1 and rep['suggested'] == 1 and rep['unmatched'] == 0


def test_star_matches_every_pair_that_leads_to_the_item():
    pairs = pd.DataFrame([_pair('c', 'person', 'c:p', 'gnd', '1', 'Q1'), _pair('c', 'person', 'c:p', 'viaf', '2', 'Q1'),
                          _pair('c', 'person', 'c:p', 'wikipedia', 'en:Y', 'Q2')])
    out, _ = MO.apply_id_overrides(pairs, _ov(('c', 'person', 'c:p', '*', '', 'Q1', 'accept', 'MR1', 'n')),
                                   pd.DataFrame({'coll_slug': ['c'], 'entity_key': ['c:p'], 'name': ['p'], 'n_mentions': [1]}), 'person')
    assert list(out['status']) == ['accepted', 'accepted', 'suggested']


def test_a_pair_the_gates_never_made_is_created_only_for_a_known_qid():
    pairs = pd.DataFrame([_pair('c', 'person', 'c:p', 'gnd', '1', 'Q1')])
    ents = pd.DataFrame({'coll_slug': ['c'], 'entity_key': ['c:p'], 'name': ['Pat'], 'n_mentions_all': [4]})
    ov = _ov(('c', 'person', 'c:p', 'wikidata', 'Q7', 'Q7', 'accept', 'MR1', 'replaces the wrong item'),
             ('c', 'person', 'c:q', 'wikidata', 'Q8', 'Q8', 'accept', 'MR2', 'unknown entity'))
    out, rep = MO.apply_id_overrides(pairs, ov, ents, 'person', known_qids={'Q7', 'Q8'})
    new = out[out['scheme'] == 'wikidata'].iloc[0]
    assert new['qid'] == 'Q7' and new['status'] == 'accepted' and new['reason'] == 'manual_review' and new['n_mentions'] == 4 and new['name'] == 'Pat'
    assert rep['created'] == 2          # the second entity is not in the table: the pair exists, the build ignores entities it does not have
    out2, rep2 = MO.apply_id_overrides(pairs, ov, ents, 'person', known_qids={'Q8'})
    assert rep2['unknown_qid'] == 1 and len(out2) == len(pairs) + 1
    out3, rep3 = MO.apply_id_overrides(pairs, _ov(('c', 'person', 'c:p', 'gnd', '404', '', 'reject', 'MR3', 'n')), ents, 'person')
    assert rep3['unmatched'] == 1 and list(out3['status']) == ['suggested']


def test_effective_verify_merges_the_manual_rejections(tmp_path):
    rd, dest = tmp_path / 'rd', tmp_path / 'out'
    rd.mkdir()
    base = tmp_path / 'verify.tsv'
    pd.DataFrame({'coll_slug': ['a'], 'entity_key': ['a:x'], 'rule': ['r1']}).to_csv(base, sep='\t', index=False)
    assert MO.effective_verify(base, rd, dest) == base                       # no manual file: unchanged
    (rd / 'geocode_rejects.tsv').write_text('coll_slug\tentity_key\trule\tnote\nb\tb:y\tmanual_review\twhy\na\ta:x\tmanual_review\tdup\n')
    p = MO.effective_verify(base, rd, dest)
    v = pd.read_csv(p, sep='\t')
    assert set(zip(v['coll_slug'], v['entity_key'])) == {('a', 'a:x'), ('b', 'b:y')} and v.set_index('entity_key').loc['a:x', 'rule'] == 'r1'
    assert MO.effective_verify(None, rd, dest) is not None


def test_build_applies_the_overrides(tmp_path):
    v2, rd = _make(tmp_path)
    out = tmp_path / 'out'
    # Luther's GND rejected by hand; Johann Casimir (a mismatch in the fixture) accepted by hand; Spain gets a QID that replaces the wrong one
    (rd / 'id_overrides.tsv').write_text(
        '# test\n' + '\t'.join(MO.COLS) + '\n'
        + '\t'.join(['colla', 'person', 'colla:luther', 'gnd', '118575449', '', 'reject', 'MRT1', 'test reject']) + '\n'
        + '\t'.join(['collb', 'person', 'collb:jc', 'viaf', '1', '', 'accept', 'MRT2', 'test accept']) + '\n'
        + '\t'.join(['colla', 'place', 'colla:spain', 'tgn', '7002327', '', 'reject', 'MRT3', 'wrong Getty id']) + '\n'
        + '\t'.join(['colla', 'place', 'colla:spain', 'wikidata', 'Q220', 'Q220', 'accept', 'MRT3', 'replacement']) + '\n')
    (rd / 'geocode_rejects.tsv').write_text('coll_slug\tentity_key\trule\tnote\ncolla\tsurf:galli\tmanual_review\tno\n')
    gc = pd.DataFrame({'coll_slug': ['colla', 'collb'], 'entity_key': ['surf:galli', 'surf:france'], 'name': ['Galli', 'France'],
                       'gc_lat': [46.6, 46.6], 'gc_lon': [2.5, 2.5], 'gc_source': ['name_exonym'] * 2, 'gc_conf': ['medium'] * 2,
                       'gc_radius_km': [350.0, 350.0], 'gc_cand': ['exonym:France'] * 2, 'gc_cand_name': ['France'] * 2,
                       'gc_evidence': ['exonym'] * 2})
    gcp = tmp_path / 'geocode_places.parquet'
    gc.to_parquet(gcp)
    rep = V.build(v2, rd, out, geocode=gcp)
    pe = pd.read_parquet(out / 'persons_local_global.parquet').set_index('entity_key')
    assert pd.isna(pe.loc['colla:luther', 'qid'])                          # accepted by the gates, rejected by hand
    assert pe.loc['collb:jc', 'qid'] == 'Q719626'                          # suggested by the gates, accepted by hand
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index('entity_key')
    assert geo.loc['colla:spain', 'qid'] == 'Q220'
    assert geo.loc['surf:galli', 'coord_source'] == 'none' and geo.loc['surf:france', 'coord_source'] == 'name_exonym'
    lg = pd.read_csv(out / 'gate_log.tsv', sep='\t')
    assert ((lg['reason'] == 'manual_review') & (lg['entity_key'] == 'colla:luther') & (lg['status'] == 'rejected')).any()
    assert rep['manual_overrides']['rows'] == 4 and rep['manual_overrides']['place']['created'] == 1


def test_shipped_tables_are_well_formed():
    ov = MO.load_id_overrides(DATA)
    assert len(ov) > 200 and set(ov['action']) <= set(MO.ACTIONS)
    assert ov['case_id'].str.match(r'^(MR\d{4}|V22-\d{2})$').all()      # v2.2 adds V22-nn for the sample check's own cases
    # every created pair names an item (the build skips an unknown one but a typo must not ship)
    new = ov[(ov['scheme'] == 'wikidata') & (ov['action'] == 'accept') & ~ov['qid'].isin(['', '-'])]
    assert (new['id'] == new['qid']).all() and new['id'].str.match(r'^Q[1-9]\d*$').all()
    # no pair is both accepted and rejected through two spellings of the key
    k = ov.groupby(['coll_slug', 'kind', 'entity_key', 'scheme', 'id'])['action'].nunique()
    assert (k == 1).all()
    gr = MO.read_tsv(DATA / 'geocode_rejects.tsv')
    assert {'coll_slug', 'entity_key', 'rule'} <= set(gr.columns) and not gr.duplicated(['coll_slug', 'entity_key']).any()
    co = MO.read_tsv(DATA / 'coord_overrides.tsv')
    assert not co.duplicated(['coll_slug', 'entity_key']).any()
    kept = co[co['lat'].str.strip() != '-']                                     # v2.2: '-' drops the register point (both columns)
    assert pd.to_numeric(kept['lat']).between(-90, 90).all() and pd.to_numeric(kept['lon']).between(-180, 180).all()
    assert ((co['lat'].str.strip() == '-') == (co['lon'].str.strip() == '-')).all()


def test_the_confirmed_corrections_are_there():
    ov = MO.load_id_overrides(DATA)
    an = ov[ov['coll_slug'] == 'tagebuecher_christian_ii_anhalt']
    # Mehmed IV carried the GND of Ibrahim I: rejected and replaced by Mehmed IV (Q170514)
    mehmed = an[an['entity_key'].str.endswith('sultan_mehmed_iv')]
    assert set(zip(mehmed['action'], mehmed['qid'])) == {('reject', 'Q168776'), ('accept', 'Q170514')}
    # the Shuisky dynasty is not the Romanov dynasty
    assert ((an['entity_key'].str.endswith('schuiski_dynastie')) & (an['action'] == 'reject') & (an['qid'] == 'Q112707')).any()
    # the grand_tour register swaps two GND ids (Gyldenløve / Gyldenstolpe)
    gt = ov[ov['coll_slug'] == 'grand_tour']
    assert {'Q70892', 'Q4346653'} <= set(gt.loc[gt['action'] == 'accept', 'qid'])
    # the homonym gate rejected Endorf, the village of the Quenstedt quarter in the Anhalt diary
    assert ((an['entity_key'].str.endswith(':endorf')) & (an['action'] == 'accept') & (an['id'] == '2930369')).any()
    # a disco_sonnets namesake accepted by the gates (the conquistador, 1500-1542, for a poet of the 17th century)
    assert ((ov['coll_slug'] == 'disco_sonnets') & (ov['action'] == 'reject') & (ov['id'] == '23382263')).sum() == 2


def test_coordinate_override_with_a_dash_drops_the_register_point_that_came_from_a_wrong_id(tmp_path):
    """v2.2: Bullinger's register coordinates of Gärnitz, Longueville and Neuburg are the GeoNames points of the wrong homonyms (hard overrides reject the ids); a row
    of coord_overrides.tsv with lat = '-' drops the register point; a row with coordinates replaces it (Marignano = Melegnano)."""
    from test_reconcile_v2 import _entity, _make, _mention
    A = 'colla'
    rows = [_mention(A, 'place', 'Gärnitz', key='colla:g', reg=True), _mention(A, 'place', 'Marignano', key='colla:m', reg=True),
            _mention(A, 'place', 'Rest', key='colla:r', reg=True)]
    ents = [_entity(A, 'place', 'colla:g', 'Gärnitz', 51.28, 12.23), _entity(A, 'place', 'colla:m', 'Marignano', 44.87, 9.89), _entity(A, 'place', 'colla:r', 'Rest', 50.0, 10.0)]
    v2, rd = _make(tmp_path, rows, ents)
    (rd / 'coord_overrides.tsv').write_text('coll_slug\tentity_key\tlat\tlon\tnote\ncolla\tcolla:g\t-\t-\tdropped\ncolla\tcolla:m\t45.3578\t9.3236\tMelegnano\n')
    out = tmp_path / 'out'
    V.build(v2, rd, out)
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index('entity_key')
    assert geo.loc['colla:g', 'coord_source'] == 'none' and pd.isna(geo.loc['colla:g', 'lat'])
    assert geo.loc['colla:m', 'coord_source'] == 'register' and abs(geo.loc['colla:m', 'lat'] - 45.3578) < 1e-6
    assert geo.loc['colla:r', 'lat'] == 50.0
