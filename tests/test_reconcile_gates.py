import json

import numpy as np
import pandas as pd
import pyarrow as pa

from ratlas import reconcile_gates as RG
from ratlas.model import auth_to_arrow


def _wd(qid, labels, classes=('Q5',), aliases=None, lat=None, lon=None, birth=None, death=None, redirect=None, dup=None,
        occupations=()):
    return dict(qid=qid, missing=False, redirect_to=redirect, dup_of=dup, classes=list(classes), occupations=list(occupations),
                labels=json.dumps(labels), aliases=json.dumps(aliases or {}), lat=lat, lon=lon, n_coords=int(lat is not None),
                country_qid=None, country_iso=None, birth=birth, birth_prec=9 if birth else None, death=death,
                death_prec=9 if death else None, n_claims=10)


def _auth(tmp_path, wd, mapping, classes=None, gnd=None, gn=None, same=None):
    d = tmp_path / 'rd'
    d.mkdir(exist_ok=True)
    pd.DataFrame([{'scheme': s, 'id': i, 'qids': '|'.join(q)} for (s, i), q in mapping.items()],
                 columns=['scheme', 'id', 'qids']).to_csv(d / 'wd_map.tsv', sep='\t', index=False)
    pd.DataFrame(wd).to_parquet(d / 'wd_entities.parquet', index=False)
    cl = {'Q5': 'human', 'Q515': 'city', 'Q4167410': 'Wikimedia disambiguation page', 'Q783794': 'company', 'Q164950': 'dynasty',
          'Q202444': 'given name', 'Q6256': 'country', 'Q1348305': 'biographical article', 'Q4271324': 'mythical character'}
    cl.update(classes or {})
    pd.DataFrame({'qid': list(cl), 'label_en': list(cl.values())}).to_csv(d / 'wd_class_labels.tsv', sep='\t', index=False)
    pd.DataFrame(gnd or [], columns=['gnd', 'name', 'variants', 'types', 'birth', 'death', 'qids', 'info']).to_parquet(d / 'gnd_records.parquet', index=False)
    pd.DataFrame(gn or [], columns=['gid', 'name', 'names', 'lat', 'lon', 'pop', 'cc', 'fcode', 'fclass']).to_parquet(d / 'gn_ids.parquet', index=False)
    pd.DataFrame(same or [], columns=['key', 'gid', 'lat', 'lon', 'pop', 'cc', 'fcode', 'names']).to_parquet(d / 'gn_samename.parquet', index=False)
    return RG.AuthData(d)


def _vote(coll, kind, ek, scheme, idv, n=3, n_hi=None, reg=False, sug=False):
    return dict(coll_slug=coll, kind=kind, entity_key=ek, scheme=scheme, id=idv, n=n, n_ip=n, n_hi=n if n_hi is None else n_hi,
                reg=reg, sug=sug)


def _ent(coll, ek, name, variants=(), person_class=None, note_type=None, birth=None, death=None, last_year=1600, lat=None, lon=None,
         n_all=3):
    return dict(coll_slug=coll, entity_key=ek, name=name, variants=list(variants), person_class=person_class, note_type=note_type,
                birth=birth, death=death, last_year=last_year, lat=lat, lon=lon, n_mentions_all=n_all)


def _run(tmp_path, wd, mapping, votes, ents, kind, **kw):
    auth = _auth(tmp_path, wd, mapping, **kw)
    out = RG.run(pd.DataFrame(votes), pd.DataFrame(ents), auth, kind, log=lambda m: None)['pairs']
    return out, auth


def test_person_gates(tmp_path):
    C = 'cavriana'
    wd = [_wd('Q9554', {'en': 'Martin Luther', 'de': 'Martin Luther'}, birth='+1483-11-10T00:00:00Z'),
          _wd('Q719626', {'en': 'George, Duke of Saxony'}, birth='+1471-08-27T00:00:00Z'),
          _wd('Q27583572', {'de': 'Kirsten, Michael'}, classes=['Q1348305']),
          _wd('Q111', {'en': 'Wilhelm Abc'}, classes=['Q4167410']),
          _wd('Q222', {'en': 'Georg Hohenlohe'}, classes=['Q164950']),
          _wd('Q333', {'en': 'Jan Novak'}, birth='+1794-01-01T00:00:00Z'),
          _wd('Q444', {'en': 'Hans Müller'}), _wd('Q555', {'en': 'Hans Müller'})]
    mapping = {('gnd', '118575449'): ['Q9554'], ('viaf', '1'): ['Q719626'], ('gnd', '5'): ['Q27583572'],
               ('gnd', '6'): ['Q111'], ('gnd', '7'): ['Q222'], ('viaf', '8'): ['Q333'], ('gnd', '9'): ['Q444'], ('viaf', '9'): ['Q444']}
    votes = [_vote(C, 'person', 'luther', 'gnd', '118575449'), _vote(C, 'person', 'pfalz', 'viaf', '1'),
             _vote(C, 'person', 'kirsten', 'gnd', '5'), _vote(C, 'person', 'abc', 'gnd', '6'),
             _vote(C, 'person', 'hohenlohe', 'gnd', '7'), _vote(C, 'person', 'novak', 'viaf', '8'),
             _vote(C, 'person', 'hm', 'gnd', '9'), _vote(C, 'person', 'hm', 'viaf', '9'),
             _vote(C, 'person', 'lowcert', 'gnd', '118575449', n=2, n_hi=0),
             _vote(C, 'person', 'noid', 'viaf', '777'),
             _vote(C, 'person', 'birthplace', 'geonames', '2988507')]
    ents = [_ent(C, 'luther', 'Martin Luther'), _ent(C, 'pfalz', 'Johann Casimir von der Pfalz', ['Duca di Duponte']),
            _ent(C, 'kirsten', 'Kirsten, Michael'), _ent(C, 'abc', 'Wilhelm Abc'),
            _ent(C, 'hohenlohe', 'Hohenlohe', person_class='dynasty'), _ent(C, 'novak', 'Jan Novak', last_year=1572),
            _ent(C, 'hm', 'Hans Mueller', ['Muller']), _ent(C, 'lowcert', 'Martin Luther'), _ent(C, 'noid', 'Somebody'),
            _ent(C, 'birthplace', 'Somebody')]
    pairs, auth = _run(tmp_path, wd, mapping, votes, ents, 'person')

    def st(ek, scheme=None):
        x = pairs[(pairs.entity_key == ek) & ((pairs.scheme == scheme) if scheme else True)]
        return list(zip(x['status'], x['reason']))
    assert st('luther')[0][0] == 'accepted'
    assert st('pfalz') == [('suggested', 'name_mismatch')]               # cavriana: wrong VIAF
    assert st('kirsten')[0][0] == 'rejected'                             # a biographical article is not a human
    assert st('abc') == [('rejected', 'class_disambiguation')]
    assert st('hohenlohe')[0][0] == 'accepted'                           # dynasty class allowed for a dynasty tag
    assert st('novak') == [('suggested', 'name_mismatch')] or st('novak')[0][1] in ('name_mismatch', 'chronology')
    assert [s for s, _ in st('hm')] == ['accepted', 'accepted']
    assert st('lowcert') == [('suggested', 'cert')]
    assert st('noid') == [('suggested', 'unverified')]
    assert st('birthplace')[0][0] == 'ignored'
    ch = RG.choose(pairs, pd.DataFrame(votes))
    q = ch['qid'].set_index('entity_key')
    assert q.loc['luther', 'qid'] == 'Q9554' and 'pfalz' not in q.index and 'kirsten' not in q.index
    assert q.loc['hm', 'qid'] == 'Q444' and q.loc['hm', 'qid_via'] == 'gnd+viaf'
    assert ch['suggested'].set_index('entity_key').loc['pfalz', 'qid_suggested'] == 'Q719626'


def test_chronology_and_register_dates(tmp_path):
    C = 'x'
    wd = [_wd('Q1', {'en': 'Jan Novak'}, birth='+1794-01-01T00:00:00Z'),
          _wd('Q2', {'en': 'Hans Meyer'}, birth='+1462-01-01T00:00:00Z', death='+1515-01-01T00:00:00Z')]
    votes = [_vote(C, 'person', 'a', 'wikidata', 'Q1'), _vote(C, 'person', 'b', 'wikidata', 'Q2')]
    ents = [_ent(C, 'a', 'Jan Novak', last_year=1572), _ent(C, 'b', 'Hans Meyer', birth='1555', death='1584', last_year=1580)]
    pairs, _ = _run(tmp_path, wd, {}, votes, ents, 'person')
    assert list(zip(pairs.entity_key, pairs.status, pairs.reason)) == [('a', 'suggested', 'chronology'), ('b', 'suggested', 'register_dates')]


def test_place_gates(tmp_path):
    C = 'cavriana'
    wd = [_wd('Q193190', {'en': 'Famagusta', 'tr': 'Gazimağusa'}, classes=['Q515'], lat=35.12, lon=33.95),
          _wd('Q29', {'en': 'Spain', 'es': 'España'}, classes=['Q6256'], lat=40.0, lon=-4.0),
          _wd('Q30343751', {'en': 'Thomson Reuters (Switzerland)'}, classes=['Q783794']),
          _wd('Q2090', {'en': 'Nuremberg', 'de': 'Nürnberg'}, classes=['Q515'], lat=49.45, lon=11.08),
          _wd('Q32060456', {}, classes=['Q21286738'], dup=['Q2090']),
          _wd('Q71', {'en': 'Geneva'}, classes=['Q515'], lat=46.2, lon=6.14)]
    gn = [dict(gid='2660646', name='Geneva', names='geneva|geneve|genf|ginevra', lat=46.2, lon=6.14, pop=183000, cc='CH', fcode='PPLA', fclass='P'),
          dict(gid='2893838', name='Kalkar', names='kalkar', lat=50.60, lon=6.77, pop=0, cc='DE', fcode='PPL', fclass='P'),
          dict(gid='9', name='Tortosa', names='tortosa', lat=40.8, lon=0.5, pop=30000, cc='ES', fcode='PPL', fclass='P')]
    same = [dict(key='kalkar', gid='2893837', lat=51.74, lon=6.29, pop=13000, cc='DE', fcode='PPL'),
            dict(key='kalkar', gid='2893838', lat=50.60, lon=6.77, pop=0, cc='DE', fcode='PPL')]
    mapping = {('tgn', '7002327'): ['Q193190'], ('tgn', '1000001'): ['Q29'], ('geonames', '2660646'): ['Q30343751'],
               ('geonames', '2861650'): ['Q32060456'], ('geonames', '2893838'): []}
    votes = [_vote(C, 'place', 'spagna', 'tgn', '7002327'), _vote(C, 'place', 'espana', 'tgn', '1000001'),
             _vote(C, 'place', 'genf', 'geonames', '2660646'), _vote(C, 'place', 'nbg', 'geonames', '2861650'),
             _vote(C, 'place', 'kalkar', 'geonames', '2893838'), _vote(C, 'place', 'tolosa', 'geonames', '9'),
             _vote(C, 'place', 'pleiades', 'pleiades', '555')]
    ents = [_ent(C, 'spagna', 'Spagna', ['Ispagna']), _ent(C, 'espana', 'Spain', ['Spagna', 'España']),
            _ent(C, 'genf', 'Genff', ['Genf']),
            _ent(C, 'nbg', 'Nürnberg', ['Norimberga']), _ent(C, 'kalkar', 'Kalkar'),
            _ent(C, 'tolosa', 'Tolosa'), _ent(C, 'pleiades', 'Something')]
    pairs, auth = _run(tmp_path, wd, mapping, votes, ents, 'place', gn=gn, same=same)
    # the register has places on the lower Rhine (v2.1: the homonym gate needs five placed places of the edition to decide)
    anchors = [_ent(C, f'anchor{i}', f'Ort{i}', lat=la, lon=lo) for i, (la, lo) in enumerate(
        [(51.79, 6.14), (51.65, 6.60), (51.85, 6.40), (51.55, 6.30), (51.70, 6.05)])]
    ents2 = pd.DataFrame(ents + anchors)
    out = RG.run(pd.DataFrame(votes), ents2, auth, 'place', log=lambda m: None)['pairs']
    # ... and the same edition without them cannot tell: the pair is only suggested
    out_nc = RG.run(pd.DataFrame(votes), pd.DataFrame(ents), auth, 'place', log=lambda m: None)['pairs']
    assert sorted(zip(out_nc[out_nc.entity_key == 'kalkar'].status, out_nc[out_nc.entity_key == 'kalkar'].reason)) == \
        [('suggested', 'homonym_unverified')] * len(out_nc[out_nc.entity_key == 'kalkar'])

    def st(ek):
        x = out[out.entity_key == ek]
        return sorted(zip(x['src'], x['status'], x['reason']))
    assert st('spagna') == [('wd', 'suggested', 'name_mismatch')]                # Spain -> Famagusta stays a suggestion
    assert st('espana')[0][1] == 'accepted'
    genf = st('genf')
    assert ('wd', 'rejected', 'class_non_place') in genf and ('geonames', 'accepted', 'fold') in genf or \
        any(s == 'geonames' and a == 'accepted' for s, a, _ in genf)         # the GeoNames record itself is right
    assert st('nbg')[0][1] == 'accepted'                                    # duplicate followed to Q2090
    assert out[out.entity_key == 'nbg'].iloc[0]['qid'] == 'Q2090' and out[out.entity_key == 'nbg'].iloc[0]['qid_raw'] == 'Q32060456'
    assert all(a == 'rejected' and r == 'homonym' for _, a, r in st('kalkar'))
    assert st('tolosa') == [('geonames', 'suggested', 'name_mismatch')]         # Tolosa carries the id of Tortosa
    assert st('pleiades') == [('', 'suggested', 'unverified')]
    ch = RG.choose(out, pd.DataFrame(votes))
    assert 'spagna' not in set(ch['qid']['entity_key']) and 'kalkar' not in set(ch['ids']['entity_key'])
    assert ch['qid'].set_index('entity_key').loc['nbg', 'qid'] == 'Q2090'
    ids = ch['ids'].set_index(['entity_key', 'scheme'])
    assert ('genf', 'geonames') in ids.index


def test_second_id_and_near_evidence(tmp_path):
    C = 'x'
    wd = [_wd('Q7', {'en': 'Trier'}, classes=['Q515'], lat=49.75, lon=6.64), _wd('Q8', {'en': 'Mainz'}, classes=['Q515'], lat=50.0, lon=8.27)]
    mapping = {('gnd', '1'): ['Q7'], ('viaf', '1'): ['Q7']}
    votes = [_vote(C, 'place', 'augusta', 'gnd', '1'), _vote(C, 'place', 'augusta', 'viaf', '1'),
             _vote(C, 'place', 'moguntia', 'wikidata', 'Q8')]
    # Treveris: names differ from the English label but two ids agree
    ents = [_ent(C, 'augusta', 'Augusta Treverorum'), _ent(C, 'moguntia', 'Moguntiacum', lat=50.01, lon=8.28)]
    pairs, _ = _run(tmp_path, wd, mapping, votes, ents, 'place')
    a = pairs[pairs.entity_key == 'augusta']
    assert set(a.status) == {'accepted'} and set(a.reason) == {'second_id'}
    # Moguntiacum: the register point sits next to the item's point, but a coordinate is not evidence of identity (v2 audit, check 1)
    m = pairs[pairs.entity_key == 'moguntia'].iloc[0]
    assert m.status == 'suggested' and m.reason == 'name_mismatch'


def test_near_evidence_needs_one_feature_with_both_names(tmp_path):
    """Venedig and Venezia are names of one GeoNames feature: accepted. Halle and Altstadt (the district Wikidata reaches through Halle's GeoNames
    id), Balingen and Bisingen are not: the item is another place, whatever the distance."""
    C = 'x'
    wd = [_wd('Q641', {'en': 'Venice', 'it': 'Venezia'}, classes=['Q515'], lat=45.44, lon=12.33),
          _wd('Q445500', {'de': 'Altstadt'}, classes=['Q515'], lat=51.482, lon=11.97),
          _wd('Q31914615', {'de': 'Bisingen'}, classes=['Q515'], lat=48.31, lon=8.917)]
    mapping = {('geonames', '3164603'): ['Q641'], ('geonames', '2911522'): ['Q445500'], ('geonames', '2953089'): ['Q31914615']}
    gn = [dict(gid='3164603', name='Venezia', names='venezia|venedig|venice', lat=45.44, lon=12.33, pop=260000, cc='IT', fcode='PPLA', fclass='P'),
          dict(gid='2911522', name='Halle', names='halle|halle saale', lat=51.4825, lon=11.97, pop=230000, cc='DE', fcode='PPLA3', fclass='P'),
          dict(gid='2953089', name='Balingen', names='balingen', lat=48.275, lon=8.855, pop=34000, cc='DE', fcode='PPLA4', fclass='P')]
    same = [dict(key='venedig', gid='3164603', lat=45.44, lon=12.33, pop=260000, cc='IT', fcode='PPLA', names='venezia|venedig|venice'),
            dict(key='halle', gid='2911522', lat=51.4825, lon=11.97, pop=230000, cc='DE', fcode='PPLA3', names='halle|halle saale'),
            dict(key='balingen', gid='2953089', lat=48.275, lon=8.855, pop=34000, cc='DE', fcode='PPLA4', names='balingen')]
    votes = [_vote(C, 'place', 'venedig', 'wikidata', 'Q641'), _vote(C, 'place', 'halle', 'wikidata', 'Q445500'),
             _vote(C, 'place', 'balingen', 'wikidata', 'Q31914615')]
    ents = [_ent(C, 'venedig', 'Venedig', lat=45.43, lon=12.34), _ent(C, 'halle', 'Halle', lat=51.48, lon=11.97),
            _ent(C, 'balingen', 'Balingen', lat=48.28, lon=8.86)]
    pairs, _ = _run(tmp_path, wd, mapping, votes, ents, 'place', gn=gn, same=same)
    got = {r.entity_key: (r.status, r.reason) for r in pairs.itertuples()}
    assert got['venedig'] == ('accepted', 'near_geonames_alias')
    assert got['halle'][0] == 'suggested' and got['balingen'][0] == 'suggested'


def test_votes_normalise_suggested_viaf_and_cert():
    d = pd.DataFrame({'coll_slug': ['a', 'a', 'a'], 'kind': ['person'] * 3, 'entity_key': ['k'] * 3,
                      'scheme': ['viaf-suggested', 'viaf', 'deutsche-biographie'], 'raw': ['123', '456', '118575449'],
                      'ip': [True] * 3, 'hi': [True] * 3})
    n = RG._normalise(d.rename(columns={'raw': 'raw'}))
    rows = {(r.scheme, r.id): r.sug for r in n.itertuples()}
    assert rows[('viaf', '123')] is True or rows[('viaf', '123')] == True      # noqa: E712
    assert rows[('viaf', '456')] == False                                      # noqa: E712
    assert ('gnd', '118575449') in rows                                         # Deutsche Biographie ids are GND numbers


def test_viaf_only_is_a_suggestion_in_sandrart(tmp_path):
    wd = [_wd('Q1', {'en': 'Hans Meyer'}), _wd('Q2', {'en': 'Hans Meyer'})]
    mapping = {('viaf', '1'): ['Q1'], ('viaf', '2'): ['Q2'], ('gnd', '2'): ['Q2']}
    votes = [_vote('sandrart', 'person', 'a', 'viaf', '1'), _vote('sandrart', 'person', 'b', 'viaf', '2'),
             _vote('sandrart', 'person', 'b', 'gnd', '2'), _vote('other', 'person', 'c', 'viaf', '1')]
    ents = [_ent('sandrart', 'a', 'Hans Meyer'), _ent('sandrart', 'b', 'Hans Meyer'), _ent('other', 'c', 'Hans Meyer')]
    pairs, _ = _run(tmp_path, wd, mapping, votes, ents, 'person')
    got = {(r.entity_key, r.scheme): (r.status, r.reason) for r in pairs.itertuples()}
    assert got[('a', 'viaf')] == ('suggested', 'viaf_unreliable')
    assert got[('b', 'viaf')][0] == 'accepted' and got[('c', 'viaf')][0] == 'accepted'


def test_an_id_on_a_minority_of_a_surface_is_not_the_surface(tmp_path):
    # corpus_corporum: one Pleiades link of the Roman province Europa on 1 of 12 mentions of 'Europa' (the continent)
    wd = [_wd('Q46', {'en': 'Europe', 'la': 'Europa'}, classes=['Q6256']), _wd('Q99', {'la': 'Europa', 'en': 'Europa'}, classes=['Q515'], lat=41, lon=27)]
    mapping = {('pleiades', '991376'): ['Q99'], ('wikidata', 'Q46'): []}
    votes = [_vote('cc', 'place', 'surf:europa', 'pleiades', '991376', n=1), _vote('cc', 'place', 'surf:europa', 'wikidata', 'Q46', n=8),
             _vote('cc', 'place', 'colla:rome', 'pleiades', '991376', n=1, reg=True)]
    ents = [_ent('cc', 'surf:europa', 'Europa', n_all=12), _ent('cc', 'colla:rome', 'Europa', n_all=12)]
    pairs, _ = _run(tmp_path, wd, mapping, votes, ents, 'place')
    got = {(r.entity_key, r.scheme): (r.status, r.reason) for r in pairs.itertuples()}
    assert got[('surf:europa', 'pleiades')] == ('suggested', 'minority_id')
    assert got[('surf:europa', 'wikidata')][0] == 'accepted'
    assert got[('colla:rome', 'pleiades')][0] == 'accepted'               # a register entry's own id is never a minority


def test_chronology_also_stops_an_acceptance_by_a_second_id(tmp_path):
    # v2 audit, check 3: a poem dated 1501-1700 (author's lifespan) cites a man born in 1730; two ids agree on him, the names do not
    wd = [_wd('Q16493296', {'es': 'Francisco Nieto Molina'}, birth='+1730-01-01T00:00:00Z'),
          _wd('Q5', {'es': 'Francisco Quevedo'}, birth='+1580-09-17T00:00:00Z')]
    mapping = {('viaf', '1'): ['Q16493296'], ('gnd', '1'): ['Q16493296'], ('viaf', '2'): ['Q5'], ('gnd', '2'): ['Q5']}
    votes = [_vote('disco', 'person', 'a', 'viaf', '1'), _vote('disco', 'person', 'a', 'gnd', '1'),
             _vote('disco', 'person', 'b', 'viaf', '2'), _vote('disco', 'person', 'b', 'gnd', '2')]
    ents = [_ent('disco', 'a', 'Poeta anonimus', last_year=1700), _ent('disco', 'b', 'Poeta anonimus', last_year=1700)]
    pairs, _ = _run(tmp_path, wd, mapping, votes, ents, 'person')
    got = {(r.entity_key, r.scheme): (r.status, r.reason) for r in pairs.itertuples()}
    assert got[('a', 'viaf')] == ('suggested', 'chronology') and got[('a', 'gnd')] == ('suggested', 'chronology')
    assert got[('b', 'viaf')] == ('accepted', 'second_id')


def test_a_country_of_the_register_does_not_take_a_village(tmp_path):
    # v2 audit, check 5: the register's Hessen (type country) took the Ortsteil Hessen; Luxemburg (country AND city) keeps its item
    wd = [_wd('Q896361', {'de': 'Hessen'}, classes=['Q123']), _wd('Q1221156', {'de': 'Hessen'}, classes=['Q6256']),
          _wd('Q1842', {'de': 'Luxemburg'}, classes=['Q6256', 'Q515'])]
    mapping = {}
    votes = [_vote('x', 'place', 'hessen', 'wikidata', 'Q896361'), _vote('x', 'place', 'hessen2', 'wikidata', 'Q1221156'),
             _vote('x', 'place', 'lux', 'wikidata', 'Q1842'), _vote('x', 'place', 'ort', 'wikidata', 'Q896361')]
    ents = [_ent('x', 'hessen', 'Hessen', note_type='country'), _ent('x', 'hessen2', 'Hessen', note_type='country'),
            _ent('x', 'lux', 'Luxemburg', note_type='country'), _ent('x', 'ort', 'Hessen', note_type='place')]
    pairs, _ = _run(tmp_path, wd, mapping, votes, ents, 'place', classes={'Q123': 'Ortsteil'})
    got = {r.entity_key: (r.status, r.reason) for r in pairs.itertuples()}
    assert got['hessen'] == ('rejected', 'type_mismatch')
    assert got['hessen2'][0] == 'accepted' and got['lux'][0] == 'accepted' and got['ort'][0] == 'accepted'


def _tgn_dir(tmp_path):
    d = tmp_path / 'tgn'
    d.mkdir(exist_ok=True)
    places = [('7000874', 'Regensburg', 'inhabited place', 49.0, 12.1), ('7003980', 'Ancona', 'inhabited place', 43.6, 13.5),
              ('7011179', 'Venezia', 'inhabited place', 45.44, 12.33), ('7004444', 'Ravenna', 'inhabited place', 44.4, 12.2)]
    pd.DataFrame([dict(subject_id=i, name=n, name_norm=n.lower(), place_type=t, place_types=t, record_type='A', merged_stat='M', lat=la, lon=lo,
                       parent_id=None, parent_name=None, hierarchy=n + ', Europe, World', ancestor_ids='', depth=2, nation='x')
                  for i, n, t, la, lo in places]).to_parquet(d / 'tgn_places.parquet', index=False)
    names = [('7000874', 'Regensburg', True), ('7000874', 'Ratisbona', False), ('7000874', 'Ratisbon', False), ('7000874', 'Castra Regina', False),
             ('7003980', 'Ancona', True), ('7003980', 'Ancon', False), ('7011179', 'Venezia', True), ('7011179', 'Venedig', False),
             ('7011179', 'Venice', False), ('7004444', 'Ravenna', True)]
    pd.DataFrame([dict(subject_id=i, term_id=str(k), name=n, name_norm=n.lower(), language='la', is_preferred=p)
                  for k, (i, n, p) in enumerate(names)]).to_parquet(d / 'tgn_names.parquet', index=False)
    return d


def test_tgn_ids_are_verified_against_the_local_tgn_names(tmp_path):
    """v2.1: a TGN id nobody can check through Wikidata is read in the local Getty index: compatible names = accepted (Ratisbona is Regensburg),
    names of another place = rejected (tgn_name_mismatch), a subject the index does not hold = still suggested."""
    C = 'cavriana'
    votes = [_vote(C, 'place', 'ratisbona', 'tgn', '7000874'), _vote(C, 'place', 'ancona', 'tgn', '7003980'),
             _vote(C, 'place', 'rimini', 'tgn', '7004444'), _vote(C, 'place', 'venezia', 'tgn', '7011179'),
             _vote(C, 'place', 'gone', 'tgn', '9999999')]
    ents = [_ent(C, 'ratisbona', 'Ratisbona'), _ent(C, 'ancona', 'Ancona'), _ent(C, 'rimini', 'Rimini'), _ent(C, 'venezia', 'Venedig'),
            _ent(C, 'gone', 'Atlantis')]
    auth = _auth(tmp_path, [], {})
    auth.tgn_dir = _tgn_dir(tmp_path)
    out = RG.run(pd.DataFrame(votes), pd.DataFrame(ents), auth, 'place', log=lambda m: None)['pairs']
    st = {ek: sorted(zip(g['src'], g['status'], g['reason'])) for ek, g in out.groupby('entity_key')}
    assert st['ratisbona'][0][:2] == ('tgn', 'accepted')                         # a Latin name of the subject
    assert st['ancona'][0][:2] == ('tgn', 'accepted')
    assert st['venezia'][0][:2] == ('tgn', 'accepted')                           # Venedig is a German name of Venezia
    assert st['rimini'] == [('tgn', 'rejected', 'tgn_name_mismatch')]            # the id is Ravenna
    assert st['gone'] == [('tgn', 'suggested', 'unverified')]
    ch = RG.choose(out, pd.DataFrame(votes))
    ids = ch['ids']
    assert set(ids[ids.scheme == 'tgn']['entity_key']) == {'ratisbona', 'ancona', 'venezia'}
    assert set(auth.tgn) == {'7000874', '7003980', '7004444', '7011179'} and auth.tgn['7000874']['lat'] == 49.0
    # without the index nothing changes: unverified
    auth0 = _auth(tmp_path, [], {})
    out0 = RG.run(pd.DataFrame(votes), pd.DataFrame(ents), auth0, 'place', log=lambda m: None)['pairs']
    assert set(zip(out0['status'], out0['reason'])) == {('suggested', 'unverified')}


def test_homonym_candidate_in_the_editions_territory_is_accepted_and_isolated_one_rejected(tmp_path):
    C = 'x'
    gn = [dict(gid='1', name='Kalkar', names='kalkar', lat=50.60, lon=6.77, pop=0, cc='DE', fcode='PPL', fclass='P'),
          dict(gid='3', name='Kalkar', names='kalkar', lat=47.60, lon=9.77, pop=0, cc='DE', fcode='PPL', fclass='P')]
    same = [dict(key='kalkar', gid='2', lat=51.74, lon=6.29, pop=13000, cc='DE', fcode='PPL', names='kalkar'),
            dict(key='kalkar', gid='1', lat=50.60, lon=6.77, pop=0, cc='DE', fcode='PPL', names='kalkar'),
            dict(key='kalkar', gid='3', lat=47.60, lon=9.77, pop=0, cc='DE', fcode='PPL', names='kalkar')]
    votes = [_vote(C, 'place', 'in_eifel', 'geonames', '1'), _vote(C, 'place', 'far_away', 'geonames', '3')]
    ents = [_ent(C, 'in_eifel', 'Kalkar'), _ent(C, 'far_away', 'Kalkar')] + [
        _ent(C, f'a{i}', f'A{i}', lat=la, lon=lo) for i, (la, lo) in enumerate([(50.62, 6.75), (50.55, 6.80), (50.70, 6.70), (50.50, 6.90), (50.65, 6.60)])]
    out, _ = _run(tmp_path, [], {}, votes, ents, 'place', gn=gn, same=same)
    st = {ek: sorted(zip(g['status'], g['reason'])) for ek, g in out.groupby('entity_key')}
    assert st['in_eifel'][0][0] == 'accepted'
    assert st['far_away'] == [('rejected', 'homonym')]


def test_homonym_village_with_a_larger_namesake_in_the_same_region_is_suggested_not_accepted(tmp_path):
    """v2.2: the edition has places near the Eifel hamlet AND near the Kalkar of the Lower Rhine (131 km): the gate cannot tell which one the
    editor means, so the id is only suggested; with places near the hamlet alone it is accepted (v2.1 behaviour)."""
    C = 'x'
    gn = [dict(gid='1', name='Kalkar', names='kalkar', lat=50.60, lon=6.77, pop=0, cc='DE', fcode='PPL', fclass='P')]
    same = [dict(key='kalkar', gid='2', lat=51.74, lon=6.29, pop=13000, cc='DE', fcode='PPL', names='kalkar'),
            dict(key='kalkar', gid='1', lat=50.60, lon=6.77, pop=0, cc='DE', fcode='PPL', names='kalkar')]
    votes = [_vote(C, 'place', 'kalkar', 'geonames', '1')]
    near_eifel = [(50.62, 6.75), (50.55, 6.80), (50.70, 6.70), (50.50, 6.90), (50.65, 6.60)]
    near_both = near_eifel + [(51.79, 6.14), (51.65, 6.60)]
    for pts, expect in ((near_eifel, ('accepted', 'exact')), (near_both, ('suggested', 'homonym_unverified'))):
        ents = [_ent(C, 'kalkar', 'Kalkar')] + [_ent(C, f'a{i}', f'A{i}', lat=la, lon=lo) for i, (la, lo) in enumerate(pts)]
        out, _ = _run(tmp_path, [], {}, votes, ents, 'place', gn=gn, same=same)
        g = out[out.entity_key == 'kalkar']
        assert set(zip(g['status'], g['reason'])) == {expect}, (expect, set(zip(g['status'], g['reason'])))


def _tgn_dir2(tmp_path):
    d = tmp_path / 'tgn2'
    d.mkdir(exist_ok=True)
    rows = [  # subject_id, name, type, lat, lon, parent_id
        ('7011054', 'Sífnos, Nísos', 'island', 36.9675, 24.7117, None),
        ('7215870', 'Alaunwerk', 'inhabited place', 52.7833, 13.9833, None),
        ('7002689', 'Khaniá', 'regional division', 35.3333, 24.0, None),
        ('7011021', 'Palaikastro', 'deserted settlement', 39.4089, 21.6186, '7002689'),
        ('7030347', 'Roman Empire', 'former nation/state/empire', 41.8925, 12.485, None),
        ('7030114', 'Lusitania', 'province', 38.91624, 6.33905, '7030347'),
        ('7001111', 'Piave', 'river', 46.0, 12.5, None),
        ('7007070', 'Stuttgart', 'inhabited place', 48.78, 9.18, None)]
    pd.DataFrame([dict(subject_id=i, name=n, name_norm=n.lower(), place_type=t, place_types=t, record_type='A', merged_stat='M', lat=la, lon=lo,
                       parent_id=p, parent_name=None, hierarchy=n + ', World', ancestor_ids='', depth=2, nation='x')
                  for i, n, t, la, lo, p in rows]).to_parquet(d / 'tgn_places.parquet', index=False)
    names = [('7011054', 'Sífnos, Nísos'), ('7011054', 'Sifnos'), ('7011054', 'Siphnos'), ('7215870', 'Alaunwerk'), ('7002689', 'Khaniá'),
             ('7011021', 'Palaikastro'), ('7011021', 'Aptera'), ('7030347', 'Roman Empire'), ('7030114', 'Lusitania'), ('7001111', 'Piave'),
             ('7007070', 'Stuttgart')]
    pd.DataFrame([dict(subject_id=i, term_id=str(k), name=n, name_norm=n.lower(), language='la', is_preferred=(k % 2 == 0))
                  for k, (i, n) in enumerate(names)]).to_parquet(d / 'tgn_names.parquet', index=False)
    return d


def test_tgn_v22_fuzzy_threshold_qualifier_head_and_coordinate_checks(tmp_path):
    """v2.2: Kythnos/Sifnos (fuzzy 0.83) is only suggested; 'Bad Düben, Alaunwerk' is not the Alaunwerk of Brandenburg; a TGN point 500 km from its
    own parent (Aptera) or at the wrong side of the meridian (Lusitania) is not used while the id stays accepted."""
    C = 'sandrart'
    gn = [dict(gid='10', name='Bad Düben', names='bad duben', lat=51.59, lon=12.58, pop=8000, cc='DE', fcode='PPL', fclass='P')]
    same = [dict(key='bad duben', gid='10', lat=51.59, lon=12.58, pop=8000, cc='DE', fcode='PPL', names='bad duben'),
            dict(key='bad duben', gid='11', lat=51.586, lon=12.56, pop=7700, cc='DE', fcode='ADM4', names='bad duben')]      # the commune: not an area
    votes = [_vote(C, 'place', 'cythnos', 'tgn', '7011054'), _vote(C, 'place', 'alaun', 'tgn', '7215870'), _vote(C, 'place', 'stut', 'tgn', '7007070'),
             _vote(C, 'place', 'aptera', 'tgn', '7011021'), _vote(C, 'place', 'lusi', 'tgn', '7030114')]
    ents = [_ent(C, 'cythnos', 'Cythnos'), _ent(C, 'alaun', 'Bad Düben, Alaunwerk'), _ent(C, 'stut', 'Stutgard'),
            _ent(C, 'aptera', 'Aptera'), _ent(C, 'lusi', 'Lusitaniâ')]
    auth = _auth(tmp_path, [], {}, gn=gn, same=same)
    auth.tgn_dir = _tgn_dir2(tmp_path)
    out = RG.run(pd.DataFrame(votes), pd.DataFrame(ents), auth, 'place', log=lambda m: None)['pairs']
    st = {ek: sorted(zip(g['status'], g['reason'])) for ek, g in out.groupby('entity_key')}
    assert st['cythnos'] == [('suggested', 'tgn_fuzzy')]
    assert st['alaun'] == [('rejected', 'tgn_head_far')]
    assert st['stut'] == [('suggested', 'tgn_fuzzy')]                      # Stutgard / Stuttgart is 0.88: below the threshold, no independent evidence
    # the hierarchy / exonym checks (TGN.coord_issue) on the loaded subjects
    from ratlas import tgn as TGN
    from ratlas.reconcile_v2 import exonym_keys
    assert 'parent_far' in TGN.coord_issue(auth.tgn['7011021'], exonym_keys('Aptera'))
    assert 'exonym_far' in TGN.coord_issue(auth.tgn['7030114'], exonym_keys('Lusitaniâ'))
    assert TGN.coord_issue(auth.tgn['7011054'], exonym_keys('Cythnos')) is None and TGN.coord_issue(None) is None
    assert TGN.parent_limit_km('regional division') == 300 and TGN.parent_limit_km('nation') is None and TGN.parent_limit_km('continent') is None
