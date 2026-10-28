"""v2.4 (fixes of the independent check of v2.3): Latin homonyms (Lugdunum = Lyon or Leiden), polity heads, the Orbis Latinus alone, the individual corrections, the
display data (continents, Casebooks low candidates, extra.json)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ratlas import exonyms as EX
from ratlas import latin_geo as LG
from ratlas import latin_homonyms as LH
from ratlas import manual_overrides as MO
from ratlas import polities as POL
from ratlas import reconcile_build as B
from ratlas import reconcile_geocode as GC
from ratlas import reconcile_v2 as V
from ratlas import regions as RGN
from test_v23_geocode import _f, _geocoder, _row, _run

DATA = Path(__file__).resolve().parents[1] / 'data'
CLOUD_NL = [_row('ckcc_epistolarium', f'n{i}', f'Stad {i}', lat=52.0 + 0.05 * i, lon=5.0 + 0.05 * i, n=3) for i in range(12)]
CLOUD_FR = [_row('ckcc_epistolarium', f'f{i}', f'Lieu {i}', lat=45.5 + 0.05 * i, lon=4.6 + 0.05 * i, n=3) for i in range(12)]


def _pairs(n_rows, target=0, cloud_from=1):
    """The target row and the cloud rows share document 0; other rows have a document of their own."""
    docs = [0 if (i == target or i >= cloud_from) else i + 1 for i in range(n_rows)]
    return pd.DataFrame({'doc': docs, 'row': list(range(n_rows))})


# ---- fix 1: Lugdunum = Lyon or Leiden ------------------------------------------------------------------------------------------------------------
def test_exonym_table_no_longer_gives_the_bare_latin_forms_to_lyon():
    for k in ('lugdunum', 'lugduni', 'lugdunensis', 'lugduno', 'lugdunenses'):
        assert EX.lookup(k, 'ckcc_epistolarium') is None, k
    assert EX.lookup('lugduni batavorum')[3] == 'Leiden' and EX.lookup('lugduni bat')[3] == 'Leiden' and EX.lookup('lugduno batavorum')[3] == 'Leiden'
    assert EX.lookup('lugdunum segusianorum')[3] == 'Lyon' and EX.lookup('lyons')[3] == 'Lyon'
    assert EX.lookup('traiectum ad rhenum')[3] == 'Utrecht' and EX.lookup('traiectum ad mosam')[3] == 'Maastricht'
    assert EX.lookup('mare hadriaticum')[3] == 'Adriatic Sea' and EX.FEATURE_KIND['Adriatic Sea'] == 'n'
    # the reread of the table: forms that name two places or the wrong one are out
    for k in ('vienne', 'indias', 'ruthenia', 'rutheni', 'regiomontanus'):
        assert EX.lookup(k) is None, k
    assert EX.lookup('wien') is None or EX.lookup('wien')[3] == 'Vienna'
    assert EX.lookup('vindobona')[3] == 'Vienna' and EX.lookup('russia')[3] == 'Muscovy' and EX.lookup('indes occidentales')[3] == 'Americas'


def test_family_of_matches_the_bare_forms_only():
    for n in ('Lugduni', 'Lugdunum', 'Lugdunenses', 'Lugduno', 'Lugdunensi', 'LUGDUNI', 'Traiecti', 'Trajectum', 'Augusta', 'Hadriae', 'Hadria'):
        assert LH.family_of(n) is not None, n
    for n in ('Lugduni Batavorum', 'Lugduni Bat.', 'Lugdunum in Gallia', 'Augusta Vindelicorum', 'Lyon', None, 'Traiectum ad Rhenum'):
        assert LH.family_of(n) is None, n
    assert LH.family_of('Lugduno').name == 'lugdunum' and LH.family_of('Hadriae').name == 'hadria'


def test_decide_by_passage_and_documents():
    fam = LH.family_of('Lugduni')
    nl = {'Leiden': 0.9, 'Lyon': 0.0}
    fr = {'Leiden': 0.0, 'Lyon': 0.9}
    # explicit support for Leiden ('Batavorum', 'Academia Lugdunensi'), none for Lyon
    d = LH.decide(fam, ['Lugduni Batavorum, 12 Novembr. 1633', 'ad bibliopolas nostros Batavo- Lugdunenses'], nl)
    assert d.cand.label == 'Leiden' and d.reason == 'cue' and d.n_cue['Leiden'] == 2
    # explicit support for Lyon (Rhodanus) and none for Leiden: Lyon
    d = LH.decide(fam, ['viso Brisaco, Geneva, Lugduno ivit visurus regem apud Rhodanum'], fr)
    assert d.cand.label == 'Lyon' and d.reason == 'cue'
    # cues for both: nothing
    d = LH.decide(fam, ['Lugduni Batavorum', 'Lugduni ad Rhodanum'], nl)
    assert d.cand is None and d.reason == 'mixed_cues'
    # no cue: the Dutch documents say Leiden; the French documents do not make it Lyon (Lyon needs a cue)
    assert LH.decide(fam, ['Lugduni convenarum seditio'], nl).cand.label == 'Leiden'
    d = LH.decide(fam, ['Lugduni convenarum seditio'], fr)
    assert d.cand is None and d.reason == 'needs_cue'
    assert LH.decide(fam, [], {}).cand is None                                    # nothing known at all
    # a cue the documents flatly contradict is not enough
    d = LH.decide(fam, ['Lugduno Rhodani'] + ['x'] * 20, {'Leiden': 0.0, 'Lyon': 0.0})
    assert d.cand is None and d.reason == 'no_support'                            # 1 of 21 passages: below CUE_SHARE_MIN and no support from the documents
    d = LH.decide(fam, ['Lugduno Rhodani'] + ['x'] * 20, {'Leiden': 0.0, 'Lyon': 0.5})
    assert d.cand is None and d.reason == 'no_support'                            # Lyon needs a cue in nearly every passage: 1 of 21 is not enough, whatever the documents say
    assert LH.decide(fam, ['Lugduno Rhodani', 'Lugduno Rhodani', 'x', 'x'], {'Leiden': 0.0, 'Lyon': 0.5}).cand.label == 'Lyon'     # half of the passages and the documents' support
    assert LH.decide(fam, ['Lugduno Rhodani', 'Lugduno Rhodani', 'x', 'x'], {'Leiden': 0.0, 'Lyon': 0.0}).cand is None
    aug = LH.family_of('Augusta')
    assert LH.decide(aug, ['Augusta Taurinorum obsessa'] * 11 + ['Francofurtum, Augusta, Norimberga'] * 9, {}).cand is None       # the ckcc 'Augusta': Turin 11 times, Augsburg 9
    d = LH.decide(fam, ['Lugduno Batavorum'] + ['x'] * 4, {'Leiden': 0.0, 'Lyon': 1.2})
    assert d.cand is None and d.reason == 'contradicted'                          # 1 of 5 passages, and the documents say Lyon
    d = LH.decide(fam, ['Lugduno Batavorum', 'Lugduni Batavorum'], {'Leiden': 0.0, 'Lyon': 1.2})
    assert d.cand.label == 'Leiden'                                               # every passage names Leiden
    # the sea is the default of Hadria; the town of Adria needs its cue
    h = LH.family_of('Hadriae')
    assert LH.decide(h, ['inquieti turbidus Hadriae'], {}).cand.label == 'Adriatic Sea'
    assert LH.decide(h, ['episcopus Hadriae in Polesine, Rhodigii'], {}).cand.label == 'Adria'


def test_lugduni_in_the_letters_of_the_dutch_republic_is_leiden_or_nothing_never_lyon_by_default():
    rows = [_row('ckcc_epistolarium', 'a', 'Lugduni'), _row('ckcc_epistolarium', 'b', 'Lugdunum'), _row('ckcc_epistolarium', 'c', 'Lugdunenses'),
            _row('ckcc_epistolarium', 'd', 'Lugduno')] + CLOUD_NL
    n = len(rows)
    docs = [0, 0, 0, 0] + [0] * 12                                                # one letter with a Dutch cloud
    pairs = pd.DataFrame({'doc': docs, 'row': list(range(n))})
    aux = {0: {'windows': ['liberato Lugduni Batavorum', 'Academia Lugdunensi']},          # Leiden by the passage
           1: {'windows': ['Lugdunum ivit cum rege Galliae', 'statim Lugdunum proficisci cogor ad Rhodanum']},   # every passage names Lyon (Galliae, Rhodanum), none Leiden; Dutch documents
           2: {'windows': ['Lugdunenses Batavi burgemeesters', 'Lugdunenses calumniae Carentonie, Gallia']},                      # cues for both
           }
    G, out = _run(rows, [], pairs=pairs, resources={'mention_aux': aux})
    a, b, c, d = (_f(out, i) for i in range(4))
    assert a and a['tier'] == 'latin_homonym' and abs(a['lat'] - 52.16) < 0.01 and a['conf'] == 'medium'                 # Leiden
    assert b and abs(b['lat'] - 45.764) < 0.01                                                    # only Lyon is named by the passage: Lyon
    assert c is None and G.why.get(2) == 'latin_homonym_mixed_cues'                              # both: not placed
    assert d and abs(d['lat'] - 52.16) < 0.01                                                     # no cue, Dutch documents: Leiden
    # the same four forms in a French cloud with no cue: nothing
    rows2 = [_row('ckcc_epistolarium', 'a', 'Lugduni')] + CLOUD_FR
    G2, out2 = _run(rows2, [], pairs=pd.DataFrame({'doc': [0] * 13, 'row': list(range(13))}))
    assert _f(out2, 0) is None and G2.why.get(0) == 'latin_homonym_needs_cue'


def test_traiectum_and_augusta_follow_the_same_rule():
    rows = [_row('ckcc_epistolarium', 'a', 'Traiecti'), _row('ckcc_epistolarium', 'b', 'Traiectum'), _row('ckcc_epistolarium', 'c', 'Augusta')] + CLOUD_NL
    pairs = pd.DataFrame({'doc': [0, 0, 1] + [0] * 12, 'row': list(range(15))})
    aux = {1: {'windows': ['Traiectum ad Mosam transiit']}}
    G, out = _run(rows, [], pairs=pairs, resources={'mention_aux': aux})
    assert _f(out, 0) and abs(_f(out, 0)['lat'] - 52.091) < 0.01                                   # Dutch documents: Utrecht
    assert _f(out, 1) and abs(_f(out, 1)['lat'] - 50.851) < 0.01                                   # the passage names the Meuse: Maastricht
    assert _f(out, 2) is None                                                                       # Augusta alone in a letter without documents' places: nothing
    rows = [_row('ckcc_epistolarium', 'c', 'Augusta')] + [_row('ckcc_epistolarium', f'b{i}', f'Ort {i}', lat=48.2 + 0.05 * i, lon=10.6 + 0.05 * i, n=3) for i in range(12)]
    G, out = _run(rows, [], pairs=pd.DataFrame({'doc': [0] * 13, 'row': list(range(13))}))
    assert _f(out, 0) and abs(_f(out, 0)['lat'] - 48.371) < 0.01                                   # Bavarian documents: Augsburg


def test_hadriae_is_the_sea_not_the_town():
    rows = [_row('corpus_corporum', 'a', 'Hadriae'), _row('corpus_corporum', 'b', 'Hadria')]
    G, out = _run(rows, [], pairs=pd.DataFrame({'doc': [0, 1], 'row': [0, 1]}), resources={'mention_aux': {1: {'windows': ['episcopus Hadriae, Rhodigii in Polesine']}}})
    a, b = _f(out, 0), _f(out, 1)
    assert a and a['cand_name'] == 'Adriatic Sea' and a['radius'] >= 200 and abs(a['lat'] - 42.8) < 0.1
    assert b and b['cand_name'] == 'Adria' and abs(b['lat'] - 45.05) < 0.1


# ---- fix 2: polity_gn -----------------------------------------------------------------------------------------------------------------------------
def test_polity_table_has_the_reviewed_entries_of_the_check():
    for name, label, lat, lon in (('Stolberg, Grafschaft', 'County of Stolberg', 51.57, 10.95), ('Stolberg im Harz, Grafschaft', 'County of Stolberg', 51.57, 10.95),
                                  ('Gaster, Landvogtei', 'Bailiwick of Gaster', 47.16, 9.04), ('Oberstein, Herrschaft', 'Lordship of Oberstein', 49.71, 7.31),
                                  ('Hohenlohe, Grafschaft', 'County of Hohenlohe', 49.2, 9.5), ('Falkenberg, Herrschaft (in Oberschlesien)', 'Lordship of Falkenberg (Niemodlin)', 50.64, 17.58)):
        p = POL.lookup(POL.parse_qualified(name))
        assert p is not None and p.label == label and abs(p.lat - lat) < 0.03 and abs(p.lon - lon) < 0.03 and p.radius >= 25 and p.basis, name
    assert POL.lookup(POL.parse_qualified('Werdenberg, Grafschaft')) is None                       # the table does not pin it
    assert POL.lookup(POL.parse_qualified('Falkenberg')) is None


def test_polity_head_is_never_a_tgn_only_locality():
    """Stolberg in Lippe (51.97 N 9.05 E) was a TGN-only locality; the county of the Harz is the table's. Without the table the head is not placed."""
    rows = [_row('tagebuecher_christian_ii_anhalt', 'a', 'Mittelburg, Grafschaft'), _row('tagebuecher_christian_ii_anhalt', 'b', 'Stolberg, Grafschaft'),
            _row('tagebuecher_christian_ii_anhalt', 'c', 'Altenburg, Grafschaft')]
    gn = [('mittelburg', 'tgn7001234', True, 'Mittelburg', 51.97, 9.05, 'L', 'LCTY', '', '', 0),                    # a TGN-only locality
          ('stolberg', 'tgn7001235', True, 'Stolberg', 51.97, 9.05, 'L', 'LCTY', '', '', 0),
          ('altenburg', '2', True, 'Altenburg', 50.98, 12.43, 'A', 'ADM2', 'DE', '15', 90000)]                      # a GeoNames region: accepted
    G = _geocoder(rows, gn)
    G.gn.tgn = np.array([str(g).startswith('tgn') for g in G.gn.gid])
    out = GC.run_pipeline(G, np.array([], dtype=int), log=lambda m: None)
    assert _f(out, 0) is None or _f(out, 0)['conf'] == 'low'
    b = _f(out, 1)
    assert b and b['cand_name'] == 'County of Stolberg' and abs(b['lat'] - 51.5667) < 0.01 and b['radius'] >= 25                 # the table's, not Lippe
    c = _f(out, 2)
    assert c and abs(c['lat'] - 50.98) < 0.01 and c['conf'] in ('medium', 'high')


# ---- fix 3: the Orbis Latinus alone, adjectives, surface ------------------------------------------------------------------------------------------
def _wdl(*pairs):
    w = LG.WikidataLatin()
    for q, la in pairs:
        w.add(q, {'la': la}, {})
    return w


def test_latin_stoplist_adjectives_are_not_places():
    assert 'palatinus' in LG.NOT_PLACES and 'tiberina' in LG.NOT_PLACES
    w = _wdl(('Q1', 'Palatinus'))
    rows = [_row('corpus_corporum', 'p', 'Palatinus'), _row('sandrart', 'r', 'Palatine Hill', lat=41.888, lon=12.487, n=21)]
    G, out = _run(rows, [], resources={'wdlatin': w}, extra_cols={'qid': [None, 'Q1'], 'coord_kind': [None, None], 'coord_radius_km': [np.nan, np.nan]})
    assert _f(out, 0) is None or _f(out, 0)['conf'] == 'low'


def test_surface_that_differs_from_the_name_stops_the_latin_path():
    """The text says Tagum (the Tagus), the register standardised Fagum: the Latin path must not read 'Fagum' (Foug in Lorraine)."""
    assert GC.surface_compatible('Fagum', 'Fagum') and GC.surface_compatible('Hagam', 'Hagae') and GC.surface_compatible('Lugduni Batavorum', 'Lugduno Batavorum')
    assert GC.surface_compatible('Lugdunum', 'Lugd.') and not GC.surface_compatible('Fagum', 'Tagum') and not GC.surface_compatible('Fagus', 'Tagus')
    w = _wdl(('Q2', 'Fagus'))
    cloud = [_row('cidtc', f'c{i}', f'Ort {i}', lat=48.6 + 0.05 * i, lon=5.7 + 0.05 * i, n=3) for i in range(12)]
    rows = [_row('cidtc', 'pl1438', 'Fagum'), _row('cidtc', 'foug', 'Foug', lat=48.68, lon=5.79, n=2)] + cloud
    n = len(rows)
    ex = {'qid': [None, 'Q2'] + [None] * 12, 'coord_kind': [None] * n, 'coord_radius_km': [np.nan] * n}
    pairs = pd.DataFrame({'doc': [0] * n, 'row': list(range(n))})
    G, out = _run(rows, [], pairs=pairs, resources={'wdlatin': w}, extra_cols=ex)
    assert _f(out, 0) and _f(out, 0)['method'] == 'name_latin'                                    # without the surface check the name would be placed
    G, out = _run(rows, [], pairs=pairs, resources={'wdlatin': w, 'mention_aux': {0: {'surf_ok': 0.0}}}, extra_cols=ex)
    assert _f(out, 0) is None and G.why.get(0) == 'latin_surface_mismatch'


def test_load_mention_aux_reads_passages_and_surface_mismatch(tmp_path):
    rows = []
    for ctx in ('Lugduni Batavorum, 12 Nov.', 'Academia Lugdunensi'):
        rows.append(dict(coll_slug='ckcc_epistolarium', kind='place', tag='placeName', local_key=None, surface='Lugduni', canonical=True, excluded=False, context=ctx))
    for sfc in ('Tagum', 'Tagum', 'Fagum'):
        rows.append(dict(coll_slug='cidtc', kind='place', tag='placeName', local_key='cidtc:pl1438', surface=sfc, canonical=True, excluded=False, context='Toleti supra ' + sfc))
    p = tmp_path / 'm.parquet'
    pd.DataFrame(rows).to_parquet(p)
    P = pd.DataFrame({'coll_slug': ['ckcc_epistolarium', 'cidtc'], 'entity_key': ['surf:lugduni', 'cidtc:pl1438'], 'name': ['Lugduni', 'Fagum']})
    aux = GC.load_mention_aux(p, P, log=lambda m: None)
    assert aux[0]['windows'] == ['Lugduni Batavorum, 12 Nov.', 'Academia Lugdunensi'] and 'surf_ok' not in aux[0]
    assert abs(aux[1]['surf_ok'] - 1 / 3) < 1e-9 and 'windows' not in aux[1]


def test_two_families_far_from_the_documents_are_a_suggestion_unless_a_table_agrees():
    """Nordovico (Noordwijk, the editor says) = Norwich for the atlas and the Orbis Latinus, in Dutch letters: two families, 1 mention, far from every place of the letter."""
    w = _wdl(('Q3', 'Nordovicum'))
    orbis = LG.OrbisLatinus([('Nordovicum', 'Norwich', 'city', 'England', 'A', '1')])
    gn = [('norwich', '9', True, 'Norwich', 52.63, 1.30, 'P', 'PPLA2', 'GB', 'ENG', 130000)]
    cloud_by = [_row('corpus_corporum', f'b{i}', f'Ort {i}', lat=48.2 + 0.05 * i, lon=10.6 + 0.05 * i, n=3) for i in range(12)]       # the letter names Bavarian places
    rows = [_row('corpus_corporum', 'u', 'Nordovico'), _row('corpus_corporum', 'nw', 'Norwich', lat=52.63, lon=1.30, n=3)] + cloud_by
    n = len(rows)
    ex = {'qid': [None, 'Q3'] + [None] * 12, 'coord_kind': [None] * n, 'coord_radius_km': [np.nan] * n}
    far = pd.DataFrame({'doc': [0, 1] + [0] * 12, 'row': list(range(n))})
    G, out = _run(rows, gn, pairs=far, resources={'wdlatin': w, 'orbis': orbis}, extra_cols=ex)
    assert _f(out, 0) is None or _f(out, 0)['conf'] == 'low'                                       # a letter of Bavarian places, an English point
    alone = pd.DataFrame({'doc': list(range(n)), 'row': list(range(n))})                          # no document names a neighbour: nothing contradicts
    G, out = _run(rows, gn, pairs=alone, resources={'wdlatin': w, 'orbis': orbis}, extra_cols=ex)
    assert _f(out, 0) and _f(out, 0)['conf'] in ('medium', 'high') and abs(_f(out, 0)['lat'] - 52.63) < 0.01


# ---- fix 4: individual corrections -------------------------------------------------------------------------------------------------------------
def test_lucena_cordoba_is_the_village_not_the_province_capital():
    gn = [('lucena', '1', True, 'Lucena', 37.4089, -4.4853, 'P', 'PPL', 'ES', '51', 42000),
          ('cordoba', '2', True, 'Córdoba', 37.8916, -4.7727, 'P', 'PPLA2', 'ES', '51', 325000)]
    cloud = [_row('disco_sonnets', f'c{i}', f'Lugar {i}', lat=37.85 + 0.01 * i, lon=-4.75 + 0.01 * i, n=3) for i in range(10)]     # the poems name places round Córdoba
    rows = [_row('disco_sonnets', 'l', 'Lucena (Córdoba)')] + cloud
    pairs = pd.DataFrame({'doc': [0] * 11, 'row': list(range(11))})
    G = _geocoder(rows, gn, pairs)
    lat = G.P['lat'].to_numpy('float64')
    lon = G.P['lon'].to_numpy('float64')
    G.set_anchors(lat, lon)
    G.set_context(lat, lon, np.log1p(G.weight))
    f = G.head_alt(0)
    assert f and abs(f['lat'] - 37.4089) < 0.01 and f['tier'] == 'head_alt_head_locator' and f['conf'] == 'medium'      # Lucena, 59 km from the city that locates it
    # the head found by propagation (a placed 'Lucena' of another edition: no feature code in its result) is a locator case too
    rows3 = rows + [_row('letradas', 'lu', 'Lucena', lat=37.4089, lon=-4.4853, n=4)]
    G3 = _geocoder(rows3, gn, pd.DataFrame({'doc': [0] * 11 + [12], 'row': list(range(12))}))
    lat3, lon3 = G3.P['lat'].to_numpy('float64'), G3.P['lon'].to_numpy('float64')
    G3.set_anchors(lat3, lon3)
    G3.set_context(lat3, lon3, np.log1p(G3.weight))
    f3 = G3.head_alt(0)
    assert f3 and abs(f3['lat'] - 37.4089) < 0.01
    # a head 300 km from the bracketed place is not a locator: the old rule (documents decide) applies
    gn2 = [('lucena', '1', True, 'Lucena', 39.9, -0.9, 'P', 'PPL', 'ES', '1', 42000), gn[1]]
    G2 = _geocoder(rows, gn2, pairs)
    G2.set_anchors(lat, lon)
    G2.set_context(lat, lon, np.log1p(G2.weight))
    f2 = G2.head_alt(0)
    assert f2 is None or f2['tier'] != 'head_alt_head_locator'


def test_inns_take_the_point_of_the_same_inn_with_the_same_qualifier():
    rows = [_row('map_of_early_modern_london', 'CRKE1', 'Cross Keys Inn (Gracechurch Street)', lat=51.5122, lon=-0.0852),
            _row('map_of_early_modern_london', 'CRKE3', 'Cross Keys Inn (Holborn)'), _row('map_of_early_modern_london', 'CRKE2', 'Cross Keys Inn (St. John’s Street)'),
            _row('map_of_early_modern_london', 'CRKE9', 'Cross Keys Inn (Gracechurch Street)')]
    G, out = _run(rows, [])
    assert _f(out, 1) is None and G.why.get(1) == 'parish_qualifier' and _f(out, 2) is None             # other inns of the name: not Gracechurch Street's point
    assert G.church_sig(0) is not None and G.church_sig(0) == G.church_sig(3) != G.church_sig(1)


def test_county_discs_keep_out_the_points_nearer_to_a_neighbouring_county():
    norfolk = (52.65, 1.00, 75.0)
    assert not RGN.in_disc([51.88], [0.93], [norfolk])[0]                                                # The Hythe of Colchester (Essex), 86 km from the centre of Norfolk
    assert RGN.in_disc([52.63], [1.30], [norfolk])[0] and RGN.in_disc([52.95], [1.30], [norfolk])[0]     # Norwich, Cromer
    yorks = (54.00, -1.20, 140.0)
    assert RGN.in_disc([53.9], [-1.6], [yorks])[0]                                                       # the whole Yorkshire family is one county
    assert RGN.in_disc([53.9], [-1.6], [(53.85, -1.65, 85.0)])[0]                                         # the West Riding
    unknown = (52.0, -1.0, 40.0)
    assert RGN.in_disc([51.88], [-0.9], [unknown])[0]                                                    # a disc that is not a county of the list keeps the plain rule


def test_hearth_tax_headings_are_scoped_to_the_county_of_their_file():
    from ratlas.collections import hearth_tax as HT
    assert HT.place_key('Kirkby', 'Lancashire') == 'surf:kirkby @lancashire'
    assert HT.place_key('Kirkby', 'North Riding of Yorkshire') == 'surf:kirkby @north riding of yorkshire' != HT.place_key('Kirkby', 'Lancashire')

    class H:
        def __init__(self, tag, surface, zone='txt', kind='place'):
            self.tag, self.surface, self.zone, self.kind = tag, surface, zone, kind
    hits = [H('placeName@county', 'Lancashire'), H('placeName@parish', 'Kirkby'), H('placeName', 'Bristol', 'hdr')]
    assert HT.file_county(hits) == 'Lancashire'
    assert HT.file_county([H('placeName', 'Bristol', 'hdr'), H('placeName@ward', 'Ward of St Michael')]) == 'Bristol'
    assert HT.file_county([H('placeName@parish', 'Kirkby')]) == ''


def test_hearth_tax_extractor_gives_two_kirkbys_two_keys():
    from ratlas import helpers_batch1 as H
    NS = 'xmlns="http://www.tei-c.org/ns/1.0"'

    def doc(county, year):
        return (f'<TEI {NS}><teiHeader><fileDesc><titleStmt><title>Hearth tax <date when="{year}">{year}</date></title></titleStmt></fileDesc></teiHeader>'
                f'<text><body><div><head><placeName type="county">{county}</placeName></head><head><placeName type="parish">Kirkby</placeName></head>'
                f'<p><persName ana="h_tp">John Smith</persName><measure unit="hearths">2</measure></p></div></body></text></TEI>')
    keys = []
    for county in ('Lancashire', 'North Riding of Yorkshire'):
        ms, _ = H.synthetic('hearth_tax', doc(county, 1664))
        keys += [m.local_key for m in ms if m.kind == 'place' and m.surface == 'Kirkby']
        assert any(m.surface == county and m.local_key is None for m in ms)                         # the county heading itself is not split
    assert keys == ['surf:kirkby @lancashire', 'surf:kirkby @north riding of yorkshire']


def test_h15corpus_contradictory_geos_of_one_name_are_not_a_register_point():
    from ratlas.collections import h15corpus as H15
    pts = {'italia, milán': [(45.4642, 9.1895), (45.7819, 10.8638), (45.4642, 9.1895)], 'italia , milán': [(45.4642, 9.1895)], 'italia, nápoles': [(40.85, 14.27), (40.84, 14.25)]}
    assert H15.contradictory(pts) == {'italia, milán'}


def test_data_rows_of_the_check_are_loaded_by_the_loaders(tmp_path):
    ov = B.load_overrides(DATA)
    k = {(r.coll_slug, r.entity_key): (r.ov_lat, r.ov_lon) for r in ov.itertuples()}
    thorney = k[('map_of_early_modern_london', 'map_of_early_modern_london:THOR4')]
    assert abs(thorney[0] - 51.4995) < 0.01 and abs(thorney[1] + 0.1273) < 0.01                        # the island of Westminster, not Colnbrook
    pancras = k[('map_of_early_modern_london', 'map_of_early_modern_london:STPA105')]
    assert abs(pancras[0] - 51.513) < 0.01 and abs(pancras[1] + 0.092) < 0.01                          # the City parish, not Camden
    heath = k[('casebooks', 'surf:heath')]
    assert abs(heath[0] - 51.945) < 0.01 and abs(heath[1] + 0.657) < 0.01                              # Heath and Reach, Bedfordshire
    eff = MO.effective_verify(None, DATA, tmp_path)
    rej = pd.read_csv(eff, sep='\t', dtype=str, keep_default_na=False)
    keys = set(zip(rej['coll_slug'], rej['entity_key']))
    for key in (('cidtc', 'cidtc:pl1170'), ('corpus_corporum', 'surf:aegea'), ('corpus_corporum', 'surf:becam'), ('ckcc_epistolarium', 'surf:nordovico'), ('corpus_corporum', 'surf:palatinus'),
                ('cidtc', 'cidtc:pl1438'), ('old_bailey', 'surf:new street'), ('old_bailey', 'surf:st helens'), ('old_bailey', 'surf:bow'),
                ('religionsfrieden_quellenedition', 'religionsfrieden_quellenedition:e000001_augsburger_religionsfrieden#werdenberg_grafschaft')):
        assert key in keys, key
    kinds = V.load_kind_overrides(DATA)
    assert kinds[('sandrart', 'sandrart:place-1080')] == 'n' and kinds[('tagebuecher_christian_ii_anhalt', 'tagebuecher_christian_ii_anhalt:jettenbach_schloss')] == 'b'


# ---- fix 5: display data -----------------------------------------------------------------------------------------------------------------
def _pl(rows):
    cols = dict(coll_slug='x', entity_key='k', qid=None, qid_exo=None, coord_source_geo='wikidata', geo_lat=48.69, geo_lon=9.14, coord_radius_km=150.0, coord_kind='area',
                place_kind='area', n_in_period=10)
    return pd.DataFrame([cols | r for r in rows])


def test_continents_get_one_point_and_one_radius_whatever_the_source():
    pl = _pl([dict(entity_key='a', qid='Q46'), dict(entity_key='b', qid=None, qid_exo='Q46', geo_lat=50.0, geo_lon=15.0, coord_radius_km=1500.0, coord_source_geo='name_exonym'),
              dict(entity_key='c', qid='Q142', geo_lat=46.6, geo_lon=2.5), dict(entity_key='d', qid='Q828', geo_lat=19.0, geo_lon=-96.0),
              dict(entity_key='e', qid='Q46', coord_source_geo='none', geo_lat=np.nan, geo_lon=np.nan)])
    rep = {}
    out = V.apply_continents(pl, rep).set_index('entity_key')
    assert (out.loc['a', 'geo_lat'], out.loc['a', 'geo_lon'], out.loc['a', 'coord_radius_km']) == (50.0, 15.0, 1500.0) == (out.loc['b', 'geo_lat'], out.loc['b', 'geo_lon'], out.loc['b', 'coord_radius_km'])
    assert out.loc['c', 'geo_lat'] == 46.6                                                         # France is not a continent
    assert (out.loc['d', 'geo_lat'], out.loc['d', 'geo_lon'], out.loc['d', 'coord_radius_km']) == (5.0, -75.0, 4000.0)
    assert np.isnan(out.loc['e', 'geo_lat'])                                                       # an unplaced local stays unplaced
    assert rep['continents']['locals'] == 3 and V.EXONYM_QID['Europe'] == 'Q46' and V.EXONYM_QID['Americas'] == 'Q828'


def test_continents_and_rivers_have_their_own_types():
    from ratlas import placetypes as PT
    assert PT.TYPES['continent'].parent == 'area' and PT.TYPES['river'].parent == 'natural' and PT.kind_of('continent') == 'a' and PT.kind_of('river') == 'n'
    assert PT.wd_type('Q5107', 'continent') == 'continent'


def test_casebooks_low_candidates_are_listed_not_placed():
    pl = pd.DataFrame({'coll_slug': 'casebooks', 'entity_key': ['casebooks:P1', 'casebooks:P2', 'casebooks:P3', 'casebooks:P4'], 'name': ['a', 'b', 'c', 'd'],
                       'n_in_period': [30, 5, 7, 9], 'cb_place_id': ['P1', 'P2', 'P3', 'P4'], 'cb_entry': ["St George's Lane", 'Stondon', 'Bedfordshire', 'Exeter College'],
                       'cb_county': ['London', 'Bedfordshire', 'Bedfordshire', 'Oxfordshire'], 'cb_parish': None, 'cb_match': ['high', 'high', 'medium', 'low'],
                       'cb_geo_conf': ['low', 'high', 'low', 'low'], 'cb_geo_method': 'x', 'cb_lat': [51.556, 52.0, 52.1, 51.75], 'cb_lon': [-0.128, -0.3, -0.4, -1.25],
                       'cb_radius_km': [2.0, 2.0, 25.0, 2.0], 'cb_used': [False, True, True, False], 'coord_source_geo': ['none', 'casebooks_directory', 'casebooks_directory', 'none'],
                       'geo_lat': np.nan, 'geo_lon': np.nan})
    out = V.casebooks_low_candidates(pl)
    assert list(out['entity_key']) == ['casebooks:P1']                                             # matched, low geocoding, point-sized, not used; P4's match itself is low
    assert bool(out['london'].iloc[0]) and out['suggested_radius_km'].iloc[0] == 15.0 and out['cb_entry'].iloc[0] == "St George's Lane"
    assert list(V.casebooks_low_candidates(pl.iloc[0:0]).columns)[:2] == ['coll_slug', 'entity_key']


def test_a_continent_has_one_point_in_the_global_table_too(tmp_path):
    from test_reconcile_gates import _wd
    from test_reconcile_v2 import _entity, _make, _mention
    rows = [_mention('colla', 'place', 'Europa', key='colla:eu', reg=True, auth={'wikidata': 'Q46'}), _mention('colla', 'place', 'Europa', key='colla:eu', reg=True, auth={'wikidata': 'Q46'})]
    ents = [_entity('colla', 'place', 'colla:eu', 'Europa', 48.69, 9.14, auth={'wikidata': 'Q46'})]
    wd = [_wd('Q46', {'en': 'Europe', 'de': 'Europa'}, classes=['Q5107'], lat=48.69, lon=9.14)]
    v2, rd = _make(tmp_path, rows, ents, extra_wd=wd)
    out = tmp_path / 'out'
    rep = V.build(v2, rd, out)
    pg = pd.read_parquet(out / 'places_global.parquet').set_index('id')
    assert (pg.loc['pl:Q46', 'lat'], pg.loc['pl:Q46', 'lon'], pg.loc['pl:Q46', 'coord_radius_km']) == (50.0, 15.0, 1500.0) and pg.loc['pl:Q46', 'coord_kind'] == 'area'
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index(['coll_slug', 'entity_key'])
    assert (geo.loc[('colla', 'colla:eu'), 'lat'], geo.loc[('colla', 'colla:eu'), 'lon']) == (50.0, 15.0) and rep['continents']['locals'] == 1
