"""Pure functions of the globe data shards (no data needed)."""
import hashlib

from ratlas import cards_globe as G


def test_shards():
    assert G.doc_shard(0) == 0 and G.doc_shard(64) == 0 and G.doc_shard(130) == 2
    q = 'Q72'
    assert G.img_shard(q) == int.from_bytes(hashlib.sha1(q.encode()).digest(), 'big') % 32
    assert 0 <= G.img_shard('Q1') < 32


def test_kind_letter():
    """v2.2: five-way kind c / a / n / b / u."""
    assert G.kind_letter('city') == 'c' and G.kind_letter('area') == 'a' and G.kind_letter('other') == 'u'       # the atlas's old three-way words still convert
    assert G.kind_letter('other', wd_labels=['river']) == 'n'
    assert G.kind_letter('other', gn_fcodes=['ISL']) == 'n'
    assert G.kind_letter('other', tgn_types=['inhabited place']) == 'c'
    assert G.kind_letter('other', tgn_types=['sea']) == 'n'
    assert G.kind_letter('other', wd_labels=['castle']) == 'b'
    assert G.kind_letter('other', notes=['fluss']) == 'n'
    assert G.kind_letter('u', notes=['stre']) == 'b' and G.kind_letter('u') == 'u'


def test_original_name_is_never_a_common_noun():
    assert G.orig_name('Newp', 'Newp', 'Newport Pagnell') == 'Newp'
    assert G.orig_name(None, 'this towne', 'Great Linford') == 'Great Linford'
    assert G.orig_name('alhier', 'Magdeburg', 'Magdeburg') == 'Magdeburg'


def test_letters_network_edges_coverage_and_ambiguity():
    import pandas as pd
    ends = pd.DataFrame([
        # doc 1: Zürich -> Basel (1549), doc 2: Zürich -> Basel (1551), doc 3: Basel -> Zürich, doc 4: only a sender, doc 5: sender on the globe, addressee not
        # doc 6: sender ends at two points (ambiguous), doc 7: inside one city, doc 8: out of the quarters
        dict(coll_slug='bullinger', source_path='1', doc_id='a', role='sent_from', entity_key='z', doc_year=1549.0),
        dict(coll_slug='bullinger', source_path='1', doc_id='a', role='sent_to', entity_key='b', doc_year=1549.0),
        dict(coll_slug='bullinger', source_path='2', doc_id='a', role='sent_from', entity_key='z', doc_year=1551.0),
        dict(coll_slug='bullinger', source_path='2', doc_id='a', role='sent_to', entity_key='b', doc_year=1551.0),
        dict(coll_slug='ckcc_epistolarium', source_path='3', doc_id='a', role='sent_from', entity_key='b2', doc_year=1630.0),
        dict(coll_slug='ckcc_epistolarium', source_path='3', doc_id='a', role='sent_to', entity_key='z2', doc_year=1630.0),
        dict(coll_slug='bullinger', source_path='4', doc_id='a', role='sent_from', entity_key='z', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='5', doc_id='a', role='sent_from', entity_key='z', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='5', doc_id='a', role='sent_to', entity_key='nowhere', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='6', doc_id='a', role='sent_from', entity_key='z', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='6', doc_id='a', role='sent_from', entity_key='b', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='6', doc_id='a', role='sent_to', entity_key='b', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='7', doc_id='a', role='sent_from', entity_key='z', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='7', doc_id='a', role='sent_to', entity_key='z', doc_year=1550.0),
        dict(coll_slug='bullinger', source_path='8', doc_id='a', role='sent_from', entity_key='z', doc_year=1801.0),
        dict(coll_slug='bullinger', source_path='8', doc_id='a', role='sent_to', entity_key='b', doc_year=None)])
    row_of = {('bullinger', 'z'): 0, ('bullinger', 'b'): 1, ('ckcc_epistolarium', 'z2'): 0, ('ckcc_epistolarium', 'b2'): 1}
    sidx = {'bullinger': 0, 'ckcc_epistolarium': 1}
    edges, cov = G.build_letters(ends, row_of, sidx)
    e = {(f, t): (n, q, sl) for f, t, n, q, sl in edges}
    assert e[(0, 1)][0] == 3 and e[(0, 1)][2] == [[0, 3]] and e[(0, 1)][1][(1549 - 1450) // 25] == 1 and e[(0, 1)][1][(1551 - 1450) // 25] == 1
    assert sum(e[(0, 1)][1]) == 2                                      # the letter of 1801 / without a year is an edge but in no quarter
    assert e[(1, 0)][0] == 1 and e[(1, 0)][2] == [[1, 1]] and e[(1, 0)][1][(1630 - 1450) // 25] == 1
    assert e[(0, 0)][0] == 1                                           # a letter inside one city
    assert edges[0][2] == 3 and [x[2] for x in edges] == sorted((x[2] for x in edges), reverse=True)
    c = {x['slug']: x for x in cov}
    assert c['bullinger']['letters'] == 7 and c['bullinger']['sender_place'] == 7 and c['bullinger']['addressee_place'] == 6
    assert c['bullinger']['both_places'] == 6 and c['bullinger']['ambiguous'] == 1 and c['bullinger']['both_on_globe'] == 4
    # the denominator adds the typed letters that have no end at all
    ld = pd.DataFrame({'doc_key': ['x1', 'x2'], 'coll_slug': ['ckcc_epistolarium', 'ckcc_epistolarium']})
    c2 = {x['slug']: x for x in G.build_letters(ends, row_of, sidx, ld)[1]}
    assert c2['ckcc_epistolarium']['letters'] == 3 and c2['ckcc_epistolarium']['share_both_on_globe'] == round(1 / 3, 4)
    # with_docs: the letters of each edge, in the order of the edges
    edges3, _, docs3 = G.build_letters(ends, row_of, sidx, with_docs=True)
    assert edges3 == edges and [len(d) for d in docs3] == [x[2] for x in edges]
    dk = {(f, t): d for (f, t, *_), d in zip(edges3, docs3)}
    assert len(dk[(0, 1)]) == 3 and len(set(dk[(0, 1)])) == 3 and len(dk[(0, 0)]) == 1


def test_meaningful_types():
    assert G.meaningful_types(['human settlement', 'city', 'Q5', 'City', 'big city', 'port', 'free imperial city', 'x']) == \
        ['city', 'port', 'free imperial city']


def test_pick_artist():
    assert G.pick_artist('<a href="x">Jane Doe</a>', '') == 'Jane Doe'
    assert G.pick_artist('', 'Photo by <b>A. B.</b>') == 'Photo by A. B.'
    assert G.pick_artist('', 'Own work') == 'unknown author'
    assert G.pick_artist(None, None) == 'unknown author'


def test_context_only_for_t1_t2():
    assert G.clean_context('a  b\nc', 'T1') == 'a b c'
    assert G.clean_context('a b c', 'T3') is None
    assert len(G.clean_context('x' * 500, 'T2')) == G.CONTEXT_MAX


def test_inception_and_names():
    assert G.inception_value('+1250-00-00T00:00:00Z', 1250.0, 9.0) == 1250
    assert G.inception_value('13th century', 1250.0, 7.0) == '13th century'
    assert G.inception_value(None, None, None) is None
    assert G.lang_names('Roma', ['roma', 'Urbs', 'Roma ', 'A', 'B', 'C'], cap=4) == ['Roma', 'Urbs', 'A', 'B']


def test_doc_sort_key_orders_by_year_then_date():
    rows = [['b', 't', '1600-01-01', 1600], ['a', 't', None, None], ['c', 't', '1550', 1550]]
    assert [r[0] for r in sorted(rows, key=G.doc_sort_key)] == ['c', 'b', 'a']


def test_surface_form_strips_leading_preposition():
    assert G.surface_form('of  Boughton ') == 'Boughton'
    assert G.surface_form('in Oxford sheere') == 'Oxford sheere'
    assert G.surface_form('Inzersdorf') == 'Inzersdorf'
    assert G.surface_form('at X') == 'at X'


def test_unnamed_constant_and_generic_check():
    from ratlas import generic_terms as GT
    assert G.UNNAMED == 'Unnamed place' and GT.place_generic('this towne') == 'common_noun' and GT.place_generic(G.UNNAMED) is None
