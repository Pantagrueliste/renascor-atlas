"""Tests of the pure logic of the extra-documents step of the NER layer (readers, duplicate control, decisions)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import ner2_layer_extra_dedup as D  # noqa: E402
import ner2_layer_extra_extract as E  # noqa: E402
import ner2_layer_extra_finalize as F  # noqa: E402
import ner2_layer_extra_readers as R  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------------------------------------------------
def test_decode_any_utf16_and_nul():
    assert R.decode_any('<a>é</a>'.encode('utf-16')) == '<a>é</a>'
    assert R.decode_any(b'ab\x00cd') == 'abcd'
    assert R.decode_any('caf\xe9'.encode('cp1252')) == 'café'


def test_parse_xml_loose_duplicate_ids_and_declaration():
    xml = '<?xml version="1.0" encoding="UTF-8"?><TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p xml:id="a">x</p><p xml:id="a">y</p></body></text></TEI>'
    root = R.parse_xml_loose(xml.encode('utf-16'))                       # declaration says UTF-8, bytes are UTF-16: the declaration is dropped
    assert len(list(root.iter('{*}p'))) == 2
    with pytest.raises(R.Skip):
        R.parse_xml_loose(b'')


def test_has_place_markup_sees_front_and_not_header():
    head = '<teiHeader><fileDesc><pubPlace><placeName>Paris</placeName></pubPlace></fileDesc></teiHeader>'
    ns = 'xmlns="http://www.tei-c.org/ns/1.0"'
    plain = f'<TEI {ns}>{head}<text><body><p>Paris</p></body></text></TEI>'
    front = f'<TEI {ns}>{head}<text><front><stage><placeName>Paris</placeName></stage></front><body><p>x</p></body></text></TEI>'
    typed = f'<TEI {ns}><text><body><p><rs type="place">Paris</rs></p></body></text></TEI>'
    assert not R.has_place_markup(R.parse_xml_loose(plain.encode()))
    assert R.has_place_markup(R.parse_xml_loose(front.encode()))
    assert R.has_place_markup(R.parse_xml_loose(typed.encode()))


def test_top_level_bodies_and_first_body_text():
    ns = 'xmlns="http://www.tei-c.org/ns/1.0"'
    xml = f'<TEI {ns}><text><body><p>un deux</p></body></text><text><body><p>un deux<placeName>Rome</placeName></p></body></text></TEI>'
    root = R.parse_xml_loose(xml.encode())
    assert len(R.top_level_bodies(root)) == 2
    txt, place, has_body, extra = E.tei_root_text(root)
    assert txt.strip() == 'un deux' and place and has_body and extra >= 2        # place found in the second body, text from the first only


def test_ref_text_keeps_alphabetic_tokens_only():
    xml = '<tok_dipl id="1" utf="Hie"/><tok_dipl id="2" utf="."/><tok_dipl id="3" utf="nach"/><tok_dipl id="4" utf="12"/>'
    assert R.ref_text(xml) == 'Hie nach'
    assert R.ref_sigle('<x sigle="F028" y="">', 'zz') == 'F028'


def test_tml_text_drops_chrome_and_boilerplate():
    html = ('<html><head><title>t</title></head><body><nav>menu</nav><p>Dionysius De arte musicali</p>'
            '<p>Previous part --&gt;</p><p>The Center for the History of Music Theory and Literature is a research center of the Indiana University.</p></body></html>')
    out = R.tml_text(html)
    assert 'menu' not in out and 'Center' not in out and out.startswith('Dionysius')
    assert R.TML_SITE_PAGES.match('index') and not R.TML_SITE_PAGES.match('GENTSPE')


def test_pepys_text_cuts_to_the_diary_and_drops_editor_notes():
    raw = 'licence\nPREFACE\nblah\nJANUARY 1659-1660\nI lived in Axe Yard [the street was near Whitehall] and went home.\nEND OF THE DIARY.\nAPPENDIX'
    t = R.pepys_text(raw)
    assert 'Axe Yard' in t and 'Whitehall' not in t and 'APPENDIX' not in t and 'blah' not in t
    with pytest.raises(R.Skip):
        R.pepys_text('no landmarks')


def test_bartoli_year_and_text():
    md = '---\nlayout: letter\ndocdate: 07-01-1563\nplace: Venezia\n---\n\n[001r]  \nIllustrissimo Signor\n---\nCon la di'
    assert R.bartoli_docdate_year(md) == 1563
    t = R.bartoli_md_text(md)
    assert 'Venezia' not in t and '001r' not in t and 'Illustrissimo Signor' in t and '---' not in t


def test_bible_pages():
    kjv = 'For other versions see X . ← Numbers The King James Bible (1769) Numbers — Chapter XVI Layout 1 &#8203; 1 Now Korah, the son of Izhar &#32;'
    assert R.bible_page_text(kjv, 'kjv').startswith('1 Now Korah')
    dio = 'Questo testo è da formattare . <dc:title> Sacra Bibbia </dc:title> Sacra Bibbia &#x25c4; Esodo - capitolo 1 Esodo - capitolo 3 &#x25ba; CAP. II 1 OR un uomo andò'
    assert R.bible_page_text(dio, 'diodati').startswith('CAP. II')
    lu = ('«&#160; Martin Luther (Übersetzer): Das Newe Testament Deutzsch Fertig! Die Schreibweise folgt dem Originaltext. (in eckigen Klammern). '
          '&#x5b; 9 &#x5d; &#x5b; WA.DB 6,14 &#x5d; Euangelion Sanct Matthes. ' + 'Djs ist das buch von der gepurt. ' * 30 + ' ↑ Vorlage: zihist . Hier nach WA.DB korrigiert.')
    t = R.bible_page_text(lu, 'luther')
    assert t.startswith('Euangelion') and 'WA.DB' not in t and 'Vorlage' not in t


def test_diodati_check_rejects_modern_text_and_index_pages():
    old = ('CAP. II 1 OR un uomo della famiglia di Levi andò, e prese una figliuola. ' + 'perciocchè egli avea figliuoli, e Iddio disse. ' * 30)
    assert R.diodati_check(old) == old
    modern = 'In principio Dio creò il cielo e la terra. La terra era informe e deserta. ' * 20
    with pytest.raises(R.Skip) as e:
        R.diodati_check(modern)
    assert e.value.status == 'translation'
    with pytest.raises(R.Skip) as e:
        R.diodati_check('Indice Capitolo I Capitolo II Capitolo III')
    assert e.value.status == 'site_page'


def test_fold_name_matches_catalogue_and_file_names():
    assert R.fold_name('Matth_usevangelium') == R.fold_name('Matthäusevangelium')
    assert R.fold_name('1__Johannes') == R.fold_name('1_Johannes')


def _emothe(xml):
    return R.emothe_info(R.parse_xml_loose(xml.encode()))


def test_emothe_translation_rules():
    ns = 'xmlns="http://www.tei-c.org/ns/1.0"'
    orig = f'<TEI {ns}><teiHeader><fileDesc><sourceDesc><bibl>Lope, Arte nuevo, ed. Arellano, Madrid 2020.</bibl></sourceDesc></fileDesc></teiHeader><text/></TEI>'
    assert R.emothe_decision(_emothe(orig))[0] is True
    modern = (f'<TEI {ns}><teiHeader><fileDesc><titleStmt><title type="traduccion">Mudarra</title><editor role="traductor">Eugène Baret</editor></titleStmt>'
              '<sourceDesc><bibl>Traduit par Baret. Paris: Didier, 1869.</bibl></sourceDesc></fileDesc></teiHeader><text/></TEI>')
    keep, why = R.emothe_decision(_emothe(modern))
    assert keep is False and '1869' in why
    early = (f'<TEI {ns}><teiHeader><fileDesc><titleStmt><editor role="translator">John Dancer</editor></titleStmt>'
             '<sourceDesc><bibl>Translated by John Dancer. London: John Starkey, 1660.</bibl></sourceDesc></fileDesc></teiHeader><text/></TEI>')
    assert R.emothe_decision(_emothe(early))[0] is True


def test_wikisource_plain():
    t = 'Questo testo è riletto e controllato . <dc:title> Ecloga </dc:title> <dc:date></dc:date> //it.wikisource.org/w/index.php?title=X&oldid=- Testo vero.'
    assert R.wikisource_plain(t) == 'Testo vero.'
    assert R.wikisource_plain('Verso. Esta obra se encuentra en dominio público . Public domain false') == 'Verso.'


# ---------------------------------------------------------------------------------------------------------------------
# extractor helpers
# ---------------------------------------------------------------------------------------------------------------------
def test_match_key_exact_prefix_and_size():
    keys = {'abc', 'x' * 45 + 'AAA', 'x' * 45 + 'BBB'}
    assert E.match_key('abc', keys) == 'abc'
    assert E.match_key('zzz', keys) is None
    cut = 'x' * 45
    sizes = {'x' * 45 + 'AAA': 100, 'x' * 45 + 'BBB': 5000}
    assert E.match_key(cut, keys, sizes, 4900) == 'x' * 45 + 'BBB'              # several candidates: closest in length
    assert E.match_key(cut, keys, sizes, 4900, taken={'x' * 45 + 'BBB'}) == 'x' * 45 + 'AAA'
    assert E.match_key(cut, keys) is None                                         # ambiguous without sizes


def test_catalogue_name_keys():
    assert E.ws_key('WS_it_Del_modo_di_trattare_i_popoli_della_Valdichiana_ribellati.tx', 'it') == 'Del_modo_di_trattare_i_popoli_della_Valdichiana_ribellati'
    assert E.bible_key('WS_en_Bible_pid4926081_Bible__King_James__Numbers_Chapter_16') == 'en:4926081'
    assert E.bible_key('WS_it_Bible_Diodati160_pid52473_Sacra_Bibbia__Diodati_1607__Vecchio_Testamento_Esodo_capitol') == 'it:52473'
    assert E.bible_key('WS_de_LutherNT1522_Matth_usevangelium') == 'luther:matthusevangelium'
    assert E.stem('gams_o_srbas.1543.txt') == 'gams_o_srbas.1543'
    assert E.stem('latin_xmls\\12135.xml.txt') == '12135'


# ---------------------------------------------------------------------------------------------------------------------
# shingles and duplicate decisions
# ---------------------------------------------------------------------------------------------------------------------
def _text(n, seed):
    rng = np.random.default_rng(seed)
    return ' '.join(f'w{int(x)}' for x in rng.integers(0, 5000, n))


def test_shingle_hashes_fold_and_sampling():
    a = D.shingle_hashes('Vn ſtraunge Cittie of Rome and the Pope, in the Citie of Paris, with many Men.' * 50)
    b = D.shingle_hashes('vn straunge cittie of rome and the pope in the citie of paris with many men ' * 50)
    assert len(a) and np.array_equal(a, b)                                       # case, long s, punctuation do not matter
    assert len(D.shingle_hashes('too short')) == 0
    big = D.shingle_hashes(_text(40000, 1))
    assert 0.18 < len(big) / 40000 < 0.32                                       # one shingle in four


def test_cand_index_containment_and_union():
    base = _text(8000, 2)
    a = base                                                                     # candidate A is a piece of two references
    r1 = base[: len(base) // 2] + ' ' + _text(4000, 3)
    r2 = base[len(base) // 2:] + ' ' + _text(4000, 4)
    ha = D.shingle_hashes(a)
    idx = D.CandIndex(['A', 'B'], [ha, D.shingle_hashes(_text(3000, 5))])
    c1, m1 = idx.scan(D.shingle_hashes(r1))
    c2, m2 = idx.scan(D.shingle_hashes(r2))
    assert 0.35 < c1[0] / idx.nsh[0] < 0.65 and 0.35 < c2[0] / idx.nsh[0] < 0.65 and 1 not in c1
    u = idx.union_per_doc(np.concatenate([m1, m2]))
    assert u[0] / idx.nsh[0] > 0.95 and u[1] == 0                                # the union holds A, none of B
    c, m = idx.scan(D.shingle_hashes(_text(5000, 9)))
    assert c == {}


def test_cand_index_roundtrip(tmp_path):
    idx = D.CandIndex(['A', 'B'], [D.shingle_hashes(_text(900, 6)), D.shingle_hashes(_text(900, 7))])
    idx.save(tmp_path / 'c.npz')
    j = D.CandIndex.load(tmp_path / 'c.npz')
    assert j.names == ['A', 'B'] and np.array_equal(j.U, idx.U) and np.array_equal(j.D, idx.D)


def test_decide_references_and_intra_order():
    layer = {'max': {'A': (0.9, 'EEBO/A1'), 'B': (0.1, 'x')}, 'union': {'A': 0.9, 'B': 0.8, 'C': 0.0}}
    tagged = {'max': {'D': (0.5, 'cidtc:f.xml')}, 'union': {}}
    intra = {'E': [('F', 0.95)], 'F': [('E', 0.30)], 'G': [('E', 0.1)]}
    nsh = {'A': 10, 'B': 10, 'C': 10, 'D': 10, 'E': 5, 'F': 50, 'G': 7}
    out = F.decide(['A', 'B', 'C', 'D', 'E', 'F', 'G'], nsh, layer, tagged, intra, 0.3)
    assert out['A'][0] == 'dup_layer' and out['A'][1] == 'EEBO/A1'
    assert out['B'][0] == 'dup_layer' and 'union' in out['B'][1]                  # union of several documents
    assert 'C' not in out
    assert out['D'][0] == 'dup_tagged'
    assert out['E'][0] == 'dup_extra' and out['E'][1] == 'F'                      # the larger one is kept
    assert 'F' not in out and 'G' not in out


def test_guess_lang_flags_only_clear_mismatches():
    it = 'Il mio signore che sono nel della di per con non è la le gli che ' * 40
    la = 'et in est non ad cum ut quod qui quae quam sed si ex de per ab ' * 40
    assert F.lang_flag(F.guess_lang(it), 'it') == ''
    assert F.lang_flag(F.guess_lang(la), 'it') == 'la'
    assert F.lang_flag(F.guess_lang('Zxq plmk wvrt'), 'fr') == ''
