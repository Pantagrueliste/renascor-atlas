"""Tests of the NER passage extractor on small TEI fixtures (ratlas.ner.tei, ner.langs, ner.editions)."""
from ratlas.ner import editions as E
from ratlas.ner import tei
from ratlas.ner.langs import guess_lang, norm_lang

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def tei_doc(body, header='', front='', back='', lang=None):
    la = f' xml:lang="{lang}"' if lang else ''
    return (f'<?xml version="1.0"?><TEI {NS}><teiHeader><fileDesc><titleStmt><title>T <placeName>Rom</placeName></title></titleStmt>'
            f'{header}</fileDesc></teiHeader><text{la}>{front}<body>{body}</body>{back}</text></TEI>').encode('utf-8')


def run(body, opts=None, **kw):
    r = tei.document_passages(tei_doc(body, **kw), opts=opts)
    assert r['error'] is None
    return r


def surfaces(r, label=None):
    return [(s['label'], p['text'][s['start']:s['end']]) for p in r['passages'] for s in p['spans'] if label in (None, s['label'])]


def alltext(r):
    return '\n'.join(p['text'] for p in r['passages'])


FILL = ' lorem ipsum dolor sit amet consectetur'


def test_spans_match_the_text_and_header_is_not_read():
    r = run('<p>Iohannes de <placeName>Paris</placeName> et <persName>Petrus</persName>' + FILL + '</p>')
    assert surfaces(r) == [('LOC', 'Paris'), ('PER', 'Petrus')]
    assert 'Rom' not in alltext(r)                       # the title in the header
    sp = r['passages'][0]['spans'][0]
    assert sp['tag'] == 'placeName' and sp['has_ref'] is False


def test_all_place_tags_and_typed_name_rs():
    body = ('<p><settlement>Basel</settlement> <country>Helvetia</country> <region>Alsatia</region> <geogName>Rhenus</geogName> '
            '<name type="place">Roma</name> <rs type="person">Cicero</rs> <name type="org">Senatus</name> <name>Anon</name> ' + FILL + '</p>')
    r = run(body)
    assert surfaces(r, 'LOC') == [('LOC', 'Basel'), ('LOC', 'Helvetia'), ('LOC', 'Alsatia'), ('LOC', 'Rhenus'), ('LOC', 'Roma')]
    assert surfaces(r, 'PER') == [('PER', 'Cicero')]       # the org and the untyped name are left as O


def test_choice_reads_the_diplomatic_form_and_flags_it():
    body = ('<p>Visitavit <choice><abbr>Lond.</abbr><expan>Londinum</expan></choice> et <choice><orig>Lõdres</orig><reg>Londres</reg></choice> '
            'et <choice><sic>Pariss</sic><corr>Paris</corr></choice>' + FILL + '</p>')
    r = run(body)
    t = alltext(r)
    assert 'Lond.' in t and 'Lõdres' in t and 'Pariss' in t
    assert 'Londinum' not in t and 'Londres' not in t and 'Paris ' not in t
    assert 'choice' in r['passages'][0]['flags']


def test_choice_without_text_in_the_diplomatic_child_falls_back_to_the_editorial_reading():
    body = '<p>tem<choice><orig><g ref="#crossedp"/></orig><reg>por</reg></choice>e ' + FILL + '</p>'
    r = run(body)
    assert 'tempore' in alltext(r)
    assert 'choice_fallback' in r['passages'][0]['flags']


def test_subst_app_supplied_ex_del():
    body = ('<p>alpha <subst><del>old</del><add>new</add></subst> <app><lem>lemma</lem><rdg>variant</rdg></app> '
            'be<supplied>ta</supplied> gam<ex>ma</ex> <del>gone</del> end' + FILL + '</p>')
    r = run(body)
    t = alltext(r)
    assert 'new' in t and 'lemma' in t
    for gone in ('old', 'variant', 'ta ', 'gone'):
        assert gone not in t
    assert 'gam ' in t or 'gam\n' in t
    fl = set(r['passages'][0]['flags'])
    assert {'subst', 'app', 'ex_dropped', 'supplied_dropped', 'del_dropped'} <= fl


def test_line_break_without_break_joins_the_word():
    body = '<p>Lon-<lb break="no"/>\n   dres and Pa<lb/>ris' + FILL + '</p>'
    t = alltext(run(body))
    assert 'Lon-dres' in t and 'Pa ris' in t


def test_non_primary_parts_are_not_read():
    body = ('<div type="letter"><p>Main <placeName>Basel</placeName>' + FILL + '</p>'
            '<note>Editor note about <placeName>Berlin</placeName></note>'
            '<note type="footnote">Fussnote <placeName>Wien</placeName></note>'
            '<note place="margin">Margin <placeName>Napoli</placeName></note>'
            '<note type="marginal">Marginal <persName>Marcus</persName></note>'
            '<fw type="header">RUNNING HEAD <placeName>Linz</placeName></fw>'
            '<listPlace><place><placeName>Register</placeName></place></listPlace>'
            '<app><lem>L</lem><rdg>Apparatus <placeName>Graz</placeName></rdg></app></div>'
            '<div type="commentary"><p>Kommentar <placeName>Bonn</placeName></p></div>'
            '<div type="translation"><p>Translation <placeName>Munich</placeName></p></div>'
            '<div decls="#translation"><p>Translated <placeName>Cologne</placeName></p></div>')
    r = run(body, front='<front><p>Front <placeName>Prague</placeName></p></front>', back='<back><p>Back <placeName>Brno</placeName></p></back>')
    got = {x[1] for x in surfaces(r)}
    assert got == {'Basel', 'Napoli', 'Marcus'}
    assert 'Berlin' not in alltext(r) and 'Prague' not in alltext(r)


def test_nested_outermost_wins_and_comma_lists_are_split():
    body = ('<p><persName>Johannes von <placeName>Zürich</placeName></persName> reiste nach '
            '<placeName><settlement>Oxford</settlement>, <country>England</country></placeName> und '
            '<placeName>Sankt <settlement>Gallen</settlement></placeName>' + FILL + '</p>')
    r = run(body)
    assert surfaces(r) == [('PER', 'Johannes von Zürich'), ('LOC', 'Oxford'), ('LOC', 'England'), ('LOC', 'Sankt Gallen')]
    p = r['passages'][0]
    assert [p['text'][n['start']:n['end']] for n in p['nested']] == ['Zürich', 'Gallen']


def test_org_is_not_a_label_but_a_place_inside_keeps_its_own():
    r = run('<p><orgName>University of <placeName>Basel</placeName></orgName> and <orgName>Senate</orgName>' + FILL + '</p>')
    assert surfaces(r) == [('LOC', 'Basel')]
    p = r['passages'][0]
    assert len(p['orgs']) == 2


def test_generic_and_placeholder_tags_are_ignored_not_labelled():
    r = run('<p>He went to <rs type="place">the town</rs> with <persName>his brother</persName> and <persName>S.n.</persName> in <placeName>Wien</placeName>' + FILL + '</p>')
    assert surfaces(r) == [('LOC', 'Wien')]
    assert len(r['passages'][0]['ignored']) == 3


def test_type_map_and_strip_lead():
    body = '<p>Mrs Churchyeard <rs type="address" key="P1">of Stantone</rs> a tailor' + FILL + '</p>'
    assert surfaces(run(body)) == []                      # 'address' is unknown without the edition's type_map
    o = tei.Options(type_map={'address': 'place'}, strip_lead=('of', 'at'))
    assert surfaces(run(body, opts=o)) == [('LOC', 'Stantone')]


def test_passages_are_at_most_200_words_and_never_cut_a_span():
    words = ' '.join(f'w{i}' for i in range(150))
    paras = ''.join(f'<p>{words} <placeName>Ort{k}</placeName> end.</p>' for k in range(8))
    r = run(paras)
    ps = r['passages']
    assert len(ps) > 1 and all(p['words'] <= 200 for p in ps)
    assert sorted(s for _, s in surfaces(r)) == sorted(f'Ort{k}' for k in range(8))
    for p in ps:
        for s in p['spans']:
            assert p['text'][s['start']:s['end']].startswith('Ort')


def test_one_long_paragraph_is_cut_at_sentence_ends():
    sent = ' '.join(f'w{i}' for i in range(30)) + '.'
    body = '<p>' + ' '.join(sent for _ in range(20)) + ' Dann <placeName>Wien</placeName>.</p>'
    r = run(body)
    assert len(r['passages']) >= 3 and all(p['words'] <= 200 for p in r['passages'])
    assert all(p['text'].rstrip().endswith('.') for p in r['passages'][:-1])
    assert surfaces(r) == [('LOC', 'Wien')]


def test_a_span_with_a_break_inside_is_one_span():
    r = run('<p><persName>Maria<lb/>Theresia</persName> und <placeName>Neu<lb break="no"/>stadt</placeName>' + FILL + '</p>')
    assert surfaces(r) == [('PER', 'Maria Theresia'), ('LOC', 'Neustadt')]


def test_too_long_span_marks_the_passage():
    long = ' '.join(f'x{i}' for i in range(30))
    r = run(f'<p><placeName>{long}</placeName> and more words to fill{FILL}</p>')
    assert r['passages'][0]['dropped_span'] is True and surfaces(r) == []


def test_language_xml_lang_is_inherited_and_doc_language_is_returned():
    body = '<div xml:lang="de"><p>Er kam nach <placeName>Wien</placeName> und blieb dort lange Zeit bei Freunden</p></div><div xml:lang="la"><p>Venit Romam et mansit ibi per multos annos cum amicis</p></div>'
    r = run(body, lang='fr')
    assert r['doc_lang'] == 'fr'
    p = r['passages'][0]
    assert p['lang'] in ('de', 'la') and p['lang_src'] == 'xml:lang'
    assert set(p['langs']) == {'de', 'la'} and 'mixed_lang' in p['flags']


def test_deterministic():
    body = '<p>' + 'a <placeName>Paris</placeName> b <persName>Jean</persName> ' * 40 + '</p>'
    assert run(body)['passages'] == run(body)['passages']


def test_unparseable_and_empty_documents():
    assert tei.document_passages(b'not xml at all')['passages'] == []
    assert run('<p></p>')['passages'] == []


def test_lang_helpers():
    assert norm_lang('lat') == 'la' and norm_lang('deu') == 'de' and norm_lang('frm') == 'fr' and norm_lang('de-Latf') == 'de'
    assert norm_lang('und') is None and norm_lang(None) is None
    assert guess_lang('Il est venu a Paris avec les amis et le roi dans la ville pour une grande fete du peuple et des seigneurs de la cour') == 'fr'
    assert guess_lang('short') is None


def test_london_lives_and_medici_are_never_read():
    assert not E.allowed_path('london-lives/xml/1.xml')
    assert not E.allowed_path('raw/LondonLives/x.xml')
    assert not E.allowed_path('medici-archive/x.xml')
    assert E.allowed_path('croala/txts/a.xml')
    assert E.overlap_group('freem_bvh') == E.overlap_group('epistemon_bvh') != E.overlap_group('croala')


def test_pseudo_xml_chunks_one_document_per_letter():
    one = lambda name, place: (f'### {name}.xml\n<TEI><teiHeader><meta type="id" value="{name}"/></teiHeader><text xml:lang="la"><body><p>Scripsi ad te '
                               f'<placeName>{place}</placeName> heri et hodie multa alia' + FILL + '</p></body></text></TEI>\n')
    r = tei.document_chunks((one('0001', 'Roma') + one('0002', 'Lugdunum')).encode('utf-8'))
    assert [p['chunk'] for p in r['passages']] == ['0001.xml', '0002.xml']
    assert surfaces(r) == [('LOC', 'Roma'), ('LOC', 'Lugdunum')]
    assert r['passages'][0]['doc_lang'] == 'la'
    plain = tei.document_chunks(tei_doc('<p>Ad <placeName>Romam</placeName>' + FILL + '</p>'))
    assert 'chunk' not in plain['passages'][0]
