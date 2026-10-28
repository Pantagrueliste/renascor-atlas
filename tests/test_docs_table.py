"""Tiny checks of the document-table rules (no raw data needed)."""
from ratlas import docs_extract as X
from ratlas import docs_rules as R
from ratlas.docs_table import doc_key


def test_iso_from_text():
    assert X.iso_from_text('1688 noviembre 12') == ('1688-11-12', 'day')
    assert X.iso_from_text('12 novembre 1688') == ('1688-11-12', 'day')
    assert X.iso_from_text('Oktober 1623') == ('1623-10', 'month')
    assert X.iso_from_text('pago 1520') == (None, '')
    assert X.iso_from_text('1417-01-19') == ('1417-01-19', 'day')


def test_human_date():
    assert R.human_date('1596-12-22') == '22 Dec 1596'
    assert R.human_date('1525-03-26/1525-03-31') == '26 Mar 1525 - 31 Mar 1525'
    assert R.human_date('1621-05-10/..') == 'after 10 May 1621'


def test_choose_date_prefers_precise_header_date_and_drops_edition_dates():
    info = {'dates': [['sent', None, 'year', '1551', 1551, 1551], ['creation', '1551-11-03', 'day', '3 nov 1551', 1551, 1551],
                      ['source', '2014-01-31', 'day', '2014', 2014, 2014]]}
    d, p, a, b, src, conflict = R.choose_date(info, 1551, 1551)
    assert (d, p) == ('1551-11-03', 'day') and not conflict
    d, p, a, b, src, conflict = R.choose_date({'dates': [['source', '2006', 'year', '2006', 2006, 2006]]}, 1650, 1700)
    assert (d, p, src, conflict) == ('1650/1700', 'range', 'mentions', True)


def test_split_letter_title():
    assert R.split_letter_title('bullinger', 'Konrad Gessner an Bullinger, [Zürich, 1547]') == \
        ('Konrad Gessner', 'Zürich', 'Bullinger', None)
    assert R.split_letter_title('sozinianische_briefwechsel',
                                'Jacques Delelande (Hamburg?) an Stanisław Lubieniecki, 11. März 1669')[0] == 'Jacques Delelande'
    assert R.split_letter_title('cidtc', 'Ioannes DANTISCUS to Bona Sforza') == ('Ioannes DANTISCUS', None, 'Bona Sforza', None)


def test_url_patterns():
    c = R.Ctx('ferdinand_i', 'x/files/A001.xml', 'A001', {})
    assert R.URLS['ferdinand_i'](c)[0] == 'https://ferdinand-korrespondenz.acdh.oeaw.ac.at/A001.html'
    c = R.Ctx('postscriptum', 'postscriptum/ES1500/PS4012_TEIP5.xml', 'PS4012_TEIP5', {})
    assert R.URLS['postscriptum'](c)[0].endswith('cid=PS4012')
    c = R.Ctx('karlstadt_kritische_gesamtausgabe', 'a/files/xml/004A_transcript.xml', 'x', {})
    assert R.URLS['karlstadt_kritische_gesamtausgabe'](c)[0].endswith('/texts/004A/004A_transcript.xml')


def test_doc_key_is_stable():
    assert doc_key('a', 'b', 'c') == doc_key('a', 'b', 'c') != doc_key('a', 'b', 'd')
    assert len(doc_key('a', 'b', 'c')) == 12


def test_bullinger_addressee_has_no_date():
    assert R.split_letter_title('bullinger', 'Johannes Blasius / Chur an Heinrich Bullinger, 20. September 1547') == \
        ('Johannes Blasius', 'Chur', 'Heinrich Bullinger', None)
    assert R.split_letter_title('bullinger', 'Heinrich Bullinger / Zürich an Oswald Myconius (Geißhüsler) / Basel, 3. August 1548') == \
        ('Heinrich Bullinger', 'Zürich', 'Oswald Myconius (Geißhüsler)', 'Basel')


def test_placeholders_are_not_names():
    assert R.names(['s.n.']) is None and R.names(['S.l.']) is None and R.names(['o.O.']) is None and R.names(['¿?']) is None
    assert R.names(['NN']) is None
    assert R.names(['España¿?']) == 'España'
    assert R.names(['Zürich', 's.l.']) == 'Zürich'


def test_clean_title_drops_an_unclosed_parenthesis_fragment():
    assert R.clean_title('Jacob Jordaens wird zum Prodekan ernannt (28.') == 'Jacob Jordaens wird zum Prodekan ernannt'
    assert R.clean_title('Brucker Pacification (28. Januar 1578') == 'Brucker Pacification'
    assert R.clean_title('Declaratio Ferdinandea (24. September 1555)') == 'Declaratio Ferdinandea (24. September 1555)'


def test_title_padding_around_place_and_date_children():
    from lxml import etree
    el = etree.fromstring('<title>An Friedrich <placeName>Torgau</placeName><date>1516, vor 5. Juni</date></title>')
    assert X.text_no(el, drop=('note',), pad=('placeName', 'date')) == 'An Friedrich Torgau, 1516, vor 5. Juni'


def test_textlang_in_words_and_pubdate():
    from lxml import etree
    root = etree.fromstring('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>t</title></titleStmt>'
                            '<publicationStmt><date when="1740-07-26">1740-07-26</date></publicationStmt>'
                            '<sourceDesc><msDesc><msContents><msItem><textLang>Frühneuhochdeutsch</textLang></msItem></msContents>'
                            '<head xml:lang="de">Clauda – Verhör</head></msDesc></sourceDesc></fileDesc></teiHeader></TEI>')
    info = X.parse_header(root)
    assert info['textlang_words'] == ['de']
    assert info['pubdates'] == ['1740-07-26']
    assert info['msheads'] == [('de', 'Clauda – Verhör')]


def test_nested_place_in_persname_is_not_part_of_the_name():
    from lxml import etree
    root = etree.fromstring('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><profileDesc><correspDesc><correspAction type="received">'
                            '<persName><persName ref="#a">Bartolomeo Concino</persName> <placeName ref="#f">Florence</placeName></persName>'
                            '</correspAction></correspDesc></profileDesc></teiHeader></TEI>')
    info = X.parse_header(root)
    assert info['addressees'] == ['Bartolomeo Concino'] and info['place_to'] == ['Florence']


def test_bullinger_addressee_date_and_weekday_tail():
    assert R.split_letter_title('bullinger', 'Joachim Vadian an Bullinger, St. Gallen, Donnerstag, 12. Januar 1548') == \
        ('Joachim Vadian', 'St. Gallen', 'Bullinger', None)
    assert R.split_letter_title('bullinger', 'Berchtold Haller / [Bern] an Heinrich Bullinger, 16., 17. und 22. März [1532]') == \
        ('Berchtold Haller', 'Bern', 'Heinrich Bullinger', None)
    assert R.split_letter_title('bullinger', 'Heinrich Stoll (Seebach) / [Winterthur] an Heinrich Bullinger, Leo Jud, 20. Juni 1536')[2] == \
        'Heinrich Bullinger, Leo Jud'


def test_title_comma_between_place_and_date_children():
    from lxml import etree
    el = etree.fromstring('<title>An Friedrich <placeName>Torgau</placeName><date>1516, vor 5. Juni</date></title>')
    assert X.text_no(el, drop=('note',), pad=('placeName', 'date')) == 'An Friedrich Torgau, 1516, vor 5. Juni'


def test_bullinger_addressee_undated_forms():
    assert R.split_letter_title('bullinger', 'Berchtold Haller / Bern an Heinrich Bullinger, zwischen [3.] Juni 1532 und [4.] Juni 1532')[2] == 'Heinrich Bullinger'
    assert R.split_letter_title('bullinger', 'Melchior Wirz / Basel an Heinrich Bullinger, s.d. s.d. 1533')[2] == 'Heinrich Bullinger'
    assert R.split_letter_title('bullinger', '[Heinrich Bullinger] / Kappel am Albis an Wolfgang Rüpplin (Joner), Peter Simler (Simmler), [Oktober/November] 1524')[2] == \
        'Wolfgang Rüpplin (Joner), Peter Simler (Simmler)'
