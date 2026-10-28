from ratlas.reconcile_ids import norm_id, query_forms, back_form


def test_norm_basic():
    assert norm_id('gnd', 'https://d-nb.info/gnd/118532537') == ('gnd', '118532537')
    assert norm_id('gnd', '4094052-4') == ('gnd', '4094052-4')
    assert norm_id('viaf', '1098154380921130290284') is None
    assert norm_id('url', 'http://www.viaf.org/2477678') == ('viaf', '2477678')
    assert norm_id('url', 'http://www.wikidata.org/wiki/Q1741?uselang=en') == ('wikidata', 'Q1741')
    assert norm_id('url', 'http://www.perseus.tufts.edu/x') is None
    assert norm_id('wikipedia', 'en:Lighthouse_of_Alexandria') == ('wikipedia', 'en:Lighthouse of Alexandria')
    assert norm_id('wikidata', 'Q36') == ('wikidata', 'Q36')
    assert norm_id('wikidata', 'x') is None
    assert norm_id('hls', '001622') == ('hls', '1622')
    assert norm_id('bnf', 'cb10647130x') == ('bnf', '10647130x')
    assert norm_id('isni', '0000000114443046') == ('isni', '0000000114443046')
    assert norm_id('deutsche-biographie', 'sfz35425') is None
    assert norm_id('deutsche-biographie', 'gnd118532537') == ('deutsche-biographie', '118532537')
    assert norm_id('cerl', 'cnp01302116') == ('cerl', 'cnp01302116')
    assert norm_id('tgn', '7006876') == ('tgn', '7006876')
    assert norm_id('ulan', '50012') is None


def test_forms():
    assert query_forms('hls', '1622') == ['1622', '001622']
    assert query_forms('isni', '0000000114443046')[0] == '0000 0001 1444 3046'
    assert back_form('hls', '001622') == '1622'
    assert back_form('isni', '0000 0001 1444 3046') == '0000000114443046'
