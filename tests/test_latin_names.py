import pytest

from ratlas import latin_names as L


def first(form, **kw):
    r = L.lemmatize(form, **kw)
    return r[0].lemma if r else None


@pytest.mark.parametrize('form,expected', [
    ('Leidam', 'Leida'),
    ('Hagam', 'Haga'),
    ('Rhenum', 'Rhenus'),
    ('Rheni', 'Rhenus'),
    ('Rheno', 'Rhenus'),
    ('Budam', 'Buda'),
    ('Colmariae', 'Colmaria'),
    ('Colmariam', 'Colmaria'),
    ('Transilvaniam', 'Transilvania'),
    ('Vallis Telinae', 'Vallis Telina'),
    ('Vallem Telinam', 'Vallis Telina'),
    ('Lugduno', 'Lugdunum'),
    ('Masoviae', 'Masovia'),
    ('Stocholmiae', 'Stocholmia'),
    ('Mosellam', 'Mosella'),
    ('Scotiam', 'Scotia'),
    ('Romae', 'Roma'),
    ('Wittenbergae', 'Wittenberga'),
])
def test_first_candidate(form, expected):
    assert first(form) == expected


@pytest.mark.parametrize('form,expected', [
    ('Hagae Comitis', 'Haga Comitis'),                 # head inflected, genitive qualifier kept
    ('Haga Comitis', 'Haga Comitis'),
    ('Lugduno Batavorum', 'Lugdunum Batavorum'),
    ('Lugdunum Batavorum', 'Lugdunum Batavorum'),
    ('Montem Pessulanum', 'Mons Pessulanus'),          # head Mons (m.), adjective agrees
    ('Montis Pessulani', 'Mons Pessulanus'),
    ('Sancti Galli', 'Sanctus Gallus'),                # adjective before the noun
    ('Aquae Sextiae', 'Aquae Sextiae'),                # plural-only head: unchanged
    ('Novam Hollandiam', 'Nova Hollandia'),
    ('apud Rhenum', 'Rhenus'),                         # leading preposition dropped
    ('in Leida', 'Leida'),
])
def test_multiword(form, expected):
    assert first(form) == expected


def test_declensions():
    assert first('Narbonem') == 'Narbo' and first('Carthaginis') == 'Carthago'
    assert first('Montem') == 'Mons' and first('Pontis') == 'Pons'
    assert first('Neapolim') == 'Neapolis' and first('Tiberim') == 'Tiberis'
    assert first('Hamburgi') == 'Hamburgum' and first('Argentorati') == 'Argentoratum'    # neuter towns
    assert first('Mediolani') == 'Mediolanum'
    assert 'Athenae' in L.nominatives('Athenis') and first('Athenae') == 'Athenae'          # plural-only names stay
    assert 'Alexander' in L.nominatives('Alexandri')
    assert 'Delphi' in L.nominatives('Delphos')
    assert first('Londinensem') == 'Londinensis'
    assert 'Hafnia' in L.nominatives('Hafniensis')
    assert 'Londinium' in L.nominatives('Londinensis', max_candidates=10)
    assert first('LEIDAM') == 'LEIDA'


def test_nominative_is_kept_and_ranked_first():
    assert first('Roma') == 'Roma'
    assert first('Lugdunum') == 'Lugdunum'            # neuter town, not Lugdunus
    assert first('Rhenus') == 'Rhenus'
    assert L.nominatives('Rhenum')[:2] == ['Rhenus', 'Rhenum']
    assert L.lemmatize('Roma', include_self=False) == []


def test_scores_and_limits():
    r = L.lemmatize('Rheni', max_candidates=2)
    assert len(r) == 2 and r[0].score >= r[1].score
    assert all(0 <= c.score <= 1 for c in L.lemmatize('Vallis Telinae'))
    assert any(c.rule == 'a1_acc' for c in L.lemmatize('Leidam'))
    assert 'Leida' not in L.nominatives('Leidam', max_candidates=1)[1:]


@pytest.mark.parametrize('bad', ['', None, '123', 'Rhenum1', 42, '   '])
def test_bad_input(bad):
    assert L.lemmatize(bad) == []


def test_no_spurious_double_vowels():
    for f in ('Colmariae', 'Masoviae', 'Athenae', 'Stocholmiae'):
        for c in L.lemmatize(f):
            assert 'aa' not in c.lemma and 'ais' not in c.lemma, c


def test_spelling_variants():
    assert first('Leidæ') == 'Leida'                    # ligature
    assert first('Lugdūnum') == 'Lugdunum'              # macron
    assert first('Colmarie') == 'Colmaria'              # medieval e for ae
    assert first('Hagę') == 'Haga'                      # e caudata


def test_non_latin_context_not_handled():
    # Moree (French) and Delfos (Spanish) look like Latin forms: the rules must not be applied outside Latin context
    assert L.lemmatize_in_context('Moree', False) == []
    assert L.lemmatize_in_context('Delfos', False) == []
    assert L.lemmatize_in_context('Leidam', False) == []
    assert L.lemmatize_in_context('Leidam', True)[0].lemma == 'Leida'
    ctx = L.is_latin_context('casebooks', lang='en')
    assert not ctx and L.lemmatize_in_context('Moree', ctx) == []


def test_is_latin_language():
    for c in ('la', 'lat', 'LA', 'Latin', 'la-x-medieval', 'neo-latin'):
        assert L.is_latin_language(c) is True
    for c in ('de', 'deu', 'ger ', 'fr', 'fra', 'frm', 'en', 'eng', 'nl', 'es', 'spa', 'it', 'gez', 'ota', 'de-gml'):
        assert L.is_latin_language(c) is False
    for c in (None, '', '?', 'und', 'mul', float('nan')):
        assert L.is_latin_language(c) is None


def test_is_latin_context_order():
    # the mention's own language wins, then the document's, then the edition
    assert L.is_latin_context('casebooks', lang='la')
    assert not L.is_latin_context('croala', lang='de')
    assert L.is_latin_context('casebooks', doc_language='lat')
    assert not L.is_latin_context('croala', lang='?', doc_language='fr')
    assert L.is_latin_context('croala', lang=None, doc_language=None)
    assert L.is_latin_context('corpus_corporum', lang='?', doc_language='und')
    assert L.is_latin_context('bullinger') and L.is_latin_context('sozinianische_briefwechsel')
    assert L.is_latin_context('camena_poemata')                       # no lang at all in the tables: entry says [Latin]
    # non-Latin editions, unknown collections, nothing known
    assert not L.is_latin_context('casebooks') and not L.is_latin_context('ssrq_swisslaw')
    assert not L.is_latin_context('hainhofer_reisen')                 # a German edition with scattered lang='la'
    assert not L.is_latin_context('no_such_collection') and not L.is_latin_context()
    # explicit override of the edition share
    assert L.is_latin_context('no_such_collection', edition_share=0.9)
    assert L.is_latin_context('cidtc') and not L.is_latin_context('cidtc', min_share=0.7)


def test_edition_table():
    assert L.edition_latin_share('croala') > 0.9
    assert L.edition_latin_share('casebooks') < 0.05
    assert L.edition_latin_share('unknown') is None
    for slug, (share, src) in L._EDITIONS.items():
        assert 0 <= share <= 1 and src in ('m', 'e'), slug


def test_latin_share_from_languages():
    assert L.latin_share_from_languages(['Latin']) == 1.0
    assert L.latin_share_from_languages(['Latin', 'German']) == 0.5
    assert L.latin_share_from_languages(['German', 'Latin']) == 0.25
    assert L.latin_share_from_languages(['German']) == 0.0
    assert L.latin_share_from_languages(None) == 0.0


def test_fold_key():
    assert L.fold_key('Lugdūnum') == L.fold_key('Lugdunum')
    assert L.fold_key('Vitemberga') == L.fold_key('Vitemberga')
    assert L.fold_key('Leyda') == L.fold_key('Leida')
    assert L.fold_key('Coloniæ') == L.fold_key('Colonie')
    assert L.fold_key('Vallis  Telina,') == 'uallis telina'


def test_resolve_against_lexicon():
    lex = L.LatinLexicon({'Leida': 'Q43631', 'Haga Comitis': 'Q36600', 'Rhenus': 'Q584', 'Lugdunum Batavorum': 'Q43631',
                          'Colmaria': 'Q6602', 'Vitemberga': 'Q2966'})
    assert [r.matched for r in L.resolve('Leidam', lex)] == ['Leida']
    assert L.resolve('Leidam', lex)[0].payload == 'Q43631'
    assert L.resolve('Hagae Comitis', lex)[0].matched == 'Haga Comitis'
    assert L.resolve('Rheni', lex)[0].matched == 'Rhenus'
    assert L.resolve('Lugduno Batavorum', lex)[0].matched == 'Lugdunum Batavorum'
    assert L.resolve('Colmariae', lex)[0].matched == 'Colmaria'
    assert L.resolve('Wittenbergae', lex) == []                 # candidate Wittenberga is not in the lexicon (Vitemberga is)
    assert L.resolve('Moree', lex) == [] and L.resolve('Roma', lex) == []
    assert 'Rheno' not in lex and 'Rhenus' in lex and len(lex) == 6
    # the form itself is never returned, even when it is in the lexicon
    assert L.resolve('Rhenus', lex) == []


def test_collatinus_optional():
    if not L.collatinus_available():
        assert L.collatinus_lemmas('Rhenum') == []
        pytest.skip('pycollatinus not installed (optional)')
    assert 'Rhenus' in L.collatinus_lemmas('Rhenum')
    assert 'Lutetia' in L.collatinus_lemmas('Lutetiam')
    assert L.collatinus_lemmas('Hagam') == []
    assert 'Rhenus' in L.nominatives('Rhenum', collatinus=True)


def test_context_levels():
    assert L.latin_context_level('croala') == 'latin'
    assert L.latin_context_level('casebooks', lang='la') == 'latin'
    assert L.latin_context_level('croala', lang='de') == 'non_latin'
    assert L.latin_context_level('ckcc_epistolarium') == 'mixed'          # Latin among 7 languages, mention language unknown
    assert L.latin_context_level('ckcc_epistolarium', lang='?') == 'mixed'
    assert L.latin_context_level('casebooks') == 'non_latin'
    assert L.latin_context_level('nope') == 'unknown' and L.latin_context_level() == 'unknown'
    for slug in ('croala', 'bullinger', 'ckcc_epistolarium', 'casebooks'):
        assert (L.latin_context_level(slug) == 'latin') == L.is_latin_context(slug)


def test_resolve_in_context():
    lex = L.LatinLexicon(['Leida', 'Rhenus', 'Colmaria', 'Lerna', 'Deli'])
    assert [r.matched for r in L.resolve_in_context('Leidam', lex, 'corpus_corporum')] == ['Leida']
    assert [r.matched for r in L.resolve_in_context('Leidam', lex, 'ckcc_epistolarium')] == ['Leida']            # mixed: strong rule
    assert L.resolve_in_context('Leidam', lex, 'ckcc_epistolarium', lang='fr') == []                             # French mention
    assert L.resolve_in_context('Leidam', lex, 'casebooks') == [] and L.resolve_in_context('Leidam', lex, 'nope') == []
    assert L.resolve_in_context('Rheno', lex, 'ckcc_epistolarium')[0].score == 0.78                             # threshold is inclusive
    assert L.resolve_in_context('Colmarie', lex, 'ckcc_epistolarium') == []                                      # 0.62 rule: not in mixed
    assert L.resolve_in_context('Rheni', lex, 'ckcc_epistolarium')[0].matched == 'Rhenus'
    # weak rules never match in a Latin edition either (Lerne is the Lerna spring or German, Delos is Greek)
    assert L.resolve_in_context('Lerne', lex, 'corpus_corporum') == []
    assert L.resolve_in_context('Delos', lex, 'corpus_corporum') == []
    assert L.resolve_in_context('Colmarie', lex, 'corpus_corporum')[0].matched == 'Colmaria'


def test_glosses_are_not_latin_inflection():
    # 'Löbau (Lubawa)' must not be reduced to 'Löbau' (a different town in Saxony)
    assert L.lemmatize('Löbau (Lubawa)') == [] and L.lemmatize('Leidam [sic]') == []
    assert L.lemmatize('Leidam/Hagam') == []
