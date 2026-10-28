import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from ratlas.ner import ner2_passages as P  # noqa: E402


def test_tokenize_matches_seed_convention():
    toks, starts = P.tokenize('Pontefract, "Yorke" (1530).')
    assert toks == ['Pontefract', ',', '"', 'Yorke', '"', '(', '1530', ')', '.']
    assert starts[0] == 0 and starts[3] == 13


def test_rejoin_eol_hyphens():
    assert P.rejoin_eol_hyphens('the Town of Pon-\ntefract and Yorke') == 'the Town of Pontefract and Yorke'
    assert P.rejoin_eol_hyphens('mal¬\nheur') == 'malheur'
    # an upper-case continuation is a real hyphenated compound, not a split word
    assert P.rejoin_eol_hyphens('Saint-\nDenis') == 'Saint-\nDenis'


def test_repair_dropcaps_basic_and_legit_letters():
    t, reps = P.repair_dropcaps('Il arriva a L ondon et f rance et s aine.', 'fr')
    assert 'London' in t and 'france' in t and 'saine' in t
    assert len(reps) == 3
    # real one-letter words are kept: Italian "a", "e", "i"; French "à"; Latin "a", "e"
    assert P.repair_dropcaps('va a Roma e poi a Napoli', 'it')[0] == 'va a Roma e poi a Napoli'
    assert P.repair_dropcaps('venit à Roma', 'la')[0] == 'venit à Roma'
    assert P.repair_dropcaps('ab urbe a patre e terra', 'la')[0] == 'ab urbe a patre e terra'


def test_repair_dropcaps_uses_document_vocabulary():
    doc = 'iudex iudex iudex Rome Rome london London London pontifex'
    vocab = P.doc_vocab(doc, 'la')
    # "s iudex": "iudex" is a word of the document, "siudex" is not -> left alone
    assert P.repair_dropcaps('pontifex s iudex', 'la', vocab)[0] == 'pontifex s iudex'
    # "L ondon": "ondon" is not a word of the document, "London" is -> repaired
    assert P.repair_dropcaps('in L ondon', 'en', P.doc_vocab(doc, 'en'))[0] == 'in London'
    # apostrophes and accented letters are not drop capitals
    assert P.repair_dropcaps("l'amour d'un homme", 'fr')[0] == "l'amour d'un homme"
    assert P.repair_dropcaps('è vero', 'it')[0] == 'è vero'


def test_stopwords_and_language_guess():
    en = 'the Town was taken by the King and the Lords of the Realm came to the Court with all their men'
    la = 'et in ea civitate quae est in Gallia cum Rege ad Romam non venit sed ex urbe ut dicit'
    assert P.guess_lang(en) == 'en' and P.guess_lang(la) == 'la'
    assert P.stopword_share(en, 'en') > P.stopword_share(en, 'la')


def _prose(n_sentences=14):
    names = ['Yorke', 'London', 'Rome', 'Paris', 'Antwerp', 'Venice', 'Dover', 'Bristol', 'Chester', 'Exeter']
    out = []
    for i in range(n_sentences):
        a, b = names[i % len(names)], names[(i * 3 + 1) % len(names)]
        out.append(f'The King went from {a} to {b} with all his men and the Lords of the Realm, and they were in the Town of {b} for a time before the Court.')
    return ' '.join(out)


def test_draw_passage_filters_and_is_deterministic():
    text = _prose(40)
    g1 = P.draw_passage(text, random.Random(1), 'en')
    g2 = P.draw_passage(text, random.Random(1), 'en')
    assert g1 is not None and g1 == g2
    off, seg = g1
    assert 120 <= len(seg.split()) <= 200
    assert seg[0].isupper() and seg.rstrip()[-1] in '.!?'
    ok, why = P.passage_ok(seg, 'en')
    assert ok, why


def test_passage_ok_rejects_lists_verse_and_markup():
    assert P.passage_ok('London ' * 150, 'en')[0] is False                          # too few function words
    prose = _prose(12)
    assert P.passage_ok(prose + ' <div> http://x.org', 'en', hi=400)[0] is False   # markup residue
    digits = prose + ' 1234567890' * 40
    assert P.passage_ok(digits, 'en', hi=400)[0] is False
    toc = 'I. The King. II. The Lords. III. The Town. IV. The Court. ' + prose
    assert P.passage_ok(toc, 'en', hi=400)[0] is False
    verse = '\n'.join(['O Rome of the King and the Lords'] * 12)
    assert P.is_verse(verse) is True and P.is_verse(prose) is False


def test_fingerprints_ignore_case_long_s_and_accents():
    a = P.fingerprints('Der Käyser ſagte zu den Fürſten daß er nach Wien reiſen wolle und nicht mehr')
    b = P.fingerprints('der kayser sagte zu den fursten dass er nach wien reisen wolle und nicht mehr')
    assert a and len(a & b) >= 1
    assert P.fold('Käyser ſagte') == 'kayser sagte'


def test_merge_consensus(tmp_path):
    import gzip

    import ner2_merge_predictions as M
    passages = tmp_path / 'p.jsonl'
    passages.write_text(json.dumps({'id': 'x', 'lang': 'en'}) + '\n')
    pdir = tmp_path / 'pred'
    pdir.mkdir()
    for m, conf in (('r3_mdeberta_HM', 0.99), ('r3_xlmr_large_H', 0.9), ('llm_qwen35_9b', None)):
        with gzip.open(pdir / f'{m}.jsonl.gz', 'wt') as f:
            f.write(json.dumps({'id': 'x', 'LOC': [{'s': 3, 'e': 4, 'cs': 10, 'ce': 16, 'text': 'London', 'conf': conf}], 'PER': []}) + '\n')
    sys.argv = ['x', '--passages', str(passages), '--pred-dir', str(pdir), '--out', str(tmp_path / 'o.jsonl')]
    M.main()
    rec = json.loads((tmp_path / 'o.jsonl').read_text())
    assert rec['consensus'][0]['n_encoders'] == 2 and rec['consensus'][0]['max_conf'] == 0.99
    assert set(rec['models']) == {'r3_mdeberta_HM', 'r3_xlmr_large_H', 'llm_qwen35_9b'}
