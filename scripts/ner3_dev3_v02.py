"""dev3 gold -> guide v0.2 conventions (owner decisions of 2026-10-25). Writes a new file; the v0.1 gold is kept unchanged.

    python scripts/ner3_dev3_v02.py --gold dev3/gold_dev3.jsonl --out dev3/gold_dev3_v02.jsonl --changes dev3/changements_v02.tsv

The seven decisions (renner/conventions_proprietaire_2026-10-25.md):
  1 period      final abbreviation period outside the name
  2 peoples     peoples, ethnics, place adjectives, languages: nothing (Latin -ensis stays LOC)
  3 title_place a place in a title that designates the person: the place is LOC, no PER
  4 regnal      a sovereign followed by a place: sovereign = PER, place = LOC
  5 deities     God: nothing; Christ, saints, the Devil, biblical figures, ancient gods: PER; personifications, stars: nothing
  6 titles_kin  titles, address formulae and kinship outside the PER
  7 buildings   generic buildings and places: nothing unless named

Every change is listed with its rule and reason (``--changes``) and recorded on the span (``v02``: rule, action, before).
The edits are an explicit list (``EDITS``) written after reading every span of dev3 and a scan of the text for the deity words;
the rule-6 alt removal and the period trim are mechanical. gold v2 and gold v3 are never read.
"""
from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from pathlib import Path


def nfc(x):
    return unicodedata.normalize('NFC', x)

# PER honorific prefixes (alt_start) that are epithets, not titles: the long form stays accepted.
EPITHET_PREFIX = {'magno'}

# (passage id, label, span text, occurrence index among spans with that label+text in the passage (0-based) or None = all, action, rule, reason)
# actions: drop | relabel_LOC | sure | trim_end:<n tokens> | trim_start:<n tokens> | split_regnal:<n tokens of the sovereign>
EDITS = [
    ('dev3/la/lawiki_13085_pactum_nertschiae', 'LOC', 'Sinici', None, 'drop', 'peoples', 'adjectif de lieu (« Imperii Sinici »)'),
    ('dev3/la/lawiki_13085_pactum_nertschiae', 'LOC', 'Ruthenici', None, 'drop', 'peoples', 'adjectif de lieu (« Ruthenici Imperii »)'),
    ('dev3/la/W0095', 'LOC', 'Indici', None, 'drop', 'peoples', 'adjectif (« Indici consilij », conseil des Indes)'),
    ('dev3/la/lawiki_33104_litterae_auberti_miraei_ad_bollandistas', 'LOC', 'Bethlehemicum', None, 'drop', 'peoples',
     'adjectif de lieu dans un titre d\'ouvrage (« Chronicon Bethlehemicum »)'),
    ('dev3/la/W0030#3', 'LOC', 'Florentinum', None, 'drop', 'peoples', 'adjectif de lieu, pas en -ensis (« Concilium Florentinum »)'),
    ('dev3/la/noscemus_770049', 'LOC', 'Pisana', None, 'drop', 'peoples', 'adjectif de lieu (« Pisana Acad. »)'),
    ('dev3/la/lawiki_36329_sollicitudo_pastoralis', 'LOC', 'Vallisoletano', None, 'drop', 'peoples', 'adjectif de lieu (chapitre de Valladolid)'),
    ('dev3/la/A35526', 'LOC', 'Curubitana', None, 'drop', 'peoples', 'adjectif de lieu (« civitate Curubitana »)'),
    ('dev3/la/noscemus_608643', 'LOC', 'Nicaenam', None, 'drop', 'peoples', 'adjectif de lieu (« Nicaenam synodum »)'),
    ('dev3/la/A15397', 'LOC', 'Assyrio', None, 'drop', 'peoples', 'gentilé (« Assyrio regi »)'),
    ('dev3/es/WS_es_Sue_os_y_discursos_Sue_o_del_infierno#2', 'LOC', 'Niceno', None, 'drop', 'peoples', 'adjectif de lieu (« Concilio Niceno »)'),
    ('dev3/it/bibit001650', 'LOC', 'Palazzo del Podestà', None, 'drop', 'buildings',
     'bâtiment désigné par sa fonction (palais du podestat), pas par un nom'),
    ('dev3/it/bibit000063', 'PER', 'Chevreuse', 0, 'relabel_LOC', 'title_place',
     '« madama Chevreuse » : titre qui désigne la personne ; le lieu est LOC, pas de PER'),
    ('dev3/es/Lope_Guzmán-el-Bravo', 'PER', 'Almanzor de Córdoba', None, 'split_regnal:1', 'regnal',
     'souverain suivi de son lieu : [Almanzor] PER, [Córdoba] LOC'),
    ('dev3/es/Castillo_Estafa-2#2', 'PER', 'Cristo', None, 'sure', 'deities', 'Christ = PER (règle décidée ; le doute portait sur la règle)'),
    ('dev3/la/A54390', 'PER', 'Satanae', None, 'sure', 'deities', 'le diable = PER (règle décidée)'),
    ('dev3/la/A21194', 'PER', 'Iupiter', None, 'sure', 'deities', 'dieu antique invoqué (« Proh Iupiter ») = PER ; pas l\'astre'),
    ('dev3/es/Lope_Desdicha-por-la-honra', 'PER', 'Mamut Bajá', None, 'trim_end:1', 'titles_kin', 'titre (bajá, pacha) hors du PER'),
    ('dev3/it/bibit001678', 'PER', 'Mortasà Bassà', None, 'trim_end:1', 'titles_kin', 'titre (bassà, pacha) hors du PER'),
    ('dev3/es/Cervantes__Quijote-II', 'PER', 'Cide Hamete', None, 'trim_start:1', 'titles_kin', 'titre (cide, « seigneur ») hors du PER'),
]

# (passage id prefix, token index, token text, label, certainty, rule, reason)
ADDS = [
    ('dev3/la/lawiki_17320_malleus_maleficarum', 9, 'Dæmoni', 'PER', 'sure', 'deities', 'le diable (« se ex toto Dæmoni deuouit ») = PER'),
    ('dev3/la/lawiki_17320_malleus_maleficarum', 22, 'Diaboli', 'PER', 'sure', 'deities', 'le diable (« à Diaboli potestate ») = PER'),
    ('dev3/la/A85728', 72, 'daemon', 'PER', 'doubt', 'deities', 'le diable probable (« succutiat daemon ») ; doute : démon quelconque'),
    ('dev3/es/Ubeda__PicaraJustina', 8, 'diablo', 'PER', 'sure', 'deities', 'le diable (« San Miguel, que venció al diablo ») = PER'),
    ('dev3/es/WS_es_Ep_stolas_del_Caballero_de', 55, 'diablo', 'PER', 'sure', 'deities', 'le diable (« con la culebra y con el diablo ») = PER'),
]
# Not added (read, judged generic): « el diablo » / « el otro diablo » (Quijote II, several devils), « un diablo », « diablos »,
# « el demonio » (Sueño del Juicio, one of the accusing demons), « sophista diabol[e] » (insult); « Spirito Santo », « Spiritus Sanctus »
# (God: nothing); planets in the astrological passages (Mars, Venus, Saturnus: nothing); « fortuna », « amore », « natura » (personifications
# or common nouns: nothing). Unchanged and still in doubt: bare title places that stand for a person (« di Beauvais e di Chevreuse »,
# « Beaufort essendosi... »): decision 3 covers a title phrase, decision « Saxuma » (2026-10-23) says a place for its lord is not LOC; open.


def tags(n, spans, lab):
    t = ['O'] * n
    for s in spans:
        if s['label'] != lab:
            continue
        t[s['start']] = f'B-{lab}'
        for k in range(s['start'] + 1, s['end']):
            t[k] = f'I-{lab}'
    return t


def set_span(r, s, start, end):
    T, S = r['tokens'], r['token_starts']
    s['start'], s['end'] = start, end
    s['char_start'] = S[start]
    s['char_end'] = S[end - 1] + len(T[end - 1])
    s['text'] = ' '.join(T[start:end])          # v0.1 convention: tokens joined by a space; surface = the characters
    s['surface'] = r['text'][s['char_start']:s['char_end']]


def note(s, rule, action, before, reason):
    s.setdefault('v02', []).append({'rule': rule, 'action': action, 'reason': reason,
                                    'before': {k: before.get(k) for k in ('label', 'start', 'end', 'text', 'alt_start', 'certainty')}})


def drop_alt(s):
    s.pop('alt_start', None)
    s.pop('alt_char_start', None)


def convert(rows):
    by_id = {nfc(r['id']): r for r in rows}
    changes = []

    def log(r, s, rule, action, before, reason):
        note(s, rule, action, before, reason)
        changes.append({'id': r['id'], 'lang': r['lang'], 'rule': rule, 'action': action,
                        'label_before': before.get('label') or '', 'text_before': before.get('text') or '',
                        'alt_before': ' '.join(r['tokens'][before['alt_start']:before['start']]) if before.get('alt_start') is not None else '',
                        'label_after': s['label'] if action != 'drop' else '', 'text_after': s['text'] if action != 'drop' else '',
                        'certainty_after': s.get('certainty', '') if action != 'drop' else '', 'reason': reason})

    # explicit edits
    for pid, lab, text, occ, action, rule, reason in EDITS:
        r = by_id[nfc(pid)]
        hits = [s for s in r['spans'] if s['label'] == lab and nfc(s['text']) == nfc(text)]
        if occ is not None:
            hits = [s for s in hits if s.get('alt_start') is not None] if action == 'relabel_LOC' else hits[occ:occ + 1]
        assert hits, (pid, lab, text)
        for s in hits:
            before = copy.deepcopy(s)
            if action == 'drop':
                r['spans'].remove(s)
            elif action == 'relabel_LOC':
                s['label'] = 'LOC'
                s['certainty'] = 'sure'
                drop_alt(s)
            elif action == 'sure':
                s['certainty'] = 'sure'
            elif action.startswith('trim_end:'):
                set_span(r, s, s['start'], s['end'] - int(action.split(':')[1]))
            elif action.startswith('trim_start:'):
                set_span(r, s, s['start'] + int(action.split(':')[1]), s['end'])
            elif action.startswith('split_regnal:'):
                k = int(action.split(':')[1])
                end0 = s['end']
                set_span(r, s, s['start'], s['start'] + k)
                j = s['start'] + k
                while r['tokens'][j].lower() in ('de', 'di', 'da', 'del', 'von', 'of', 'd', "'"):
                    j += 1
                loc = {'label': 'LOC', 'certainty': 'sure', 'agreement': 'v02_revision', 'pass_a': False, 'pass_b': False}
                set_span(r, loc, j, end0)
                r['spans'].append(loc)
                log(r, loc, rule, 'add', {'label': '', 'text': ''}, reason)
            else:
                raise ValueError(action)
            log(r, s, rule, action, before, reason)

    for pid, k, tok, lab, cert, rule, reason in ADDS:
        cand = [r for r in rows if nfc(r['id']).startswith(nfc(pid)) and len(r['tokens']) > k and nfc(r['tokens'][k]) == nfc(tok)]
        assert len(cand) == 1, (pid, k, tok, len(cand))
        r = cand[0]
        assert not any(s['start'] <= k < s['end'] for s in r['spans']), (pid, k)
        s = {'label': lab, 'certainty': cert, 'agreement': 'v02_revision', 'pass_a': False, 'pass_b': False}
        set_span(r, s, k, k + 1)
        r['spans'].append(s)
        log(r, s, rule, 'add', {'label': '', 'text': ''}, reason)

    # mechanical rules over every span
    for r in rows:
        T = r['tokens']
        for s in list(r['spans']):
            # 6: a title / address / honorific before a PER is no longer an accepted long form
            if s['label'] == 'PER' and s.get('alt_start') is not None:
                pre = ' '.join(T[s['alt_start']:s['start']])
                if pre.lower() not in EPITHET_PREFIX:
                    before = copy.deepcopy(s)
                    drop_alt(s)
                    rr = 'saint' if pre.lower().rstrip(' .') in ('s', 'san', 'santa', 'santo', 'sancta', 'sanct', 'ſanctus', 'diuus', 'señor san', 'señora santa') else 'title'
                    log(r, s, 'titles_kin', 'drop_alt', before, f'« {pre} » : ' + ('titre de saint hors du PER' if rr == 'saint' else 'titre ou formule hors du PER'))
            # 1: final period outside the name
            while s['end'] - s['start'] > 1 and T[s['end'] - 1] == '.':
                before = copy.deepcopy(s)
                set_span(r, s, s['start'], s['end'] - 1)
                log(r, s, 'period', 'trim_end:1', before, 'point final hors du nom')
        r['spans'].sort(key=lambda s: (s['start'], s['end'], s['label']))
        r['ner_tags_loc'] = tags(len(T), r['spans'], 'LOC')
        r['ner_tags_per'] = tags(len(T), r['spans'], 'PER')
        r['guide_version'] = '0.2'
        r['span_convention'] = (r.get('span_convention', '') + '; v0.2 (owner decisions 2026-10-25): titles, honorifics, saint titles '
                                'and kinship outside PER (alt_start kept only for epithets), final period outside, peoples and place adjectives '
                                'not LOC (Latin -ensis kept), God not annotated, Christ/saints/Devil/gods PER, generic buildings not LOC')
    return rows, changes


def check(rows):
    for r in rows:
        seen = set()
        for s in r['spans']:
            assert 0 <= s['start'] < s['end'] <= len(r['tokens']), r['id']
            assert r['text'][s['char_start']:s['char_end']] == s['surface'], (r['id'], s)
            assert ' '.join(r['tokens'][s['start']:s['end']]) == s['text'], (r['id'], s)
            for k in range(s['start'], s['end']):
                assert k not in seen, ('overlap', r['id'], s)
                seen.add(k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gold', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--changes', required=True)
    a = ap.parse_args(argv)
    if any(x in a.gold for x in ('gold2', 'gold_v2', 'gold3', 'gold_v3')):
        ap.error('gold v2 / v3 are never read here')
    rows = [json.loads(l) for l in open(a.gold, encoding='utf-8') if l.strip()]
    rows, changes = convert(rows)
    check(rows)
    Path(a.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    cols = ['id', 'lang', 'rule', 'action', 'label_before', 'text_before', 'alt_before', 'label_after', 'text_after', 'certainty_after', 'reason']
    with open(a.changes, 'w', encoding='utf-8') as fh:
        fh.write('\t'.join(cols) + '\n')
        for c in changes:
            fh.write('\t'.join(str(c[k]).replace('\t', ' ') for k in cols) + '\n')
    return rows, changes


if __name__ == '__main__':
    _, ch = main()
    from collections import Counter
    print(Counter((c['rule'], c['action']) for c in ch))
