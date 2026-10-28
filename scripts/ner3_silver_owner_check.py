"""Owner-rule compliance of the RenNER silver (decisions of 2026-10-25, guide v0.2): automatic flags per entity.
Gold v2 and gold v3 are never read. Lexicons from ner_owner_rules.py, plus a few Latin / Spanish / Italian lists below.

Flags (one entity can carry several):
  r1_period     span ends with a '.' token (decision 1)
  r2_people     LOC of one word that is a people, demonym, place adjective or language (decision 2; -ensis kept)
  r3_title_loc  PER = dignity + preposition + capitalised word (« Duca di Calabria »): should be LOC on the place, no PER (decision 3)
  r4_regnal     PER with a numeral followed by a preposition + capitalised word inside the span (« Felipe II de España ») (decision 4)
  r4_regnal_miss PER ending in a numeral, followed by preposition + capitalised word that is not LOC (decision 4)
  r5_god        PER on God (Dios, Deus, Dio, Iddio, Dominus for God...) (decision 5)
  r5_personif   PER on a personification (Fortuna, Amor, Fama, Natura, Mors...) (decision 5)
  r5_astre      PER on Sol, Luna, Phoebe, Titan as heavenly bodies (candidate; sense needed) (decision 5)
  r5_miss       Christ or the Devil not inside any PER (Cristo, Christus, Jesús, diablo, Satanás, Lucifer...) (decision 5)
  r6_title      PER starting with a title, form of address, honorific, possessive, article or kinship word (decision 6)
  r6_saint      PER starting with San, S., Santo, Sancta, Saint, diuus (decision 6, point D* of guide v0.2: Q17)
  r6_kin_tail   PER containing a kinship word after the name (« Pedro su hermano ») (decision 6)
  r7_generic    LOC of one generic building or place word (« iglesia », « templum ») (decision 7)
    python scripts/ner3_silver_owner_check.py --silver FILE [--silver FILE] [--ids ids.txt] [--sample N --seed S] --out out.json
        [--flags flags.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner_owner_rules as O  # noqa: E402

fold = O.fold
DIGNITY = O.TITLE_DIGNITY | O._set('''vescovo obispo arzobispo arcivescovo episcopus episcopi episcopo episcopum archiepiscopus archiepiscopi
cardinale cardenal cardinalis cardinal abbas abate abad prior priore patriarca patriarcha legatus legato legado nuncio nunzio governatore
gobernador virrey viceré vicerè almirante ammiraglio capitano capitán condestable connestabile podestà doge dogo''')
ADDRESS = O.TITLE_ADDRESS - O._set('st s san santo santa sancta sancto sanctus sancti sanctum sanctae sainct saint sainte ste sant sanct sankt beato beata beatus divus diuus')
SAINT = O._set('st s san santo santa sancta sancto sanctus sancti sanctum sanctae sanctam sainct saint sainte ste sant sanct sankt beato beata beatus divus diuus diua diva divi diui sto sta')
KIN = O.KIN | O._set('hermanos hijos padres primo prima abuelo abuela nieto nieta tío tía figli fratelli nonno nonna avo ava fratres filii')
PREPS = O.PREPS | O._set('de di da of von')
GOD = O.GOD | O._set('dominus domine domini domino dominum señor signore signor iddio idio dei deo deus deum dios dio jehova iehova '
                     'creador redemptor redentor salvador salvator altissimo altísimo omnipotente omnipotens')
GOD_CONTEXT_ONLY = O._set('dominus domine domini domino dominum señor signore signor')   # God only when alone in the span
PERSONIF = O.PERSONIF | O._set('fortunae fortunam fama famam virtus virtud virtù gloria victoria vittoria pax paz pace iustitia justicia '
                              'giustizia sapientia sapienza prudentia prudencia invidia envidia tempus tiempo tempo occasio ocasión occasione '
                              'nox noche notte aurora')
ASTRE = O._set('sol solis solem sole luna lunae lunam titan phoebe cynthia')
MISS5 = re.compile(r'^(crist|christ|chrys|jes[uú]|ies[uú]|gesu|gesù|giesu|ihs|xpo|xps|diabl|diavol|demoni|daemon|dæmon|satan|lucif|luzbel|belzeb|beelzeb|belial)', re.I)
MISS5_NOT = re.compile(r'^(christian|cristian|christianit|cristiand|crist[ai]n|christen|chrisma|crisma|christoph|cristof|cristob|cristóbal|chrysost|crisost|demonstr|demóstr|diabolic|diablur)', re.I)
PEOPLE_EXTRA = re.compile(r'^(español|espagnol|frances[ae]?s?$|ingl[eé]s|alem[aá]n|italian|portugu[eé]s|flamenc|tudesc|turc|moro[s]?$|indi[oa]s$|griego|romano|romani$|romana$|'
                          r'venecian|veneziano|veneziani|genov[eé]s|genoves|napolitan|milan[eé]s|florentin|fiorentin|toscan|lombard|siciliano|calabr[eé]s|'
                          r'castellan|aragon[eé]s|catal[aá]n|andaluz|vizca[ií]n|gallego|asturian|navarr[oa]s?$|valencian|toledan|sevillan|'
                          r'hebre|ebre|ebrei|jud[ií]o|giudei|iudae|graec|graeci|graecor|latin|arab|árabe|persian|persa|egipci|egizi|aegypti[ia]|'
                          r'troian|troyan|teucr|danai|achiv|grai$|graios|graium|romanor|romanis|romanos|gall[io]s?$|gallor|german[io]s?$|germanor|'
                          r'hispan[io]s?$|hispanor|britann[io]s?$|angl[io]s?$|anglor|saxon|scot[io]?s?$|helvet|batav|belg[ae]?s?$|belgar|'
                          r'thrac|scyth|parth|medor|med[io]s$|assyri|chalda|aethiop|ethiop|etiop|ind[io]s$|indor|sinens|sinic|tartar|polon|hungar|ungar|'
                          r'bohem|boem|dalmat|illyr|macedon|athenien|spartan|lacedaem|corinth|thebano|theban|cretens|cretes|cretan|sicul|sard[io]s?$|'
                          r'etrusc|tusc[io]s?$|latin[io]s?$|sabin|volsc|samnit|campan|apul|lucan[io]s?$|bruti[io]s?$|umbr[io]s?$|picen|ligur[eé]s?$|venet[io]s?$|'
                          r'mauri$|mauros|maurorum|maur[io]s$|saracen|sarracen|turch|osman|ottoman|musulm|mahomet[ae]?n|moscovit|ruthen|lituan|svec|suec|dan[io]s$|norveg|'
                          r'normann|normand|burgund|borgo[gñ]n|lusitan|cantabr|astur|vascon|celtiber|iber[io]s?$|carthagin|poen[io]s?$|punic|numid|afri$|afros|afrorum|'
                          r'americ[ae]n|mexican|peruan|chilen|chichimec|tlaxcal|azteca|mexica$|culh?ua|otom[ií])', re.I)
GENERIC = O.GENERIC_LOC | O._set('iglesias capilla capella cappella catedral catedrale duomo basilica basílica ermita colegio collegio '
                                 'universidad università universitas academia academia aula curia senado senato senatus regia alcázar '
                                 'alcazar torre tower torri turris claustro chiostro coro sacristía sagrestia refectorio hospicio ospizio '
                                 'cárcel carcel carcere prisión prigione mercado mercato plaza piazza calle via vía strada casa casas domus aedes '
                                 'templo tempio templum fanum sacellum capitolio')


def ws(r, s, e):
    return [O.wtext(r, w) for w in O.words(r, s, e)]


def cap(t):
    return t[:1].isupper()


def is_num(t):
    return bool(O.ROMAN.match(t.strip('.'))) or fold(t) in O.ORDINAL


def check_passage(r):
    lang = r['lang']
    toks = r['tokens']
    per_cov = set()
    loc_cov = set()
    for sp in r['spans']:
        (per_cov if sp['label'] == 'PER' else loc_cov).update(range(sp['start'], sp['end']))
    out = []
    for sp in r['spans']:
        s, e, lab = sp['start'], sp['end'], sp['label']
        W = ws(r, s, e)
        F = [fold(w) for w in W]
        fl = []
        if toks[e - 1] == '.':
            fl.append('r1_period')
        if lab == 'LOC':
            if len(W) == 1 and F[0] and not O.ENSIS.search(F[0]) and (O.is_people(F[0], lang == 'la') or PEOPLE_EXTRA.match(W[0])):
                fl.append('r2_people')
            if len(W) == 1 and F[0] in GENERIC:
                fl.append('r7_generic')
            elif len(W) == 2 and F[0] in O.ARTICLES and F[1] in GENERIC:
                fl.append('r7_generic')
        else:
            Fn = [f for f in F if f]
            if Fn and (Fn[0] in DIGNITY or Fn[0] in ADDRESS or Fn[0] in KIN or Fn[0] in O.POSSESSIVE or Fn[0] in O.ARTICLES) and len(Fn) > 0:
                if Fn[0] in SAINT:
                    pass
                elif len(Fn) >= 3 and Fn[0] in DIGNITY and Fn[1] in PREPS and cap(W[2]):
                    fl.append('r3_title_loc')
                else:
                    fl.append('r6_title')
            if Fn and Fn[0] in SAINT and len(Fn) >= 2:
                fl.append('r6_saint')
            for k in range(1, len(F) - 1):
                if is_num(W[k]) and F[k + 1] in PREPS and k + 2 < len(W) and cap(W[k + 2]):
                    fl.append('r4_regnal')
                    break
            if len(W) >= 2 and is_num(W[-1]) and e + 1 < len(toks) and fold(toks[e]) in PREPS and cap(toks[e + 1]) and e + 1 not in loc_cov:
                fl.append('r4_regnal_miss')
            if any(f in KIN for f in F[1:]) and not (F[0] in KIN):
                fl.append('r6_kin_tail')
            if Fn and (all(f in GOD for f in Fn) and (len(Fn) > 1 or Fn[0] not in GOD_CONTEXT_ONLY or True)):
                fl.append('r5_god')
            if len(Fn) == 1 and Fn[0] in PERSONIF:
                fl.append('r5_personif')
            if len(Fn) == 1 and Fn[0] in ASTRE:
                fl.append('r5_astre')
        out.append({'s': s, 'e': e, 'label': lab, 'text': ' '.join(toks[s:e]), 'flags': fl})
    miss = []
    for i, t in enumerate(toks):
        if i not in per_cov and MISS5.match(t) and not MISS5_NOT.match(t):
            miss.append({'s': i, 'e': i + 1, 'label': 'O', 'text': t, 'flags': ['r5_miss']})
    return out, miss


FLAGS = ['r1_period', 'r2_people', 'r3_title_loc', 'r4_regnal', 'r4_regnal_miss', 'r5_god', 'r5_personif', 'r5_astre', 'r5_miss',
         'r6_title', 'r6_saint', 'r6_kin_tail', 'r7_generic']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--silver', action='append', required=True)
    ap.add_argument('--ids', help='restrict to these ids (one per line)')
    ap.add_argument('--form', help='restrict to these forms (comma list: verse,drama)')
    ap.add_argument('--sample', type=int, default=0, help='N passages per language (after --ids / --form)')
    ap.add_argument('--seed', type=int, default=20261026)
    ap.add_argument('--out', required=True)
    ap.add_argument('--flags', help='write every flagged entity with context (jsonl)')
    ap.add_argument('--sample-ids', help='write the sampled ids here')
    a = ap.parse_args()
    keep = set(open(a.ids).read().split()) if a.ids else None
    forms = set(a.form.split(',')) if a.form else None
    by_lang = defaultdict(list)
    for p in a.silver:
        for l in open(p, encoding='utf-8'):
            r = json.loads(l)
            if keep is not None and r['id'] not in keep:
                continue
            if forms is not None and r.get('form') not in forms:
                continue
            by_lang[r['lang']].append(r)
    if a.sample:
        rnd = random.Random(a.seed)
        for lg in by_lang:
            rows = sorted(by_lang[lg], key=lambda r: r['id'])
            rnd.shuffle(rows)
            by_lang[lg] = rows[:a.sample]
        if a.sample_ids:
            open(a.sample_ids, 'w').write('\n'.join(r['id'] for lg in sorted(by_lang) for r in by_lang[lg]) + '\n')
    res = {}
    ff = open(a.flags, 'w', encoding='utf-8') if a.flags else None
    for lg in sorted(by_lang):
        c = Counter()
        n = Counter()
        for r in by_lang[lg]:
            ents, miss = check_passage(r)
            n['passages'] += 1
            for x in ents:
                n['entities'] += 1
                n[x['label']] += 1
                for f in set(x['flags']):
                    c[f] += 1
                if x['flags'] and ff:
                    ff.write(json.dumps({'id': r['id'], 'lang': lg, 'form': r.get('form'), **x,
                                         'ctx': ' '.join(r['tokens'][max(0, x['s'] - 8):x['s']]) + ' [[' + x['text'] + ']] ' +
                                         ' '.join(r['tokens'][x['e']:x['e'] + 6])}, ensure_ascii=False) + '\n')
            for x in miss:
                c['r5_miss'] += 1
                if ff:
                    ff.write(json.dumps({'id': r['id'], 'lang': lg, 'form': r.get('form'), **x,
                                         'ctx': ' '.join(r['tokens'][max(0, x['s'] - 8):x['s']]) + ' [[' + x['text'] + ']] ' +
                                         ' '.join(r['tokens'][x['e']:x['e'] + 6])}, ensure_ascii=False) + '\n')
        flagged_any = None
        res[lg] = {'counts': dict(n), 'flags': {f: c[f] for f in FLAGS},
                   'rate_per_100_entities': {f: round(100 * c[f] / max(1, n['entities']), 2) for f in FLAGS},
                   'total_flags': sum(c[f] for f in FLAGS),
                   'total_per_100_entities': round(100 * sum(c[f] for f in FLAGS) / max(1, n['entities']), 2)}
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    for lg, v in res.items():
        print(lg, v['counts'], 'total/100', v['total_per_100_entities'], {f: x for f, x in v['rate_per_100_entities'].items() if x})


if __name__ == '__main__':
    main()
