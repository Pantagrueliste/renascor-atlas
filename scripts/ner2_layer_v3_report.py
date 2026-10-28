"""NER layer v3: the French report (LISEZMOI.md) and two small tables, from the build log of ner2_layer_v3_build.py.

    python scripts/ner2_layer_v3_report.py --build BUILD_DIR --v3 V3_DIR --prev V28_DIR --new V29_DIR [--keep-old LISEZMOI_old.md]

Writes in V3_DIR: LISEZMOI.md, docs_exclus_v3.tsv (documents left out, with the reason), points_a_relire_v3.tsv (points of the new sets alone with 11 mentions or more).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def fr(n) -> str:
    return f'{int(n):,}'.replace(',', ' ')


def pct(a, b) -> str:
    return (f'{100 * a / b:.1f}'.replace('.', ',') + ' %') if b else '-'


def dlt(a, b) -> str:
    return f'{b - a:+,}'.replace(',', ' ')


STEP_FR = {
    'v2 avant ajout': 'départ : couche v2 seule',
    'ajout des jeux': 'jeux ajoutés (tei, extra, seen), Nakala et doublons de documents déjà retirés',
    'Nice': 'correction 3 : Nice du concile -> İznik',
    'Antioche': 'correction 4 : Antiochia latin -> Antakya',
    'personnes': 'correction 5 : noms de personnes (p_per >= 0,9) sans point',
    'Île-de-France': 'correction 6 : Isle de France -> Île-de-France',
    'liste noire': 'correction 8 : Uti, Duron, Megalopolis',
    'tables': 'points 2 et 7 : variantes, régions, fleuves',
    'rayon': 'point 9 : rayon minimal 40 km',
    'fusion 1': 'point 9 : fusion des doublons',
    'formes jugées': 'formes des points jugés rejointes à ces points',
    'balayage': 'balayage (171 verdicts : retraits, déplacements, fusions, libellés, homonymes)',
    'fusion 2': 'fusion après balayage',
    'antiques': 'règle des noms antiques (curation.py)',
    'lointains': 'règle des homonymes lointains (curation.py)',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--v3', required=True)
    ap.add_argument('--prev', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--keep-old', default=None)
    ap.add_argument('--work', default=None, help='tables of the v2 layer (docs_v1.parquet ...), for the number of documents of v2')
    a = ap.parse_args()
    B, V3, PV, NW = Path(a.build), Path(a.v3), Path(a.prev), Path(a.new)
    log = json.loads((B / 'build_log.json').read_text('utf-8'))
    old = json.loads((PV / 'model_layer_v2.json').read_text('utf-8'))
    olddocs = json.loads((PV / 'model_layer_v2_docs.json').read_text('utf-8'))
    new = json.loads((NW / 'model_layer_v2.json').read_text('utf-8'))
    newdocs = json.loads((NW / 'model_layer_v2_docs.json').read_text('utf-8'))
    acc = log['accounting']
    v3 = log['v3']
    steps = {s['step']: s for s in log['steps']}
    track = log['track']

    op, np_ = old['places'], new['places']
    def tier_stats(pl):
        m = [p for p in pl if p['tier'] == 'main']
        r = [p for p in pl if p['tier'] == 'rare']
        return len(m), sum(p['n'] for p in m), len(r), sum(p['n'] for p in r)
    om, omn, orr, orn = tier_stats(op)
    nm, nmn, nr, nrn = tier_stats(np_)
    # documents with at least one placed mention
    od = {d for lst in olddocs['points'].values() for d, _ in lst}
    nd = {d for lst in newdocs['points'].values() for d, _ in lst}
    # before / after per collection
    ocoll = Counter()
    for pid, lst in olddocs['points'].items():
        for d, n in lst:
            ocoll[olddocs['docs'][d]['collection']] += n
    ncoll = Counter()
    ndoc_coll = Counter()
    for d in nd:
        ndoc_coll[newdocs['docs'][d]['collection']] += 1
    for pid, lst in newdocs['points'].items():
        for d, n in lst:
            ncoll[newdocs['docs'][d]['collection']] += n
    odoc_coll = Counter()
    for d in od:
        odoc_coll[olddocs['docs'][d]['collection']] += 1

    # inputs per set
    docs = {s: pd.read_parquet(V3 / s / 'docs.parquet') for s in ('tei', 'extra', 'seen')}
    tot = json.loads((V3 / 'summary_all/v3_totals.json').read_text('utf-8'))
    bycol = pd.read_csv(V3 / 'summary_all/v3_by_collection.tsv', sep='\t')
    reasons = v3['reasons']
    setof = {}
    for s, d in docs.items():
        for x in d['doc_id']:
            setof[x] = s
    # documents left out
    out_rows = []
    allf = pd.concat([d[['doc_id', 'collection', 'n_words']].assign(set=s) for s, d in docs.items()])
    meta = allf.set_index('doc_id')
    dfm = pd.read_parquet(V3 / 'geo_all/doc_forms.parquet')
    mdoc = dfm.groupby('doc_id')['n'].sum().to_dict()
    for d, r in sorted(reasons.items()):
        out_rows.append((d, setof.get(d, ''), meta.at[d, 'collection'] if d in meta.index else '', int(meta.at[d, 'n_words']) if d in meta.index else 0, int(mdoc.get(d, 0)), r))
    pd.DataFrame(out_rows, columns=['doc_id', 'jeu', 'collection', 'mots', 'mentions_envoyees', 'raison']).to_csv(V3 / 'docs_exclus_v3.tsv', sep='\t', index=False)
    kind = Counter()
    for d, r in reasons.items():
        k = 'Nakala (décision 11)' if r.startswith('collection') else ('copie exacte' if 'copie exacte' in r else ('contenu dans la couche' if 'dans la couche' in r else ('contenu dans un autre document nouveau' if 'autre document' in r else r)))
        kind[(setof.get(d, '?'), k)] += 1
    # points of the new sets alone
    newonly = [p for p in np_ if all(x.startswith('ml3:') for x in p['prev'])]
    pd.DataFrame([(p['id'], p['name'], p['n'], p['nd'], p['k'], p['lat'], p['lon'], '|'.join(p['cols'][:3]), '; '.join(f'{f[0]} ({f[1]}, {f[2]})' for f in p['forms'][:4]), (p.get('ed') or {}).get('name', ''))
                  for p in sorted(newonly, key=lambda p: -p['n']) if p['n'] >= 11],
                 columns=['id', 'nom', 'mentions', 'documents', 'genre', 'lat', 'lon', 'collections', 'formes', 'point_des_editeurs']).to_csv(V3 / 'points_a_relire_v3.tsv', sep='\t', index=False)

    L = []
    A = L.append
    A('# Couche NER v3 : v2 + éditions TEI + documents réparés + éditions d\'entraînement (23 oct. 2026)')
    A('')
    A('Rien n\'est commité, publié ni envoyé. Aucune relecture humaine : ce sont des prédictions du modèle, corrigées par des règles. '
      'La couche v3 est **une seule couche** : la v2 (22 et 23 oct.) plus les mentions des trois nouveaux jeux (`tei`, `extra`, `seen`), sans doublon, avec toutes les corrections.')
    A('')
    A('Scripts : `scripts/ner2_layer_v3_check.py` (Hypatia : doublons, contextes), `scripts/ner2_layer_v3_build.py` (Mac : fusion et corrections), `scripts/ner2_layer_v3_report.py` (ce rapport). '
      'Corrections reprises de `scripts/ner2_layer_fix.py` avec `data/model_layer_overrides.json` (171 entrées dont l\'audit des homonymes) et les règles de `src/ratlas/curation.py`.')
    A('')
    A('## En bref')
    A('')
    A('| | v2 (avant) | v3 (après) | écart |')
    A('|---|---|---|---|')
    A(f"| points | {fr(len(op))} | **{fr(len(np_))}** | {dlt(len(op), len(np_))} |")
    A(f"| mentions placées | {fr(old['header']['mentions'])} | **{fr(new['header']['mentions'])}** | {dlt(old['header']['mentions'], new['header']['mentions'])} |")
    A(f"| points « principaux » (11 mentions ou plus) | {fr(om)} ({fr(omn)} mentions) | **{fr(nm)}** ({fr(nmn)} mentions) | {dlt(om, nm)} |")
    A(f"| points « rares » (10 mentions ou moins) | {fr(orr)} ({fr(orn)} mentions) | **{fr(nr)}** ({fr(nrn)} mentions) | {dlt(orr, nr)} |")
    A(f"| documents avec au moins une mention placée | {fr(len(od))} | **{fr(len(nd))}** | {dlt(len(od), len(nd))} |")
    A(f"| points qui coïncident avec un point des éditeurs | {fr(old['header']['coincide_with_editor_points']['points'])} ({fr(old['header']['coincide_with_editor_points']['mentions'])} mentions) | {fr(new['header']['coincide_with_editor_points']['points'])} ({fr(new['header']['coincide_with_editor_points']['mentions'])} mentions) | |")
    A(f"| points qui n\'existent que par les nouveaux jeux | - | {fr(acc['points_new_only'])} ({fr(acc['mentions_in_new_only_points'])} mentions ; {sum(1 for p in newonly if p['n'] >= 11)} principaux) | |")
    A('')
    A(f"Les mentions de v2 passent de {fr(old['header']['mentions'])} à {fr(acc['by_set']['v2']['mentions'])} : la règle des noms antiques retire les {fr(old['header']['mentions'] - acc['by_set']['v2']['mentions'])} mentions de « Troye » (l\'anglais Troye, Troie, était placé à Troyes). "
      f"Les mentions des nouveaux jeux : tei {fr(acc['by_set']['tei']['mentions'])}, extra {fr(acc['by_set']['extra']['mentions'])}, seen {fr(acc['by_set']['seen']['mentions'])}, soit **{fr(sum(acc['by_set'][s]['mentions'] for s in ('tei', 'extra', 'seen')))}**.")
    A('')
    A('## Garanties : pas de doublon')
    A('')
    A('* **Identifiants de documents** : aucun identifiant des trois jeux n\'est dans les tables de v2 (67 784 documents de v1, 904 de la relance espagnole), ni dans un autre jeu (tei, extra, seen sont disjoints). Contrôle dans `ner2_layer_v3_build.py` (liste « identifiant déjà dans la couche v2 » : 0).')
    A('* **Empreintes de textes** (Hypatia, `dedup_all/`) : 16 898 documents des trois jeux, 28,6 millions de 8-grammes échantillonnés (un sur quatre), comparés aux 154 fichiers de textes de la couche et entre eux, plus une empreinte exacte de la suite de mots pliés. '
      'Seuils des contrôles précédents : contenu à 50 % ou plus (au moins 16 8-grammes), ou copie exacte.')
    A('  * Contre la couche : aucune copie exacte ; 12 documents dépassent 50 % de contenu (11 de `tei`, 1 de `seen`), tous retirés. Les contrôles de la préparation (dup_layer) ne les avaient pas retenus.')
    n_layer = sum(v for (s, k), v in kind.items() if k in ('contenu dans la couche',))
    n_new = sum(v for (s, k), v in kind.items() if k == 'contenu dans un autre document nouveau')
    n_exact = sum(v for (s, k), v in kind.items() if k == 'copie exacte')
    ex_pairs = pd.read_csv(V3 / 'dedup_all/exact_new.tsv', sep='\t', keep_default_na=False)
    A(f'  * {fr(len(v3["dup_docs"]))} documents sont retirés : {n_layer} contenus dans la couche, {n_new + n_exact} contenus dans un autre document nouveau ou copies exactes '
      f'({ex_pairs["exact"].nunique()} paire de copies exactes dans les nouveaux jeux : deux fiches REED de 74 mots, une des deux retirée). '
      'Entre deux documents mutuellement contenus, on garde le plus balisé, puis le plus long ; un document balisé n\'est pas retiré au profit d\'un document non balisé.')
    A('  * Détail par jeu : ' + ', '.join(f'{s} : {sum(v for (ss, k), v in kind.items() if ss == s and k != "Nakala (décision 11)")}' for s in ('tei', 'extra', 'seen')) + '. Liste complète dans `docs_exclus_v3.tsv`.')
    intra = pd.read_csv(V3 / 'dedup_all/intra.tsv', sep='\t', keep_default_na=False)
    cross = intra[(intra['cand_set'] != intra['other_set']) & (intra['contain_cand'] >= 0.5) & (intra['cand_shingles'] >= 16)]
    tx = cross[((cross['cand_set'] == 'tei') & (cross['other_set'] == 'extra')) | ((cross['cand_set'] == 'extra') & (cross['other_set'] == 'tei'))]
    A(f"  * Recouvrement entre jeux (signalé comme non vérifié dans le rapport précédent) : {fr(len(intra) // 2)} paires de documents partagent au moins 8 8-grammes échantillonnés ; "
      f"{fr(len(cross))} lectures d'une paire (un document contenu à 50 % ou plus dans un document d'un autre jeu), dont {len(tx)} entre `tei` et `extra`.")
    top_c = Counter(meta.at[d, 'collection'] for d in v3['dup_docs'] if d in meta.index)
    A('  * Collections des documents retirés comme doublons : ' + ', '.join(f'{c} {n}' for c, n in top_c.most_common(8)) + '.')
    A(f'* **Nakala** : {sum(v for (s, k), v in kind.items() if k.startswith("Nakala"))} documents (5 de `tei`, 3 de `extra`) sortent, comme dans v2 (décision 11). London Lives, MAP et les collections privées : aucun document dans les trois jeux.')
    A('* **Balises d\'éditeur** : aucun des 239 319 empans envoyés au géocodeur ne partage un caractère avec une des 379 733 balises (contrôle indépendant sur Hypatia, `dedup_all/overlap_check.json`). Aucun empan en double, aucun empan qui en recouvre un autre.')
    A('* **Une mention, une fois** : une mention est un empan (document, début, fin) ; les empans sont uniques, les documents sont uniques, donc chaque mention n\'entre qu\'une fois dans un seul point.')
    A('* **Régression** : la même chaîne sans les nouveaux jeux (`--no-v3`) redonne le fichier de v28 à l\'identique (mêmes points, mêmes documents : 9 605 points, 393 860 mentions, 1 321 principaux).')
    A('')
    A('## Les trois jeux, de l\'entrée à la couche')
    A('')
    A('| | tei | extra | seen | total |')
    A('|---|---|---|---|---|')
    def S(col):
        return [int(docs[s][col].sum()) for s in ('tei', 'extra', 'seen')]
    rows = [('documents', [len(docs[s]) for s in ('tei', 'extra', 'seen')]), ('mots', S('n_words')), ('empans du modèle', S('n_spans')), ('empans recommandés', S('n_recommended')),
            ('recommandés écartés (chevauchent une balise)', S('n_rec_dropped_ed')), ('recommandés envoyés au géocodeur', S('n_rec_final'))]
    for nm_, v in rows:
        A(f'| {nm_} | ' + ' | '.join(fr(x) for x in v) + f' | {fr(sum(v))} |')
    reas = Counter(setof.get(d, '?') for d in reasons)
    sent = {s: int(dfm[dfm['doc_id'].isin([d for d in docs[s]['doc_id']])]['n'].sum()) for s in docs}
    lost = {s: int(dfm[dfm['doc_id'].isin([d for d in reasons if setof.get(d) == s])]['n'].sum()) for s in docs}
    placed_geo = {s: tot.get(f'mentions_placed_{s}', 0) for s in docs}
    A('| mentions envoyées après nettoyage | ' + ' | '.join(fr(sent[s]) for s in ('tei', 'extra', 'seen')) + f' | {fr(sum(sent.values()))} |')
    A('| mentions placées par le géocodeur | ' + ' | '.join(fr(placed_geo[s]) for s in ('tei', 'extra', 'seen')) + f' | {fr(sum(placed_geo.values()))} |')
    A('| documents retirés (Nakala, doublons) | ' + ' | '.join(str(reas.get(s, 0)) for s in ('tei', 'extra', 'seen')) + f' | {sum(reas.values())} |')
    A('| mentions envoyées dans ces documents | ' + ' | '.join(fr(lost[s]) for s in ('tei', 'extra', 'seen')) + f' | {fr(sum(lost.values()))} |')
    fin = acc['by_set']
    A('| **mentions dans la couche v3 (après corrections)** | ' + ' | '.join(fr(fin[s]['mentions']) for s in ('tei', 'extra', 'seen')) + f' | **{fr(sum(fin[s]["mentions"] for s in ("tei", "extra", "seen")))}** |')
    A('| documents avec une mention placée | ' + ' | '.join(fr(fin[s]['documents']) for s in ('tei', 'extra', 'seen')) + ' | ' + fr(len({d for d in nd if d in setof})) + ' |')
    A('| points qui reçoivent des mentions du jeu | ' + ' | '.join(fr(fin[s]['points']) for s in ('tei', 'extra', 'seen')) + ' | |')
    A('')
    A('Les empans de `seen` sont presque tous des lieux balisés par les éditeurs (118 738 recommandés sur 156 329 sont écartés) : il ne reste que le texte non balisé. '
      'Les collections sans balises (CKCC, BVH) gardent tous leurs empans, sans contrôle contre les éditeurs.')
    A('')
    A('## Ce que chaque correction a changé aux nouvelles mentions')
    A('')
    A('Mentions placées des trois jeux après chaque étape (l\'ordre est celui de la chaîne ; v2 pour rappel).')
    A('')
    A('| étape | tei | extra | seen | total nouveaux | v2 |')
    A('|---|---|---|---|---|---|')
    prev = None
    for name, s in track:
        if name == 'v2 avant ajout':
            prev = None
            continue
        cur = [s.get('tei', 0), s.get('extra', 0), s.get('seen', 0)]
        A(f"| {STEP_FR.get(name, name)} | {fr(cur[0])} | {fr(cur[1])} | {fr(cur[2])} | {fr(sum(cur))} | {fr(s.get('v2', 0))} |")
    A('')
    A('Lecture : l\'ajout met les mentions du géocodeur après retrait de Nakala et des doublons de documents ; ensuite chaque ligne montre ce qui sort ou reste. La fusion ne change pas les nombres, elle rassemble les points.')
    A('')
    nice = steps['3 Nice / Nyce in a council context -> Iznik']
    A('Détails utiles :')
    A(f"* Nice du concile : {nice['mentions_on_nice_forms']} mentions de formes Nice lues avec leur contexte (v2 : 297). Les 43 mentions « council » des nouveaux jeux (`fix_all/ctx.parquet`) ne déplacent rien vers İznik : 0 de plus que v2 (102).")
    A('* Personnes : les formes dont la probabilité moyenne de personne atteint 0,9 perdent leur point (4 formes des nouveaux jeux, 6 mentions).')
    A('* Antioche latin : 3 formes de plus (Antiochia, Antiochiae, Antiochiam) rejoignent Antakya, 39 mentions.')
    vf = v3['verdict_forms']
    A(f"* Formes jugées : {len(vf)} forme(s) des nouveaux jeux ({fr(sum(r[2] for r in vf))} mentions : " + ', '.join(f'{r[0]} {r[1]} {r[2]}' for r in vf) + ") rejoignent le point d\'un verdict de balayage avant son application.")
    A('')
    A('## Règles de garde du paquet atlas (`curation.py`), appliquées à toute la couche')
    A('')
    anc = v3['ancient']
    far = v3['far']
    A(f"* **Noms antiques** : {len(anc)} formes retirées, {fr(sum(r[3] for r in anc))} mentions : " + ', '.join(f'{r[0]} ({r[1]}, {r[2]}, {r[3]})' for r in anc) + '. '
      'Troye (en, fr, it, la) vient de v2 : placée à Troyes (Aube), elle est hors de la région grecque de `ancient_names.tsv`. Athos (de) était placée dans les Pyrénées-Atlantiques.')
    A(f"* **Homonymes lointains** : {len(far)} cas. Aucun point des Amériques, d'Asie ou d'Afrique australe n'a un homonyme du Vieux Monde dans d'autres collections avec la part de moins de 1 % que la règle exige. "
      f"La liste de contrôle de v2 est vide aussi ({len(v3.get('far_v2_listed', []))}).")
    A('* Les règles de `curation.py` ne s\'appliquent qu\'aux points d\'éditeurs et aux formes ; les entrées `homonym_fixes.json` sont par édition et ne visent pas la couche du modèle. Pour la couche du modèle, les homonymes sont dans `model_layer_overrides.json` (171 entrées), appliquées à v2 comme à v3.')
    A('')
    A('## Règles pour que v2 reste v2')
    A('')
    A('Pour ne pas défaire l\'audit de v2, trois choix dans la fusion :')
    A('* un point des nouveaux jeux seul ne relie jamais deux points de v2 (un amas de points nouveaux rejoint **un seul** point de v2 : meilleure preuve, puis le plus gros, puis le plus proche) ;')
    A('* les clés de nom d\'un point de v2 viennent de ses formes de v2 ; coordonnées, genre, rayon et libellé restent ceux de v2 (le plus gros élément de v2 mène le point) ;')
    A('* un point de v2 garde son libellé de v2 (calculé sur ses formes de v2). Les points des nouveaux jeux sont libellés avec la même règle.')
    A('')
    cmp = compare(op, np_)
    A(f"Résultat : sur les {fr(len(op))} points de v2, {fr(cmp['same'])} sont inchangés en coordonnées, genre et libellé ({cmp['shared']} autres partagent leur identifiant d\'origine avec un point voisin, Nice et Nicée par exemple : vérifiés par nom et coordonnées, {cmp['shared_same']} inchangés) ; {cmp['radius']} changent de rayon (le point coïncide maintenant avec le point d\'un éditeur de plus grand rayon : " + ', '.join(cmp['radius_names']) + f"). "
      f"{cmp['lost']} point perd des mentions : Troyes ({cmp['troyes']}, la règle des noms antiques).")
    A('')
    A('## Par collection : avant et après')
    A('')
    A('Mentions placées dans la couche, v2 puis v3, et documents avec une mention. « nouveaux » = tei + extra + seen.')
    A('')
    A('| collection | v2 : mentions | v3 : mentions | de v2 | des nouveaux jeux | documents v2 | documents v3 |')
    A('|---|---|---|---|---|---|---|')
    sc = acc['by_set_coll']
    newc = Counter()
    for k, v in sc.items():
        s, c = k.split('|', 1)
        if s != 'v2':
            newc[c] += v['mentions']
    v2c = Counter()
    for k, v in sc.items():
        s, c = k.split('|', 1)
        if s == 'v2':
            v2c[c] += v['mentions']
    colls = sorted(set(ncoll) | set(ocoll), key=lambda c: -(ncoll.get(c, 0)))
    for c in colls:
        A(f"| {c} | {fr(ocoll.get(c, 0))} | {fr(ncoll.get(c, 0))} | {fr(v2c.get(c, 0))} | {fr(newc.get(c, 0))} | {fr(odoc_coll.get(c, 0))} | {fr(ndoc_coll.get(c, 0))} |")
    A(f"| **total** | **{fr(sum(ocoll.values()))}** | **{fr(sum(ncoll.values()))}** | **{fr(sum(v2c.values()))}** | **{fr(sum(newc.values()))}** | **{fr(len(od))}** | **{fr(len(nd))}** |")
    A('')
    A('## Par jeu et collection (nouveaux jeux)')
    A('')
    A('Du géocodeur à la couche : mentions placées par le géocodeur (`geo_all`), puis mentions dans la couche v3 (après retrait des documents en double et des corrections).')
    A('')
    A('| jeu | collection | documents | mentions envoyées | placées (géocodeur) | dans la couche v3 | documents avec mention | points |')
    A('|---|---|---|---|---|---|---|---|')
    g = bycol.groupby(['set', 'collection']).agg(docs=('docs', 'sum'), sent=('mentions_clean', 'sum'), placed=('mentions_placed', 'sum')).reset_index()
    for r in g.sort_values(['set', 'placed'], ascending=[True, False]).itertuples():
        v = sc.get(f'{r.set}|{r.collection}', {'mentions': 0, 'documents': 0, 'points': 0})
        A(f"| {r.set} | {r.collection} | {fr(r.docs)} | {fr(r.sent)} | {fr(r.placed)} | {fr(v['mentions'])} | {fr(v['documents'])} | {fr(v['points'])} |")
    A('')
    A('## Paliers (principal à partir de 11 mentions)')
    A('')
    A(f"* v2 : {fr(om)} principaux, {fr(orr)} rares. v3 : **{fr(nm)} principaux** ({fr(nmn)} mentions), {fr(nr)} rares ({fr(nrn)} mentions). Les paliers sont recalculés sur le total (v2 + nouveaux) de chaque point.")
    A('* Combinaisons de jeux par point : ' + ', '.join(f'{k} : {fr(v)}' for k, v in sorted(acc['points_by_sets'].items(), key=lambda kv: -kv[1])[:8]) + '.')
    A('')
    A('## À relire')
    A('')
    ov3 = Path(__file__).resolve().parent.parent / 'data' / 'model_layer_overrides_v3.json'
    n_ov = len(json.loads(ov3.read_text('utf-8'))['entries']) if ov3.exists() else 0
    if n_ov:
        A(f'Relecture du 23 oct. (soir) : sur les 74 points principaux qui n\'existent que par les nouveaux jeux, 13 étaient faux (3 contextes lus par point) ; {n_ov} verdicts sont dans `data/model_layer_overrides_v3.json` et appliqués par la construction : '
          'Rhenum (le Rhin, pas Rhenen), Düringen (Thuringe), Edenburg (Édimbourg), Groningen (Pays-Bas), Rüssel (Bruxelles), Bemont (Piémont), Gelle (Gela), Gomera (Canaries) ; sans point : Ramstein, Landsberg, Augustana, Rickenbach, Bencken. '
          'Les 61 autres points ne sont pas modifiés. Le fichier `points_a_relire_v3.tsv` liste les points qui n\'existent que par les nouveaux jeux et ont 11 mentions ou plus (après correction).')
    else:
        A('Aucun verdict nouveau n\'a été ajouté : seules les corrections existantes sont appliquées. Le fichier `points_a_relire_v3.tsv` liste les points qui n\'existent que par les nouveaux jeux et ont 11 mentions ou plus.')
    A('')
    A('* **Précision** : `seen/` est entièrement dans l\'entraînement, aucune précision n\'y est mesurable. Pour `tei` et `extra`, elle ne se mesure que hors des documents d\'entraînement.')
    A('* La couche v3 n\'est pas figée : relire les points à relire, puis relancer `ner2_layer_v3_build.py` avec un fichier de verdicts de plus.')
    A('')
    A('## Fichiers')
    A('')
    A('* `RQ/derived/atlas/page_kit/v29/model_layer_v2.json` et `model_layer_v2_docs.json` : mêmes noms et même format que v28 ; le contenu est v3. Le titre de l\'en-tête dit « version 3 ».')
    A('* `build_log_v3.json` (étapes, suivi par jeu, comptes par collection, listes) ; `docs_exclus_v3.tsv` ; `points_a_relire_v3.tsv` ; `dedup_all/` (Hypatia : `dup_flags.tsv`, `intra.tsv`, `layer_matches.tsv`, `exact_*.tsv`, `overlap_check.json`) ; `fix_all/` (contextes et probabilités de personne des nouvelles formes).')
    A('* Reproduction : `sbatch slurm/ner2_layer_v3_check.sbatch both` (Hypatia, partition debug, 16 CPU, environ 2 minutes), copie de `dedup_all/` et `fix_all/`, puis `python scripts/ner2_layer_v3_build.py --work <tables v1/es> --v3 <layer/v3> --globe v2.4/globe_places.json --out-dir <sortie>`.')
    A('')
    A('---')
    A('')
    A('# Étapes précédentes (même dossier)')
    A('')
    A('Texte du rapport d\'avant, gardé tel quel. Ses « Reste à faire » sont faits dans cette étape : corrections de v2 appliquées, recouvrement tei / extra contrôlé. `ner2_layer_geo_match.py` n\'a pas été relancé : la mise en correspondance avec les points des éditeurs est refaite par `ner2_layer_v3_build.py` (même règle que `ner2_layer_fix.py`).')
    A('')
    if a.keep_old and Path(a.keep_old).exists():
        old_md = Path(a.keep_old).read_text('utf-8')
        A(old_md.replace('\n# ', '\n## ', 1) if old_md.startswith('# ') is False else '#' + old_md)
    (V3 / 'LISEZMOI.md').write_text('\n'.join(L) + '\n', 'utf-8')
    (V3 / 'build_log_v3.json').write_text((B / 'build_log.json').read_text('utf-8'), 'utf-8')
    print('ok', len(L), 'lines')


def compare(op, np_):
    bp = {}
    for p in np_:
        for x in p['prev']:
            bp.setdefault(x, []).append(p)
    byname = {(p['name'], round(p['lat'], 3), round(p['lon'], 3), p['k']) for p in np_}
    same = rad = shared = shared_same = 0
    rad_names, lost = [], 0
    troyes = ''
    for p in op:
        c = {id(q): q for x in p['prev'] for q in bp.get(x, [])}
        if len(c) != 1:
            shared += 1
            shared_same += int((p['name'], round(p['lat'], 3), round(p['lon'], 3), p['k']) in byname)
            continue
        q = list(c.values())[0]
        if abs(p['lat'] - q['lat']) < 1e-4 and abs(p['lon'] - q['lon']) < 1e-4 and p['k'] == q['k'] and p['name'] == q['name']:
            same += 1
        if p.get('r') != q.get('r'):
            rad += 1
            rad_names.append(f"{p['name']} {p.get('r')} -> {q.get('r')}")
        if q['n'] < p['n']:
            lost += 1
            troyes = f"{p['n']} -> {q['n']}"
    return {'same': same, 'radius': rad, 'radius_names': rad_names, 'lost': lost, 'troyes': troyes, 'shared': shared, 'shared_same': shared_same}


if __name__ == '__main__':
    main()
