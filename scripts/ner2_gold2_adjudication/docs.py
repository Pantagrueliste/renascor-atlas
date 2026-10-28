import json
D='/Users/clem/Projects/Papers/RQ/derived/atlas/ner/v2/gold2/'
tabs=json.load(open('tabs.json')); cnt=json.load(open('counts.json'))
NAMES={'en':'anglais','la':'latin','it':'italien','es':'espagnol','fr':'français','de':'allemand'}
L=['en','la','it','es','fr','de']
tot={k:sum(cnt[l].get(k,0) for l in L) for k in ['passages','tokens','spans','agreed','proposed','rejected','to_check','no_place']}
def row(l):
    c=cnt[l]; return f"| {NAMES[l]} | {c['passages']} | {c['tokens']} | {c['spans']} | {c.get('agreed',0)} | {c.get('proposed',0)} | {c.get('to_check',0)} | {c.get('rejected',0)} | {c['agreed']+c.get('proposed',0)} | {c.get('no_place',0)} |"
table="| Langue | Passages | Tokens | Spans candidats | Accordés | Proposés | À vérifier | Rejetés | Référence provisoire | Passages sans lieu |\n|---|---|---|---|---|---|---|---|---|---|\n"+"\n".join(row(l) for l in L)+f"\n| **Total** | {tot['passages']} | {tot['tokens']} | {tot['spans']} | {tot['agreed']} | {tot['proposed']} | {tot['to_check']} | {tot['rejected']} | {tot['agreed']+tot['proposed']} | {tot['no_place']} |"
readme=f"""# Gold v2 : fusion provisoire (à relire par le propriétaire)

Statut : provisoire. Rien n'a été relu par un humain. Les décisions « proposed » et « rejected » sont les miennes (relecture des passages), pas les vôtres.

## Chiffres

{tot['passages']} passages, {tot['spans']} spans candidats.
{tot['agreed']} accordés, {tot['proposed']} proposés, {tot['to_check']} à vérifier, {tot['rejected']} rejetés.
Référence provisoire (accordés + proposés) : {tot['agreed']+tot['proposed']} spans.
{tot['no_place']} passages sur {tot['passages']} n'ont aucun lieu dans cette référence.

{table}

## Sources fusionnées

- Annotation à l'aveugle : `ann_en.jsonl`, `ann_la.jsonl`, `ann_it.jsonl`, `ann_es.jsonl`, `ann_fr.jsonl`, `ann_de.jsonl` (certitude « sure » ou « doubt »).
- Prédictions : `predictions.jsonl`, 4 encodeurs (mDeBERTa, mmBERT, XLM-R large, hmTEAMS), Qwen3.5-9B (zéro-shot), flair (anglais seulement).
- Passages : `passages.jsonl` (250).

## Statuts

- `agreed` : l'annotateur est « sure » et au moins 2 systèmes ont exactement le même span. Aucune relecture de ma part.
- `proposed` : un span trouvé par l'annotateur ou par au moins un système, que je tiens pour un lieu après relecture du passage.
- `rejected` : un span que je tiens pour une erreur (nom commun, gentilé, personne, variante de frontière, etc.).
- `to_check` : je n'ai pas pu trancher. C'est une question de convention ou de lecture.

Chaque span a une raison en une ligne, ses frontières en tokens (`start` inclus, `end` exclu) et en caractères, et les sources qui l'ont trouvé (`sources` : annotateur et chaque système).

## Fichiers

- `proposed_gold_v2.jsonl` : un passage par ligne, champs du passage + `spans[]`. C'est l'entrée de la page de relecture.
- `provisional_scores.md` : scores provisoires de chaque système.
- `LISEZMOI.md` : ce fichier.

## À regarder en premier

1. Les {tot['to_check']} spans `to_check`. Ils tiennent en six questions :
   - Adjectifs toponymiques latins en -ensis dans un titre ou une institution (Aurelianensis, Wintoniensem, Tubingensi x2, Spanheimensis) : 5 spans. 3 encodeurs et Qwen en marquent 4 sur 5.
   - Variantes d'un nom de lieu citées comme mots, discussion de prononciation (Caludon, Caledon, Caldon, Calden, Caldn) : 5 spans, un seul passage latin (A03834).
   - « la Corte » ou « Palacio » pour Madrid, noms communs capitalisés : 3 spans espagnols.
   - Lieu d'origine dans un surnom (Lunardo da Dresano x2, Anton da Castello) et lieu dans un nom d'événement (Concilio di Trento) : 4 spans italiens.
   - Royaume pour son seigneur (Saxuma, Arimando) : 2 spans français. Guinea dans une épithète figurée : 1 span espagnol.
   - Cas isolés : Sarum (rite), « sanct Peters münster » (église citée comme ouvrage), « vor Wald », « Blawe Thal » (étymologie de Plauen).
2. Les 18 spans `rejected` marqués par au moins 3 systèmes : 7 variantes de frontière (County of Norfolk, etc.), 5 noms communs, gentilés ou noms de famille (Kingdome, Armes, Chapelle du Chasteau, Italis, Soria), 5 églises titulaires de cardinaux de la sentence de Galilée (S. Anastasio, S. Carlo, S. Onofrio, S. Pietro in Vincoli, S. Sisto : rejetées, ce n'est pas un usage locatif) et « laCour ».
3. Les `proposed` les plus fragiles : 29 où l'annotateur doutait, 9 sans aucun système, 3 trouvés par des systèmes seuls.
4. Les 125 passages sans aucun lieu : ce sont peut-être des lieux manqués par tout le monde. Seule votre relecture peut les trouver.

## Conventions que j'ai appliquées (modifiables)

- Gentilés, adjectifs d'origine, noms de langues : rejetés.
- Lieu dans un titre de noblesse ou de souverain (rex Franciae, Duca di Milano, Lord of Salisbury) : proposé.
- Nom de famille tiré d'un lieu (von Zimbern, Gómez de Ampuero, madame de Clèves) : rejeté.
- Église, couvent ou pont nommé, employé comme repère de lieu : proposé. Église titulaire dans un titre de cardinal : rejetée.
- Mot générique (pont, rivière, abbaye) inclus quand il fait partie du nom (Brandon Bridge, Mildenhall River), exclu sinon (mare Baltico, rue Saint Nicolas, County of Norfolk).
- Planètes, astres, notions théologiques, livres bibliques, noms de personnages : rejetés.
- Variantes de frontière : l'annotateur l'emporte, sinon le découpage minimal. L'autre variante est rejetée avec la raison « variante de frontière ».

## Limites

- Les « accordés » exigent 2 systèmes : la référence favorise ce que les systèmes trouvent. Le rappel des systèmes est donc surestimé.
- Un seul annotateur (un agent), à l'aveugle. Aucun accord inter-annotateur.
- Je n'ai pas relu les 429 spans accordés un par un. J'ai parcouru la liste de leurs textes : rien d'anormal.
"""
open(D+'LISEZMOI.md','w').write(readme)
sc=f"""# Scores provisoires des systèmes sur gold v2

Statut : provisoire. La référence = spans `agreed` + `proposed` de `proposed_gold_v2.jsonl` ({tot['agreed']+tot['proposed']} spans, dont {tot['proposed']} décidés par moi). Le propriétaire n'a rien relu encore. Les chiffres bougeront.

Format des cellules : F1 (précision/rappel). Un point par span exact, sauf tableau C. Micro-moyenne sur tous les passages de la langue.

## Lecture rapide

- Le vote d'au moins 2 encodeurs a le meilleur F1 en italien, français et anglais (à égalité avec mDeBERTa), et toutes langues : 0.88 (tableau B), contre 0.85 pour le meilleur encodeur seul (mDeBERTa). En latin XLM-R seul fait mieux (0.83 contre 0.81). En allemand tout est groupé autour de 0.71-0.76.
- Qwen3.5-9B a le meilleur rappel (0.91 toutes langues) mais la plus faible précision (0.66). En espagnol il est le meilleur système (F1 0.87).
- XLM-R large, le meilleur de v1 sur le texte non balisé, s'effondre en espagnol : 4 spans prédits, rappel 0.08. mmBERT (0.68) et mDeBERTa (0.63) tiennent mieux. Les encodeurs sont tous faibles en espagnol : précision de 0.67 à 1.00, rappel de 0.08 à 0.52.
- Le latin et l'allemand sont les langues les plus dures pour les encodeurs (F1 de 0.71 à 0.83). L'italien et le français sont à 0.82-0.95.
- Le seuil « 2 encodeurs et confiance >= 0.98 » donne une précision de 0.99 toutes langues, mais un rappel de 0.36 seulement. Il ne laisse presque rien passer en anglais (rappel 0.23) et en espagnol (0.08).
- En anglais, flair est derrière les encodeurs en correspondance exacte (0.80) mais au niveau d'eux avec frontières tolérées (0.89).

{tabs['strict']}

{tabs['strict_ignoreC']}

{tabs['lenient']}

## Comment lire

- Tableau A : lecture littérale. Un système qui marque un des {tot['to_check']} spans « à vérifier » est compté en faux positif.
- Tableau B : ces {tot['to_check']} spans sont neutres (ni bon ni faux). C'est la lecture que je retiendrais.
- Tableau C : un span prédit est bon s'il recouvre un span de référence. Il mesure les lieux trouvés, sans la qualité des frontières. L'écart B/C est grand en anglais pour flair (0.80 contre 0.89), XLM-R, hmTEAMS, mmBERT et Qwen : leurs frontières diffèrent de la référence (par exemple « County of Norfolk »).
- Lignes « vote » : span prédit par au moins 2 des 4 encodeurs (puis, en plus, avec une confiance maximale >= 0.98). Ce sont les règles d'usage direct recommandées en v1.

## Biais à garder en tête

- Un span « accordé » a déjà au moins 2 systèmes d'accord : la référence sous-estime ce que les systèmes ratent ensemble. Rappels surévalués, surtout pour les votes.
- Les spans trouvés par un seul système que j'ai proposés ({tot['proposed']} au total) corrigent en partie ce biais.
- Les {tot['no_place']} passages sans lieu comptent pour la précision (tout ce que prédit un système y est un faux positif), pas pour le rappel.
- Flair n'a tourné que sur l'anglais (45 passages). Les autres lignes couvrent les 250 passages.
- Petits effectifs en espagnol (48 spans de référence), français (31) et latin (45) : un span change un score de 2 à 3 points.
"""
open(D+'provisional_scores.md','w').write(sc)
