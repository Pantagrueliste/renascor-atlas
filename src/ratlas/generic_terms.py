"""Tagged pronouns and common nouns (v2.1: excluded from the default counts and from the globe; the rows stay in the tables).

The editors of some editions tag anaphora and common nouns (Sandrart's *ich*, the Anhalt diary's *er* / *sie* / *Vatter*, GerManC's *wir*,
dta_1600's *Schloß*, *Kirche*, *die Stadt*). They are not names: they are not counted as persons or places and not drawn on the globe.
The owner's default (2026-10-17): flag them, keep the rows, ``excluded_reason = 'pronoun_or_common_noun'``.

* person mentions: a **pronoun** (de en fr it es la nl pl), a **kinship / relational noun** (Vater, brother, mère, hermano, filius ...), a
  **generic person noun** (Herr, Frau, Diener, Leute ...) or a **bare title** with no name (Kaiser, König, Madame, Majestät, empereur ...),
  also with a leading article, possessive or preposition (``sein Vater``, ``der Kayser``, ``my brother``). A surface with a name in it
  (``Kaiser Ferdinand``, ``Fürst Ludwig``) is a name and is kept.
* place mentions: a deictic (hier, alhier, anhero, there), a generic place noun (Stadt, Schloß, Kirche, town, ville, ...), also with an article or
  a preposition (``die Stadt``, ``of this towne``, ``in der Stadt``). Never a name.

``generic_kind(kind, surface)`` returns ``'pronoun'``, ``'common_noun'`` or None. Pure functions, no I/O. The sub-class is what
``scripts/generic_report.py`` and the report count; the table's reason is the single value ``pronoun_or_common_noun``.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

REASON = 'pronoun_or_common_noun'


def _key(s) -> str:
    """casefold, small superscript letters of print as letters, accents and long s removed, punctuation to spaces."""
    s = unicodedata.normalize('NFKD', str(s or '').casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('ß', 'ss').replace('ſ', 's').replace('æ', 'ae').replace('œ', 'oe').replace('ÿ', 'y')
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s).split())


# ---------------------------------------------------------------------------------------------------------------------
# persons
# ---------------------------------------------------------------------------------------------------------------------
PRONOUNS = frozenset('''
ich du er sie es wir ihr mich dich ihn ihm ihnen uns euch sich mir dir ihrer seiner meiner deiner unser unserer euer
man jemand niemand jedermann wer wen wem wessen selbst derselbe dieselbe dieselben derselben selbiger selbige selbigen
dieser diese dieses jener jene jenes welcher welche welches
i you he she it we they me him her us them who whom whose one himself herself themselves myself yourself ourselves
je tu il elle ils elles on nous vous lui eux moi toi soi celui celle ceux celles cil
io egli esso essa essi esse noi voi loro costui colui colei
yo tu usted ustedes nosotros nosotras vosotros vosotras ellos ellas
ego nos vos ille illa illi hic haec iste ipse ipsa ipsi idem
ik jij hij zij wij gij u hem haar hen zich
ja ty on ona ono my wy oni one
'''.split())
# words that are pronouns in one language and names or words in another (is ea id will mark may rose) are left out; ella, lei, el, qui ... are
# in (the audit's own pronoun list, which is a frozen yardstick, has them: a lone tag of such a word is an anaphor in these corpora)
PRONOUNS = PRONOUNS | {'elle', 'ella', 'lei', 'qui', 'quis', 'quae', 'quod', 'eu', 'ele', 'ela', 'el'}

KINSHIP = frozenset('''
vater vatter vaeter vaterlein mutter mueter mutterlein bruder bruders gebruder bruderlein schwester schwestern geschwister sohn sohne sohnen
tochter tochterlein kind kinder kindlein eltern grossvater grossmutter grosseltern enkel enkelin tante onkel oheim ohem vetter vettern base
basen schwager schwagerin schwiegersohn schwiegertochter schwiegervater schwiegermutter neffe nichte gemahl gemahlin gemal gemalin gemaehl
ehegemahl ehegemahlin ehemann ehefrau eheherr braut brautigam witwe wittib wittwe waise vormund pate patin
father mother brother brothers sister sisters son sons daughter daughters child children parents grandfather grandmother grandson
granddaughter uncle aunt nephew niece cousin cousins husband wife spouse widow orphan guardian kinsman kinswoman
pere mere frere freres soeur soeurs fils fille filles enfant enfants parents oncle tante neveu niece cousin cousine mari femme epoux epouse
veuve gendre bru beau pere belle mere beaupere bellemere
padre madre fratello fratelli sorella sorelle figlio figli figlia figlie marito moglie sposo sposa zio zia nipote cugino cugina vedova genero
hermano hermanos hermana hermanas hijo hijos hija hijas esposo esposa marido mujer tio tia sobrino sobrina primo prima viuda yerno
pater mater frater fratres soror sorores filius filii filia filiae uxor maritus coniunx avus avia nepos neptis patruus avunculus consobrinus
vader moeder broer zuster zoon dochter kind kinderen man vrouw echtgenoot echtgenote weduwe oom tante neef nicht
ojciec matka brat siostra syn corka dziecko dzieci maz zona
'''.split())

GENERIC_PERSON = frozenset('''
herr herren herrn herrschaft herrschafften frau fraw frawen frauen frauwen weib weiber mann manner leute leut volk gesinde diener dienerin
knecht magd magde bauer bauern burger burgerin einwohner untertan untertanen unterthan nachbar nachbarn freund freunde gast gaste gesandter
gesandte doktor doctor arzt medicus pfarrer prediger pastor kaplan kaplane lehrer schuler student studenten soldat soldaten kriegsvolk
gentleman gentlemen lord lady master mistress sir madam servant servants man men woman women people folk neighbour neighbours friend friends
monsieur messieurs mesdames mademoiselle seigneur dame dames homme hommes gens serviteur valet
signore signori signora donna donne uomo uomini
senor senores senora dona hombre hombres
dominus domina domini homo homines servus
heer mevrouw
'''.split())

TITLES = frozenset('''
kaiser kayser keiser keyser kaisers kaysers kaiserin kayserin keiserin kaiserliche kaiserlichen kaiserlicher kayserliche kayserlichen kayserlicher
konig konige koenig koenige konigin koenigin koniglich koenigliche koniglichen konigliche koniglicher
furst fursten furstin fursten furstliche furstlichen furstlicher fuerst fuersten fuerstin fuerstliche fuerstlichen
herzog herzogin hertzog hertzogin herzoge herzoglich herzogliche herzoglichen hertzogliche hertzoglichen
graf grafin grave graffin gravin graven grafen
churfurst churfursten kurfurst kurfursten churfurstin kurfurstin churfurstliche churfurstlichen kurfurstliche kurfurstlichen churf churfurstl
pabst papst bapst pabstin papste bischof bischoff bischofe bischoffe erzbischof cardinal kardinal abt abtissin probst propst
prinz prinzessin printz printzessin marggraf margraf markgraf landgraf pfalzgraf burggraf freiherr freyherr baron
majestat mayestat maiestat majestaet mayestaet maiestaet hoheit durchlaucht durchl durchleucht excellenz exzellenz gnaden
praesident prasident prasidenten president kanzler cantzler rat rath rate rathe
emperor empress king queen prince princess duke duchess earl count countess bishop archbishop pope cardinal majesty highness
grace lordship ladyship
empereur imperatrice roi reine prince princesse duc duchesse comte comtesse eveque archeveque pape cardinal majeste altesse madame
sire monseigneur
imperatore imperatrice re regina principe principessa duca duchessa conte contessa vescovo arcivescovo papa cardinale maesta altezza
emperador emperatriz rey reina principe princesa duque duquesa conde condesa obispo arzobispo papa cardenal majestad alteza
imperator imperatrix rex regina princeps dux ducissa comes comitissa episcopus archiepiscopus papa pontifex cardinalis maiestas
maiestatem maiestate regia regiam regius regis regem serenissimus serenissima celsitudo
keizer keizerin koning koningin vorst hertog graaf bisschop paus
cesarz krol krolowa ksiaze
'''.split())

# leading words stripped before the test: articles, possessives, demonstratives, prepositions
_LEAD = frozenset('''
der die das dem den des ein eine einen einem einer eines kein keine
sein seine seinem seinen seiner seines ihr ihre ihrem ihren ihrer ihres mein meine meinem meinen meiner unser unsere unserem unseren unserer
dein deine euer eure dieser diese dieses diesem diesen jener jene
the a an my his her our your their this that these those
le la les l un une mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs ce cette ces
il lo gli i una uno mio mia miei mie tuo tua suo sua suoi sue nostro nostra vostro vostra
el los las un una mi su sus nuestro nuestra
de het een zijn haar onze
von vom zu zum zur im in an auf bey bei mit aus nach fur vor uber of at to by from near with for
a de du des da di en ad cum apud ab ex pro
'''.split())
_AND = frozenset('und et and y e ac oder ou or'.split())


def _strip_lead(k: str, extra: frozenset = frozenset()) -> str:
    toks = k.split()
    # strip only leading function words; a name particle followed by a name stays (``de Groot`` is not stripped to ``Groot``: the head word is
    # tested against the lists only when the whole remainder is a listed word, so this strip is harmless for names)
    while len(toks) > 1 and (toks[0] in _LEAD or toks[0] in extra):
        toks = toks[1:]
    return ' '.join(toks)


def person_generic(surface) -> Optional[str]:
    """'pronoun' | 'common_noun' | None for the surface of a person mention."""
    k = _key(surface)
    if not k:
        return None
    if k in PRONOUNS:
        return 'pronoun'
    r = _strip_lead(k)
    if r in PRONOUNS:
        return 'pronoun'
    if r in KINSHIP or r in GENERIC_PERSON or r in TITLES:
        return 'common_noun'
    # two words made only of listed words: ``Herr Vater``, ``Ihre Majestät`` (possessive stripped above), ``der Kaiserin Mutter``
    toks = r.split()
    if 1 < len(toks) <= 3 and all(t in KINSHIP or t in GENERIC_PERSON or t in TITLES or t in _LEAD or t in _AND for t in toks) \
            and any(t in KINSHIP or t in GENERIC_PERSON or t in TITLES for t in toks):
        return 'common_noun'
    return None


# ---------------------------------------------------------------------------------------------------------------------
# places
# ---------------------------------------------------------------------------------------------------------------------
DEICTIC = frozenset('''hier hie da dort dorten daselbst dahin dorthin hinc hic huc ibi istic illic illuc inde unde ubi alibi ibidem ibid
ici ailleurs qui qua quivi cola here there hence thence thither whither elsewhere domi foris allhier alhier alda dasselbst anhero anher hieher
hierher hieselbst hiesig hiesige hiesigen daher dorthin hindan drinnen draussen daheim zuhause'''.split())
COMMON_NOUN = frozenset('''welt herberg herberge gang stall sonnen trauben keller kaeller kuche kueche kammer stube saal garten hoff oberamt
oberwasser palais palast isle cielo tierra mundo mundus orbis terra coelum caelum regio locus ort platz hauss haus thor tor strasse gasse markt
kirche capell kapelle schloss schlosz wasser stadt statt town towne city citie ville citta ciudad dorf village villaggio kirchen kirchsaal marckt
land lande lander landen reich konigreich kingdom royaume vaterland heimat gegend gebiet provinz
house houses church chapel castle palace street streets parish parishes
eglise chateau maison rue pays contree province rivage
provintz provintzen provintzien provinzen provincien provincias provinces provinciae landen reiche reichen
chiesa castello strada paese contrada
iglesia castillo calle pais
ecclesia castrum urbs oppidum civitas
kerk kasteel huis straat dorp stad land
kosciol zamek dom ulica miasto wies'''.split())
_PLACE_LEAD = frozenset('''the a an this that our your their these those same said aforesaid die der das dem den im in an auf zu zum zur von vom
bey bei aus nach of at to by from near within without le la les l un une ce cette el los las il lo gli i de du des da di en a al del della
dieser diese dieses diesem diesen unser unsere unserer unserem unseren seine sein seiner'''.split())


def place_generic(surface) -> Optional[str]:
    """'common_noun' | None for the surface of a place mention (deictics and generic place nouns)."""
    k = _key(surface)
    if not k:
        return None
    toks = k.split()
    # strip leading function words ('of this towne', 'in der Stadt'), keep at least one word
    while len(toks) > 1 and toks[0] in _PLACE_LEAD:
        toks = toks[1:]
    r = ' '.join(toks)
    if r in DEICTIC or r in COMMON_NOUN:
        return 'common_noun'
    # 'the same town', 'this parish here': the head noun is generic and every other word is a function word / demonstrative
    if len(toks) <= 3 and toks[-1] in COMMON_NOUN and all(t in _PLACE_LEAD or t in COMMON_NOUN or t in DEICTIC for t in toks):
        return 'common_noun'
    return None


def generic_kind(kind: Optional[str], surface) -> Optional[str]:
    """Dispatcher by mention kind (person / place; other kinds are never generic here)."""
    if kind == 'person':
        return person_generic(surface)
    if kind == 'place':
        return place_generic(surface)
    return None


def is_generic(kind: Optional[str], surface) -> bool:
    return generic_kind(kind, surface) is not None


# ---------------------------------------------------------------------------------------------------------------------
# v2.2: a mention that names several entities at once
# ---------------------------------------------------------------------------------------------------------------------
REASON_MULTI = 'multi_referent'
_REF_TOKEN = re.compile(r"^[\'\"]*(#|[A-Za-z][A-Za-z0-9_\-]{0,11}:)(?!//)([^\s#]+)")


def referents(ref_raw) -> frozenset:
    """The distinct referents a ``ref_raw`` value names: ``#id`` tokens (the entries of the edition's register) and ``scheme:id`` tokens, as (scheme, id).
    ``#emt_person_id__25 #emt_person_id__31`` is two; ``gnd:1 viaf:2`` is two identifiers of ONE entity (the schemes differ), not two referents."""
    out = set()
    for t in str(ref_raw or '').split():
        m = _REF_TOKEN.match(t)
        if m:
            out.add((m.group(1), m.group(2).lower().strip("'\"")))
    return frozenset(out)


def multi_referent(ref_raw) -> bool:
    """True when the value names two or more distinct entries of the same scheme: the tag is a collective (``Schwestern``, ``kinder``, ``Churfürsten``,
    ``Provintzien``, ``beyde``) and the extractor kept only the first as the local key. Such a mention is not a mention of that one entity: v2.2 leaves it
    out of the per-entity counts (reason ``multi_referent``, the row stays) rather than splitting it, which would count one mention several times."""
    seen: dict = {}
    for scheme, ident in referents(ref_raw):
        seen.setdefault(scheme, set()).add(ident)
    return any(len(v) >= 2 for v in seen.values())
