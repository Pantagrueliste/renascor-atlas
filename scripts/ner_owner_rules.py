"""Owner-rule view of editor tags: map the editors' LOC / PER gold spans to the owner's NER decisions of 2026-10-25
(RQ/derived/atlas/ner/v2/renner/conventions_proprietaire_2026-10-25.md, guide_annotation.md v0.2). Used by ner_editor_bench_score.py
--owner-rules. Gold side only: predictions are never changed.

Mappings (each one is counted per edition; MAPPINGS gives the codes):
  period       1  a final '.' token is removed from a gold span (LOC and PER, every edition and view)
  people_loc   2  a one-word LOC span that is a people, a demonym, a place adjective or a language is dropped from the gold
                  (lexicon PEOPLE + German -isch- adjectives + Latin adjective stems; Latin -ensis is kept)
  people_per   2  a PER whose name part is a people word (« le Turc », « il Turco » for the sultan) is dropped
  title_strip  6  leading titles, forms of address, honorific adjectives, possessives, articles and kinship words are removed from a PER
                  (lexicons TITLE_DIGNITY, TITLE_ADDRESS, KIN, FUNC; edition prefixes EDITION_PREFIX for one-letter abbreviations)
  tail_cut     6  a PER is cut before a comma or before a following title / kinship word (« Joanni Episcopo Atrabatensi » -> Joanni);
                  koenigsfelden (name-only): also before the first lower-case word that is not a name particle (office, domicile)
  title_place  3  a PER that is only a dignity + preposition + place (« Duca di Nevers », « Herzoch von Lottring », « roi de France »)
                  is dropped; the place becomes LOC (LOC_editor gold, when the edition's LOC_editor is scored)
  tail_place   3  a place named in a cut tail after a preposition or as a Latin -ensis adjective becomes LOC (same condition)
  regnal       4  « Felipe II de España »: PER cut after the numeral, the place becomes LOC (same condition); needs a numeral
  title_only   6  a PER left empty after the removal of titles, address, kinship and pronouns is dropped (« Mein brueder », « Kays Mayt »)
  god          5  a PER whose name part is God (Dieu, Dios, Deus, Gott, Iddio, createur...) is dropped
  personif     5  a PER whose name part is a personification (Fortuna, Amor, Fama, Natura, Mors...) is dropped
  generic_loc  7  a LOC made only of a generic building or place word (« Church », « garden », « Palais ») is dropped
  dropcap      -  elan_crenum: a detached drop-capital letter is joined to its word before the lexicon test (« d ieu » -> dieu)

What cannot be mapped (left as the editor tagged it; listed in UNMAPPED):
  * places the editor did not tag (koenigsfelden princely titles: Österreich, Tirol, Habsburg; places inside PER spans of editions
    whose LOC_editor is not scored): no gold to add;
  * planets vs gods (Mars, Venus, Iuppiter in corpus_corporum, elan_fnvr): needs the sense; left PER;
  * personifications outside the lexicon, antonomasias (« lírico poeta », « Crisis lusitano ») and descriptions without a title word;
  * a bare lordship for its holder (« Cuhr Cöllen » -> Cöllen, « Guisa », « Lorena »): open question Q15 of the guide; left PER;
  * sovereigns without a numeral (« Ferrante d'Aragona ») cannot be told from origin names: left one PER;
  * the Devil and Christ: no change needed (PER in the editions that tag them); untagged ones cannot be added;
  * Latin / Italian dignity + genitive place without a preposition (« dux Saxoniae », « Duci Calabrie »): left PER on the place word;
  * a lower-case place after a dignity (« cardinale di loreno »): the PER is dropped, no LOC is added (``title_place_no_loc``).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

MAPPINGS = ('period', 'people_loc', 'people_per', 'title_strip', 'tail_cut', 'title_place', 'tail_place', 'regnal', 'title_only',
            'god', 'personif', 'generic_loc', 'dropcap')

UNMAPPED = [
    'places not tagged by the editor (koenigsfelden princely titles; places inside PER where LOC_editor is not scored): no gold to add',
    'planet vs god (Mars, Venus, Iuppiter): sense needed; left PER',
    'personifications outside the lexicon, antonomasias, descriptions without a title word: left PER',
    'bare lordship for its holder (Cuhr Cöllen, Guisa, Lorena): guide Q15 open; left PER',
    'sovereign without a numeral (Ferrante d\'Aragona): cannot be told from an origin name; left one PER',
    'untagged Devil / Christ / gods: cannot be added',
    'Latin or Italian dignity + genitive place without a preposition (dux Saxoniae, Duci Calabrie, regis Hispaniarum): place and name '
    'cannot be told apart by form (ducis Alberti); left PER on the place word',
    'lower-case place after a dignity (cardinale di loreno): PER dropped, no LOC added (title_place_no_loc)',
]


def fold(s: str) -> str:
    s = s.replace('ſ', 's').replace('ˀ', '').replace('’', "'").replace('¬', '')
    if s[:2] in ('ff', 'Ff', 'FF'):          # early-modern English ff = F
        s = s[1:]
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r"[^\w']", '', s.lower()).replace('_', '')


def _set(s: str) -> frozenset:
    return frozenset(fold(w) for w in s.split())


# dignities and offices: a dignity + preposition + place is a title that designates the person (decision 3)
TITLE_DIGNITY = _set('''
re rey roi roy king rex regis regi rege regem reges reina reyna regina reginae roine royne reine queen konig könig koenig konigin königin
kaiser kayser kays kayserl kayserliche kayserlichen kaiserin keyser kaÿ kay imperatore imperadore emperador empereur emperor imperator
imperatoris imperatori imperatorem papa pape pope pontefice pontifex pontifice pontificis pontificem
duca ducha duvca duchessa duque duquesa duc ducz duchesse duke duchess dux ducis duci ducem duce herzog hertzog hertzogen herzogs herzogin
herzoch erzherzog erzherzogin archiduc archiduque arciduca
conte contessa conde condesa comte comtesse count countess earl comes comitis comiti comitem graf graff grau grafin gräfin gr
marchese marchesa marques marquesa marquis marquise markgraf marggraf marquess
principe prencipe principessa princessa príncipe prince princesse princess princeps principis principi principem prinz printz princeßin
fürst furst frst fürstin furstin fürstliche furstliche fürstl
churfurst churfürst cuhrfurst cuhrfürst cuhrfurstin cuhrfürstin cuhrfürsten kurfurst kurfürst cuhr chur kur palatino palatin palatinus
pfalzgraf lantgrave landgrave landgraf
vescovo arcivescovo obispo arzobispo eveque evesque evêque archevesque archevêque bishop archbishop episcopus episcopi episcopo episcopum
episcopus archiepiscopus bischof bischoff erzbischof antistes antistitem
cardinale cardinal cardenal cardinalis abate abad abbas abbatis abbé abbe abt abbot äbtissin ebtissin ptissin prior priore prevost prévost
provost propst
viceroi visroi viceré vicere virrey visorey viceroy ammiraglio amiraglio armiraglio almirante amiral admiral
connestable condestable chancelier cancelliere canciller kanzler hoffkanzler hofkanzler deutschmeister deütschmeister
baron barone barón freiherr seigneur signore signoria signioria lord lords granduca
rector rektor rettore presul praesul legatus legati legato legatum legat general generale général comendador commandeur comendator
secretary comptroller controller treasurer chamberlain steward regente régente regent regentin elector electori electorem electoris electeur consul consulis consuli consulem
camerdiener camerdienerin kammerdiener roys rois reyes roi
botschafter botschaffter ambassadeur ambasciatore embajador ambassador nunzio nuntius legato legat
secretario segretario secret secretaire sekretär capitano capitán capitan capitaine captain hauptmann
commendator commendatore governatore gobernador gouverneur governor vogt schultheiss bürgermeister burgermeister mayor alderman warden
pfrunder pfründer pfruͤnder hofmeister hoffmeister meister maestro maistre master magister
''')
# forms of address, honorifics, possessives, pronouns, articles: removed, but not a title that makes a place LOC
TITLE_ADDRESS = _set('''
don dona doña donna dom dueña fray frey fra frate fratre sor sor sre sr sig signor signore signora sigra messer misser ser sier madonna
madama mona monna monsignor monsignore mons monsre monseigneur monsieur monsr msr madame mme mad mademoiselle mlle dame demoiselle
señor senor señora senora sir mr mrs mres mistress maister goodman goodwife lady dame esquire esq gent gentleman
herr herrn her herre hern frau fraw frawen frow fro froͤw jungkher jungkherr junckher juncker junker ritter edel edlen edeln edlenn ersam
ersame ersamen fromen from fromme frommen erbern erben erbn erwirdig erwirdigen hochwirdigen wolgeboren gnädigsten gnadigsten gnädigen
dominus domino dominum domini domine domina dominae dn dd d dr drt doctor dotor dottore docteur licenciado bachiller reverendus
reverendissimus reverendo illustrissimus illustrissimo illmo ill ecc eccmo eccellenza excellentia excelencia excellence serenissimus
serenissimi serenissimo serenissima clarissimus clarissimi vir viri summo summus
mayt mayst mayest mayestat mayestät majestad magestad majesté majesty matie maÿ may maj alteza altesse sanctita santita sanctità santità
sanctitas beatitudine merced vuesa vuestra vuestro vostra vostro vostre votre vos sua suo sue suoi su sus son sa ses mon ma mes his her
my our your yor thy jhre jhrer jhr jhren ihre ihrer ihr ihme ihm sein seine seiner mein meine meinen meiner vnser unser unsere nostre
notre nostro nostra noster nostrum nostri tuum tuus tua meus mea communem communi commun amicum amico amicus ami amy
il lo la le les l i gli el los las the der die das den dem des ein eine un una une
st s san santo santa sancta sancto sanctus sancti sanctum sanctae sainct saint sainte ste sant sanct sankt beato beata beatus divus diuus
seliger selig
alte alter old vieux vecchio viejo feu led ledit ladite lad sayd said este questo quel quello
christianissimo christianissima christianissimus christianma cavaglier cavaliere caballero chevalier cavalier
gran grand grant gross ertz erz vice et e y and und vnd ac un une einer eltister eltiste eltisten elteren min vnsern unsern mi
ille illa illud illius illi hujus huius eius ejus tyrannus amici catholico catolico nostred gna maytt mtt durchl durchlaucht durchleuchtig
highnes highness grace lordshipp lordship worship honour heer heeren mynheer sinjeur joncker jonckvrouwe vrouwe vrauwe durluchtighe
hooggeboren hochgeborene furstinne ende l' dell' all' nell' au aux al alla ad dit dite dict dernier derniere bonne bon buono buona bueno buena good
meinem seinen seinem ihren ihrem deinen jungste iungste alteste altere alteren herren gnad gnaden gn gnadiger
gnediger gnadige regirenden regierenden regirende regierende verwittibte verwittibten geborne gebohrne geb frewlin fraulein freulin
caes maiest maiestat maiestas maiestatem maiestatis majestas majestatis sacra sacrae konigl kgl mt churf furstl suum suam sui suae suo
obligatissimo servidore servitore serviteur auctor auctori chi ella egli
''')
KIN = _set('''
padre madre figlio figliuolo figliolo figlia fratello sorella zio zia nipote moglie marito cugino cugina cognato suocero suocera
hijo hija hermano hermana tío tio tía tia sobrino sobrina mujer esposa marido suegro suegra
père pere mère mere fils fille frère frere soeur sœur oncle tante neveu nièce niece femme veuve
fathers father mothers seur compaigne compagne espouse epouse enffans enfans enfants aisne neveur weduwe huysvrouwe dochter sone broeder suster vader moeder
vater vatter mutter muetter muter sohn sohne sohn söhn söhne eltern filz fame feme primogenito primogenitus son tochter bruder brueder brüeder bruedern brudern brüdern schwester schwster schwegrin schwager vetter
oheim gemahl gemahlin witwe wittib
frater fratris fratri fratrem patris patri patrem patre pater mater matris matri matrem soror sororis filius filii filio filium filia
filiae uxor uxoris uxorem maritus avus nepos parens
father mother brother sister wife wif husband daughter uncle aunt nephew widow
''')
CONJ = _set('et e y und vnd and ac ou o oder')
ARTICLES = _set("il lo la le les l l' i gli el los las the der die das den dem des")
POSSESSIVE = _set('mio mia miei mie tuo tua suo sua suoi sue vostro vostra nostro nostra mi tu su sus vuestro vuestra nuestro '
                  'nuestra mon ma mes ton ta son sa ses notre nostre votre vostre his her my our your yor thy mein meine meinen meiner '
                  'sein seine seiner ihr ihre ihrer jhr jhre jhrer vnser unser meus mea tuus tua suus sua noster nostra')
PARTICLES = _set("uff vff auff zum zur ob vo von vom van vander de del della dei degli di da du des der d' la le y e et zu ze zuo zuͦ am im an auf of a ab ex à")
PREPS = _set("vo de di da d' del della dello dei degli du des von vom van of zu ze zuo zuͦ in im ab ex de' a à dell' dall' nell' auff auf bey bei")
GOD = _set('dieu dio iddio idio iddeo dios deus dei deo deum gott gottes gotte god godes createur créateur creator criador creatore '
           'esprit spirito espiritu espíritu spiritus geist celeste céleste')   # the Holy Spirit is God; Redeemer / Saviour = Christ, not here
PERSONIF = _set('fortuna fortunae fortunam fortune amor amore amour amors fama famae natura naturae nature mors mortis mort morte '
                'invidia envie tod')
PEOPLE = _set('''
turc turcq turcs turco turchi turcos turcae turcarum turcis turcas moscorum moscovitarum gallorum hispanorum anglorum
germanorum polonorum francorum romanorum rhomanum turk turks turke türck türcken türken turcken turken tuͤrcken moscovita moscovite moscouite
franzosen frazosen francosen zosen franzos franzose polaken schpanier schpaniern spanier spaniern portugeser portugesern portugeßern
hollender hollendern holländer niderländer tartern tartaren capuciner dominicanern florentiner erlawern nicanern
francesi francese spagnuoli spagnoli tedeschi inglesi veneziani fiorentini milanesi napoletani genovesi romani
franceses españoles ingleses moros turcos indios alemanes italianos portugueses flamencos
français françois francois françoise angloys anglois angloise espagnols espaignols allemans allemands italiens lyonnois luçois senois
jennevois luxembourgois chalonnois soissonnaise
english french spanish scottish scots welsh dutch irish italian italians spaniards frenchmen englishmen
latin latine latino latina latinus latinum greek grec greco griego graece hebrew hebraice deutsch teutsch tudesco
''')
LATIN_ADJ = re.compile(r'^(polonic|hungaric|germanic|hispanic|hispan(?!i(a|ae|am|arum|is|en|e|as)$)|lusitanic|lusitan(?!i(a|ae|am|arum|is|en|e|as)$)|turcic|gallic|anglic|italic|britannic'
                       r'|lituan|lithuan|litvan|svecic|suecic|moscovitic|moscic|teutonic|francic|neoburgic|magdeburgic|alemannic|westphalic'
                       r'|sarmatic|boic|noric|punic|getic|argolic|pharsalic|babylonic|euboic|maeotic|marmaric|massic|olympic|canopic|cretic'
                       r'|mecklenburgic|brandenburgic|w[uü]rtembergic|madagascaric|romanus|romana[em]?$|romani$|romanos|romanis|romanorum'
                       r'|africani|americani)', re.I)
GERMAN_ADJ = re.compile(r'(isch|[^aeiouäöüy]sch)(e|en|er|es|em)$', re.I)
ENSIS = re.compile(r'ens(is|e|es|i|em|ium|ibus)$', re.I)
GENERIC_LOC = _set('''
church churche chapel chappell chapell church-yard house howse hall halle aula court garden gardin lane street strete sewer sewar side
place market mill tower bridge gate
kirche kirchen kirch schloss schloß schlos rathaus palast hof hoff hofe garten brucke brücke tor thor kloster closter spital markt
chiesa chiese palazzo palagio corte tempio castello ponte porta piazza convento monastero ospedale
iglesia palacio templo castillo puente puerta plaza convento monasterio hospital
église eglise palais chasteau château chastel pont porte place couvent hôpital
ecclesia templum curia palatium arx porta pons forum monasterium
''')
ROMAN = re.compile(r'^(?=[IVXLCDM]+$)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')
ORDINAL = _set('primo primero premier primus erste secondo segundo second secundus secundo andere terzo tercero troisième tertius dritte '
               'quarto cuarto quatrième quartus vierte quinto cinquième quintus fünfte sesto sexto sixième sextus settimo séptimo septimus '
               'ottavo octavo octavus nono noveno nonus decimo décimo decimus eighth seventh sixth fifth fourth third')

EDITION_PREFIX = {   # one-letter abbreviations of titles, without a period, in these editions only
    'kaiserin_eleonora_magdalena': _set('b f h p g m v fr br bede beide beiden beden'),
    'hainhofer_reisen': _set('e f g k fr jre'),
    'reed': _set('l lo ld'),
}
EDITION_PREFIX_DIGNITY = {'reed': _set('l lo ld')}   # reed: L. = Lord (a dignity: « L of essex » -> LOC essex)
NAME_ONLY = {'koenigsfelden'}   # cut at the first lower-case word that is not a particle
DROPCAP = {'elan_crenum'}

PUNCT = {'.', ',', ';', ':', '/', '(', ')', '[', ']', '-', '–', '—', '·', '₎', '₍', '✳', '|', '¡', '!', '?', '¿', '>', '<'}
SEP = {',', ';', '/'}            # a separator cuts the name (not ':' which marks abbreviations, not brackets)
HONORIFIC = re.compile(r'issim(us|a|um|i|o|e|ae|is|os|am|orum|arum)$')
LORDSHIP = _set('señor senor sr sieur seigneur herr her lord signore signor dominus')


def words(r, s, e):
    """Token spans of the words in [s, e): contiguous tokens form one word; an apostrophe token closes a word."""
    T, S = r['tokens'], r['token_starts']
    out = []
    for k in range(s, e):
        joined = out and ((S[k] == S[k - 1] + len(T[k - 1]) and T[k - 1] not in ("'", '’') and T[k] not in PUNCT and T[k - 1] not in PUNCT)
                          or T[k - 1] == '¬')          # « Uͦl¬ rich »: a word broken at the line end
        if joined:
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1])
    return [tuple(w) for w in out]


def wtext(r, w):
    return ''.join(r['tokens'][w[0]:w[1]])


def wfold(r, w):
    return fold(wtext(r, w))


def is_punct(r, w):
    return all(t in PUNCT for t in r['tokens'][w[0]:w[1]])


def trim_period(r, sp):
    s, e = sp
    while e - s > 1 and r['tokens'][e - 1] == '.':
        e -= 1
    return s, e


def is_people(f: str, latin: bool = True) -> bool:
    """People, demonym, place adjective or language (decision 2). Latin -ensis is never a people word (kept LOC)."""
    if not f or ENSIS.search(f):
        return False
    return f in PEOPLE or (latin and bool(LATIN_ADJ.match(f))) or (len(f) >= 7 and bool(GERMAN_ADJ.search(f)) and f not in {'fischen'})


def map_loc(r, sp, log: Counter, edition: str):
    """-> mapped LOC span or None (dropped)."""
    s, e = trim_period(r, sp)
    if (s, e) != tuple(sp):
        log['period'] += 1
    ws = [w for w in words(r, s, e) if not is_punct(r, w)]
    content = [w for w in ws if wfold(r, w) not in ARTICLES]
    if len(content) == 1 and is_people(wfold(r, content[0])):
        log['people_loc'] += 1
        return None
    if content and all(wfold(r, w) in GENERIC_LOC for w in content):
        log['generic_loc'] += 1
        return None
    return (s, e)


def _norm_dropcap(r, ws):
    """elan_crenum: join a detached one-letter word with the next lower-case word (« d ieu », « c harles »)."""
    out, k = [], 0
    while k < len(ws):
        w = ws[k]
        if k + 1 < len(ws) and len(wtext(r, w)) == 1 and wtext(r, w).isalpha() and wtext(r, ws[k + 1])[:1].islower():
            out.append((w[0], ws[k + 1][1]))
            k += 2
            continue
        out.append(w)
        k += 1
    return out


def _wf(r, w):
    if isinstance(w, tuple) and len(w) == 2:
        return fold(''.join(r['tokens'][w[0]:w[1]]).replace(' ', ''))
    return ''


def _cap(t: str) -> bool:
    return t[:1].isupper()


def map_per(r, sp, log: Counter, edition: str):
    """-> (PER span or None, [LOC spans to add])."""
    s, e = trim_period(r, sp)
    if (s, e) != tuple(sp):
        log['period'] += 1
    ws = words(r, s, e)
    if edition in DROPCAP:
        n0 = len(ws)
        ws = _norm_dropcap(r, ws)
        if len(ws) != n0:
            log['dropcap'] += 1
    prefix = EDITION_PREFIX.get(edition, frozenset())
    prefix_dignity = EDITION_PREFIX_DIGNITY.get(edition, frozenset())
    T = r['tokens']

    def f(w):
        return _wf(r, w)

    def txt(w):
        return ''.join(T[w[0]:w[1]])

    def is_sep(w):
        return is_punct(r, w) and T[w[0]] in SEP

    def is_dignity(w):
        return f(w) in TITLE_DIGNITY or f(w) in prefix_dignity

    def is_title(w, extra=True):
        x = f(w)
        return (x in TITLE_DIGNITY or x in TITLE_ADDRESS or x in KIN or x in POSSESSIVE or bool(HONORIFIC.search(x))
                or (extra and x in prefix))

    def lowerish(w, ref):
        """Lower case, or upper case inside an all-capitals name (case says nothing there)."""
        t = txt(w)
        return t[:1].islower() or (len(t) > 1 and t.isupper() and all(txt(x).isupper() for x in ref if not is_punct(r, x)))

    def is_tail(w, nxt, ref):
        """A following dignity, kinship word or possessive cuts the name only when it is lower case or followed by a preposition
        (« Duke of Anjou »): a capitalised « König », « Pope », « Kaiser » after a first name is a surname."""
        x = f(w)
        if not (x in TITLE_DIGNITY or x in KIN or x in POSSESSIVE):
            return False
        return lowerish(w, ref) or (nxt is not None and f(nxt) in PREPS)

    def place_after(ws2):
        """Capitalised words after a leading preposition (and article) -> LOC token span, else None."""
        j, art = 0, False
        while j < len(ws2) and (f(ws2[j]) in PREPS or f(ws2[j]) in ARTICLES or txt(ws2[j]) in ("'", '’')):
            art |= f(ws2[j]) in ARTICLES
            j += 1
        run = []
        for w in ws2[j:]:
            if (is_punct(r, w) or is_dignity(w) or f(w) in KIN or f(w) in CONJ
                    or (run and not _cap(txt(w)) and f(w) not in PARTICLES and _cap(txt(run[0])))):
                break
            run.append(w)
        while run and f(run[-1]) in PARTICLES:
            run.pop()
        if not run:
            return None
        if not _cap(txt(run[0])):                   # « abbot of thorney » is a place; « príncipe de los poetas », « zu diser zitt » are not
            rest_words = [w for w in ws2[j:] if not is_punct(r, w)]
            if art or len(run) > 1 or not txt(run[0]).isalpha() or rest_words[-1] != run[0]:
                return None
        return (run[0][0], run[-1][1])

    # 1. leading titles, address, kinship, possessives, articles, punctuation (decision 6)
    k, dignity, stripped = 0, False, False
    while k < len(ws):
        w = ws[k]
        if is_punct(r, w):
            k += 1
            continue
        if not is_title(w):
            break
        nxt = ws[k + 1] if k + 1 < len(ws) else None
        if _cap(txt(w)) and nxt is not None and is_sep(nxt) and not stripped:
            break                                   # « Regina , geborne ... »: a capitalised word before a comma is the name
        if f(w) == 'regina' and r.get('lang') != 'la' and nxt is not None and _cap(txt(nxt)) and not is_title(nxt):
            break                                   # « Regina Elisabetha »: a first name outside Latin
        stripped = True
        if f(w) in KIN and nxt is not None and f(nxt) in PREPS and k + 2 < len(ws) and _cap(txt(ws[k + 2])):
            k += 1                                  # « wife of Iohn Burges », « la mère de Luther »: the preposition goes with the kin word
        dignity |= is_dignity(w)
        k += 1
    if edition in NAME_ONLY:                        # koenigsfelden: honorific adjectives before the name (lieber, getrùwer, wysen)
        while k < len(ws) and f(ws[k]) not in PREPS and (is_punct(r, ws[k]) or (not _cap(txt(ws[k])) and f(ws[k]) not in PARTICLES)
                                                        or is_title(ws[k])):
            stripped = stripped or not is_punct(r, ws[k])
            dignity |= is_dignity(ws[k])
            k += 1
    rest = ws[k:]
    if stripped:
        log['title_strip'] += 1
    adds = []
    if not rest or all(is_punct(r, w) for w in rest):
        log['title_only'] += 1
        return None, adds
    # 2. dignity + preposition + place: the title designates the person (decision 3)
    if dignity and f(rest[0]) in PREPS:
        loc = place_after(rest)
        log['title_place'] += 1
        if loc:
            adds.append(loc)
        else:
            log['title_place_no_loc'] += 1
        return None, adds
    # 3. name part: cut at a separator, a following title / kinship word, or (name-only editions) office and domicile
    name, tail = [], []
    for j, w in enumerate(rest):
        nxt = rest[j + 1] if j + 1 < len(rest) else None
        if name and (is_sep(w) or is_tail(w, nxt, rest)):
            tail = rest[j:]
            break
        if name and edition in NAME_ONLY and not is_punct(r, w) and (
                (not _cap(txt(w)) and f(w) not in PARTICLES) or f(w) in _set('zu ze zuo')):
            tail = rest[j:]
            break
        name.append(w)
    if tail:
        while name and (is_sep(name[-1]) or (edition in NAME_ONLY and f(name[-1]) in PARTICLES)):
            name.pop()
    if not name or all(is_punct(r, w) for w in name):
        log['title_only'] += 1
        return None, adds
    if tail:
        log['tail_cut'] += 1
        # a place in a title tail (« Duke of Anjou », « señor de Polvoranca », « schulths ze Bremgarten ») or a Latin -ensis adjective
        has_title = edition in NAME_ONLY or any(is_dignity(w) or f(w) in LORDSHIP for w in tail)
        for j, w in enumerate(tail):
            if has_title and f(w) in PREPS and j + 1 < len(tail):
                loc = place_after(tail[j:])
                if loc:
                    adds.append(loc)
                    log['tail_place'] += 1
                break
            if ENSIS.search(f(w)) and _cap(txt(w)):
                adds.append((w[0], w[1]))
                log['tail_place'] += 1
                break
    # 4. sovereign + numeral + preposition + place (decision 4)
    for j in range(1, len(name) - 1):
        x = txt(name[j])
        if ((ROMAN.match(x) and x not in {'L', 'C', 'D', 'M'}) or f(name[j]) in ORDINAL) and f(name[j + 1]) in PREPS:
            loc = place_after(name[j + 1:])
            if loc:
                adds.append(loc)
                log['regnal'] += 1
                name = name[:j + 1]
            break
    content = [w for w in name if not is_punct(r, w)]
    folded = [f(w) for w in content]
    # 5. God, personifications (decision 5); peoples (decision 2); numbers alone
    if any(x in GOD for x in folded) and all(x in GOD or x in TITLE_ADDRESS or x in PARTICLES for x in folded):
        log['god'] += 1
        return None, adds
    if len(folded) == 1 and folded[0] in PERSONIF:
        log['personif'] += 1
        return None, adds
    if all(is_people(x, latin=bool(stripped or tail)) for x in folded):
        log['people_per'] += 1
        return None, adds
    if all(x.isdigit() for x in folded):
        log['title_only'] += 1
        return None, adds
    # 6. a lone Latin -ensis adjective left after a dignity: the title designates the person (« Monasteriensem antistitem »)
    if len(content) == 1 and ENSIS.search(folded[0]) and (dignity or tail) and _cap(txt(content[0])):
        adds.append((content[0][0], content[0][1]))
        log['title_place'] += 1
        return None, adds
    return (name[0][0], name[-1][1]), adds


def owner_map(r, edition: str, loc_gold: set | None, per_gold: set | None, v2_gold: set | None, add_loc: bool):
    """Map one passage's gold sets. Returns (loc, per, v2, log, changes); changes = [(label, before, after, mappings)]."""
    log = Counter()
    changes = []
    loc_out, per_out, v2_out = None, None, None
    new_locs = []
    if per_gold is not None:
        per_out = set()
        for sp in sorted(per_gold):
            c = Counter()
            m, adds = map_per(r, sp, c, edition)
            if m:
                per_out.add(m)
            new_locs += adds
            if m != tuple(sp) or adds:
                changes.append(('PER', tuple(sp), m, dict(c), adds))
            log.update(c)
    if loc_gold is not None:
        loc_out = set()
        for sp in sorted(loc_gold):
            c = Counter()
            m = map_loc(r, sp, c, edition)
            if m:
                loc_out.add(m)
            if m != tuple(sp):
                changes.append(('LOC', tuple(sp), m, dict(c), []))
            log.update(c)
        if add_loc:
            for a in new_locs:
                if not any(a[0] < b[1] and b[0] < a[1] for b in loc_out):
                    loc_out.add(a)
                    log['loc_added'] += 1
    if v2_gold is not None:
        v2_out = set()
        for sp in sorted(v2_gold):
            c = Counter()
            m = map_loc(r, sp, c, edition)
            if m:
                v2_out.add(m)
            log.update({f'v2_{k}': n for k, n in c.items()})
    return loc_out, per_out, v2_out, log, changes
