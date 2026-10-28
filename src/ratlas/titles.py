"""Places that occur inside a title (owner decision 1 of 2026-10-28): the detector and the lists of mentions it removed.

A place mention is "inside a title" when the context shows it as the domain of a title held by a person:
King of France, Roy d'Escosse, Duke Robert of Normandy, my Lord of Salisbury, rex Franciae, Herzog zu Sachsen, il re di Francia, ...
Three categories are kept apart: ``royal`` (kings, queens, emperors, sultans, tsars, khans), ``noble`` (dukes, counts, earls, princes,
marquises, barons, lords, electors, landgraves, doges ...) and ``ecclesiastical`` (bishops, archbishops, abbots, cardinals, patriarchs).
"The kingdom of France", "in the duchy of Milan", "went to York" are not titles: the word before the place must be the title itself
(kingdom, duchy, bishopric, Königreich ... do not match: the match is on whole words).

Data (``data/curation/``, read by the page builder ``page_kit/build_v29.py`` and by ``scripts/ner2_layer_v3_build.py``):
  * ``title_rules.json``                -- the lexicon and the structures of this module, as documentation (``rules_json()``);
  * ``title_excluded_editor.parquet``   -- the editor mentions the detector removed, keyed by ``row`` = position (0-based) of the line in
                                           ``RQ/derived/atlas/v2.4/mentions.parquet``; ``doc_id`` and ``surface`` are kept to check the key;
  * ``title_excluded_model.tsv.gz``     -- the model mentions removed, keyed by (src, doc_id, lang, key) = a line of the layer's ``doc_forms``
                                           (``src`` ml = v1 run, mles = Spanish rerun, ml3 = sets of version 3), with the counts per category.
Ecclesiastical titles (bishop of, evesque de, episcopus X) are a category of their own: ``load_*_exclusions(ecclesiastical=False)`` gives
the lists without them, which puts those places back (the one flag ``ECCLESIASTICAL_REMOVED`` is the default of both loaders and of the builders).

Public API
    detect(left, surface, right, lang=None) -> None | dict(cat=..., title=..., rule=...)
    ``left`` / ``right`` are the text before / after the surface (the editor contexts are 120 characters wide, the model contexts 110 + 90).
    rules_json() -> the lexicon and the patterns as a JSON-able dict (written to rules.json).

How it decides (all on a diacritic-folded copy of the text, long s and combining marks removed, case kept for the name tokens)
  1. DIRECT   : the left text ends with  TITLE [name tokens] [,] PARTICLE [the] [Great|New|...]  and the surface follows.
                Name tokens are given names ("Duke Robert of Normandy"); they are allowed only for the strong titles and never when they
                are office words (Lord Mayor of London, Lord Chancellor of England stay places).
  2. SURFACE  : the surface itself starts with the title ("Roy de France", "Duke of Anjou" tagged as a place).
  3. CHAIN    : the place is one of a list that follows a title ("King of England, France and Ireland": France, Ireland).
  4. LATIN    : no particle in Latin: "rex Franciae", "ducem Burgundiae" (genitive endings only), also the title after the place
                ("Franciae regem") and the German genitive before the title ("Frankreichs Konig").
"""
from __future__ import annotations

import re
import unicodedata

ROYAL, NOBLE, ECCL = 'royal', 'noble', 'ecclesiastical'

# --------------------------------------------------------------------------------------------------------------------
# lexicon: category -> language group -> title words (diacritics folded, lower case, regular-expression fragments)
# 'strong' titles accept given names between the title and the particle; 'weak' ones do not (sieur de, seigneur de, Lord of ...)
# --------------------------------------------------------------------------------------------------------------------
STRONG = {
    ROYAL: {
        'en': r"kings?|kinges?|kyngs?|kynges?|kynge?s|kyng|queens?|queen|quee?nes?|quenes?|quenys?|emperou?rs?|emperours?|emperoure|emperowr|emperowrs|empress(?:es)?|emperesse|tsars?|czars?|tzars?|czar|sultans?|soltans?|soldans?|sultane|sultaness|soldan|shah|sophi|sophy|khan|cham|mogul|mogor|negus|kaiser",
        'fr': r"rois?|roys?|royne|roynes|reines?|reynes?|empereurs?|emperiers?|imperatrices?|empereur|sultans?|soldans?|souldans?|roi|tsar|czar|grand[ -]seigneur|grand[ -]sultan",
        'la': r"rex|rege|regis|regi|regem|reges|regum|regibus|regina|reginae|reginam|reginarum|imperator|imperatore|imperatoris|imperatori|imperatorem|imperatores|imperatrix|imperatricis|sultanus|sultani|soldanus|soldani|czar|tzar|magnus[ ]dominus",
        'it': r"re|regina|imperatore|imperadore|imperatrice|imperador|sultano|soldano|gran[ ]signore|gran[ ]sultano",
        'es': r"reyes|rey|reina|reynas?|reyna|rei|reis|rainha|emperador|emperatriz|imperador|sultan|soldan|gran[ ]senor",
        'de': r"konigs?|konige|konigen|konigin|konigk|kunigs?|kunige|kunigen|kunigin|kuning|kayser|kaiser|keyser|keiser|kaiserin|kayserin|keyserin|sultan|soldan|zar|zaar|konig",
        'nl': r"konings?|koninginne|koningin|coninck|conincks|coninc|coninghen|coningh|koninck|keizer|keyser|keijser|sultan|soudan|czaar|zaar",
    },
    NOBLE: {
        'en': r"dukes?|duke|duchess(?:es)?|dutchess|earls?|earle|earles|erle|erles|counts?|countes|countess(?:es)?|marquess|marquesse|marquis|marques|marquise|viscounts?|vicounts?|barons?|baroness|princes?|princesse?s?|prynce|dolphin|dauphins?|archdukes?|archduke|archdutchess|landgraves?|margraves?|marquesses|electors?|palsgraves?|palsgrave|doges?|grand[ -]dukes?|great[ -]dukes?|grand[ -]duke|infante|tetrarch",
        'fr': r"ducs?|duchesses?|comtes?|contes?|comtesses?|marquis|marquises?|vicomtes?|vidames?|barons?|baronnes?|princes?|princesses?|dauphins?|dauphines?|archiducs?|archiduchesses?|grands?[ -]ducs?|electeurs?|landgraves?|margraves?|palatins?|infantes?|doges?",
        'la': r"dux|ducis|duci|ducem|duces|ducum|ducibus|ducissa|ducissae|comes|comite|comitis|comiti|comitem|comites|comitum|comitibus|comitissa|marchio|marchioni|marchionis|marchionem|marchiones|marchisius|marchesius|princeps|principe|principis|principi|principem|principes|principum|archidux|archiducis|archiducem|vicecomes|vicecomitis|baro|baroni|baronis|baronem|barones|elector|electori|electoris|electorem|electores|landgravius|landgravii|landgravio|lantgravius|lantgravii|lantgravio|lantgravium|landgravium|dynasta|dynastae|toparcha|doge|marchisio",
        'it': r"duchi|duca|duchessa|conti|conte|contessa|marchesi|marchese|marchesa|principe|principi|principessa|barone|baroni|visconte|arciduca|granduca|gran[ ]duca|elettore|landgravio|margravio|doge|dogi",
        'es': r"duques?|duquesa|condes?|condesa|marques|marquesa|marquese|principes?|principe|princesa|barones?|baron|vizcondes?|archiduques?|archiduque|elector|infante|doge",
        'de': r"herzog|herzogs|herzoge|herzogen|herzogin|hertzog|hertzogs|hertzoge|hertzogen|hertzogin|hertzogk|hertzoch|herzoch|hertog|hertoch|graf|graff|graffs|grafen|grafe|grafin|graffin|grave|graven|graffe|graffen|grafs|markgraf|markgraff|markgrafen|marggraf|marggraff|marggrafen|marggraffen|margraf|margraff|margraffen|marckgraf|marckgraff|marckgraffen|landgraf|landgrafen|landgraff|landgraffen|lantgraf|lantgrafen|lantgraff|lantgraffen|landtgraf|landtgraff|landtgrafen|landtgraffen|landtgrave|landtgraven|landtgrav|landtgraue|landtgrauen|landgraue|landgrauen|lantgraue|lantgrauen|pfaltzgraue|pfaltzgrauen|pfalzgraue|pfalzgrauen|marggraue|marggrauen|margraue|margrauen|burggraue|burggrauen|rheingraue|wildgraue|graue|grauen|lantgrave|lantgraven|landgrave|landgraven|pfalzgraf|pfalzgrafen|pfaltzgraf|pfaltzgrafen|pfalzgraff|burggraf|burggrafen|rheingraf|wildgraf|rauhgraf|furst|fursten|furstin|furste|fuerst|kurfurst|kurfursten|churfurst|churfursten|churfurste|churfurstin|kurfurstin|kurfurste|chuerfurst|erzherzog|erzherzogen|ertzherzog|ertzhertzog|erzhertzog|erzhertzogen|freiherr|freyherr|freiherrn|freyherrn|reichsgraf|reichsgrafen|dogen|doge|prinz|prinzen|printz|princen",
        'nl': r"hertog|hertogh|hertoge|hertogen|hertoginne|hertoginnen|graaf|grave|graven|grafe|graeff|graef|markgraaf|marcgrave|prins|prince|princen|vorst|vorsten|vorstin|aartshertog|aertshertoge|keurvorst|keurvorsten|baron|burggraaf|doge",
    },
    ECCL: {
        'en': r"bishops?|bishop|bishopps?|bishoppes?|byshops?|byshopps?|bysshops?|bysshopp|bysshoppe|bysshoppes|busshops?|bisshops?|bishoppe|bishope|bpp?|archbishops?|archbyshops?|archbysshops?|archbisshops?|archebishops?|archebysshops?|archebysshoppe|archebyshops?|archebisshops?|archebishoppe|archbishoppe|archbishopp|archbishoppe|archb|archbp|abbots?|abbat|abbate|abbess(?:es)?|cardinals?|cardinall|patriarchs?|patriarches?|prelate",
        'fr': r"eveques?|evesques?|evesque|archeveques?|archevesques?|archevesque|abbes?|abbesses?|cardinals?|cardinal|patriarches?|prelats?",
        'la': r"episcopus|episcopi|episcopo|episcopum|episcopos|episcoporum|episcopis|archiepiscopus|archiepiscopi|archiepiscopo|archiepiscopum|archiepiscopos|archiepiscoporum|abbas|abbate|abbatis|abbati|abbatem|abbates|abbatum|abbatissa|cardinalis|cardinalem|cardinali|patriarcha|patriarchae|patriarcham|praesul|praesulis|presul|antistes|antistitis|pontifex[ ]maximus",
        'it': r"vescovi|vescovo|arcivescovi|arcivescovo|abate|abati|abbate|abbadessa|cardinale|cardinali|patriarca|patriarchi|prelato",
        'es': r"obispos?|arzobispos?|abades?|abadesa|abad|cardenal|cardenales|patriarca|patriarcas|prelado",
        'de': r"bischofs?|bischof|bischofe|bischofen|bischoff|bischoffs|bischoffe|bischoffen|bisschoff|bischove|bischoven|bisthof|erzbischof|erzbischofs|erzbischofe|erzbischofen|ertzbischof|ertzbischoff|ertzbischoffs|erzbischoff|erzbischoffs|abt|abte|abtes|abts|abbt|abbte|apt|abtissin|abbtissin|kardinal|kardinals|kardinale|cardinal|cardinals|cardinale|cardinall|patriarch|patriarchen",
        'nl': r"bisschop|bisschoppen|bisshop|bischop|bischoppen|biscop|biscoppen|aartsbisschop|aertsbisschop|aertsbischop|abt|abdt|abt|kardinaal|cardinael|cardinaal|patriarch",
    },
}
# weak titles: only DIRECT (title + particle + place), no given name in between
WEAK = {
    NOBLE: {
        'en': r"lords?|lorde|lordes",
        'fr': r"seigneurs?|seigneur|sieurs?|sires?|sire|monseigneur|messeigneurs|messire",
        'it': r"signori|signore|signor|sig",
        'es': r"senor|senores|sennor|sennores",
        'de': r"herr|herrn|herren|herrens|herrschaft",
        'nl': r"heer|heeren|heere|here",
    },
}
# weak words that are only a title with the 'of' structure and never alone for German/Dutch/Italian: restricted below by the particle list
WEAK_PARTICLES_STRICT = {'de': ('von', 'zu', 'zue', 'zuo', 'zum', 'zur'), 'nl': ('van',), 'it': ('di', 'de', 'del', 'della', 'd'), 'es': ('de',)}
# German "Herr(en) von X" is the lord of X, but also "die Herren von Bern" (the council): weak titles are used for German only in the singular
# with the nobiliary particle after a name-less title: kept off by default (see DROP_WEAK)
DROP_WEAK = {('de', 'herr'), ('de', 'herrn'), ('de', 'herren'), ('de', 'herrens'), ('de', 'herrschaft'), ('nl', 'heer'), ('nl', 'heeren'), ('nl', 'heere'), ('nl', 'here'),
             ('it', 'sig'), ('it', 'signori'), ('es', 'senores'), ('es', 'sennores')}

# words that make a title+name+particle an OFFICE rather than a title (Lord Mayor of London, Lord Chancellor of England)
OFFICE_WORDS = (r"mayor|maior|maire|chancellor|chancellier|chancelier|chauncellor|cancellarius|admiral|amiral|treasurer|tresorier|deputy|deputie|lieutenant|lieutenaunt|justice|"
                r"chamberlain|chamberlayn|steward|warden|keeper|protector|president|governor|gouverneur|constable|connestable|marshal|marshall|marshale|mareschal|marechal|marescal|high|privy|secretary|"
                r"ambassador|embassador|ambassadeur|legate|legat|captain|capitaine|general|proctor|bailiff|sheriff|shrive|provost|prevost|archdeacon|dean|doyen|"
                r"judge|master|maister|bailly|baillif|seneschal|senechal|viceroy|vice|vicar|vicaire|chief|great[ ]seal|councillor|counsellor|conseiller|council|"
                r"principaute|principality|duchy|duche|kingdom|royaume|empire|county|comte|bishopric|archbishopric|hith|hithe|street|lane|bridge|wharf|stairs|inn|house|hall|college|court|gate|market|church|chappel|chapel|head|arms|bench|strand|cross|row|alley|yard|"
                r"palatine[ ]of|prior|prieur|treasorer|auditor|surveyor|receiver|clerk|speaker|recorder|gentleman|esquire|knight|chevalier|ritter|vogt|amtmann|richter|"
                r"landvogt|statthalter|stadthalter|hauptmann|rat|rath|burgermeister|schultheiss|schultheis|pfleger|canzler|kanzler|marschall|hofmeister|sendbote|"
                r"gubernator|gubernatoren|gubernatore|governatore|gobernador|gouvernor|oberst|obrist|rittmeister|oberster|feldmarschall|generalissimo|generalleutnant|commissarius|commissary|kommissar|"
                r"gesandter|gesandte|legatus|orator|oratore|residente|resident|nuncio|nuntius|nunzio|sekretar|secretarius|secretaire|secretario")
PARTICLES = r"of|de|di|d['’]|du|des|del|della|dei|degli|dello|do|da|von|vonn|vom|van|zu|zum|zur|zue|zuo|in|inn|bei|bey"
LOCATIVE_DE = {'zu', 'zum', 'zur', 'zue', 'zuo', 'in', 'inn', 'bei', 'bey'}
LOCATIVE_PARTICLES = {'in', 'inn', 'bei', 'bey'}          # "Konig in Frankreich" only after German / Dutch titles
# names in between: capitalised words, Roman numerals, small connectors
NAME_CONNECT = r"the|der|le|la|il|el|ye|don|dom|dona|sant|san|saint"

# words that, directly before the title word, show a different thing ("vice-roy", "Our Lady of")
NOTRE_DAME = re.compile(r"(?:notre|nostre|our|unser|vnser|nostra|nuestra|onze|unsere|vnsere|ons)[ ]+(?:dame|lady|frau|frawen|fraw|donna|senora|vrouw|vrouwe|liebe[nr]?[ ]+fra\w*)\b", re.I)

DOMAIN_NOUN = re.compile(r"(?i)(?<![\w])(?:ducatus|ducatum|ducatu|regnum|regni|regno|comitatus|comitatum|principatus|principatum|marchionatus|episcopatus|episcopatum|dioecesis|diocesis|provincia|provinciae|provinciam|exercitus|exercitum|legatus|legatum|status|concilii|concilium)\s+$")
REALM = re.compile(r"(?i)^(?:francia|franciae|francorum|francor|gall|angl|brit|scot|hibern|hisp|pol|hungar|bohem|boem|dan|suec|sueo|norveg|portug|lusit|navarr|sicil|neapol|german|roman|burgund|sabaud|lothar|transylvan|moldav|valach|turc|pers|aegypt|ethiop|maur|tartar|moscov|russ|livon|prus|borus|saxon|bavar|palatin|helvet|ital|sardin|cypr|hierosol|jerusalem|iud|babylon|assyr|syr|macedon|thrac|graec|arag|castell|leon|asturi|granat|cordub|toledo|valent|aquitan|austri|styri|carinth|carniol|silesi|lusati|mor?av|holsat|slesvic|norweg|gothor|vandal|vend|batav|belg|flandr|brabant|holland|zeland|frisi|gelr|cliv|iuliac|wirt|vurt|hass|brandeburg|brandenburg|misni|thuring|pomer|mecklenb|luneb|brunsv|holstein|transilv|illyr|dalmat|croat|sclavon|bosn|serv|bulgar|athen|sparta|lacedaem|troi|egypt|libya|persi|indi|cathai|sin|japon|lituan|poloni|samogit|volhyn|ukrain|cosac|tauric|crim|georgi|armeni|trapezunt|constantinopol|florent|mediolan|ferrar|urbin|parm|mantu|mutin|venet|genu|luc|senens|bononi|piomb|massa|saluc|monferr|montisferr|cleric|somers|dorcest|norfolc|suffolc|northumb|nortumb|somerset|summerset|warwic|bedford|hertford|southampt|pembroch|pembrok|derbi|essex|arundel|salisbur|lincoln|cantiae|devon|cornub|wall(?:ia|iae)|ebor|lancast|cestri|westmor|cumbr|huntington|montgomer|shrewsb|worcestr|exon|oxon|cleveland|rutland|kanc|kent|sussex|surr|dunelm|wint|glocest|ormond|desmon|kildar|clanric|tirconn|ulton|hiberni|anhalt|oldenb|mansf|nassov|neoburg|bipont|megapol|mecklenb|jul(?:iae|ia|iacens)|marchiae|ravens|tirol|habsb|hapsb|veldent|sponh|moers|morsae|carinth|carnioli|stiri|styri|austriae|wurt|wirt|teck|mumpel|montbel|mompel|sultz|furstenb|hohenl|hanau|hanov|waldeck|lipp|schaumb|stolb|schwarzb|schwarzenb|bentheim|diepholt|hoya|ostfris|embd|ostfrisi|jever|kniph|lauenb|saxon|wittenb|thuring|misni|lusat|silesi|opol|ratibor|teschin|glogov|sagan|lign|brieg|oels|olsn|munsterb|brzeg|zator|oswiec|poznan|kalis|plocen|masov|cuiav|sandomir|cracov|vilnens|trocens|kiov|smolens|volhyn|podol|belz|chelm|halicz|lemburg|rutheni|russi|moscov|lituan|livon|curland|semigall|estoni|ingri|carel|finland|norveg|island|groenland|gothland|scani|hallandi|blekin|oelandi|ferroe|oreadi|hebrid|monae|manniae|insularum)")
ADJ_SUFFIX = re.compile(r"(?i)(?:ensis|ensium|ensem|ensi|anus|ani|anae|ana|ano|inus|inae|ini|ino|ina|icus|ici|icae|orum|arum|onus|ono|onum)$")
NOM_TITLES = re.compile(r"(?i)^(?:rex|dux|comes|princeps|imperator|marchio|baro|elector|episcopus|archiepiscopus|abbas|cardinalis|patriarcha|praesul|presul|regina|imperatrix|archidux|vicecomes|landgravius|ducissa|comitissa|sultanus|soldanus)$")
REALM_STRICT = re.compile(REALM.pattern.replace('|florent|mediolan|ferrar|urbin|parm|mantu|mutin|venet|genu|luc|senens|bononi|piomb|massa|saluc|monferr|montisferr|cleric', ''), re.I)
REALM = re.compile(REALM.pattern.replace('v', '[vu]'), re.I)          # all-caps inscriptions are folded V -> U (BAVARIAE -> BAUARIAE)
REALM_STRICT = re.compile(REALM_STRICT.pattern.replace('v', '[vu]'), re.I)
EMPEROR_REALM = re.compile(r"(?i)^(?:rom|roem|rome|romain|german|teutsch|deutsch|teutschland|deutschland|allemagne|almain|almaigne|alemanha|oester|oster|austria|autriche|constantinopel|konstantinopel|constantinople|constantinopl|trapezunt|trebizond|griech|grec|grece|greece|aethiop|ethiop|persi|perse|pers|russ|rus|moscau|moskau|moscow|muscov|moscov|marok|maroc|morocco|indi|inde|turk|turq|tyrk|tuerk|turcor|china|chin|japan|mexic|peru|byzant|ostreich|osterreich|oesterreich|oesterr|hispan|spani|espagn|ungar|hungar|bohem|boem|behaim|behem|polen|pollen|franck|france|frankreich|engel|angl|engl|schwed|daenemark|danemark|dennemark|denmark|dani|moldau|walach|siebenbur|transylv|jerusalem|babylon|assyri|syri|egypt|aegypt|armeni|tartar|cathay|abyss|habesch|ethiopi|mali|monomotapa|tunis|algier|fes|marokk|sicil|neapol|napol|ital|lombard|burgund|brasil|portug)")
EMPEROR = re.compile(r'(?i)^(?:kaiser|kayser|keyser|keiser|keijser|keizer|emperou?r|empereur|empress|emperesse)\w*$')
# Latin / German reverse
LA_GEN = re.compile(r"(?:ae|ii|i|orum|arum|iae|ensis|ensium|ensem|ensi)$")
LA_GEN_REV = re.compile(r"(?:ae|orum|arum|iae|ensis|ensium|ensi|ensem)$")
LA_ADJ_ECC = re.compile(r"(?:ensem|ensi|anus|ano|anum|ani|inus|ino|inum|ini)$")           # "episcopo Augustano", "archiepiscopum Eboracensem" (ecclesiastical titles only)
LA_CONS = re.compile(r"(?:[bcdfghklnprtwxz]|ck|ch|gh|rg|ng)'?$")           # "Comiti Marlborough": a non-Latin place name after a Latin title
LA_ADJ_ONLY = re.compile(r"(?:ici|icae|icorum)$")          # Italici (exercitus), Romani ...: adjectives, not a domain
DE_ADJ = re.compile(r"(?i)(?:licher|lichen|lichem|liche|isch|ischer|ischen|ischem|ische|isches)$")
NATIONALITY_BEFORE = re.compile(r"(?i)(?:spann|spaen|hispan|franz|frantz|engl|engel|schwed|dan|d\u00e4n|pol|ungar|hungar|beh|b\u00f6hm|bohm|welsch|ital|saechs|s\u00e4chs|sachs|bay|t\u00fcrk|turk|luttr|lothr)\w*isch\w*$")
DUX_FORMS = re.compile(r"(?i)^(?:dux|duce|ducis|duci|ducem|duces|ducum|ducibus)$")
PEOPLE_GEN = re.compile(r"(?i)^(?:gall|hispan|ital|german|angl|scot|pol|turc|pers|dan|suec|helvet|bohem|hungar|batav|belg|francor|lusitan|moscovit|russ)\w*(?:orum|arum)$")
# --------------------------------------------------------------------------------------------------------------------


def fold(s: str) -> str:
    """Fold long s, soft hyphen, combining marks and the usual early-modern letters; keep the case; length may change, so only used on one side of the place."""
    s = re.sub(r'(?i)\b(com|con|vicom|viscon|visconm)t[\u00e9\u00e8]\b', lambda m: m.group(0)[:-1] + 'ey', s)       # comté = county, not 'comte' = count
    s = s.replace('ſ', 's').replace('­', '').replace(' ', ' ')
    s = unicodedata.normalize('NFKD', s)
    s = s.replace('e\u0328', 'ae').replace('E\u0328', 'Ae')          # e-ogonek stands for ae in early Latin
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('ß', 'ss').replace('æ', 'ae').replace('Æ', 'Ae').replace('œ', 'oe').replace('Œ', 'Oe').replace('ʒ', 'z').replace('ů', 'u').replace('ͤ', '')
    s = re.sub(r"(?i)\b(chur|kur|erz|ertz|land|lant|pfaltz|pfalz|marck|mark|marg|burg|rhein|reichs|erb|fur|fuer|vice|ober|nider|nieder)[-\u2010]\s?(?=[a-z])", r'\1', s)       # Chur-Furst, Land-graf, Pfaltz- graf
    s = re.sub(r'\b[A-Z]{3,}\b', lambda m: m.group(0).replace('V', 'U').replace('J', 'I'), s)       # inscriptions: DVX BVRGVNDIAE = dux Burgundiae
    s = re.sub(r'\s+', ' ', s)
    return s


def _tables():
    """list of (fragment, category, lang group, strong) for every title word (the fragments are regular expressions)."""
    t = []
    for cat, byl in STRONG.items():
        for lg, frag in byl.items():
            for w in frag.split('|'):
                t.append((w, cat, lg, True))
    for cat, byl in WEAK.items():
        for lg, frag in byl.items():
            for w in frag.split('|'):
                if (lg, w) not in DROP_WEAK:
                    t.append((w, cat, lg, False))
    return t


TITLE_LIST = _tables()
_FULL = [(re.compile(w + '$', re.I), c, lg, st) for w, c, lg, st in TITLE_LIST]
_ALT = '|'.join(sorted({w for w, c, lg, st in TITLE_LIST}, key=len, reverse=True))
_CAP = r"(?-i:[A-Z][\w'’\-\.]*)"
_ROMAN = r"(?-i:[IVXLC]{1,6}\.?)"
EPITHETS = r"catolico|catholico|catholique|catholic|catholicque|cristianissimo|christianissimo|christianissime|chrestien|crestien|christian|serenissimo|serenissime|serenissimus|invictissimo|potentissimo|puissant|mighty|dread|dreade|most|right|very|tres|treschrestien|gran|grand|great"
_NAMETOK = rf"(?:{_CAP}|{_ROMAN}|(?:{NAME_CONNECT})|(?i:palatine|palatinus|palatin|pfalz|{EPITHETS}))"
_ADJ = r"(?:all|great|new|old|grand|grande|gran|gross|grosse|upper|lower|high|low|haute|basse|haut|bas|saint|st\.?|san|santa|sainte|sankt|sant|ober|nieder|nider|neu|alt|klein|lesser|greater|east|west|north|south|nova|novae|magna|major|minor|inferior|superior)"
_PART_TAIL = (rf"\s*(?P<part>{PARTICLES})\s*(?P<art>(?:the|le|la|el|les|der|die|das|dem|den|het|l['’])\s*)?(?:{_ADJ}\s+){{0,2}}$")
# DIRECT: title [epithets/name tokens] [,] particle [article] [qualifiers] <place>
_DIRECT_AT = re.compile(
    rf"(?P<title>{_ALT})(?![\w])(?P<mid>(?:\s+(?:{_NAMETOK})){{0,4}})?" + _PART_TAIL, re.U | re.I)
SURFACE_HEAD = re.compile(rf"(?i)^(?:(?:the|le|la|les|el|il|der|die|das|des|dem|den)\s+)?(?:{_ALT})\b")
_TITLE_POS = re.compile(rf"(?<![\w\-\.])(?:{_ALT})(?![\w])", re.U | re.I)
# bare Latin: title [name tokens] <place in genitive>
_LA_ALT = '|'.join(sorted({w for w, c, lg, st in TITLE_LIST if lg == 'la'}, key=len, reverse=True))
_LATIN = re.compile(rf"(?i:(?<![\w\-])(?P<title>{_LA_ALT})(?![\w]))(?P<mid>(?:\s+(?i:palatinus|palatini|palatino|magnus|magni)){{0,2}})\s*(?:(?i:de)\s+)?$", re.U)
_OFFICE = re.compile(rf"(?i:(?<![\w\-])(?:{OFFICE_WORDS})(?![\w]))")
_SEP_END = re.compile(r"(?:\s*,\s*|\s*/\s*|\s+(?:and|et|ac|atque|und|vnd|vndt|vnnd|unnd|undt|ende|e|y|&|&amp;)\s+|\s*,\s*(?:and|et|ac|und|vnd|e|y|&)\s+)(?:(?:de|di|of|d['’])\s*)?$", re.I)
# the part after the place: a Latin title after one or two genitive names, or the German genitive then the title
_R_LATIN = re.compile(rf"^\s*(?:(?:et|ac|atque)\s+(?-i:[A-Z][\w]*(?:ae|orum|arum|ium|iae|ensis))\s+){{0,2}}(?i:(?P<title>{_LA_ALT}))(?![\w])", re.U)
_DE_ALT = '|'.join(sorted({w for w, c, lg, st in TITLE_LIST if lg == 'de' and st and not re.search(r'gra[uv]', w)}, key=len, reverse=True))      # 'grauen' = engraved, 'grave' = tomb in English texts
_R_DE = re.compile(rf"^\s*(?i:(?P<title>{_DE_ALT}))(?![\w])", re.U)
_PREFIX = re.compile(rf"^(?i:(?:{PARTICLES}))\s+")


def _category_of(word: str, lg_hint: str | None = None):
    """-> (category, lang group, strong) of a matched title word; when several groups know the word, the hint (or the first) wins."""
    w = fold(word).lower().replace('-', ' ')
    hits = [(c, lg, st) for rx, c, lg, st in _FULL if rx.match(w)]
    if not hits:
        return None
    if lg_hint:
        for h in hits:
            if h[1] == lg_hint:
                return h
    if len(hits) > 1:
        pass
    return hits[0]


def _direct(L: str, right: str = '', place: str | None = None):
    L = L[-110:]
    for mt in reversed(list(_TITLE_POS.finditer(L))):
        m = _DIRECT_AT.match(L, mt.start())
        if m:
            r = _direct_one(L, m, right, place)
            if r:
                return r
    return None


PART_LANGS = {'of': ('en',), 'de': ('fr', 'es', 'it', 'la', 'en'), "d'": ('fr', 'it'), 'du': ('fr',), 'des': ('fr',), 'di': ('it',), 'del': ('it', 'es'), 'della': ('it',),
              'dei': ('it',), 'degli': ('it',), 'dello': ('it',), 'do': ('es',), 'da': ('es',), 'von': ('de',), 'vonn': ('de',), 'vom': ('de',), 'van': ('nl',), 'zu': ('de',), 'zum': ('de',),
              'zur': ('de',), 'zue': ('de',), 'zuo': ('de',), 'in': ('de', 'nl'), 'inn': ('de',), 'bei': ('de',), 'bey': ('de',)}


def _info_for(title: str, part: str):
    """(category, lang group, strong) of a title word, choosing among the groups that know the word the one that the particle belongs to."""
    w = fold(title).lower().replace('-', ' ')
    hits = [(c, lg, st) for rx, c, lg, st in _FULL if rx.match(w)]
    if not hits:
        return None
    ok = PART_LANGS.get(part)
    if ok:
        for lg in ok:
            for h in hits:
                if h[1] == lg:
                    return h
        return None
    return hits[0]


def _direct_one(L: str, m, right: str = '', place: str | None = None):
    title = m.group('title')
    info = _info_for(title, (m.group('part') or '').lower().replace('\u2019', "'"))
    if info is None:
        return None
    cat, lg, strong = info
    pre = L[:m.start('title')]
    if NATIONALITY_BEFORE.search(pre.rstrip().split(' ')[-1] if pre.strip() else ''):
        return None          # "ein Spannischer Grave zu Zerbst": a Spanish count who is at Zerbst
    if title.lower() in ('conte', 'contes') and lg == 'fr' and re.search(r"(?i)\b(?:nous|on|vous|je|il|elle|ils|qui|me|te|se|ne|y)\s*$", pre):
        return None          # "on nous conte d'Osnabruck": to tell, not a count
    if re.search(r"\w-\s*$", pre):                      # 'ordinai- re de Gelle': a word broken at the end of a line
        return None
    if re.search(r"(?i)(?:vice|vize)[ -]?$", pre):          # vice-roy de Naples, viceroy of Peru: a governor, not a king
        return None
    if NOTRE_DAME.search(L[max(0, m.start('title') - 12):m.end('title')]):
        return None
    if re.search(r"(?-i:[A-Z][\w\.]+)\s+(?:de|van)\s*$", pre) and title[:1].isupper():
        return None          # "Abraham de Graaf d'Amsterdam": a surname (review 2026-10-28: der/den/von/vom removed, they blocked 'Heinrich II. von' + Konig and 'der Bischof von')
    mid = (m.group('mid') or '').strip()
    if mid:
        if any(t.endswith('.') and len(t) > 2 and not re.fullmatch(r"[IVXLC]+\.", t) for t in mid.split()):
            return None      # "Il Principe Leopoldo. Di Firenze": a sentence ends inside the name
        if not strong:
            return None
        if _OFFICE.search(mid):
            return None
        toks = mid.split()
        if any(re.fullmatch(_ALT, fold(t), re.I) for t in toks):
            return None      # another title inside the name ("Rex Svecorum der Konig der Schweden zu ...")
        if re.fullmatch(NAME_CONNECT, toks[0], re.I):
            return None      # "l'Empereur la Principaute de Glogaw": the name must start with a capitalised word
        # a capitalised word that is another title word ("King Duke") or only connectors: no
        if all(re.fullmatch(NAME_CONNECT, t, re.I) for t in toks):
            return None
        if not any(re.match(r"[A-Z]", t) for t in toks) and not all(re.fullmatch(EPITHETS, fold(t), re.I) for t in toks):
            return None
    part = (m.group('part') or '').lower().replace('’', "'")
    if not strong:
        # weak titles require the 'domain' particle of their language
        ok = WEAK_PARTICLES_STRICT.get(lg)
        if ok and part not in ok and part not in ('of', 'de'):
            return None
    if lg == 'la' and (mid or part != 'de'):
        return None          # Latin titles take the bare genitive ("rex Franciae"); names or German particles around them are another construction
    if mid and any(t.endswith('.') and re.fullmatch(r"[IVXLC]+\.", t) for t in mid.split()[:-1]):
        return None          # "du roy Henrick IV. Ceux de Venise": the sentence ends after the numeral
    if not strong and (m.group('art') or '').strip().lower() == 'the':
        return None          # "lord of the Mannor of Whitby", "lord of the Isles": a lordship of a thing, not a domain title
    if part in LOCATIVE_PARTICLES and lg not in ('de', 'nl'):
        return None
    if mid and part == 'vom':
        return None
    if mid and lg in ('de', 'nl') and part in ('zu', 'zum', 'zur', 'zue', 'zuo', 'in', 'bei', 'bey') and \
            re.search(r'(?i)(?:konigs|herzogs|hertzogs|bischofs|bischoffs|grafs|kaisers|keysers|abts|abtes)$', fold(title)):
        return None          # "des Konigs Galeria zu Fontainebleau": a possessive, not a title with a domain
    if re.search(r'(?i)^cardinal|^kardinal', title) and part in ('zu', 'zum', 'zur', 'zue', 'zuo', 'in', 'bei', 'bey', 'at'):
        return None          # "die Cardinale zu Rom"

    if part not in LOCATIVE_DE and re.match(r"^[\s,;:.]*(?:aus|auss)\b", fold(right or ''), re.I):
        return None          # "von Krucheln aus": where he writes from
    if part in LOCATIVE_DE and _verb_follows(right):
        return None          # "der Churfurst zu Stain ankommen", "der Printz zu Augspurg verritten": where the prince is, not his title
    if place is not None and EMPEROR.match(fold(title)) and not EMPEROR_REALM.match(fold(place)):
        return None          # "l'empereur de Ratisbone", "der Kaiser von Regenspurg": where the emperor comes from, not his realm
    if part in ('zu', 'zum', 'zur', 'zue', 'zuo', 'in', 'bei', 'bey', 'at') and EMPEROR.match(fold(title)):
        return None          # "der Kayser zu Regensburg": where the emperor is, not a domain
    if lg == 'it' and title.lower() == 're' and part not in ('di', "d'", 'de', 'del'):
        return None
    # the German 'in': 'Konig in Frankreich' fine; but 'Bischof in the Kirche' is not caught since the place follows
    return {'cat': cat, 'title': title, 'rule': 'direct', 'lg': lg, 'pre': pre, 'mid': mid, 'part': part}


_CONT = re.compile(r"^\s*(?:,\s*)?(?:(?i:and|et|und|vnd|vndt|undt|vnnd|unnd|ac|y|e|&)\s+)?(?-i:[A-Z])|^\s*(?:,\s*)?(?i:etc)")


def _chain(L: str, depth: int = 0, surf: str = '', right: str = '', orig: str | None = None):
    orig = L if orig is None else orig
    """place in a list that follows a title: "King of England, France and Ireland"."""
    if depth > 4:
        return None
    ms = _SEP_END.search(L)
    if not ms:
        return None
    sep = ms.group(0).strip().lower()
    if sep in (',', '/') or sep.startswith(','):
        if not _CONT.match(right) and depth == 0 and sep == ',':
            return None          # "der Konig in Schweden, Magdeburgk eingenommen": a comma is not a list unless the list goes on
    toks = L[:ms.start()].split(' ')
    for n in (1, 2, 3):
        if len(toks) < n or not all(re.match(r"[A-Z][^.]*$", t) for t in toks[-n:]) or DE_ADJ.search(toks[-1]):
            break
        before = ' '.join(toks[:-n]) + ' '
        r = _direct(before, right)
        if r:
            if r['lg'] in ('de', 'nl') and '/' not in orig[-(len(L) - len(before) + 160):]:
                return None      # German / Dutch titulatures list their lands with slashes; commas in running text are clauses
            r = dict(r)
            r['rule'] = 'chain'
            return r
        if LA_GEN.search(toks[-1].lower()) and LA_GEN.search(surf.lower().rstrip('.,;:')) and not LA_ADJ_ONLY.search(surf.lower().rstrip('.,;:')):
            r = _latin_left(before)
            if r:
                r = dict(r)
                r['rule'] = 'chain'
                return r
        r = _chain(before.rstrip(), depth + 1, surf, right, orig) if _SEP_END.search(before.rstrip() + ' ') else None
        if r:
            return r
    return None


def _latin_left(L: str):
    m = _LATIN.search(L)
    if not m:
        return None
    info = _category_of(m.group('title'))
    if not info:
        return None
    mid = (m.group('mid') or '').strip()
    if mid and _OFFICE.search(mid):
        return None
    return {'cat': info[0], 'title': m.group('title'), 'rule': 'latin', 'lg': 'la'}


def detect(left: str, surface: str, right: str = '', lang: str | None = None, coll: str | None = None):
    r = _detect(left, surface, right, lang)
    if r and r.get('pre') is not None:
        if _surname(r['pre'], coll, r['title']):
            return None
        place = re.sub(r"(?i)^(?:of|de|di|d['\u2019]|du|des|von|van|zu)\s+", '', fold(surface or '').strip())
        tl = fold(r['title']).lower()
        if tl.startswith('baron') and tl.endswith('s') and r.get('part') == 'of' and place.lower() == 'london':
            return None          # "the Barons of London": the citizens, not a title
        if not _is_strong(r) and tl.endswith(('s', 'i', 'en')) and REPUBLICS.match(place.split(' ')[0]):
            return None          # "les seigneurs de Venise", "i signori di Genova": a government
    if r and coll and coll.lower() in SURNAME_COLLS and r.get('lg') == 'la':
        return None              # 'Comes', 'Duke', 'Dux' as a surname in an English register
    if r and r.get('pre') is not None:
        mid = r.get('mid') or ''
        if mid and lang_is_en(r) and r['title'].lower().endswith('s') and not r['title'].lower().endswith('ss'):
            return None              # "Abbots Langley", "Kings Lynn": a place whose name begins with a title word
        if mid and coll and coll.lower() in SURNAME_COLLS:
            return None              # "Duke Dunne of Shendly", "Cæsar Bellingeam of ..."
    if r:
        r['support'] = needs_support(r)
        r.pop('pre', None)
        r.pop('mid', None)
    return r


def needs_support(res: dict) -> bool:
    """True for the detections that the corpus-level recurrence filter must confirm (see ``recurrence_filter``): a German / Dutch noble or royal title joined to the
    place by a locative preposition (zu, in, bei): 'Furst Friedrich zu Venedig, seine ferien helt' reads like a title but says where he is."""
    if not res or res.get('rule') != 'direct' or res.get('lg') not in ('de', 'nl') or res.get('part') not in LOCATIVE_DE:
        return False
    return res.get('cat') in (ROYAL, NOBLE) or bool(res.get('mid'))


def recurrence_filter(keys, flags, min_support: int = 3):
    """keys: iterable of (collection, place string); flags: iterable of bool (needs_support). -> list of bool keep.
    A place that is really the domain of a title recurs in its collection after titles (Plotzkau, Sachsen, Polen); a place where a prince merely is does not.
    Detections that need support are kept when the same (collection, folded place) has at least ``min_support`` such detections."""
    from collections import Counter
    ks = [(c, fold(p).lower().strip()) for c, p in keys]
    flags = list(flags)
    cnt = Counter(k for k, f in zip(ks, flags) if f)
    return [(not f) or cnt[k] >= min_support for k, f in zip(ks, flags)]


def _is_strong(r) -> bool:
    return _info_for(r['title'], r.get('part') or '') is not None and _info_for(r['title'], r.get('part') or '')[2]


def lang_is_en(r):
    return r.get('lg') == 'en'


def _detect(left: str, surface: str, right: str = '', lang: str | None = None):
    """-> None or dict(cat, title, rule, lg). ``left``/``right``: text around the surface."""
    left = fold(left or '')
    surf = fold(surface or '').strip()
    right = fold(right or '')
    if not surf:
        return None
    # 2. the surface starts with the title: try each token boundary of the surface as the place start
    toks = surf.split(' ')
    for k in range(1, min(len(toks), 6)):
        head = ' '.join(toks[:k])
        place = ' '.join(toks[k:])
        if not place or not re.match(r"[A-Z]", place[:1]):
            continue
        if not SURFACE_HEAD.match(head):
            continue            # 'Palast des Konigs zu Fontainebleau': a building, the title is not the first word
        r = _direct(left + head + ' ', right, place)
        if r is None:
            r = _latin_left(left + head + ' ') if re.search(r"(?i)^(?:" + _LA_ALT + r")$", head.split(' ')[-1]) else None
        if r:
            r = dict(r)
            r['rule'] = 'surface'
            return r
    # particle at the start of the surface ("of Sandye", "de France", "von Sachsen"): it belongs to the left text
    mp = _PREFIX.match(surf)
    L = left
    pl = surf
    if mp:
        L = left + surf[:mp.end()]
        pl = surf[mp.end():]
    else:
        L = left if left.endswith(' ') or not left else left + ' '
    # a place that is itself an adjective of a title ("Anglorum", "Gallicus") is not handled here
    if not re.match(r"[A-Za-zÀ-ɏ]", pl[:1] or ' '):
        return None
    if DE_ADJ.search(pl):
        return None
    r = _direct(L, right, pl)
    if r:
        return r
    if DOMAIN_NOUN.search(L):
        return None
    # 4. Latin without particle (genitive endings only)
    pls = pl.lower().rstrip('.,;:')
    if pl.isupper() and len(pl) > 3:
        r = _latin_left(L)
        if r and re.search(r"[A-Z]{3,}\s*$", L):
            return r             # inscriptions: 'COMES HAPSBVRGI'
    if LA_GEN.search(pls) and not LA_ADJ_ONLY.search(pls):
        r = _latin_left(L)
        if r and DUX_FORMS.match(r['title']) and PEOPLE_GEN.match(pls) and not pls.startswith('venet'):
            return None      # 'dux Hispanorum': a commander of the Spaniards
        if r and (REALM.match(pls) or ADJ_SUFFIX.search(pls)):
            return r
    if LA_ADJ_ECC.search(pls) or LA_CONS.search(pls):
        r = _latin_left(L)
        if r and (r['cat'] == ECCL or (LA_CONS.search(pls) and pl[:1].isupper())):
            return r
    r = _chain(L, 0, pl, right)
    if r:
        return r
    # reverse order: the place (Latin genitive) followed by the title; German genitive in -s followed by the title
    if LA_GEN_REV.search(pls):
        m = _R_LATIN.search(right)
        if m and DUX_FORMS.match(m.group('title')) and PEOPLE_GEN.match(pls) and not pls.startswith('venet'):
            m = None
        if m and (REALM_STRICT.match(pls) or ADJ_SUFFIX.search(pls)):
            info = _category_of(m.group('title'))
            if info:
                return {'cat': info[0], 'title': m.group('title'), 'rule': 'latin_after', 'lg': 'la'}
    if re.search(r'[bcdfghklmnpqrtwz]s$', pls):
        m = _R_DE.search(right)
        if m:
            info = _category_of(m.group('title'))
            if info:
                return {'cat': info[0], 'title': m.group('title'), 'rule': 'de_genitive', 'lg': 'de'}
    return None


HONORIFIC = re.compile(r"(?i)(?<![\w])(?:goody|goodwife|goodman|mr|mrs|mres|mris|mistress|mistris|widow|old|young|master|miss|doctor|dr|mons|monsr|monsieur|maistre|maitre|magister)\.?$")
SURNAME_COLLS = {'casebooks'}           # registers of patients ("Mary Abbot of Chichly", "Goody King of Luton"): a title word after a given name is a surname
STOP_BEFORE = re.compile(r"(?i)^(?:the|my|our|ye|his|her|that|this|of|and|et|to|by|from|said|late|right|honourable|honorable|most|noble|lady|lord|sir|kinge?|king's|duke|earl|earle|countess|countes|count|prince|princess|queen|lord's|bishop|archbishop|abbot|as|for|with|was|is|were|are|then|when|where|which|who|whom|whose)$")


_FOLLOW_OK = {'und', 'vnd', 'vndt', 'undt', 'unnd', 'vnnd', 'vn', 'etc', 'rc', 'oder', 'der', 'die', 'das', 'den', 'dem', 'des', 'ist', 'sind', 'hat', 'hatt', 'haben', 'hatte', 'war',
              'wird', 'wirdt', 'sol', 'soll', 'wil', 'will', 'wolle', 'sein', 'seiner', 'sine', 'ein', 'eine', 'als', 'auch', 'so', 'dann', 'dan', 'aber', 'noch', 'zu', 'von', 'bey', 'in', 'mit',
              'sampt', 'sambt', 'samt', 'wegen', 'sey', 'seye', 'sye', 'ist.', 'gewesen', 'regiert', 'regierte', 'gnaden', 'gnedigen', 'gnedigem', 'gnadigen', 'gnadigem', 'lieben', 'getrewen', 'unsern', 'unserm', 'vnsern', 'vnserm'}


_LOC_SIGNAL = {'aus', 'auss', 'seye', 'sey', 'sye', 'gewesen', 'bey', 'beÿ', 'beym', 'bei', 'im', 'an', 'auf', 'wo', 'da', 'gelegen', 'ankommen', 'angekommen', 'abgeritten', 'verritten', 'ritt', 'kommen', 'kommt',
               'komt', 'gekommen', 'reiset', 'reist', 'gereist', 'zog', 'logiert', 'losirt', 'logirt', 'bleibt', 'geblieben', 'hielt', 'helt', 'gehalten', 'ligt', 'liegt', 'wohnt', 'verreist',
               'verhindere', 'aufgebrochen', 'ankomen', 'ankommet', 'angelangt', 'eingeritten', 'ingeritten', 'einkommen', 'weg', 'wegk', 'vorbey', 'durch', 'nach', 'nacher', 'wider', 'wieder'}


def _verb_follows(right: str) -> bool:
    """True when the word after the place says where the holder is or moves ("der Churfurst zu Stain ankommen", "die Fursten zu Heydelberg bey der Hochzeit")."""
    m = re.match(r"^[\s,;:.]*([a-zA-Z\u00C0-\u024F]+)", re.sub(r"^\s*\[[^\]]{0,12}\]", "", fold(right or '')))
    if not m:
        return False
    w = m.group(1).lower()
    return w in _LOC_SIGNAL or (w.startswith(('ge', 'ver', 'an', 'ein', 'ab', 'auf', 'aus')) and w.endswith(('en', 'et', 'ten')) and len(w) > 6 and w not in ('gnaden', 'gnedigen', 'gnadigen', 'gnedigem'))


FEMALE_NAMES = re.compile(r"(?i)^(?:mary|marie|anne?|anna|elizabeth|elisabeth|elizabet|margaret|margery|margarett|jane|joan|joane|jone|alice|alis|agnes|katherine|katharine|catherine|susan|susanna|sarah|sara|dorothy|dorothea|ellen|helen|eleanor|elinor|isabel|isabell|isabella|frances|rebecca|rachel|martha|grace|amy|bridget|barbara|judith|hester|ursula|cicely|cecily|maud|mawd|mabel|joyce|winifred|prudence|lucy|blanche|mildred|audrey|christian|ciceley)$")
MASCULINE_TITLES = re.compile(r"(?i)^(?:king|kinge|kyng|duke|earl|earle|erle|prince|bishop|bysshop|byshop|abbot|abbat|lord|count|baron|marquess|emperor|knight|sultan|pope)$")
REPUBLICS = re.compile(r"(?i)^(?:venise|venize|venice|venetie|veneetie|venetia|venedig|gene|genes|genoa|genua|gen\u00e8s|berne|bern|zurich|zurych|geneve|geneva|genf|basel|lucerne|luzern|raguse|ragusa|lucques|lucca)$")


def _surname(pre: str, coll: str | None, title: str = '') -> bool:
    """True when the title word is in fact a surname: after an honorific (Goody King), or, in the patient registers, after a given name."""
    pre = pre.rstrip()
    if not pre:
        return False
    if HONORIFIC.search(pre):
        return True
    if title and MASCULINE_TITLES.match(title) and FEMALE_NAMES.match(pre.split(' ')[-1].rstrip('.:;')):
        return True              # "Mary King of Kensington": a king is not a woman
    if coll and coll.lower() in SURNAME_COLLS:
        toks = pre.split(' ')
        tok = toks[-1]
        if tok == '.' or tok.endswith('.') and len(tok) == 1:
            return True                        # "added by Michael Hawkins . King of haversham": the register line starts with the patient
        if tok.endswith(','):
            return False
        t = tok.rstrip('.:;')
        if STOP_BEFORE.match(t) and t.lower() not in ('a', 'an', 'of'):
            if len(toks) >= 2 and re.match(r"[A-Z][a-z]+$", toks[-2]) and not STOP_BEFORE.match(toks[-2]):
                return True                    # "Ione Lord Lord of Laundon"
            return False
        if t.lower() == 'of':
            return True                        # "Edward of King of Steep Claydon"
        if re.match(r"[A-Za-z]", t) and (t[:1].isupper() or tok.endswith(('.', ':')) or t.lower() in ('le', 'old')):
            return True
        if t.islower() and len(t) >= 1:        # 'lawde Lord', 'goody king', 'Alce hu King', 'king k King': a lower-case given name or nickname
            return True
    return False


def split_context(context: str, surface: str, near: int = 57):
    """Editor contexts are a 120-character window; the surface starts around position 57. -> (left, right) or None."""
    if not context or not surface:
        return None
    best = None
    start = 0
    while True:
        i = context.find(surface, start)
        if i < 0:
            break
        if best is None or abs(i - near) < abs(best - near):
            best = i
        start = i + 1
    if best is None:
        return None
    return context[:best], context[best + len(surface):]


def rules_json():
    """The lexicon and the structures, for rules.json."""
    out = {'categories': [ROYAL, NOBLE, ECCL], 'titles': {}, 'weak_titles': {}, 'particles': PARTICLES.split('|'), 'office_words_blocking_names': OFFICE_WORDS.split('|'),
           'name_connectors': NAME_CONNECT.split('|'), 'latin_genitive_endings': LA_GEN.pattern, 'dropped_weak': sorted(f'{a}:{b}' for a, b in DROP_WEAK), 'recurrence_min_support': 3}
    for cat, byl in STRONG.items():
        out['titles'][cat] = {lg: sorted(set(frag.split('|'))) for lg, frag in byl.items()}
    for cat, byl in WEAK.items():
        out['weak_titles'][cat] = {lg: sorted(set(w for w in frag.split('|') if (lg, w) not in DROP_WEAK)) for lg, frag in byl.items()}
    return out


# --------------------------------------------------------------------------------------------------------------------
# the lists of removed mentions (v29)
# --------------------------------------------------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / 'data' / 'curation'
EDITOR_FILE = DATA / 'title_excluded_editor.parquet'
MODEL_FILE = DATA / 'title_excluded_model.tsv.gz'
ECCLESIASTICAL_REMOVED = False       # the one switch (owner, 28 Oct 2026: bishops' sees stay places); True takes York, Winchester, Cluny ... out


def removed_categories(ecclesiastical: bool | None = None) -> tuple:
    """The categories whose places leave the atlas: royal and noble always, ecclesiastical unless the flag is off."""
    ecc = ECCLESIASTICAL_REMOVED if ecclesiastical is None else ecclesiastical
    return (ROYAL, NOBLE, ECCL) if ecc else (ROYAL, NOBLE)


def load_editor_exclusions(path=None, ecclesiastical: bool | None = None) -> pd.DataFrame:
    """The editor mentions to remove: columns row, coll_slug, doc_id, surface, category, title_word, rule. ``row`` is unique."""
    x = pd.read_parquet(path or EDITOR_FILE)
    x = x[x['category'].isin(removed_categories(ecclesiastical))]
    if x['row'].duplicated().any():
        raise ValueError('title_excluded_editor: duplicate row key')
    return x.reset_index(drop=True)


def check_editor_keys(excl: pd.DataFrame, doc_id, surface) -> int:
    """Check the keys against the mentions table (``doc_id`` and ``surface`` of the whole table, by position); returns the number of keys checked.
    Raises ValueError when the table is not the one the keys were made on."""
    n = len(doc_id)
    if int(excl['row'].max()) >= n:
        raise ValueError(f'title_excluded_editor: row {int(excl["row"].max())} is outside a table of {n} lines')
    d = pd.Series(doc_id).fillna('').iloc[excl['row'].to_numpy()].to_numpy()
    s = pd.Series(surface).fillna('').iloc[excl['row'].to_numpy()].to_numpy()
    bad = (d != excl['doc_id'].fillna('').to_numpy()) | (s != excl['surface'].fillna('').to_numpy())
    if bad.any():
        raise ValueError(f'title_excluded_editor: {int(bad.sum())} keys do not match the mentions table (first: row {int(excl["row"].to_numpy()[bad][0])})')
    return len(excl)


def load_model_exclusions(path=None, ecclesiastical: bool | None = None) -> pd.DataFrame:
    """The model mentions to remove, one line per (src, doc_id, lang, key): n_excluded of the n_total mentions of the form in the document.
    Columns src, doc_id, lang, key, n_total, n_royal, n_noble, n_ecclesiastical, n_excluded (lines with nothing left to remove are dropped)."""
    x = pd.read_csv(path or MODEL_FILE, sep='\t', keep_default_na=False)
    for c in ('n_total', 'n_royal', 'n_noble', 'n_ecclesiastical'):
        x[c] = x[c].astype(int)
    ecc = ECCLESIASTICAL_REMOVED if ecclesiastical is None else ecclesiastical
    x['n_excluded'] = x['n_royal'] + x['n_noble'] + (x['n_ecclesiastical'] if ecc else 0)
    if (x['n_excluded'] > x['n_total']).any():
        raise ValueError('title_excluded_model: more mentions removed than the form has in the document')
    if x.duplicated(['src', 'doc_id', 'lang', 'key']).any():
        raise ValueError('title_excluded_model: duplicate key')
    return x[x['n_excluded'] > 0].reset_index(drop=True)
