"""What a place string is: a place, or a celestial body, a people, a god, a common noun (v2; pure functions).

The audit found places that are not places (bdpn: 1,784 mentions of Sol, Luna, cielo; ckcc: 3,949 Galli / Hispani; epistemon:
Palais, Isle; germanc: hier, Welt; hainhofer: herberg, gang, stall; ssrq: Keller, Oberamt) and a geocoder that put them on the
nearest village of that name. ``place_class`` names the reason; the geocoder never looks up a place with a class.

classes: ``celestial`` ``mythical`` ``biblical`` ``fictional`` ``ethnonym`` ``common_noun`` ``deictic`` ``repository`` or None.
``ethnonym`` and ``area`` names of the exonym table are handled by ``exonyms``: a people's name maps to its country only as an
area, flagged.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .namegate import small_letters

SKIP_CLASSES = frozenset({'celestial', 'mythical', 'biblical', 'fictional', 'ethnonym', 'common_noun', 'deictic'})


def key(s) -> str:
    s = unicodedata.normalize('NFKD', small_letters(str(s)).casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c)).replace('ß', 'ss')
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s).split())


# words that are not places whatever the edition says (single normalised keys, or the whole key)
DEICTIC = frozenset('''hier hie da dort dorten daselbst dahin dorthin hinc hic huc ibi istic illic illuc inde unde ubi alibi ibidem
ibid idem ebenda ici ailleurs lo qui qua quivi colà here there hence thence thither whither elsewhere ibi domi foris
allhier alda dasselbst
north south east west northern southern eastern western nord sud ost est norden suden osten westen oriens occidens septentrio meridies'''.split())     # v2.3: directions (MoEML's 'North' became the Département du Nord)
COMMON_NOUN = frozenset('''welt herberg herberge gang stall sonnen trauben keller kaeller kuche kueche kammer stube saal garten hoff oberamt
oberwasser palais palast isle cielo tierra mundo mundus orbis terra coelum caelum regio locus ort platz hauss thor tor strasse gasse
markt kirche capell kapelle schloss wasser stadt statt town city ville citta ciudad dorf village villaggio kirchen kirchsaal marckt
konigreich koenigreich kingdom royaume'''.split())
# lone generic words that are real places too ('Hof', 'Burg', 'See') are not in the list: only the editorial register decides

# biblical places whose names are also modern villages (abacus: Hai/Ai at the Province du Hainaut): never placed by name
BIBLICAL_PLACES = frozenset('hai kana naim babel sodom sodoma gomorra gomorrha gomorrah ninive emmaus gilgal bethania bethanien bethany'.split())

_ETHNONYM_STEMS = '''galli gallis gallos gallorum hispani hispanis hispanos hispanorum batavi batavis batavos batavorum belgae
belgis belgas belgarum germani germanis germanos germanorum angli anglis anglos anglorum itali italis italos italorum sueci suecis
suecos suecorum sueones dani danis danos danorum poloni polonis polonos polonorum boemi bohemi boemis bohemis bohemos hungari
hungaris hungaros hungarorum turcae turcis turcas turcarum turci graeci graecis graecos graecorum iudaei iudaeis iudaeos romani
romanis romanos romanorum scoti scotis scotos hiberni hibernis helvetii helvetiis helvetios burgundi burgundis saxones saxonibus
franci francis francos lusitani lusitanis lusitanos britanni britannis britannos siculi siculis veneti venetis venetos athenienses
atheniensibus atheniensium lacedaemonii lacedaemoniis lacedemonii spartani spartanis thebani thebanis carthaginienses persae persis
persas persarum medi medis aegyptii aegyptiis macedones macedonibus thraces thracibus scythae scythis sarmatae sarmatis
tartari tartaris moscovitae moschi muscovitae rutheni ruthenis lituani lithuani livones prussi prussis hollandi frisii
frisiis flandri flandris brabanti zelandi cimbri cimbris teutones teutonibus vandali vandalis gothi gothis longobardi longobardis
galles gallici hispanici espanol espanoles francesi francois francais anglois allemands italiens hollandois flamands
spanier spanische franzosen teutschen welsche schweizer niderlander
atheniens lacedemoniens grecs romains gaulois perses egyptiens carthaginois troyens thebains'''.split()
ETHNONYM = frozenset(_ETHNONYM_STEMS)

_NOTE_CLASS = (
    (re.compile(r'celestial|planet|star\b|zodiac'), 'celestial'),
    (re.compile(r'myth|legend|deity|divine|god\b|goddess'), 'mythical'),
    (re.compile(r'bibl'), 'biblical'),
    (re.compile(r'fiction|imaginary|literary|utopi'), 'fictional'),
    (re.compile(r'ethnon|demonym|people|nation|tribe|gentilic'), 'ethnonym'),
    (re.compile(r'repositor'), 'repository'),
    (re.compile(r'epithet'), 'common_noun'),             # cidtc: 'terra', 'fluminis', 'mari Hispano': a description, not a name
)


ARTICLES = frozenset('die der das dem den des ein eine the a an la le les el los las il lo gli un una une de het'.split())


def classify(name: Optional[str], note_type: Optional[str] = None, person_class: Optional[str] = None) -> Optional[str]:
    """Class of a place entity (None = an ordinary place)."""
    person_class = person_class if isinstance(person_class, str) else ''
    note_type = note_type if isinstance(note_type, str) else ''
    name = name if isinstance(name, str) else ''
    pc_ = person_class.lower()
    if pc_ in ('celestial',):
        return 'celestial'
    if pc_ in ('mythological', 'mythical'):
        return 'mythical'
    if pc_ == 'biblical':
        return 'biblical'
    if pc_ in ('character',):
        return 'fictional'
    if pc_ == 'people':
        return 'ethnonym'
    nt = note_type.lower()
    if nt:
        for rx, c in _NOTE_CLASS:
            if rx.search(nt):
                return c
    k = key(name)
    if not k:
        return None
    # 'die Stadt', 'the town', 'la ville': an article before a deictic or a common noun is still a deictic or a common noun (v2 audit: 'die Stadt'
    # of the GerManC news was geocoded to Magdeburg and to a Slovak town)
    head, _, rest = k.partition(' ')
    ks = [k] + ([rest] if rest and head in ARTICLES else [])
    for x in ks:
        if x in DEICTIC:
            return 'deictic'
        if x in COMMON_NOUN:
            return 'common_noun'
    if k in BIBLICAL_PLACES:
        return 'biblical'
    return None


def is_ethnonym_surface(name: Optional[str]) -> bool:
    k = key(name or '')
    return bool(k) and k in ETHNONYM
