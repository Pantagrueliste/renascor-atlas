"""The five-way kind of a place of the globe (v2.2; pure functions).

    c   cities, towns, villages, city quarters                       (a settlement)
    a   countries, regions, territories                              (a political or administrative area)
    n   rivers, lakes, seas, islands, mountains, valleys, passes, capes, forests   (a natural feature)
    b   buildings, castles, monasteries, streets, squares, gates, bridges, sites    (a built feature or a site)
    u   other (no source says what it is)

``kind5`` reads, in this order of weight, the Wikidata classes of the accepted item (English labels), the GeoNames feature code, the Getty TGN place type
and the register's own type of the entry (``note_type``). Each source votes for the kinds its descriptions name: the **head noun** of a label decides
('island country' is a country, 'river island' an island, 'municipality with town privileges in the Czech Republic' a municipality). The atlas's three-way
kind (city / area / other) breaks ties and answers when no source says anything. Pure functions, no I/O.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional

KINDS = ('c', 'a', 'n', 'b', 'u')
KIND_LABEL = {'c': 'city / town / village / city quarter', 'a': 'country / region / territory',
              'n': 'river / lake / sea / island / mountain / valley / pass / cape / forest', 'b': 'building / castle / monastery / street / square / gate / bridge / site',
              'u': 'other'}
_BASE = {'city': 'c', 'area': 'a', 'other': 'u', 'c': 'c', 'a': 'a', 'n': 'n', 'b': 'b', 'u': 'u'}
#: tie-break when two kinds have the same weight: a thing that is both an island and a municipality is drawn as the island; a castle town as the town
PRIORITY = ('b', 'n', 'c', 'a', 'u')


def _words(s: str) -> frozenset:
    return frozenset(s.split())


_N = _words('''river stream brook watercourse tributary creek rivulet canal lake lakes reservoir pond lagoon sea ocean bay gulf strait channel fjord sound inlet cove
island islands isle isles islet archipelago peninsula cape headland promontory mountain mountains mount hill hills volcano range massif ridge peak summit
valley pass gorge canyon ravine forest wood woods woodland jungle desert plain moor heath plateau glacier spring springs waterfall falls beach coast cliff delta marsh
swamp bog sandbank reef atoll isthmus steppe oasis cave grotto lowland highland upland meadow dune rock cataract lough loch mere firth estuary basin shoal
tarn fell down downs weald chase mountain-range massif alp alps''')
_B = _words('''building buildings castle fortress fort fortification citadel stronghold palace church cathedral chapel basilica monastery abbey convent priory nunnery cloister
temple mosque synagogue shrine tower bridge gate street road lane alley avenue boulevard square plaza piazza market marketplace inn tavern hotel hospital school
university college academy library theatre theater museum prison jail gaol mill manor villa house hall mansion garden gardens park cemetery churchyard port
harbour harbor dock wharf pier quay lighthouse arena stadium amphitheatre barracks arsenal armoury embassy courthouse guildhall hospice almshouse farm estate
grange ruins ruin battlefield monument memorial pillar obelisk fountain well aqueduct dam wall rampart bastion
site residence townhall brewery pub alehouse playhouse bridewell compter vineyard factory foundry workshop mint warehouse cloth-hall belfry cloisters
chateau castrum structure complex rathaus festung kloster schloss kirche theatre-building''')
_A = _words('''country nation state kingdom empire duchy principality county province region department canton republic territory district governorate prefecture
margraviate landgraviate electorate bishopric diocese archbishopric archdiocese lordship barony earldom shire hundred wapentake riding division oblast
voivodeship krai land continent subcontinent landscape dominion colony protectorate confederation commonwealth palatinate county-palatine march
mandate emirate sultanate khanate caliphate marquisate viscounty county-borough liberty rape lathe ward-district
satrapy viceroyalty vice-royalty unit bailiwick seigneury seignory commandery landgraviate-county''')
_C = _words('''city town village hamlet municipality commune settlement borough suburb neighbourhood neighborhood quarter township locality conurbation metropolis
capital seat comune gemeinde stadt dorf ortsteil stadtteil parish-town burgh polis''')
_TWO = {'market town': 'c', 'human settlement': 'c', 'urban area': 'c', 'inhabited place': 'c', 'populated place': 'c', 'city district': 'c',
        'civil parish': 'c', 'big city': 'c', 'port city': 'c', 'deserted settlement': 'c', 'lost settlement': 'c', 'ghost town': 'c', 'city state': 'c',
        'free city': 'c', 'imperial city': 'c', 'mountain range': 'n', 'mountain system': 'n', 'mountain pass': 'n', 'natural region': 'n',
        'archaeological site': 'b', 'historic site': 'b', 'historical site': 'b', 'world heritage': 'b', 'country house': 'b', 'town hall': 'b',
        'market square': 'b', 'historical region': 'a', 'cultural region': 'a', 'geographic region': 'a', 'general region': 'a', 'historical country': 'a',
        'sovereign state': 'a', 'federal state': 'a', 'first level subdivision': 'a', 'second level subdivision': 'a', 'regional division': 'a',
        'dependent state': 'a', 'autonomous region': 'a', 'administrative division': 'a', 'administrative territorial entity': 'a',
        'political entity': 'a', 'imperial abbey': 'a', 'prince bishopric': 'a', 'free state': 'a', 'holy see': 'a'}
_SPECIAL_HEAD = {'ward': 'c', 'quarter': 'c', 'borough': 'c', 'port': 'b', 'harbour': 'b', 'harbor': 'b', 'manor': 'b', 'estate': 'b', 'abbey': 'b',
                 'march': 'a', 'hundred': 'a', 'liberty': 'a', 'see': 'a'}
_OF = re.compile(r'\s+(?:of|in|within|on|at|from)\s+.*$', re.I)
_PAREN = re.compile(r'\([^)]*\)')
_SPLIT = re.compile(r'[^a-z]+')
_HEAD: dict = {}
for _tab, _k in ((_C, 'c'), (_A, 'a'), (_N, 'n'), (_B, 'b')):          # later wins: a word in two tables is the more specific kind
    for _w in _tab:
        _HEAD[_w] = _k
_HEAD.update(_SPECIAL_HEAD)
_TWO.update({'city centre': 'c', 'city center': 'c', 'old town': 'c', 'town quarter': 'c', 'urban district': 'c', 'city state': 'c'})


def _fold(s) -> str:
    """lower case without accents ('château' -> 'chateau')."""
    s = unicodedata.normalize('NFKD', str(s or '').lower())
    return ''.join(c for c in s if not unicodedata.combining(c)).replace('ß', 'ss')


def head_kind(label) -> Optional[str]:
    """Kind a description names by its head noun, or None. 'municipality with town privileges in the Czech Republic' -> c; 'island country' -> a;
    'river' -> n; 'church building' -> b; 'castle town' -> c; 'region of Italy' -> a; 'big city' -> c; 'city-state' -> c."""
    if not isinstance(label, str) or not label.strip():
        return None
    s = _fold(_PAREN.sub(' ', label)).replace('_', ' ').replace(',', ' ')
    s = _OF.sub('', s)
    s = re.sub(r'\bwith\b.*$', '', s)                                 # 'municipality with town privileges'
    toks = [t for t in _SPLIT.split(s.replace('-', ' ')) if t]
    if not toks:
        return None
    if ' '.join(toks[-2:]) in _TWO:
        return _TWO[' '.join(toks[-2:])]
    last = toks[-1]
    for w in (last, last[:-1] if last.endswith('s') else last, last[:-2] if last.endswith('es') else last, last[:-3] + 'y' if last.endswith('ies') else last):
        if w in _HEAD:
            return _HEAD[w]
    return None


# ---------------------------------------------------------------------------------------------------------------------
# GeoNames feature codes
# ---------------------------------------------------------------------------------------------------------------------
_GN_N = frozenset('''STM STMA STMB STMC STMD STMH STMI STMIX STMM STMQ STMS STMSB STMX CNL CNLB LK LKC LKI LKN LKNI LKO LKOI LKS LKSB LKSC LKSI LKSN LKSNI
LKX LGN LGNS LGNX BAY BAYS BGHT COVE CREK CHN CHNM CHNN FJD FJDS GULF INLT OCN OCNS PND PNDS POOL RSV SEA SPNG SPNS SPNT STRT SWMP WTRH FLLS RPDS GLCR ICEF
MT MTS MTSU MTU HLL HLLS HLLU KNLL KNL PK PKS PKSU PKU PEN PENX ISL ISLS ISLET ISLF ISLM ISLT ISLX ATOL CAPE PT PASS VAL VALS VALG VALX VALU CNYN GRGE
RDGE RDGB RDGU RDGEU BCH BCHS CLF CLFS PLN PLNS PLAT DSRT DUNE CAV CAVE CRTR VLC MND MNDS RK RKS SPIT SLP SLPS PRMT RAV TRGD UPLD LAVA FRST FRSTF FRSU GRSLD HTH MDW
MOOR SCRB WOOD ESTY BAR BNK BNKX REEF RF SHOL SBED SEAU SLID TRNH TUND MRSH MRSHN PEAT OAS CST ANCH ANCH ATOLL BNKU BYLD CNYU'''.split())
_GN_B = frozenset('''BLDG BLDO BP BRKS BRKW BSTN BTYD BUR CARN CSNO CSTL CTHSE CH CHRY CMP CMPMN CMPSC CMPTX FT FTHSE GDN GOSP HSE HSEC HSP HSTS HTL ML MKT MNMT MSQE MSSN
MSSNQ MSTY MTRO MUS OBPT OBS OPRA PAL PGDA PLDR PS PSH PYR PYRS RECG RECR RET RHSE RLG RUIN SCH SCHA SCHC SHRN SQR STDM STNB TMPL TMB TOWR UNIV VETF WALL WHRF YD ZOO
RD RDA RDB RDCUT RDJCT RDST ST STKR TNL TNLN TNLRD TNLS BDG BDGQ DAM DAMQ DAMSB PRK PRKGT PRKHQ PRN PRNJ PRNQ HBR HBRX DCK DCKB PRT QUAY JTY CNLA CNLD CNLI CNLN
CNLQ CNLSB CNLX AIRP AIRB AIRF AIRH AIRQ ASTR BANK BNKR CMTY DIKE FRM FRMS FRMQ GATE ESTT FRMT HOSP LTHSE MN MLM MLSG MLSGQ MLWTR OILR PIER PKLT PO POST PP PPQ PRKGT
RSRT REST SNTR STNC SPA SWT TMPL THTR UNIV USGE VLC'''.split())
_GN_A = frozenset('''ADMD ADM1 ADM2 ADM1H ADM2H ADMDH PCL PCLD PCLF PCLH PCLI PCLIX PCLS TERR RGN RGNE RGNH RGNL AREA CONT LTER ZN ZNB'''.split())
_GN_C = frozenset('ADM3 ADM4 ADM5 ADM3H ADM4H ADM5H PRSH STLMT'.split())          # communes, municipalities, parishes: the settlement (GeoNames' own third to fifth level)
_GN_U = frozenset('LCTY VINS RES RESA RESF RESH RESN RESP RESV RESW LAND AGRC GRAZ INDS MILB MNA OILF CTRB DEVH FLD GVL TRB NVB'.split())


def gn_kind(fcode) -> Optional[str]:
    """Kind of a GeoNames feature code (PPLA2 -> c, PCLI -> a, STM -> n, CSTL -> b), None for an empty or unknown code."""
    f = fcode if isinstance(fcode, str) else ''
    if not f or f in ('None', 'nan'):
        return None
    if f.startswith('PPL') or f in _GN_C:
        return 'c'
    if f.startswith('PCL') or f in _GN_A:
        return 'a'
    if f in _GN_B:
        return 'b'
    if f in _GN_N:
        return 'n'
    if f in _GN_U:
        return 'u'
    return None


# ---------------------------------------------------------------------------------------------------------------------
# register types
# ---------------------------------------------------------------------------------------------------------------------
_NOTE_N = frozenset('''moun mountain berg mons gebirge alpen rive river fluss flus flumen fleuve bach stream lake see lac lacus sea meer mare insel island isola insula ile
place_ea_river place_ea_mountain place_ea_lake place_ea_sea place_ea_island place_ea_forest wald forest valley tal'''.split())
_NOTE_B = frozenset('''stre street strasse straße weg platz burg kloster church kirche castle schloss schloß mühle muhle hof gut building monastery abbey bridge brücke
gate tor place_f_street place_f_church place_f_residence place_f_school place_f_townhall place_f_guildhall place_f_victuallinghouse place_f_religioushouse
place_f_property place_f_openarea place_f_hospital place_f_prison place_f_theatre place_f_market place_f_bridge place_f_gate place_f_inn place_f_palace
palace palast capell kapelle spital haus acker flurname'''.split())
_NOTE_A = frozenset('''country coun region dist district province landscape landschaft county place_county canton kanton kingdom duchy continent place_ea_diocese
place_country place_ea_region place_ea_county place_ea_province gerichtsherrschaft herrschaft landvogtei grafschaft'''.split())
_NOTE_C = frozenset('''city stadt vill dorf gemeinde gemeindeteil settlement town village place_pa_settlement place_pa_ward place_pa_village place_pa_town place_pa_city
place_pa_suburb quarter ortsteil'''.split())


def note_kind(note_type) -> Optional[str]:
    t = str(note_type or '').strip().lower()
    if not t or t in ('none', 'nan', 'place'):
        return None
    for table, kind in ((_NOTE_N, 'n'), (_NOTE_B, 'b'), (_NOTE_A, 'a'), (_NOTE_C, 'c')):
        if t in table:
            return kind
    if t.startswith('place_f_'):
        return 'b'
    if t.startswith('place_pa_'):
        return 'c'
    # 'eidgenössischer ort, kanton', 'herrschaft, amt, gemeine herrschaft, grafschaft': a list of types; the first that names a kind
    for part in re.split(r'[,;/]', t):
        part = part.strip()
        for table, kind in ((_NOTE_N, 'n'), (_NOTE_B, 'b'), (_NOTE_A, 'a'), (_NOTE_C, 'c')):
            if part in table:
                return kind
    return head_kind(t)


# ---------------------------------------------------------------------------------------------------------------------
# the name itself: "Kloster Zinna", "St. Paul's Cathedral", "Ballenstedt, Amt", "The Thames"
# ---------------------------------------------------------------------------------------------------------------------
_NAME_B = frozenset('''burg kloster abtei stift schloss schlosz kirche kapelle cathedral church chapel abbey priory convent monastery nunnery college hospital spital
palace palast castle chateau castello castillo palazzo bridge brucke gate street strasse gasse lane square platz market inn tavern tower turm hall house bastei festung
fortress fort rathaus playhouse prison gaol wharf stairs alley yard court'''.split())
_NAME_A = frozenset('''amt herrschaft grafschaft herzogtum furstentum kurfurstentum konigreich markgrafschaft landgrafschaft bistum erzbistum hochstift kreis provinz county
shire duchy kingdom diocese bishopric province region regnum ducatus comitatus diocesis principality landschaft landvogtei vogtei'''.split())
_NAME_COMPOUND = (('b', ('strasse', 'gasse', 'kirche', 'kloster', 'schloss', 'brucke')), ('n', ('see', 'meer', 'fluss')))      # v2.3: no 'bach' (Rorbach, Rohrbach, Offenbach, Ansbach are villages)


_NAME_RIVERS = frozenset('''thames themse rhine rhin danube donau elbe weser maas meuse scheldt seine loire rhone tiber tevere tagus ebro vistula weichsel volga'''.split())
_RIVER_FILLER = frozenset('''the der die das le la les l river fluss rivier fleuve flumen fluvius riviere'''.split())
_NAME_PEAK = frozenset('berg mount mountain mons monte'.split())
_NAME_N_PLAIN = frozenset('''fluss bach river riviere fleuve flumen fluvius lake lac lacus meer sea mare gebirge wald forest insel island isola insula ile tal valley pass cape kap'''.split())


def name_kind(name) -> Optional[str]:
    """Kind a place NAME says by itself (None for a plain name): a word of the name that is a building, an administrative area or a natural feature
    ('Kloster Zinna', 'St. Paul's Cathedral', 'Ballenstedt(-Hoym), Amt', 'The Thames', 'Bodensee'). Whole words only, and German compounds with a few endings of
    six letters or more beyond the first part ('Königsstraße'); 'Burg' as a whole word only ('Hamburg' is a town); no 'See' / 'Tor' / 'Dom' as words
    ('Holy See', 'Le Thor'). A river's name counts only when the name is the river and nothing else ('The Thames', not 'Dorchester on Thames'); *Berg*, *Mount*
    and *Monte* only as the first word ('Monte Somma', not 'Évora Monte')."""
    if not isinstance(name, str) or not name.strip():
        return None
    toks = [t for t in _SPLIT.split(_fold(name).replace("'", ' ').replace('-', ' ')) if t]
    if not toks:
        return None
    if any(t in _NAME_B for t in toks):
        return 'b'
    if 'parish' in toks and any(t in ('st', 'saint', 'sainte', 'sankt', 's') for t in toks):          # v2.3: 'Parish of St. Martin (Ludgate)' is a church
        return 'b'
    if any(t in _NAME_A for t in toks):
        return 'a'
    rest = [t for t in toks if t not in _RIVER_FILLER]
    if len(rest) == 1 and rest[0] in _NAME_RIVERS:
        return 'n'
    if toks[0] in _NAME_PEAK or any(t in _NAME_N_PLAIN for t in toks):
        return 'n'
    for t in toks:
        for kind, ends in _NAME_COMPOUND:
            if any(t.endswith(e) and len(t) >= len(e) + 4 for e in ends):
                return kind
    return None


# ---------------------------------------------------------------------------------------------------------------------
def kind5(base=None, wd_labels: Iterable = (), gn_fcodes: Iterable = (), tgn_types: Iterable = (), notes: Iterable = (), exonym_kind: Optional[str] = None,
          names: Iterable = (), weights=(1.0, 1.0, 0.8, 0.6, 1.0)) -> str:
    """The five-way kind of a place from its sources; see the module docstring. ``base``: the atlas's kind ('city' / 'area' / 'other' or a letter)."""
    votes: Counter = Counter()
    ww, wg, wt, wn, wm = weights
    labs = [head_kind(l) for l in wd_labels if isinstance(l, str)]
    for k in labs:
        if k:
            votes[k] += ww
    for f in gn_fcodes:
        k = gn_kind(f)
        if k and k != 'u':
            votes[k] += wg
    for t in tgn_types:
        if isinstance(t, str):
            for part in t.split(';')[:2]:
                k = head_kind(part)
                if k:
                    votes[k] += wt / (1 + 0.0)
    for n in notes:
        k = note_kind(n)
        if k:
            votes[k] += wn
    for nm in list(names)[:3]:
        k = name_kind(nm)
        if k:
            votes[k] += wm
    if exonym_kind in KINDS:
        votes[exonym_kind] += 2.0
    if votes:
        top = max(votes.values())
        best = [k for k in PRIORITY if abs(votes.get(k, 0.0) - top) < 1e-9]
        b = _BASE.get(str(base)) if base is not None else None
        if len(best) > 1 and b in best:
            return b
        return best[0]
    return _BASE.get(str(base), 'u') if base is not None else 'u'
