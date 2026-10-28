"""Latin and old exonyms, curated (v2): consulted by the name-based geocoder BEFORE any GeoNames alternate-name lookup.

Why. GeoNames lists 'Batavis' as an alternate name of Passau (Batavis, Castra Batava), so 751 mentions of the Dutch ('Batavis nihil
prodest') of Grotius' letters were placed at Passau; 'Batavia' went to Jakarta; 'Nicaea' (the council) to Nice; 'Perse' (Persia) to a
hamlet in the Yonne; 'Prussia' to a hamlet in Piedmont; 'Galli' (the French) were geocoded as a country. This table maps such forms to the
place the early-modern text means. Areas (countries, regions, peoples named after their country) get a coarse point and a radius:
they are never drawn on a capital.

Each entry: (label, lat, lon, radius_km, kind, [forms]). kind: ``city`` (a point) or ``area``. Forms are matched on the geocoder's
normalised keys (``merge._norm``): lower case, no diacritics, punctuation to spaces.
"""
from __future__ import annotations

from .merge import _norm
from .namegate import small_letters

# fmt: off
# v2.4 (reread of the table, check of v2.3): removed 'vienne' (Vienna; but the French town of Vienne: the register of ckcc_epistolarium has it too, 48 + 4 mentions), 'indias' (the West
# OR the East Indies), 'rutheni ruthenia' (Muscovy: Ruthenia is the Rus' of Poland-Lithuania) and 'regiomontanus' (an adjective; Regiomontanus is of Königsberg in Franconia). The
# bare Latin forms of Lyon (and Leiden), Utrecht and Augsburg: see ``latin_homonyms``. Still to be read by the owner: 'urbs' (Rome), 'megalopolis' (Mecklenburg), 'francofurtum' (Main).
_RAW = [
    # ---- areas: countries and peoples named after them
    ('Low Countries / Netherlands', 52.1, 5.1, 160, 'area',
     'batavi batavis batavos batavorum batavia batavus batava bataviae batavicus batavicum batavica hollandia hollandiae hollandi holland hollant '
     'hollande hollandt belgica belgicae belgium belgae belgas belgis belgarum pays-bas paesi_bassi paesi nederlandt nederland nederlanden '
     'nederlant niderland niderlande niederlande niderlanden germania_inferior germaniae_inferioris pays_bas netherlands the_netherlands'.split()),
    ('France', 46.6, 2.5, 350, 'area',
     'gallia galliae gallias galli gallis gallos gallorum gallicus gallicum gallica francia franciae francie france frankreich frantzreich '
     'franckreich frankreych franzosen francois-royaume gallien'.split()),
    ('Spain', 40.2, -3.7, 350, 'area',
     'hispania hispaniae hispani hispanis hispanos hispanorum hispanicus hispanicum hispanica spagna spanien espagne espana espanya spayne '
     'ispagna spania spanjen spain'.split()),
    ('Italy', 42.5, 12.5, 350, 'area',
     'italia italiae itali italis italos italorum italie italien italy italya italicus'.split()),
    ('Germany', 51.0, 10.0, 350, 'area',
     'germania germaniae germani germanis germanos germanorum teutschland teutsches_land teutschen deutschland deutsches allemagne alemania '
     'germanien germaniam almaine alemaigne germany'.split()),
    ('England', 52.6, -1.5, 250, 'area',
     'anglia angliae angli anglis anglos anglorum england engellandt engelland engelant angleterre inghilterra inglaterra britannia '
     'britanniae britanni britannis britannos'.split()),
    ('Scotland', 56.8, -4.2, 160, 'area', 'scotia scotiae scoti scotis scotos scotland scotlande escosse scozia schottland schotland'.split()),
    ('Ireland', 53.3, -7.8, 160, 'area', 'hibernia hiberniae hiberni hibernis ireland irelande irlande irlanda irland'.split()),
    ('Poland', 52.0, 19.5, 300, 'area', 'polonia poloniae poloni polonis polonos polonorum polen pologne polska polonie poland'.split()),
    ('Bohemia', 49.8, 15.0, 130, 'area',
     'bohemia boemia boemiae bohemiae boemi bohemi boemis bohemis behmen boehmen bohmen bohême boheme boemen bohemen'.split()),
    ('Hungary', 47.2, 19.0, 220, 'area', 'hungaria hungariae hungari hungaris hungaros hungarorum ungern ungarn vngern hongrie ungheria ungria hungary'.split()),
    ('Sweden', 62.0, 15.0, 400, 'area', 'suecia sueciae sueci suecis suecos suecorum sueones sueonia schweden suede svecia sverige swedes sweden'.split()),
    ('Denmark', 56.0, 10.0, 200, 'area', 'dania daniae dani danis danos danorum dänemark danemark denmark dannemark danimarca'.split()),
    ('Norway', 61.0, 9.0, 400, 'area', 'norvegia norvegiae norwegen norvege norway norwegia noruega'.split()),
    ('Switzerland', 46.8, 8.2, 150, 'area',
     'helvetia helvetiae helvetii helvetiis helvetios schweiz suisse svizzera switzerland suiza eidgenossenschaft'.split()),
    ('Austria', 47.6, 14.5, 150, 'area', 'austria austriae osterreich oesterreich österreich autriche austrie ostreich'.split()),
    ('Ottoman Empire', 40.0, 32.0, 700, 'area',
     'turcia turciae turcae turci turcis turcas turcarum turcicus turkei türkei turkey turquie turchia turquia ottoman'.split()),
    ('Greece', 39.0, 22.5, 250, 'area',
     'graecia graeciae graeci graecis graecos graecorum griechenland grece grecia grèce graecus greece'.split()),
    ('Persia', 32.5, 53.5, 700, 'area', 'perse persia persiae persae persis persas persarum persien persien persian persie persa'.split()),
    ('Egypt', 26.5, 30.8, 500, 'area', 'aegyptus aegypti aegyptii aegyptiis egypte aegypte egypten aegypten egitto egipto egypt'.split()),
    ('Portugal', 39.5, -8.0, 200, 'area', 'lusitania lusitaniae lusitani lusitanis lusitanos portugallia portugal portugalia portogallo'.split()),
    ('Muscovy', 56.0, 38.0, 600, 'area', 'moscovia moscoviae moscovitae muscovia muscoviae muscovitae moschi moskau moscovie moscovy russia russland'.split()),
    ('Prussia', 54.0, 20.0, 200, 'area', 'prussia prussiae prussi prussis preussen preußen preussia borussia borussiae prusse prusia'.split()),
    ('Livonia', 57.0, 25.0, 200, 'area', 'livonia livoniae livones livland liefland lifland livonie'.split()),
    ('Lithuania', 55.3, 24.0, 200, 'area', 'lithuania lituania lithuaniae lituani lithuani littau littauen lithuanie litauen'.split()),
    ('Europe', 50.0, 15.0, 1500, 'area', 'europa europae europe europaeus europaea europe'.split()),
    ('Asia', 34.0, 90.0, 3000, 'area', 'asia asiae asie asien'.split()),
    ('Africa', 5.0, 20.0, 3500, 'area', 'africa africae afrique afrika'.split()),
    ('Americas', 5.0, -75.0, 4000, 'area', 'america americae amerique amerika nuevo_mundo novus_orbis indes_occidentales'.split()),
    ('New Spain', 19.4, -99.1, 600, 'area', 'nueva_espana nueva_españa nova_hispania nouvelle_espagne'.split()),
    ('Peru', -9.2, -75.0, 600, 'area', 'peru perú pirú piru'.split()),
    ('Babylon', 32.54, 44.42, 120, 'area', 'babylon babylonia babilonia babylonien babilonie babylone'.split()),
    ('Phoenicia', 33.5, 35.5, 120, 'area', 'phoenicia phoenicien phoenice phoenicie phenicie fenicia phoenizien phonizien'.split()),
    ('Hanseatic League', 53.9, 10.7, 280, 'area',
     'hanse hansa hansae hansea hanseatica hanseaticae hanseaticarum hanseatici hansestadte hansestaedte hansestadten hansestet hansestett '
     'hansastadte hanseatic hanseaten hanseatische hansetownes'.split()),
    ('Alps', 46.6, 10.4, 320, 'area', 'alpen alpes alpi alpibus alpium alps'.split()),
    # v2.4: Hadria / Hadriae in Latin texts are the Adriatic Sea (Horace's 'inquieti Hadriae'); the town of Adria needs its own cue (``latin_homonyms``). The point is the middle of the sea
    ('Adriatic Sea', 42.8, 15.5, 250, 'area', 'mare_hadriaticum maris_hadriatici mari_hadriatico mare_adriaticum maris_adriatici adriatic_sea mer_adriatique adriatisches_meer'.split()),
    # v2.3 (check of v2.2): the Veluwe was a Gelderland hamlet of the name 45 km south; the Holy Land of a voyage was an inland village near Nijmegen
    ('Veluwe', 52.135, 5.750, 25, 'area', 'veluwe veluwen veluuwe veluwezoom'.split()),                       # GeoNames 2745666 (AREA)
    # old French / Latin names of Greek lands and of Brazil, which the French humanist editions use ('la Morée' was a hamlet of the Loir-et-Cher, 'Lacedemone' nothing)
    ('Peloponnese', 37.5, 22.3, 110, 'area', 'moree morea peloponnese peloponnesus peloponneso peloponnesos peloponnes'.split()),
    ('Macedonia', 40.8, 22.0, 120, 'area', 'macedoine macedonia macedoniae macedonien macedonie macedonias'.split()),
    ('Brazil', -10.0, -55.0, 450, 'area', 'bresil brasil brazil brasilia bresils'.split()),
    ('Sparta', 37.074, 22.430, 8, 'city', 'lacedemone lacedemon lacedaemon lacedaemonia lacedaemone lacedaemonem sparta spartae'.split()),
    ('Holy Land', 31.9, 35.2, 120, 'area',
     'heilige_land heiliges_land das_heilige_land terra_sancta terrae_sanctae terram_sanctam terre_sainte tierra_santa terra_santa holy_land palaestina palestina palestine'.split()),
    # ---- regions
    ('Saxony', 51.0, 13.0, 160, 'area', 'saxonia saxoniae saxones saxonibus saxonum sachsen sajonia saxe sassonia'.split()),
    ('Silesia', 51.0, 17.0, 130, 'area', 'silesia silesiae schlesien silesie slesia slask'.split()),
    ('Pomerania', 54.0, 15.8, 120, 'area', 'pomerania pomeraniae pommern pomerellen pomorze'.split()),
    ('Moravia', 49.5, 16.8, 90, 'area', 'moravia moraviae mähren mahren moravie'.split()),
    ('Flanders', 51.0, 3.5, 100, 'area', 'flandria flandriae flandern flandres flandre flandri vlaanderen fiandra flandrie'.split()),
    ('Zeeland', 51.5, 3.8, 50, 'area', 'zelandia zelandiae zeeland zeelandt seeland-nl zelanda'.split()),
    ('Friesland', 53.1, 5.8, 100, 'area', 'frisia frisiae frisii frisiis friesland frise frisland'.split()),
    ('Westphalia', 51.6, 8.0, 100, 'area', 'westphalia westphaliae westfalen westphalen westfalia westphalie'.split()),
    ('Bavaria', 48.9, 11.8, 130, 'area', 'bavaria bavariae baiern bayern baiaria bauaria bauariae baviere baviera'.split()),
    ('Swabia', 48.3, 9.5, 100, 'area', 'suevia sueviae suevi suevis schwaben swabia souabe svevia'.split()),
    ('Franconia', 49.8, 10.5, 90, 'area', 'franconia franconiae franken franconie franconi'.split()),
    ('Tyrol', 47.2, 11.4, 100, 'area', 'tirolis tyrolis tirol tyrol tirolia tyrolia tirolensis'.split()),
    ('Styria', 47.2, 15.0, 100, 'area', 'styria styriae steiermark steir steyermark steyr-land stiria estiria'.split()),
    ('Carinthia', 46.7, 13.9, 80, 'area', 'carinthia carinthiae kärnten karnten kaernten carintia'.split()),
    ('Carniola', 46.1, 14.5, 80, 'area', 'carniola carniolae krain carniole'.split()),
    ('Palatinate', 49.3, 8.3, 70, 'area', 'palatinatus palatinatus-rheni pfalz palatinat palatinatu kurpfalz churpfalz pfaltz chur_pfalz chur_pfaltz kur_pfalz kur_pfaltz churpfaltz'.split()),
    ('Alsace', 48.3, 7.5, 70, 'area', 'alsatia alsatiae elsass elsaß alsace alsazia elsas'.split()),
    ('Lorraine', 48.9, 6.2, 70, 'area', 'lotharingia lotharingiae lothringen lorraine lorena'.split()),
    ('Savoy', 45.6, 6.4, 70, 'area', 'sabaudia sabaudiae savoyen savoie savoia savoye savoy'.split()),
    ('Burgundy', 47.0, 4.8, 120, 'area', 'burgundia burgundiae burgundi burgundis burgund burgundt bourgogne borgogna borgona'.split()),
    ('Brabant', 50.9, 4.7, 60, 'area', 'brabantia brabantiae brabant brabancon braband brabandt'.split()),
    ('Hesse', 50.8, 9.2, 80, 'area', 'hassia hassiae hessen hesse assia hessia'.split()),
    ('Thuringia', 51.0, 11.0, 70, 'area', 'thuringia thuringiae duringen düringen thüringen thuringen turingia thuringe'.split()),
    ('Mecklenburg', 53.7, 12.3, 70, 'area', 'megalopolis megapolis meckelburg mecklenburg mecklenburgia'.split()),
    ('Holstein', 54.1, 9.8, 60, 'area', 'holsatia holsatiae holstein holsten holsace'.split()),
    ('Transylvania', 46.0, 24.5, 130, 'area', 'transylvania transylvaniae siebenburgen siebenbürgen transilvania transilvanie'.split()),
    ('Dalmatia', 43.7, 16.5, 120, 'area', 'dalmatia dalmatiae dalmatien dalmazia dalmatie'.split()),
    ('Sicily', 37.6, 14.0, 150, 'area', 'sicilia siciliae siculi siculis sizilien sicile sicilie'.split()),
    ('Crete', 35.25, 24.9, 130, 'area', 'creta cretae candia kreta crete'.split()),
    ('Cyprus', 35.0, 33.2, 100, 'area', 'cyprus cypri zypern chypre cipro cipri'.split()),
    ('Tuscany', 43.4, 11.1, 100, 'area', 'etruria etruriae tuscia tusciae toscana toskana tosc'.split()),
    ('Lombardy', 45.5, 9.8, 100, 'area', 'lombardia lombardiae longobardia longobardi lombardei lombardie'.split()),
    ('Apulia', 41.0, 16.3, 130, 'area', 'apulia apuliae puglia pouille apulie'.split()),
    ('Calabria', 39.0, 16.4, 100, 'area', 'calabria calabriae calabre kalabrien'.split()),
    # ---- cities: Latin and old forms
    ('Paris', 48.857, 2.352, 15, 'city', 'lutetia lutetiae lutecia parisii parisiorum parisiis parisius parrhisiis paryz pariser'.split()),
    ('Strasbourg', 48.584, 7.751, 12, 'city', 'argentoratum argentorati argentorato argentina argentinae argentoratensis straßburg strassburg strasburg strasburgk strasbourg strazburg'.split()),
    ('Augsburg', 48.371, 10.898, 12, 'city', 'augusta_vindelicorum augustae_vindelicorum augsburg augspurg augspurgk augstburg augusta-vindelicum ausburgum'.split()),
    ('Vienna', 48.208, 16.373, 15, 'city', 'vindobona vindobonae wien wienn vienna viennae wyen'.split()),
    ('Cologne', 50.938, 6.960, 15, 'city', 'colonia_agrippina coloniae_agrippinae colonia-agrippina köln koln coln cöln cologne colonie cullen'.split()),
    ('Mainz', 50.000, 8.271, 10, 'city', 'moguntia moguntiae moguntiacum moguntinus mainz maintz mentz mayence magontia'.split()),
    ('Trier', 49.750, 6.637, 10, 'city', 'treveris trevirorum treviri trier trèves treves augusta_treverorum augusta-treverorum'.split()),
    ('Basel', 47.559, 7.588, 10, 'city', 'basilea basileae basel basle bale'.split()),
    ('Zürich', 47.377, 8.541, 10, 'city', 'tigurum tiguri turicum turici zürich zurich zürych zuerich tigurinus'.split()),
    ('Geneva', 46.204, 6.143, 10, 'city', 'genava genavae geneva genf genff geneve genève ginevra gēnf'.split()),
    # v2.4 (check of v2.3): 'lugdunum lugduni lugdunensis' were Lyon here, and 290 mentions of the Leiden letters of ckcc_epistolarium with them: the bare Latin forms moved to
    # ``latin_homonyms`` (Lyon OR Leiden, decided by the passage and the document). The explicit forms stay: 'Lugdunum Batavorum' is Leiden, 'Lugdunum Segusianorum' Lyon.
    ('Lyon', 45.764, 4.836, 12, 'city', 'lyon lyons lione lugdunum_segusianorum lugduni_segusianorum lugdunum_galliae lugduni_galliae'.split()),
    ('Leiden', 52.160, 4.494, 8, 'city', ('lugdunum_batavorum lugduni_batavorum lugduno_batavorum lugdunum-batavorum lugduni_batav lugduno_batav lugdunum_batav lugduni_bat lugduno_bat '
                                         'lugdunum_bat lugduni_batavae lugdunensis_batavorum leiden leyden leyde').split()),
    ('Copenhagen', 55.676, 12.568, 12, 'city', 'hafnia hafniae kopenhagen kjobenhavn copenhagen copenhague kobenhavn kiøbenhavn'.split()),
    ('Stockholm', 59.329, 18.069, 12, 'city', 'holmia holmiae stockholm stokholm stockholmia'.split()),
    ('Gdańsk', 54.352, 18.646, 12, 'city', 'gedanum gedani dantiscum dantisci danzig dantzig dantzick gdansk gedanensis'.split()),
    ('Kraków', 50.065, 19.945, 12, 'city', 'cracovia cracoviae cracovie krakau krakow cracow cracouia cracaw krakowia'.split()),
    ('Wrocław', 51.108, 17.039, 12, 'city', 'vratislavia vratislaviae wratislavia breslau breßlau bresslau wroclaw breslavia'.split()),
    ('Prague', 50.088, 14.420, 12, 'city', 'praga pragae prag prague praha prage pragensis'.split()),
    ('Leipzig', 51.340, 12.375, 10, 'city', 'lipsia lipsiae leipzig leipsig leipsic leipzigk lipsiensis'.split()),
    ('Wittenberg', 51.866, 12.646, 8, 'city', 'vitemberga vitembergae wittemberga wittenberga wittenberg wittemberg wittenbergk wittebergae witebergae vittemberga vitebergae'.split()),
    ('Magdeburg', 52.131, 11.640, 10, 'city', 'magdeburgum magdeburgi magdeburg magdenburg magdeburgk magdenburgk magdeburgensis parthenopolis'.split()),
    ('Heidelberg', 49.399, 8.672, 8, 'city', 'heidelberga heidelbergae heidelberg heydelberg heidelbergensis'.split()),
    ('Tübingen', 48.521, 9.054, 8, 'city', 'tubinga tubingae tübingen tubingen tuwingen tubingensis'.split()),
    ('Ingolstadt', 48.763, 11.425, 8, 'city', 'ingolstadium ingolstadii ingolstadt ingolstat ingolstatt'.split()),
    ('Nuremberg', 49.452, 11.077, 10, 'city', 'norimberga norimbergae noribergae nuremberg nürnberg nurnberg nurenberg nuernberg norenberga'.split()),
    ('Regensburg', 49.013, 12.101, 10, 'city', 'ratisbona ratisbonae regensburg regenspurg regenspurgk ratisbone ratisbon ratisponae'.split()),
    ('Frankfurt am Main', 50.110, 8.682, 10, 'city', 'francofurtum_ad_moenum francofurti_ad_moenum francofurtum francoforte-sul-meno frankfurt_am_main franckfurt_am_main frankfurt_am_mayn francfort_sur_le_main'.split()),
    ('Frankfurt (Oder)', 52.347, 14.550, 8, 'city', 'francofurtum_ad_viadrum francofurti_ad_viadrum frankfurt-oder'.split()),
    ('Königsberg', 54.710, 20.510, 10, 'city', 'regiomontum regiomonti königsberg konigsberg koningsberg'.split()),
    ('Riga', 56.950, 24.105, 10, 'city', 'riga rigae riga-livoniae'.split()),
    ('Tallinn', 59.437, 24.745, 10, 'city', 'revalia revaliae reval tallinn revel'.split()),
    ('Szczecin', 53.428, 14.553, 10, 'city', 'stetinum stetini stettin stettinum szczecin'.split()),
    ('Rostock', 54.088, 12.140, 8, 'city', 'rostochium rostochii rostock rostoch'.split()),
    ('Lübeck', 53.866, 10.686, 10, 'city', 'lubeca lubecae lübeck lubeck lubecensis lübeckh'.split()),
    ('Hamburg', 53.551, 9.994, 12, 'city', 'hamburgum hamburgi hamburg hamburgh hambourg amburgo hamburgensis'.split()),
    ('Bremen', 53.079, 8.802, 10, 'city', 'brema bremae bremen bremensis'.split()),
    ('Cambridge', 52.205, 0.119, 8, 'city', 'cantabrigia cantabrigiae cantabrigiensis'.split()),
    ('Oxford', 51.752, -1.258, 8, 'city', 'oxonia oxoniae oxonium oxoniensis'.split()),
    ('London', 51.507, -0.128, 15, 'city', 'londinium londinum londini londres londra londen london-town londoniae'.split()),
    ('York', 53.960, -1.081, 8, 'city', 'eboracum eboraci eboracensis'.split()),
    ('Canterbury', 51.279, 1.080, 8, 'city', 'cantuaria cantuariae cantuariensis durovernum'.split()),
    ('Edinburgh', 55.953, -3.189, 10, 'city', 'edinburgum edinburgi edenburgh edinburgh edimbourg edimburgo edinburg'.split()),
    ('Dublin', 53.350, -6.260, 10, 'city', 'dublinum dublini dubliniae eblana'.split()),
    ('Venice', 45.440, 12.316, 12, 'city', 'venetiae venetiarum venetia venetiis venedig venetzia venise venezia venecia venetijs'.split()),
    ('Padua', 45.407, 11.877, 10, 'city', 'patavium patavii patavina padua padoue padova pàdua'.split()),
    ('Bologna', 44.494, 11.343, 10, 'city', 'bononia bononiae bononiensis bologna bolonia bologne'.split()),
    ('Florence', 43.770, 11.256, 10, 'city', 'florentia florentiae florenz florence firenze fiorenza florencia'.split()),
    ('Milan', 45.464, 9.190, 12, 'city', 'mediolanum mediolani mediolanensis mailand milan milano milán meiland'.split()),
    ('Turin', 45.070, 7.687, 10, 'city', 'augusta_taurinorum taurinum taurini turin torino turino'.split()),
    ('Genoa', 44.407, 8.934, 10, 'city', 'genua genuae genova gênes genes genoa genua-liguriae gennes'.split()),
    ('Naples', 40.852, 14.268, 12, 'city', 'neapolis neapolitanus neapel naples napoli nápoles napels'.split()),
    ('Rome', 41.903, 12.496, 15, 'city', 'roma romae romam rhomae rhoma rom rome roma-caput-mundi romanus-urbs urbs'.split()),
    ('Pisa', 43.716, 10.402, 8, 'city', 'pisae pisarum pisa pise pisani'.split()),
    ('Siena', 43.318, 11.331, 8, 'city', 'sena senae siena sienne senensis sena-iulia'.split()),
    ('Ferrara', 44.836, 11.619, 8, 'city', 'ferraria ferrariae ferrara ferrare ferrariensis'.split()),
    ('Mantua', 45.156, 10.791, 8, 'city', 'mantua mantuae mantova mantoue mantue'.split()),
    ('Pavia', 45.185, 9.155, 8, 'city', 'ticinum ticini papia papiae pavia pavie papien'.split()),
    ('Trento', 46.066, 11.121, 8, 'city', 'tridentum tridenti tridentinus trient trento trente tridentini'.split()),
    ('Konstanz', 47.663, 9.175, 8, 'city', 'constantia constantiae konstanz costnitz costentz constance constanz costantz'.split()),
    ('Worms', 49.633, 8.359, 8, 'city', 'vormatia vormatiae wormatia wormatiae worms wormbs'.split()),
    ('Speyer', 49.318, 8.431, 8, 'city', 'spira spirae spiera speyer speier spier espire spiers'.split()),
    ('Bamberg', 49.892, 10.887, 8, 'city', 'bambergae bambergensis bamberga bamberg bamberc'.split()),
    ('Würzburg', 49.794, 9.929, 8, 'city', 'herbipolis herbipoli wirceburgum würzburg wurzburg wirtzburg wirzburg'.split()),
    ('Antwerp', 51.219, 4.402, 10, 'city', 'antverpia antverpiae antwerpen anvers antwerp anversa antorf antdorff'.split()),
    ('Brussels', 50.850, 4.352, 10, 'city', 'bruxellae bruxellensis brussel bruxelles brussels bruselas brüssel'.split()),
    ('Leuven', 50.879, 4.701, 8, 'city', 'lovanium lovanii lovaniensis loven löwen löven leuven louvain lovaina lowen'.split()),
    ('Ghent', 51.054, 3.725, 8, 'city', 'gandavum gandavi gent gand gante gendt'.split()),
    ('Bruges', 51.209, 3.225, 8, 'city', 'brugae brugarum brugge bruges brujas'.split()),
    ('Amsterdam', 52.373, 4.890, 10, 'city', 'amstelodamum amstelodami amstelredamum amstelredam amsterdam amsterdamm amstelodamensis'.split()),
    ('The Hague', 52.078, 4.311, 8, 'city', 'haga_comitis hagae_comitis haga-comitum hagae hague den_haag la_haye s-gravenhage gravenhage haia'.split()),
    ('Utrecht', 52.091, 5.122, 8, 'city', ('trajectum_ad_rhenum traiectum_ad_rhenum traiecti_ad_rhenum trajecti_ad_rhenum ultraiectum ultraiecti ultraiecto ultrajectum ultrajecti '
                                          'trajectum-inferius traiectum_inferius utrecht uytrecht').split()),
    ('Maastricht', 50.851, 5.690, 8, 'city', 'traiectum_ad_mosam traiecti_ad_mosam trajectum_ad_mosam trajecti_ad_mosam traiectum_superius trajectum_superius maestricht'.split()),
    ('Franeker', 53.187, 5.543, 6, 'city', 'franequera franequerae franeker'.split()),
    ('Rotterdam', 51.924, 4.478, 8, 'city', 'roterodamum roterodami rotterdam roterdam'.split()),
    ('Dordrecht', 51.813, 4.690, 8, 'city', 'dordracum dordraci dordrecht dort dordrech'.split()),
    ('Lisbon', 38.722, -9.139, 12, 'city', 'olisipo olisiponis ulyssippo lisbona lisboa lisbonne lissabon lisbon lisbone'.split()),
    ('Madrid', 40.417, -3.704, 12, 'city', 'matritum matriti'.split()),
    ('Seville', 37.389, -5.984, 10, 'city', 'hispalis hispali sevilla sevilha siviglia séville seville sibilia sibille'.split()),
    ('Toledo', 39.862, -4.027, 8, 'city', 'toletum toleti'.split()),
    ('Salamanca', 40.970, -5.664, 8, 'city', 'salmantica salmanticae salamanca salmanticensis'.split()),
    ('Alcalá de Henares', 40.482, -3.364, 8, 'city', 'complutum complutensis alcala alcalá alcala-de-henares'.split()),
    ('Barcelona', 41.386, 2.170, 12, 'city', 'barcino barcinonis barcinone barcelona barcelone'.split()),
    ('Valencia', 39.470, -0.376, 10, 'city', 'valentia_edetanorum'.split()),
    ('Córdoba', 37.888, -4.779, 8, 'city', 'corduba cordubae'.split()),
    ('Zaragoza', 41.649, -0.888, 10, 'city', 'caesaraugusta caesaraugustae saragossa zaragoza saragosse'.split()),
    ('Coimbra', 40.211, -8.429, 8, 'city', 'conimbriga conimbrica conimbricae conimbricensis coimbra'.split()),
    ('Jerusalem', 31.768, 35.214, 10, 'city', 'hierosolyma hierosolymae hierosolymitanus jerusalem ierusalem hierusalem jerusalén gerusalemme jerusalem-ierusalem'.split()),
    ('Istanbul', 41.008, 28.978, 15, 'city', 'constantinopolis constantinopoli byzantium byzantii konstantinopel constantinople constantinopla istanbul stambul stanbul'.split()),
    ('Athens', 37.984, 23.728, 10, 'city', 'athenae athenarum athenis athenas athens athènes atene atenas athen'.split()),
    ('Alexandria', 31.200, 29.919, 10, 'city', 'alexandria alexandriae alexandrie alessandria alexandrien'.split()),
    ('Carthage', 36.853, 10.323, 10, 'city', 'carthago carthaginis carthage cartagine karthago'.split()),
    ('Troy', 39.957, 26.239, 10, 'city', 'ilium ilii troia troiae troy troja troie'.split()),
    ('Syracuse', 37.075, 15.287, 8, 'city', 'syracusae syracusarum siracusa syracuse syrakus'.split()),
    ('Corinth', 37.906, 22.879, 10, 'city', 'corinthus corinthi corinth korinth corinthe'.split()),
    ('İznik', 40.429, 29.721, 10, 'city', 'nicaea nicaeae nicea nicaenus nicaena nicäa nizäa iznik'.split()),
    ('Lima', -12.046, -77.043, 12, 'city', 'lima ciudad_de_los_reyes'.split()),
    ('Mexico City', 19.433, -99.133, 15, 'city', 'mexico mexico-tenochtitlan tenuxtitlan tenochtitlan mexico-city ciudad_de_mexico'.split()),
    ('Kamień Pomorski', 53.969, 14.771, 8, 'city', 'camin cammin kamien-pomorski kamien pomorski caminum'.split()),
    ('Eichstätt', 48.892, 11.186, 8, 'city', 'eystett eystet eichstaett eichstätt eichstadium eystadium'.split()),
    ('Donauwörth', 48.714, 10.774, 8, 'city', 'donawert donauwerth donauwörth donawörth donauverda'.split()),
    ('Passau', 48.575, 13.463, 8, 'city', 'passavium passavii passau bossau'.split()),
    ('Salzburg', 47.800, 13.045, 8, 'city', 'iuvavum salisburgum salisburgi salzburg saltzburg'.split()),
    ('Thermopylae', 38.797, 22.537, 8, 'city',
     'thermopylae thermopylai thermopylis thermopylarum thermopylas thermopyles thermopylen thermopolae pas_des_thermopyles destroict_des_thermopyles '
     'passage_des_thermopyles'.split()),
    ('Windsor', 51.482, -0.604, 6, 'city', 'windesore windesor windlesore windlesora windleshora windsore wyndsor wyndesore windeshore'.split()),
    ('Vlissingen', 51.4425, 3.5736, 8, 'city', 'flushing flessingen flissingen flissinga'.split()),             # GeoNames 2745392; Old Bailey: 'Flushing, a Town in Holland'
    ('Nowe nad Wisłą', 53.143, 18.740, 8, 'city', 'nowe nowe-nad-wisla castri_novensis novam_civitatem neuenburg-westpr'.split()),
]
# fmt: on


# plain native / modern names are the business of the GeoNames lookup and of propagation, not of this table
_PLAIN = frozenset(_norm(x) for x in '''wien köln koln nürnberg nurnberg nuernberg praha firenze milano napoli genova venezia torino padova
mantova lisboa sevilla kobenhavn kjobenhavn kiøbenhavn antwerpen gent brugge kraków krakow hamburg bremen leipzig heidelberg
trier mainz lyon basel zürich zurich riga rostock oxford london lima'''.split())
_KEEP = frozenset({'lima', 'thermopylae'})        # bdpn: Lima is unplaced in v1.1 (homonyms), the table decides; Thermopylae is a pass, GeoNames has a village of the name in France


def _build():
    table: dict = {}
    for label, lat, lon, rad, kind, forms in _RAW:
        nl = _norm(label)
        for f in forms:
            k = _norm(small_letters(f).replace('_', ' ').replace('-', ' '))
            if k not in _KEEP and ((kind == 'city' and k == nl) or k in _PLAIN):
                continue
            if len(k) >= 3 and k not in table:
                table[k] = (lat, lon, float(rad), label, kind)
    return table


TABLE = _build()


# v2.2: forms whose meaning depends on the edition. 'Savoy' is the duchy for the French, German and Latin editions (the general table) and the Savoy
# liberty, hospital and palace of the Strand in the London editions: MoEML's "Savoy Manor" was placed in Savoie, Old Bailey's "Savoy" (8 mentions) too.
LONDON_COLLECTIONS = ('map_of_early_modern_london', 'old_bailey', 'reed')
_BY_COLL_RAW = [
    (LONDON_COLLECTIONS, 'Savoy (London)', 51.5104, -0.1205, 1, 'city',
     'savoy the_savoy savoy_manor savoy_hospital savoy_palace savoy_precinct savoy_liberty savoy_prison savoy_chapel'.split()),
]
#: v2.3: Swiss editions' old French spelling of Neuchâtel ('Neufchastel': 729 mentions in the Swiss law sources); elsewhere it can be Neufchâtel-en-Bray or -Hardelot
SWISS_COLLECTIONS = ('ssrq_swisslaw', 'bullinger', 'st_galler_missiven', 'urfehdebuecher_basel_x', 'koenigsfelden')
_BY_COLL_RAW.append((SWISS_COLLECTIONS, 'Neuchâtel', 46.9899, 6.9293, 8, 'city', 'neufchastel neufchatel neuchastel'.split()))
BY_COLL: dict = {}
for _colls, _label, _lat, _lon, _rad, _kind, _forms in _BY_COLL_RAW:
    for _f in _forms:
        _k = _norm(small_letters(_f).replace('_', ' ').replace('-', ' '))
        for _c in _colls:
            BY_COLL.setdefault(_c, {}).setdefault(_k, (_lat, _lon, float(_rad), _label, _kind))

# kind of the places the table describes, when it is not a settlement or a country / region (the globe's five-way kind): label -> c a n b u
FEATURE_KIND = {'Adriatic Sea': 'n', 'Alps': 'n', 'Thermopylae': 'n', 'Hanseatic League': 'a', 'Savoy (London)': 'c', 'Windsor': 'c', 'Veluwe': 'a', 'Holy Land': 'a', 'Peloponnese': 'a', 'Macedonia': 'a', 'Neuchâtel': 'c'}


def lookup(key: str, coll: str = None):
    """Normalised key -> (lat, lon, radius_km, label, kind) or None. ``coll``: the edition's own forms (Savoy in London) come first."""
    if coll is not None:
        h = BY_COLL.get(coll, {}).get(key)
        if h is not None:
            return h
    return TABLE.get(key)
