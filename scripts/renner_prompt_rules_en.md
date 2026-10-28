Condensed RenNER rules (from guide_annotation_en.md v0.1; no example taken from gold v2 or dev3).

Annotate PROPER NAMES only, in context, every occurrence, in the exact spelling of the text (old spelling, OCR errors, split words such as "Fran conia" are copied as written). Never translate, modernise or add names that are not in the text. Case does not decide: a lower-case name is still a name; a capitalised common noun is not.

LOC = a place: town, village, region, country, kingdom, duchy, province, island, river, sea, mountain, continent, ancient place, named street, gate, bridge, church or convent used as a place marker ("in Santa Croce", "a San Lorenzo").
- A place inside a TITLE or office IS a place: "re di [Francia]", "duca di [Milano]", "rey de [Castilla]", "obispo de [Cuenca]", "episcopus [Cracoviensis]", "el de [Saboya]" (= the duke of Savoy). Annotate only the place, not the title.
- A place inside an EVENT is a place: "Concilio di [Trento]", "batalla de [Pavía]".
- Latin toponymic adjectives in -ensis/-ense are places: "canonicus [Posnaniensis]", "in academia [Lovaniensi]", "synodo [Varsoviensi]". Other adjectives (Romanus, Gallicus, Hispano, veneziano, Austriacam) are NOT places.
- A place inside a PERSON'S NAME is NOT a place: "Lodovico da Ferrara", "Petrus ex Gandavo", "Juan de Ávila", "Mathias a Kalisz" are one PER each, with no LOC inside.
  Test: name + da/di/de/von/of/ex/a + place, with no title word between = origin name (all PER). Title word (re, duca, rey, conde, obispo, episcopus, dominus of a see) + place = the place is LOC.
- NOT places: demonyms and peoples (los Moros, i Veneziani, Galli, Teutones), languages, the sun, moon, planets, stars, constellations, "Cielo", Heaven, Hell, Paradise, biblical books, "la Corte", "Palacio", "il Palazzo", a country name that stands for its ruler as a single person, generic nouns without a name (la città, el río, the island).
- Generic words: leave out "città di", "ciudad de", "urbs", "isla de", "mare" when the name exists alone; keep them when they are part of the name ("Monte Cassino", "Rio Grande"). Keep "San/Santa/S." in church and place names ("San Martino").
- Articles stay outside ("il [Friuli]"), except fixed names ("La Rochelle").

PER = a person: real, fictional, biblical, mythological or ancient individuals, saints, authors cited as authorities ("ut ait [Plinius]", "come dice [Plutarco]"), speaker labels in dialogues and plays, a nickname used as a name.
- The PER span covers the FULL NAME, with surname, patronymic, origin name, byname, epithet and numeral: "Lodovico da Ferrara", "Francisco de Padilla", "Lorenzo il Magnifico", "Filippo IV", "Carlos Quinto", "Alexander Magnus".
- Titles, offices and forms of address stay OUTSIDE the span: don, doña, fray, señor, messer, signor, sier, dominus, D., Dr, papa, re, duca, conde, cardinal, il Generale. "don [Fernando Páez]", "messer [Andrea Doria]", "il Generale [Tilly]".
  Exception: a title fixed in a fictional name ("Don Quijote").
- "San/Santo/Santa/S./Sanctus" before a saint is part of the span ("San Giovanni", "S. Agostino").
- NOT persons: titles or offices without a name (el Rey, il Papa, Su Majestad, S. A. R., Sua Altezza, Caesar as a title), God (Dios, Deus, Dio, Iddio, Dominus), personified virtues, animals, peoples, adjectives from names (ciceroniano, Lutheranus), titles of works.

Spans are flat: one label per span, never nested, never overlapping. Each name in a list is its own span. Do not include final punctuation, but keep an abbreviation dot that is part of the name ("Gio. Sinibaldo").
