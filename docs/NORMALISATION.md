# Normalisation of place types and names (v2.4 preparation)

Status: part of the v2.4 build (`slurm/cards_globe_v24.sbatch` runs `python -m ratlas.name_variants extra` and `python -m ratlas.placetypes inventory` after `cards_globe build`; CONTRACT section 20.4). Numbers below are from the v2.3 tables
(`RQ/derived/atlas/v2.3`, 474,076 placed in-period mentions of 28,629 places on 14,798 globe points), analysis read-only on the Mac.

## 1. What was added

| File | Role |
|---|---|
| `src/ratlas/placetypes.py` | the vocabulary (`TYPES`), the mappings from each source's own terms (`wd_type`, `gn_type`, `tgn_type`, `orbis_type`, `register_type`, `qualifier_type`, `name_type`), the weighted vote `classify`, the analysis (`inventory`) |
| `data/placetypes.tsv` | the **reviewed** mapping: one row per source term, with counts, the type, `status` (`reviewed` = read by a human against the counts, `auto` = the rule's answer, not yet read), `note` |
| `src/ratlas/name_variants.py` | folded key, variants, reference form, `build_extra` (writes `extra.json` and a per-global-place TSV) |
| `tests/test_placetypes.py`, `tests/test_name_variants.py` | unit tests and a toy-table test of `build_extra` |

Nothing in `core.json` changes. `extra.json` sits next to it: `{"types": {code: [label_en, parent_code, coarse_kind]}, "rows": {"<row index>": [type_code, [[variant, lang, n], ... top 10], name_based_mentions]}}`;
the row index is the position in `core.json["rows"]` (= `globe_places.json`).

## 2. One vocabulary, 54 codes

Five groups, each a valid but *generic* type ("we know it is a settlement, not which sort"), and 49 leaves. The coarse kind (`c a n b u`) is a property of the type, so today's five-way kind is
derived from the fine type and cannot contradict it.

| Group (kind) | Leaves |
|---|---|
| `area` (a) | continent, country, kingdom, empire, republic, duchy, county (countship), landgraviate, margraviate, principality, electorate, bishopric, canton, lordship (Herrschaft), bailiwick (Amt, Vogtei), district (contado, Kreis, hundred), shire (English/Scottish county), province, region |
| `settlement` (c) | city, town, market_town, village, hamlet, city_quarter, parish |
| `natural` (n) | river, lake, sea, bay, island, mountain, pass, valley, forest, cape, marsh |
| `built` (b) | church, monastery, castle, palace, inn, street, square, gate, bridge, fort, building, site |
| `other` (u) | other |

Each type carries a Wikidata class QID (`PlaceType.qid`) where a sound one exists (margraviate and the generic natural group have none verified; left empty).

## 3. Mapping rules

Every source gives a function returning a type code or `None` (no information; never a guess):

* **Wikidata P31** (`wd_type(qid, label)`): the reviewed row of the QID first, else rules on the English class label (head noun: "island country" is a country, "river island" an island;
  phrases such as "free imperial city" -> city; "X of Switzerland" is read as X). Classes that say nothing ("tourist destination", "financial center") are `-` (reviewed, no type).
  No subclass walk: the local tables hold only the direct P31 classes of each item (the label rules do the work a subclass chain would do; 988 classes, 439 with 20+ mentions, all read).
* **GeoNames feature code** (`gn_type(code, pop)`): table of ~400 codes. A populated place is `city` for PPLC/PPLA/PPLA2 and from 50,000 inhabitants, `town` from 5,000, `village` below 1,000 (today's population,
  a proxy), else generic `settlement`, because the administrative level (PPLA3, PPLA4) says little about size in 1500.
* **Getty TGN place type** (`tgn_type`): table of ~80 types, falling back on the label rules.
* **Orbis Latinus** (`orbis_type`): the book's abbreviations (St., Mfl., D., Fl., Gfsch., Hschf. ...) and the parser's normalised `place_type`.
* **Qualifier word after a comma** (`qualifier_word`): Königreich, Herzogtum, Grafschaft, Landgrafschaft, Markgrafschaft, Fürstentum, Kurfürstentum, Herrschaft, Amt, Landvogtei, Hochstift, Kanton, contado,
  ducato, comté, bailliage, seigneurie, shire ... in German, French, Italian, Spanish, Latin and English forms. Weighted highest (1.2): an editor wrote the type next to the name.
* **Words of the name** (`name_word`): whole words only (Kloster, Schloss, Cathedral, Street, Bridge, Palace ...); "Burg"/"Berg"/"Bach" inside a name are the name (Hamburg, Offenbach), "Inn" after a German
  preposition is the river; "Monte Rosa" is a mountain, a bare "Berg" is not.
* **Editions' register types** (`register_type`): the `note_type` values actually present (`place_f_church`, `place_pa_parish`, `herrschaft, kastlanei`, `flurname`, `klarissenkloster` ...), lists read left to right,
  technical labels (`place`, `editor_slug`, `from_text_forms`) are `None`. The Casebooks directory `entry_type` (parish, village, hamlet, manor) has its own row set (source `casebooks`).
* **English item label** (`qualifier_type_of_label`): "Duchy of Savoy", "Prince-Bishopric of Liège" vote for their type.

`classify(Evidence)` is a weighted vote (wd 1.0, gn 1.0, qualifier 1.2, label 0.9, tgn/register/casebooks 0.8, orbis 0.7, name 0.6, inherited 0.5; a generic vote counts 0.3 and lifts the specific types of
its group). **The atlas's coarse kind restricts the vote**: the fine type is chosen among the types of that kind; with no vote of the right kind the answer is the group ("settlement"). The evidence of all
the local places of one globe point is pooled before voting. `data/placetypes.tsv` overrides the rules row by row (the file is the authority; the rules supply first drafts for new terms).

## 4. Coverage on v2.3

Table: `/private/tmp/claude-501/-Users-clem-Projects-Papers-Renascor/e0c9bdc2-9edc-46d5-8bde-77513f9fb822/scratchpad/atlas/v24prep/placetype_coverage.tsv` (sections `summary`, `type`, `term`).

| Measure (placed in-period mentions, n = 474,076) | Mentions | Share |
|---|---|---|
| fine type other than `other` (generic group counted) | 474,021 | 99.99% |
| a **leaf** type (not `area`/`settlement`/`natural`/`built`) | 429,018 | 90.5% |
| generic group only | 45,003 | 9.5% |
| type decided by at least one source term (not by the atlas kind alone) | 450,932 | 95.1% |
| free vote had another coarse kind than the atlas kind (fine type kept inside the atlas kind) | 12,683 | 2.7% |

The first line is nearly trivial (the kind alone gives a group); the second and fourth are the honest ones. Leaf share by coordinate source: exonym table 98.3%, propagation 96.3%, Wikidata 97.1%,
register 93.1%, GeoNames id 91.5%, GeoNames name match 68.7%, Casebooks 61.3% (parish/settlement only: the directory gives no size).

Source terms: Wikidata classes 988 (439 with 20+ mentions), GeoNames codes 126 (52), TGN types 68 (23), register types 106 (37), qualifier words 59 (40), name words 96 (45), Casebooks 10 (6), Orbis 9 (4).
**Every term with 20 or more placed mentions has a reviewed row**, and the table has no `auto` row at or above that threshold (tested). A new build is checked with `python -m ratlas.placetypes inventory --final DIR --out coverage.tsv` (exit status 2 and a list when a term with 20+ mentions has no row).

Largest types by mention: city 211,614; country 78,014; town 52,962; settlement (generic) 42,695; village 18,970; region 14,827; province 8,144; parish 6,760; city_quarter 6,685; river 4,661.

## 5. Names

* **Folded key** (`fold_key`): lower case, "ß" -> ss, ligatures, diacritics removed, ae/oe/ue -> a/o/u, u=v, i=j, y=i, punctuation to space. It is for matching **inside one place** only (`group_variants` is called
  with the forms of one place): two places whose names fold together stay two places. Measured on the 63,084 surface forms of the 14,798 points: 2,606 pairs of forms of one point merge, 137 pairs of forms of
  different points would collide (19:1); w=u would add 54 inside and 54 across (1:1), so it is not done.
* **Variants** (`group_variants`): forms with the same key merge into one entry (most frequent spelling shown; accented spelling wins a tie); language = most frequent known language of the merged mentions, taken from the
  edition's dominant language (>= 80% of >= 50 tagged mentions) where a mention has none (`Variant.n_inferred_lang` says how many: such a language is the edition's, not the word's; "Gefell" can show `fr` in a French edition).
  Common nouns and deictics ("alhier", "this towne") are dropped. `extra.json` holds the top 10.
* **Reference form** (`reference_form`): English Wikidata label, else the register's name, else the exonym table's English label, else the most frequent variant. Bot labels ("lsjbot ...") and common nouns never count.
  On the 59,786 global places: Wikidata 5,959, register 23,288, exonym 753, surface 29,143, none 643 (all forms unusable). 19% of global places have more than one variant.
* **Name-based mentions** (fix 5 of the v2.3 check): mentions whose local place was located by name (`name_propagation`, `name_geonames`, `name_exonym`, `name_latin`): 137,425 of 474,076 (29.0%). This is counted on the mention's own
  place, so Latin names that the globe files under Wikidata or the register are included; the row's source code alone would show 29,094.

## 6. Known limits and open questions

1. **Type follows the atlas kind.** Where the kind is wrong the fine type is forced inside it: "Walbrook River" (kind b) is a street, "Stepney" (kind b) a church, "Zepzig" (a GeoNames farm, FRM) a building. These are errors of kind or placement, visible now, to fix there.
2. **Several valid types for one place.** Glarus, Thurgau, Savoyen (canton / province / landgraviate / district / region) differ with the source and the date. The vote returns one; the page may want to show the type "at the time" or the strongest source.
   GeoNames ADM1 -> `province` is generic and may deserve a lower weight against a specific Wikidata class (canton).
3. **City vs town.** Size from today's population (50,000 / 5,000) is a proxy, and the 1500 reality differs (Bernburg 10,002 mentions is a town, London and Zürich cities). Do we want a size class at all, or only "settlement"?
4. **PPLA / PPLA2 -> city** is a choice (seats of first/second order divisions); PPLA3 and below are generic.
5. **`parish`** is a settlement type here (English parishes named after their village, Casebooks); decide if it should be an area.
6. **`site` and `building`** are catch-alls for London detail (wharves, halls, hospitals); a finer split is possible but small (2,651 mentions together).
7. `shire` vs `county`: a separate type for the British Isles (353 mentions) keeps "county" for Grafschaft; `county` of Flanders (359+) and `Thurgau` as landgraviate show the cost of any such choice.
8. Wikidata subclass walking is not available locally (direct P31 only); if a full class hierarchy is added to the tables, `wd_label_type` can be replaced by an ancestor lookup, `data/placetypes.tsv` stays the override.
9. `status = reviewed` means the build agent read the row against the counts (with the labels and the names of the places behind it); it is not yet a read by the project owner. Read `data/placetypes.tsv` (sorted by mentions) before the types appear on the page.
10. **To do (owner request 2026-10-21, not started): one historical entity per polity.** The same country is shown under several names and points,
    some of them anachronistic. Russia in v2.4: "Muscovy" (rows 161 and 662, exonym table, 428 + 77 mentions), "Russiae" (264, Latin name, 261),
    "Russia" (1600, Wikidata, 22; 5487, 3), "Russy" (1742, 20) and "Russian Federation" (10710, 1). Wanted: one point per historical polity with a
    period-appropriate English name (e.g. Muscovy / Tsardom of Russia, not Russian Federation), all spellings attached as variants, and modern states
    (Q-ids of present-day countries) never used as the display name for 1450-1700. Same sweep needed for the other countries (Prussia / Duchy /
    Kingdom of Prussia, Holy Roman Empire vs Germany, Poland-Lithuania, Ottoman Empire vs Turkey, Netherlands, Italy...).
