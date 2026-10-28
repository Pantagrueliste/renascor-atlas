# Extraction contract (Renascor Atlas v1, with the v2 additions of section 13)

Audience: agents writing collection modules in parallel. Read this once, then work only in your own files.

## 0. Ground rules

* Add only `src/ratlas/collections/<slug>.py` and `tests/test_<slug>.py`. A new helper file
  `src/ratlas/helpers_<label>.py` is allowed. **Never edit** `model.py`, `common.py`, `runner.py`,
  `collections/__init__.py`, `collections/generic_tei.py`, `__main__.py`, `scripts/`, `slurm/`, `pyproject.toml`.
  If you need a change there, list it in your report ("common.py changes wanted") and work around it in your module.
* Mac: read files, edit code, run unit tests and tiny checks (`nice -n 19`, one process, under ~10 s; e.g.
  `ratlas extract <slug> --limit 5 --out <scratch>`). Anything bigger runs on Hypatia through Slurm.
* Read-only on `/Users/clem/Projects/Papers/RQ` (write only under `RQ/derived/atlas/`). Python that reads harvested
  files runs as `python -I` (`.venv/bin/python -I ...`). Nothing from the harvest is executed. Scratch only under
  `/private/tmp/claude-501/-Users-clem-Projects-Papers-Renascor/e0c9bdc2-9edc-46d5-8bde-77513f9fb822/scratchpad/atlas/<label>/`.
* No commits, pushes or publication. No network unless your stage says so.
* London Lives and the Medici Archive Project are excluded (the code refuses them). Do not work around that.
* Worked example to copy: `src/ratlas/collections/grand_tour.py` and `tests/test_grand_tour.py`.

## 1. Inputs

* `RQ/data/raw` (read-only), full XML inventory `RQ/docs/etudes_2026-10-16/scan.jsonl.gz`: one record per XML file
  (`path` relative to raw, `!` separates archive and member; `coll`; `sha1`; `dup`; `c` counts `zone|tag` with zones
  `hdr`/`txt`/`reg`; `refs`; `dates`; `langs`; `words`). Assessments: `colls.json` (what the tags mark, how refs
  resolve, register paths, licence, problems), `etude_bases_derivees.md`. Tiers: `RQ/release/text_archive_v1.1/LICENSE_TIERS.tsv`.
* The labels of `colls.json` are descriptions, not scan keys. Find the scan selector with
  `.venv/bin/python -I -m ratlas inspect <prefix>` (prints the coll keys, archives, tag counts, sample paths).
  A selector in `COLLS` matches a record when it equals the record's `coll` or is a string prefix of its `path`.
  Examples: `eng_dh/casebooks`, `bullinger`, `originals_2026-10-15/oct-binary/oct__ferdinand-i-familienkorrespondenz`,
  `originals_2026-10-15/repository-bulk-downloads/CKCC_Epistolarium`.
* Duplicates (`dup=true`) are skipped by default. One module per slug, one slug per output directory.

## 2. Module API (`src/ratlas/collections/<slug>.py`)

Required module constants:

| name | meaning |
|---|---|
| `SLUG` | `[a-z0-9_]+`, equal to the file name; used for `local_key` and `out/<slug>/` |
| `COLLS` | list of selectors (see above) |
| `RENASCOR_IDS` | catalogue ids (`Renascor/data/entries/<id>.json`); joined by `;` in `renascor_id` |
| `LICENCE` | SPDX-ish string, from the entry / LICENSE_TIERS / TEI headers (say which in a comment) |
| `TIER` | `'T1'`, `'T2'` or `'T3'` (LICENSE_TIERS) |

Optional: `DESCRIPTION`, `OPTIONS` (dict), `EXTRACTOR`, and module-level `iter_files`, `prepare`, `extract`.

* `OPTIONS` keys: `default_years=(from, to)` (used with `date_source='collection'` when a file has no date),
  `filename_dates=True` (try a year in the file name when the header has none), `opaque_keys=True`
  (see 4.4), `type_map={'tag-value': 'person|place|org'}` (extra `rs`/`name` `@type` values); any other key is
  free for your module and appears in `ctx.options`.
* `EXTRACTOR = MyExtractor()` where `MyExtractor` subclasses `collections.generic_tei.GenericTEI`. With no
  `EXTRACTOR` and no functions the plain `GenericTEI` is used (constants only is a valid module).
* Hooks of `GenericTEI` (override what you need; all take `ctx: common.Context`):
  `iter_files(ctx, include_dups=False) -> [FileRecord]`, `is_register_file(rec, ctx)`,
  `parse_register_file(rec, ctx, root) -> [(Entity, aliases)]`, `prepare(ctx) -> [Entity]`,
  `doc_date(root, rec, ctx) -> DocDate`, `scan(root, rec, ctx, counters) -> [Hit]`,
  `adjust(mention, hit) -> Mention | None` (per-mention fix-ups), `extract_root(root, rec, ctx)`, `extract(rec, ctx)`.
* Module-level functions override the extractor: `iter_files(ctx, include_dups=False)`, `prepare(ctx) -> [Entity]`,
  `extract(rec, ctx) -> (list[Mention], list[Entity])`.
* Lifecycle (runner): build `ctx` from the constants -> `iter_files` -> `prepare(ctx)` (loads shared registers into
  `ctx.shared`, returns their entities) -> for each file `extract(rec, ctx)` -> `Writer`. `ctx.all_files` holds all
  records (not truncated by `--limit`). `extract` must be a pure function of `(rec, ctx)` (workers are forked after
  `prepare`; state set in `extract` is lost). Raise on an unreadable file: the runner records it in `stats.json` and
  continues. `rec.read()` gives the bytes; `common.read_raw('archive.zip!member')` reads any other file (JSON, CSV).
* Default register detection (`is_register_file`): >= 5 `reg|person/place/org` entries and no tagged entities in
  the text. Override it if registers are named differently or per-document (e.g. Casebooks case headers).
  Register files return `([], [])` from `extract` (their entities come from `prepare`).

## 3. Output tables (exact schemas in `src/ratlas/model.py`)

### Mention (`out/<slug>/mentions.parquet`)

| field | type | rule |
|---|---|---|
| `coll_slug` | str | `SLUG` (the writer enforces it) |
| `renascor_id` | str? | `;`-joined catalogue ids; if a collection covers several entries, set the right one per file in `adjust` |
| `source_path` | str | `rec.path` (relative to raw, `!` for members) |
| `doc_id` | str | root `@xml:id`, else file stem (`common.doc_id_of`) |
| `doc_year`, `doc_year_from`, `doc_year_to` | int? | see 6; `doc_year` = the exact year, else the midpoint of the range |
| `date_source` | str? | `header`, `filename`, `collection`, `letter` (correspAction / dateline / opener), or `text` (dates in the running text; our addition for diaries and itineraries); None if undated |
| `in_period` | bool? | `[from, to]` overlaps 1450-1700; **None when undated** (do not guess) |
| `kind` | str | `person`, `place`, `org` (geogName, settlement, country... are `place`) |
| `tag` | str | `persName`, `placeName`, `orgName`, `geogName`, `rs@place`, `name@person`... (`rs`/`name` carry the `@type` value, lower-cased) |
| `role` | str | see 5 |
| `zone` | str | `txt` (running text, also back matter notes) or `hdr` (teiHeader) |
| `surface` | str | whitespace-normalised text of the tag (diplomatic reading: `abbr`/`sic`/`orig` kept, `expan`/`corr`/`reg` dropped when an alternative exists) |
| `ref_raw` | str? | the raw `@ref @key @corresp @sameAs` values, space-joined (kept even when unresolved) |
| `local_key` | str? | see 4; `'<slug>:<id>'` (shared register) or `'<slug>:<doc_id>#<id>'` (register inside the document) |
| `in_register` | bool? | True if `local_key` points to an entity written to `entities.parquet`; False for an opaque key (4.4); None if no key |
| `auth` | struct | `gnd viaf wikidata geonames tgn ulan pleiades hls` (str or null) + `other` (list of `'scheme:id'`). Ids on the tag itself **plus** the ids of the resolved entity |
| `context` | str? | <= 120 chars around the tag, **only for T1/T2** (the writer drops it for T3 and truncates) |
| `lang` | str? | nearest `xml:lang`, else the document language |
| `licence`, `tier` | str | from the module constants (the writer fills them) |

### Entity (`out/<slug>/entities.parquet`) one per register entry, edition-scoped

`local_key`, `coll_slug`, `renascor_id`, `kind`, `name`, `variants` (list), `lat`, `lon` (float64, WGS84; null if
unknown), `coord_source` (`tei:geo`, `json:<file>`, `csv:<file>`, ... never invent coordinates), `birth`, `death`
(strings exactly as given), `auth` (same struct), `note_type` (`@type`/`@role` of the entry, lower-cased, e.g.
`person`, `place`, `mythological`, `biblical`, `author_cited`, `fictional`; normalise in your module if the
register has such a notion), `source_path` (file of the register entry), `licence`, `tier`, `n_mentions` (filled by
the writer). Entities are deduplicated by `local_key` (first wins). Fictional characters and mythological figures
are kept but must be flagged in `note_type`.

## 4. Resolving references

1. The scanner reads `@ref`, `@key`, `@corresp`, `@sameAs` (space-separated tokens).
2. Each token is classified by `common.classify_token`: authority ids (normalised: GND `118540238`, Wikidata `Q90`,
   GeoNames `2988507`, VIAF, TGN, ULAN, Pleiades, HLS; other known authorities and external URLs go to
   `auth.other`) versus local ids (`#id`, `file.xml#id`, `prefix:id`, bare `id`).
3. Local ids are looked up in the file-local register first, then in `ctx.shared`, by alias (xml:id; the
   `prefix:` and path forms are also tried without prefix). A hit sets `local_key`, `in_register=True` and merges the
   entity's authority ids into `auth`. If the entity kind differs from the mention kind, a same-kind entity wins; the
   mismatch is accepted otherwise.
4. A miss leaves `local_key=None`. **Opaque ids**: when an edition has stable local ids but no register file
   (Bullinger `p####`, CIDTC numeric keys), set `OPTIONS['opaque_keys']=True`: `local_key='<slug>:<id>'` and
   `in_register=False`. Use it only when the id is stable inside the edition; otherwise leave None and keep `ref_raw`.
5. Odd conventions (a GeoNames id in a bare `@ref`, a name in `@key`, `psn:` slugs, ids in `@n`) are handled in your
   module: `adjust(mention, hit)` for per-mention fixes, `parse_register_file` / extra aliases for registers.
   Authority ids you can read from the register are always preferred over guessing.
6. Do not create entities that are not in a register and do not merge across editions: global identity is a later stage.

## 5. Roles

Standard: `mention` (default), `author` (header `author`), `sender`, `addressee`, `sent_from`, `sent_to`
(`correspAction` `sent`/`received`; places give `sent_from`/`sent_to`), `signer` (`<signed>`), `sent_from` also for a
place in `<dateline>`, `cast` (castList/castItem), `subject` (use it when the edition marks the subject of a text).
Other values are allowed if documented in your module docstring (the writer counts them as `nonstandard_role`).
Header caveat: `titleStmt/author` is the author of the *edited text*; in letter editions it is sometimes the
edition's principal, not the sender: check on a sample and use `adjust` to correct. People in `respStmt`, `editor`,
`funder`, `publicationStmt`, `editionStmt`, `revisionDesc`, `encodingDesc` (modern editors) are never mentions.
Register entries (listPerson, standOff...) are entities, not mentions. A same-kind tag nested in another tag
without its own reference is not counted twice.

## 6. Dating

Order (generic): `correspAction` date (`letter`) > creation / origDate / docDate / settingDesc dates (`header`) >
dates under `sourceDesc` (imprint of the source) (`header`) > first `opener`/`dateline` date or `docDate` in the text
(`letter`) > module fallback > `filename_dates` (`filename`) > `default_years` (`collection`). Publication, edition and
revision dates of the digital edition are never used. Years come from `@when/@notBefore/@notAfter/@from/@to` first,
then from 4-digit years in the text (1000-2100). Override `doc_date` for per-collection logic (diaries: dates in the
text -> `DocDate(min, max, 'text')`). Do not set `in_period` yourself: it is derived from the years. When the edition
only gives a period for the whole collection (e.g. 1596-1634) use `default_years` and say so in the docstring.
Use `common.year_from_filename(name, lo, hi)` for a filename year.

## 7. Licence handling

* `LICENCE` and `TIER` are on every row. T3 (and ND licences, e.g. CC-BY-NC-ND) are extracted for internal work
  but remain flagged by their tier; no `context` for T3 (enforced).
* Context snippets <= 120 chars only for T1/T2 (`ctx.want_context`). Do not store longer text or full sentences.
* If the licence in the TEI headers is stricter than LICENSE_TIERS, use the stricter and mention it in the report.
* A collection with mixed licences per file: split by file in `adjust` (set `m.licence`; keep the module `TIER` the
  strictest) and say so in the docstring.

## 8. What a module must get right (checklist)

1. `COLLS` selects every file of the edition including separate register files (use `ratlas inspect`).
2. Registers parsed (all `person/place/org` entries, names, variants, ids, coordinates, dates); coordinates only from
   the source (`tei:geo`, a JSON/CSV in the edition); `n_mentions > 0` for most entities (the stats list the rest).
3. Mentions of persons, places and orgs (and mythological/biblical figures, flagged where the register says so) in
   text and header; correct roles for letters; no modern editors.
4. Refs resolved as in section 4; report resolved share, auth share, unresolved top refs.
5. Dates per section 6; report the `in_period` share and the `undated` count.
6. Counts reconcile with `colls.json` / the scan (e.g. number of tagged elements by zone); explain differences
   (nested tags, untyped `rs`/`name`, tags inside registers, duplicates).
7. No exceptions on the whole collection; encoding oddities and broken XML are handled (the parser recovers).

## 9. Tests (`tests/test_<slug>.py`)

Tiny, fast (< 10 s, one process), with `pytest.mark.skipif` when the raw data is missing. Include:
(a) a synthetic XML fragment exercising your special cases, run through your extractor with a `common.Context`;
(b) a check on 1-3 real files (`C.read_raw(path)` of a known member): counts of mentions by kind, a resolved
example, a dated example, an authority id, the role of a sender / addressee when relevant;
(c) schema sanity: `Mention.to_row()` / the writer on your output (see `tests/test_grand_tour.py`).
Run: `nice -n 19 .venv/bin/python -I -m pytest tests/test_<slug>.py`.

## 10. Running on Hypatia

```bash
# on the Mac, from /Users/clem/Projects/Papers/renascor-atlas
scripts/push_coll.sh -n <coll-or-prefix>...        # dry run: list + size
scripts/push_coll.sh <coll-or-prefix>...           # rsync the raw files (archives whole) to hypatia:~/renascor-atlas-data/raw/
scripts/push_coll.sh --siblings --extra 'dir/*.csv' <coll>   # also non-XML registers next to the XML / extra globs
scripts/push_repo.sh                               # code + scan.jsonl.gz to hypatia:~/renascor-atlas, uv sync
# quick test on a compute node (never on the login node):
ssh hypatia 'cd ~/renascor-atlas && srun -p debug -c 4 --mem=16G --time=00:20:00 \
  env RATLAS_DATA=$HOME/renascor-atlas-data RATLAS_RAW=$HOME/renascor-atlas-data/raw \
  RATLAS_SCAN=$HOME/renascor-atlas-data/scan.jsonl.gz RATLAS_OUT=$HOME/renascor-atlas-data/out \
  .venv/bin/python -I -m ratlas extract <slug> --limit 200 --workers 4'
# full run, one array task per slug (debug partition, 8 cpus, 16G, 30 min by default):
ssh hypatia 'cd ~/renascor-atlas && slurm/submit.sh <slug> [<slug>...]'
#   overrides: PARTITION=small-short TIME=04:00:00 MEM=32G CPUS=8 LIMIT=500 CONCURRENT=2 slurm/submit.sh <slug>
ssh hypatia 'squeue -u cag30; tail ~/renascor-atlas-data/logs/ratlas-<jobid>_0.out'
```

Outputs land in `~/renascor-atlas-data/out/<slug>/{mentions.parquet,entities.parquet,stats.json}`; a run with `--limit` and no `--out` writes to `out/_limit/<slug>/` so it cannot overwrite a full result. Bring them back
with `rsync -a hypatia:renascor-atlas-data/out/<slug>/ /Users/clem/Projects/Papers/RQ/derived/atlas/out/<slug>/`
(`nice -n 19`). Raw files land under `~/renascor-atlas-data/raw/` with the same relative paths as `RQ/data/raw`.
The login node only does `ls/cat/rsync/sbatch/squeue/uv sync`. If you change code, run `scripts/push_repo.sh` again
(it uses `rsync --delete`, so keep your files inside the repo). Keep the Mac and Hypatia code identical.
Memory: results of one file are held in RAM until written; a 100 MB XML with 500k mentions needs ~2-4 GB.
Large archives (tar.gz) are read sequentially once per run, so prefer one run per slug over many small runs.

`--workers N` (<= 8) parallelises over files; ordering of rows is not deterministic with N > 1.

## 11. Stats and what to report

`stats.json`: `files_read`, `file_errors` (+ first 50 errors), `mentions` (by kind / role / zone / kind|role),
`resolved_share`, `in_register_share`, `auth_share`, `with_ref_unresolved`, `no_ref`, `dated`, `undated`,
`in_period_share` (of dated), `date_sources`, `context_rows`, `entities` (by kind), `entities_with_auth_share`,
`entities_with_coords_share`, `entities_never_mentioned`, `top_unresolved_refs`, `invalid_values` (must be empty),
`nonstandard_role_rows`, `seconds`.

Your report to the orchestrator: slug(s), `COLLS`, licence/tier and where you read it, mention and entity counts by
kind and zone against the numbers of `colls.json`, resolved / auth / in-period shares, undated count, top
unresolved refs and why, special handling, known gaps, test results, Hypatia job ids and run time, and
"common.py changes wanted" (or none).

## 12. Quick reference of `common.py`

`FileRecord` (`path`, `coll`, `c`, `refs`, `count('txt|persName', ...)`, `read()`, `archive`, `member`, `stem`),
`iter_scan`, `iter_collection_files`, `iter_with_data`, `read_raw`, `list_members`, `parse_xml`, `norm`, `ln`,
`surface_text`, `name_text`, `classify_token`, `parse_auth`, `merge_auth`, `idno_to_token`, `header_dates`,
`date_range`, `year_from_filename`, `resolve_doc_date`, `DocDate`, `scan_mentions` -> `Hit`, `role_for`,
`corresp_roles`, `kind_of`, `TYPE_MAP`, `make_context`, `doc_language`, `doc_id_of`, `Register`, `Context`,
`parse_register`, `make_key`, `resolve_refs`, `build_mentions`, `Writer`, `raw_root()`, `scan_path()`, `out_root()`.
Environment: `RATLAS_RAW`, `RATLAS_SCAN`, `RATLAS_OUT`, `RATLAS_DATA` (defaults: Mac paths when
`RQ/data/raw` exists, else `~/renascor-atlas-data/...`).


## 13. v2 (rebuild of 2026-10-17: audit of 2026-10-16)

Read this section after the rest: it changes four things.

### 13.1 Mention columns

| field | rule |
|---|---|
| `zone` | **refined** (was `txt`/`hdr`): `text`, `header`, `editorial_commentary`, `apparatus`, `footnote`, `regest`, `repository`. From the edition's own markup: `note/@type` and `@resp` of the enclosing note, `rdg`/`witDetail` (the `lem` is the base text), `profileDesc/abstract` (regest), `msIdentifier` (repository), per-module rules (`OPTIONS['note_zone']`, `['div_zone']`, `['zone_fn']`). Marginal notes of the manuscript (`note[@place=margin]`) stay `text`. |
| `part` | the structural zone of v1 (`txt` / `hdr`). Extraction code still sets `Mention.zone` to `txt`/`hdr`; the writer publishes it as `part` and `Mention.zone_class` as `zone`. |
| `person_class` | class of the thing mentioned: `human`, `character` (cast, fictional), `mythological`, `biblical`, `dynasty`, `role`, `metonym` (a person tag whose register entry is a place or people), `celestial` (places), `people` (ethnonyms; kind org or person). From the register entry's `note_type`, the tag type (`name@myth`, `rs@biblical`), the cast list; persons default to `human`, places and orgs to null. |
| `excluded`, `excluded_reason` | **True = left out of the default atlas counts**. Reasons (`;`-joined): the five excluded zones, `character`, `celestial`, `placeholder` (S.l., Pl0, UNKNOWN, Anónimo-NN, N. N., unbekannt*), `empty_surface`, the roles `provenance` / `quote_source`, and module reasons (`duplicate_letter`, `parallel_witness`, `vacant`, `chunk_header`). The rows stay in the table; filter on `excluded`. |
| `cert` | the editor's `@cert` (high / medium / low) of the tag, when given. |
| `doc_year`, `in_period` | a range wider than 20 years has **no `doc_year`** (the range stays in `doc_year_from/to`); `in_period` is True/False when the range lies inside/outside 1450-1700 and for a narrow overlap, **None for a wide range that straddles a boundary**. `DATE_SOURCES` gains `title`, `imprint_century`, `author_lifespan`. |

`entities.parquet` gains `person_class`; the `*_local.parquet` tables gain `person_class`, `n_default`, `n_in_period_default`.
Roles added: `written_at` (Hainhofer Stammbuch `Ort`), `provenance` (REED headings), `quote_source` (empty quotation markers).

### 13.2 Harvest paths

The re-harvest of 2026-10-17 (`RQ/data/raw/originals_2026-10-17/<slug>/`, `SOURCES.tsv` and `MANIFEST.tsv` in each folder) replaces the
v1 copy for cavriana, bullinger, dracor_eng, abacus, pceec2, rup, kaiserin_eleonora_magdalena and qhod_sema. `helpers_v2.pick(legacy, new)`
chooses the selectors; `RATLAS_HARVEST=legacy` keeps the v1 paths. The folders are not in `scan.jsonl.gz`: `scripts/scan_extra.py`
(the study's scan.py with a folder list) writes `scan_extra.jsonl.gz` (`RQ/derived/atlas/v2/`, `~/renascor-atlas-data/`), read after the main
scan by `common.iter_scan` (`RATLAS_SCAN_EXTRA`). **Selectors end with `/`** (`.../cavriana/letters/`) because `...-registers/` folders
share the prefix. Registers fetched by path (not scanned): `helpers_v2.load_registers(ctx, paths, fix=...)`.

### 13.3 Rules that apply to every module

* tag kind vs register entry kind: the mention keeps its kind, inherits no id (`common.drop_inherited`), and takes the key
  `<entry key>~<kind>` (in_register False); person tags on place entries are class `metonym`;
* a tag with no letter and no reference is dropped; trailing punctuation and line-break hyphens are removed from the surface; an editorial
  `<note>` nested in a tag is not part of the surface;
* a module that dates by running text uses `helpers_v2.RunningDate` (the last `<date>` before the element, typo-robust); never a midpoint of
  a wide range (`model.best_year`).

### 13.4 Merge

`ratlas merge --v1-collections v1_collections.tsv` writes, besides the v1 files: `counts_by_zone_class.tsv` (slug, kind, zone, class,
excluded, n), the `zone_*`, `class_*`, `*_default` and `excluded_reasons` columns of `collections.tsv`, and `deviations_vs_v1.tsv`
(v1 -> v2 per collection, with the reason; the text is `merge.V2_REASONS`). `deviations.tsv` stays the comparison with the scan inventory.


## 14. Reconcile v2 (2026-10-17: gates, arbitration by names, geocoder barriers)

Code: `authority_fetch.py` (network), `namegate.py` (pure gates), `reconcile_gates.py` (the decisions), `reconcile_gnscan.py`,
`reconcile_v2.py` (`reconcile build2`), `placeclass.py`, `exonyms.py`, and the changes in `geotext.py` / `reconcile_geocode.py`.
Rebuild order on Hypatia (Slurm, debug partition): `ratlas reconcile ids` (Mac) -> `ratlas reconcile authority fetch` (Mac, network, niced) ->
`slurm/gnscan.sbatch` -> `slurm/build2.sbatch` (pass 1) -> `slurm/namecheck_v2.sbatch` -> `build2` (pass 2, with the namecheck) ->
`geocode_index_v2` -> `geocode_run_v2` -> `build2` (+ geocode, `pre_verify`) -> `verify_v2` -> `build2` (+ geocode + verify = final).

### 14.1 Gates (every authority id of every entity, places and persons)

A *pair* is (entity, scheme, id, qid): the id and the Wikidata item it leads to (`wd_map.tsv`; Wikidata redirects and P2959 permanent
duplicates are followed to the real item; qid empty for an id's own record: GND via lobid, GeoNames via the dump). Each pair is
`accepted`, `suggested` (kept in `ids_suggested` / `qid_suggested`, never used for coordinates, labels or merging) or `rejected`.
Every pair that is not accepted is one row of `gate_log.tsv` with its `reason`. Reasons: see the table in the docstring of `namegate.py`.

* **class**: persons need P31 human, except `person_class` dynasty / mythological / biblical / character / people (their own classes);
  disambiguation pages, given names, lists, offices, organisations and places are rejected. Places: no people, companies, taxa, works;
  a GND type `Person` on a place (and the reverse) is rejected.
* **names**: the authority's labels and aliases (en fr de it es la nl pt pl cs + mul), GND names, GeoNames names must agree with the
  entry's names (`namegate.compare_person`, `place_relation`, the audit's matchers). Otherwise a second *independent* id must agree
  (another scheme leading to the same item; for places also a register coordinate or a same-name GeoNames feature within 15 / 10 km
  of the authority's point). Otherwise `suggested` (`name_mismatch`, `given_name_only`, `unverified` when no label source exists).
* **cert**: an id given only by tags with `@cert` low / medium, or written `viaf-suggested:` by the extractor: `suggested` (`cert`).
* **chronology** (persons): born after the last document of the entity; register dates > 50 years from Wikidata's: `suggested`.
* **homonyms** (GeoNames): a feature with population < 100 and a same-name town (>= 1000 inhabitants, >= 10x) more than 50 km away,
  while the feature is more than 50 km from every register-placed place of the edition and the town is nearer: `rejected` (`homonym`).
* **Wikipedia ids** (`lang:Title`): an id only if the title matches the entry. **Sandrart**: a VIAF id counts only with a second id.
* TGN: `vocab.getty.edu/robots.txt` is `Disallow: /`, so nothing is fetched there; TGN ids that Wikidata carries (P1667) are checked through
  the item's labels, the others are `suggested` / `unverified`.

### 14.2 Counts

`reconcile build2` writes **atlas counts**: excluded mentions (commentary, apparatus, footnotes, regests, repositories, characters,
placeholders, celestial bodies ...) are left out of `n_mentions`, `n_in_period`, the quarters, the clusters and the globe;
`n_mentions_all` / `n_in_period_all` keep the canonical total. Locals with no atlas mention have `scope = 'excluded'`.

### 14.3 Tables (`RQ/derived/atlas/v2/`, `~/renascor-atlas-data/out_v2/reconcile/final/`)

| file | content |
|---|---|
| `places_local_geo.parquet` | one row per edition place: gated `qid`, `qid_suggested`, `ids_suggested`, `auth_name_level`, coordinates and their `coord_source` (register, wikidata, geonames, name_propagation, name_geonames, name_exonym, none), `coord_kind` (point / area), `coord_radius_km`, `place_kind` (city / area / other), `kind_source`, `place_class`, arbitration columns, `flag_100km`, `gc_*`, `verify_rule` |
| `places_global.parquet` | clusters (`global_id`, `globe_id`, `place_kind`, `place_class`, `classes`, ids, coordinates) |
| `persons_local_global.parquet`, `persons_global.parquet` | gated QIDs, `person_class`, `wd_category`, labels, dates, occupations |
| `globe_places.json` | v1.1 format plus `k` (city / area / other), `r` (radius of an area), `cq` ({collection: 10 quarter counts}), source code `e` (exonym table) |
| `gate_log.tsv`, `gate_pairs.parquet` | every id rejected or downgraded by a gate, with the reason; every pair (accepted ones too) |
| `gate_counts_by_reason.tsv`, `gate_counts_by_edition.tsv` | the gate log counted |
| `geocode/` | `geocode_places.parquet` (every name-based result with its evidence), hold-out self-checks, `geocode_verify.tsv` (taken back by the audit rules) |
| `licences.tsv` | the licence authority table (14.7) |
| `reconcile_report.json`, `coord_conflicts.tsv`, `unplaced_*.tsv`, `version_comparison.json` | numbers |

### 14.4 Arbitration by names

A register-versus-authority conflict (> 50 km) is decided by the distance of each point to the nearest GeoNames feature of the same
name (`reconcile namecheck`), as before, AND by the name gate: an authority whose names match strongly (exact / fold / stem) keeps its
QID and ids when its coordinate loses (`auth_rejected` without `identity_dropped`; rivers, countries and regions); a weaker match loses its
identity too. When neither point has a same-name feature and the conflict exceeds 100 km, a strongly matching authority wins
(`authority_tie_100km`). Every conflict over 100 km is flagged (`flag_100km`, `coord_conflicts.tsv`).

### 14.5 Areas and kinds

Register coordinates of countries (`note_type` country / coun; the regions of CKCC) are the capital's: dropped. An area takes the GeoNames
point of an accepted id (country / first-order region centroids), else the Wikidata point, flagged `coord_kind = area` with a radius,
else none; the geocoder gives areas a GeoNames country / region feature or an exonym-table area. `place_kind` comes from the Wikidata
classes of the accepted item, the GeoNames feature code and the register type.

### 14.6 Geocoder barriers (`reconcile_geocode.py`, `geotext.py`, `placeclass.py`, `exonyms.py`)

Never geocoded: celestial / mythical / biblical / fictional / ethnonym classes (`placeclass.classify`), deictics and common nouns
(`hier`, `Welt`, `herberg`, `gang`, `cielo`, ...), micro-toponyms (`flurname`, `acker`, `weg`, `stre`), administrative areas (`... Hundred`,
`... Ward`, `honour of ...`). The curated exonym table (`exonyms.py`: Latin and old names, peoples named after their country; ~1,000 forms) is consulted
before any GeoNames alternate name, on the head name only (not on the parenthetical locator, not on a leading comma part, variants only when the name
is unusable); its areas get a radius and `coord_kind = area`. Strict regional prior per collection (`HARD_REGIONS`: a candidate outside the box is rejected unless it
has a million inhabitants or is a country / first-order region / capital). The register type (`moun`, `rive`, `coun`) restricts the
GeoNames classes. `X (Y)`: the head name is searched near Y; for cidtc the bracketed modern name is the first key. The combining small
letters of print (U+0363-036F) become letters; the skeleton folds `jh`, `gk`, `rh`, `mb/nb`.

### 14.7 Licences (rapport section 2, point 13)

`data/licences.tsv` is the one authority table: slug, catalogue ids, licence, tier, whether context snippets are stored (T1/T2 yes, T3 no),
ND flag, open audit point, the owner's decision where one exists, and the audit's note and fix. The modules keep their `LICENCE` / `TIER`
constants (the writer stamps every row with them) and `tests/test_licences.py` fails when a module and the table disagree: change a licence
in the table and in the module together, and re-read the table before any release. Open points are owner decisions (nothing was changed here):
ND editions (bullinger, gongora_epistolario, obvil_critique, polemicagongorina_obvil, freem_other rows), kaiserin / magdeburger / elan_crenum /
paesi_novamente / emmo_folger tier questions, bdpn CC0 scope. T3 rows (14 editions) must be dropped or aggregated at release.


## 15. Manual review (2026-10-17: the audit's doubtful cases, read by hand)

Code: `manual_overrides.py`. Report: `RQ/derived/atlas/v2/manual_review.tsv` (one row per case: `group`, the pair or entity, `verdict` correct / wrong /
undecidable, `confidence`, `evidence` (the passage or the dates that decide), `v2_before` / `v2_final_state_before` (what the gates and the build did),
`override` and `override_value`, `v2_after`, `outcome`). `outcome` classifies the gate: `true_accept`, `false_accept` (accepted and wrong),
`true_suggest` / `true_reject` / `true_drop` (not used and wrong), `false_suggest` / `false_reject` / `false_miss` (not used and correct),
`undecidable`, and for the class and placement checks `true_class`, `true_exclusion`, `true_placement`, `false_placement`, `old_id_*`, `new_id_*`.

Corrections that a person confirmed are data in the `--rdir` of `reconcile build2`; the repository's `data/` holds the source:

| file | content |
|---|---|
| `id_overrides.tsv` | `coll_slug kind entity_key scheme id qid action case_id note`. `action` accept / reject / suggest on the gate rows of the pair; `scheme = *` matches every pair of the entity that leads to `qid`; `accept` with `qid = -` keeps the id and drops the Wikidata item it leads to (bot duplicates, wrong mapping); `accept` with `scheme = wikidata`, `id = qid = Q...` creates a pair the gates never made (a QID that replaces a wrong one, a Wikipedia link lost at extraction). A manual decision has the reason `manual_review`, counts as a strong name match in the arbitration, and is in `gate_log.tsv` when it is not an acceptance. |
| `coord_overrides.tsv` | unchanged: a coordinate corrected by hand (the new rows put Lisbon and Vienna in place of the country points a name propagation had given them). |
| `geocode_rejects.tsv` | `coll_slug entity_key rule note`: a name-based result taken back by hand, like the audit rules' `geocode_verify.tsv` (merged into it by `build2`; the place stays unplaced). |

`reconcile_report.json` gets `manual_overrides` (rows read, pairs accepted / rejected / suggested / created, unmatched rows). A row that matches no gate row is
logged and ignored (an entity key that changed after a re-extraction must be re-checked against `manual_review.tsv`). `slurm/build2_manual.sbatch` runs the
final build on a frozen copy of the inputs (`~/renascor-atlas-data/snapshot_manual/`), so that a re-extraction in parallel cannot change them.

Effect of the overrides on the same build without them (`RQ/derived/atlas/v2/manual_review_effect.tsv`; tables in `v2/manual_review_build/`): 257 override rows
matched or created gate pairs (0 unmatched); 99 local places gained a QID, 116 were placed (253 mentions in period), 59 local persons gained a QID, 10 lost one
(the false accepts), and 2 name-based placements were taken back by hand.



## 16. Audit tests of v2 (2026-10-17: the nine checks of the audit as blocking tests)

Code: `audit_v2.py` (the nine checks), `audit_namecmp.py` and `audit_compat.py` (the audit's own matchers, **frozen**: the yardstick does not
move when `namegate.py` does), `audit_editions.py` (fresh per-edition sample). Tests: `tests/test_audit_v2.py` (always: every check on tiny
tables, one case that must fail and one that must pass; with `RATLAS_AUDIT_V2=1`: the nine checks on the real tables, zero failures).
Run on Hypatia with `slurm/audit_v2.sbatch` (CLI `ratlas audit v2`, writes `out_v2/audit_v2/check<N>_<name>.tsv` = the failures,
`audit_v2_summary.json`, `gn_final_dist.tsv`), then `slurm/audit_editions.sbatch`. Reviewed exceptions live in `data/audit_v2_allow.tsv`
(check, coll_slug, key, reason); a test requires a reason of more than 15 characters. Inputs kept in `data/` (generated once by
`scripts/make_audit_tables.py` from the audit folder): `audit_confirmed_wrong.tsv`, `freshness_audit.tsv`, `legacy_sources.tsv`.

| # | rule | notes |
|---|---|---|
| 1 | no accepted id the audit's matcher calls a mismatch | persons: `audit_namecmp.compare` on labels in 10 languages; places: `audit_compat.relation`. A second independent id excuses a mismatch. **Coordinates alone do not**: a place accepted because a point lies near the item must have a GeoNames feature that carries the entry's name AND one of the item's names within 10 km (Venedig and Venezia: one feature; Halle and Altstadt: two). No pair the audit confirmed as wrong (337 rows) is accepted again; for Balingen the id is right and the item behind it (Bisingen) is the error (`bad_qid`). |
| 2 | no non-human QID on a person unless the **published** class allows it | the class is the global table's `person_class_final` (it reads the item's own category: mythological, biblical, dynasty, character, people); disambiguation pages, given names, lists, offices never; a `metonym` has no id |
| 3 | no person born after the latest document that cites them | Wikidata birth, precision-aware (a century is a range), 5 years of slack; the latest document is the END of the date range of the default mentions (wide ranges have no `doc_year`) |
| 4 | no point farther than 50 km from every GeoNames feature that carries the place's name | names folded with the audit's phonetic fold on both sides (Constantinopel = Konstantinopel). Not judged: no namesake at all, areas, the curated exonym table, an authority point whose name agrees strongly (exact / fold / stem), a register point within 50 km of that authority. **Editorial points** (the editors' coordinates, or a name-propagated copy of one: MoEML's Newgate, REED's playhouses; GeoNames does not carry the toponym) are listed in `check4_review_editorial_points.tsv`, not failed: the register is the editors' override. |
| 5 | no capital-for-country | register type country, GeoNames country / first-order region without a city class, a Wikidata state / country class: an area; plus every point whose name only countries or first-order regions carry in GeoNames (`polity_name_points`) |
| 6 | nothing of the zones commentary / apparatus / footnote / regest / repository, no character, celestial body, placeholder, provenance or quotation source in the default counts; the tables' `n_mentions` are the sums of the default mentions | **Pronouns** tagged by the editors (Sandrart's *ich*, the Tagebuch's *er*) are counted and reported (`info`), not failed: whether anaphora are mentions is the owner's decision |
| 7 | no duplicated mention: Casebooks has one person / org mention per (case, entity) | repeated header rows of other editions are information only (a multi-letter file repeats its senders) |
| 8 | no midpoint date for a range wider than 20 years; `doc_year` inside its range; `in_period` follows the range | wide ranges inside / outside the period have it, wide ranges across a boundary have none |
| 9 | every file of every edition comes from a harvest folder with a `SOURCES.tsv`, or is listed in `data/legacy_sources.tsv` | an edition that changed upstream after its harvest is never legacy; a manifest that does not match its folder fails |

`audit_editions.py` draws 3 source files per edition (seeded, not the audit's samples) and compares the tagged elements of the raw file with the
mention rows and the dates of the header with the row dates; and lists the 10 places with the most default mentions of each edition with the
accepted item, the point and its source, the distance to the nearest namesake and flags (`unplaced`, `name_vs_label`, `far_namesake`,
`area_point`, `suggested_only`). The flagged rows are read by hand.

Rules the tests forced into the pipeline (all with tests): `kind_of_classes` (a federal state is an area, also when Wikidata calls it a ward;
a city-state stays a point); a GeoNames country or first-order region is an area unless the item is a city; a name-based point copied from an
area is an area with the same radius; the near-coordinate evidence of the name gate needs one GeoNames feature that carries both names
(`gn_samename.parquet` has a `names` column); the chronology gate also stops an acceptance by a second id and reads the end of the date range
of the default mentions (`last_year`); a character is a character in every mention of its key (`merge.unify_character_keys`).

### 16.1 What the tests found in the v2 tables of 2026-10-17 (first run) and what became of it

First run on the tables of the reconcile chain (job 5046250): checks 1 to 9 failed on 3, 1116, 1, 658, 105, 397, 9, 0, 12 rows. Reading them:

* **fixed in the code and rebuilt** (merge, gnscan, build2 pass 1 and 2, namecheck, geocode, verify, final; Slurm, ~25 minutes): the kind of Bavaria
  (a "ward" made it a city), Navarre and the Polish voivodeships (areas); GeoNames countries and first-order regions are areas; a point copied
  from an area is an area (Polen, Königreich on Krakow; Galliam); free imperial cities and city-states are points (Nuremberg, Hamburg were 150 km
  areas); Halle (Saale) took the district Altstadt (808 mentions in the Tagebuecher), Balingen the town Bisingen, Lenzburg a Roman vicus, Neapel
  a street, Gollnow a gate, Ellwangen a "do not use" item (coordinate-only evidence: now a GeoNames feature must carry both names); the register's
  Hessen (type country) took an Ortsteil; Nieto Molina (born 1730) of a poem dated 1501-1700 (chronology gate blind to wide ranges and to second
  ids); Panurge and four more characters counted as persons; the MoEML placeholder id LLLL1 gathered 302 mentions of 300 places; CKCC's Sweden
  (2,440 mentions) and Poland were unplaced (English country names added to the exonym table); `die Stadt` was geocoded to Magdeburg;
  Egmond (CKCC, 259 mentions of Descartes' letters, 195 km off: Boxmeer for Egmond-Binnen), Bernburg (Fruchtbringende Gesellschaft, 301 km off) and the Châteaux de Ternier (Bullinger) corrected in `coord_overrides.tsv`; 2 geocoder
  results taken back (Segnie = Senj, not Segni; Chez les Belin, not Berlin).
* **not an error of v2, the yardstick moved**: persons of a mythological / biblical / dynasty class (1,116 locals) are published with that class
  (the check reads `person_class_final`); the audit's `name_mismatch` rows for persons were judged on four label languages (Mehmed IV);
  authority points and Latin / old forms that GeoNames does not carry (Troye, Gandavo, Naple, Sidonis, Locris ...) are vouched for by the
  authority or the exonym / propagation chain; `ward`, `parish` and `seat` words; Sicily's centroid happens to be Enna.
* **left as information, not failed** (the owner decides): pronouns tagged by the editors (4,553 default person mentions in 29 editions:
  Sandrart 1,656, Tagebuecher 937, GerManC 605, CIDTC 475, Kaiserin 418); common nouns tagged as places (dta_1600: Schloß, Kirche, Stadt ...; they
  are never geocoded but are counted); editorial register points far from every namesake (329 places, 2,728 mentions, mostly MoEML and REED
  London toponyms that GeoNames lacks; a few are register errors, see the report); later scholars cited in editorial notes that the module leaves
  in zone `text` (lettres_calvin 5, reed 1, sozinianische 2; confirmed by hand, reason `manual_review`).
* **allow list** (`data/audit_v2_allow.tsv`, 14 rows with reasons): 11 old or foreign forms of check 4 that were read against the sources, 3 names of check 5.

Last run (job 5048592, tables of the full chain 5047460-5047469 and a final build with the Egmond override): all nine checks without failure (check 4: 11 allowed, check 5: 3 allowed); the whole suite passes on Hypatia with the real-data tests (359 passed) and on the Mac (350 passed, 9 real-data tests skipped).


## 17. v2.1: the final build (2026-10-17: overrides merged, Casebooks directory, local TGN, recalibrated gates, pronouns)

One clean chain from scratch (`slurm/v21_chain.sh`, steps in `slurm/v21.sbatch`; extraction of all collections into `out_v2_1/`, then merge, gnscan,
build2 pass 1, namecheck, pass 2, geocode index and run, build2 pre_verify, verify rules, build2 final, the nine checks, the whole test suite).
Output `~/renascor-atlas-data/out_v2_1/` and `RQ/derived/atlas/v2.1/` (the file set of v2 plus `version_comparison.json` for v1, v1.1, v2, v2.1,
`generic_terms.tsv`, `calibration_v2_1.json`). v1, v1.1, v2 and `RQ/data/raw` are never written. The authority tables (Wikidata, GND, GeoNames features) are
those fetched on 2026-10-17 (`reconcile_v2/`, copied to `reconcile_v2_1/`; no network in this build).

| change | where |
|---|---|
| manual-review overrides are part of the main build | `data/id_overrides.tsv`, `coord_overrides.tsv`, `geocode_rejects.tsv` are copied into `--rdir` by the chain (v2's `overrides/` folder was a subset of them) |
| tagged pronouns and common nouns are excluded by default | `generic_terms.py` (lists for de en fr it es la nl pl; kinship nouns, generic person nouns, bare titles; deictics and generic place nouns), `classify.finalize`, `merge.exclude_generic`: `excluded` True, `excluded_reason` `pronoun_or_common_noun` (only when no other reason applies); a common noun that carries an authority id is an identification and is kept, a pronoun never; the rows stay; `generic_terms.tsv` counts them by edition, kind, class and surface; audit check 6 now fails on a default pronoun |
| Casebooks directory | `reconcile_v2.apply_casebooks`: PLACE ids the curated map matched with confidence high / medium get `coord_source` `casebooks_directory` (`coord_conf` high only when the directory's own geocoding is high / medium, else medium; radius 20 km or more = area); low stays a suggestion (`cb_lat`, `cb_lon`, `cb_used` False); it replaces the name-based result (kept in `gc_*`) and never overrides a register or authority coordinate. Columns `cb_*` in `places_local_geo.parquet`; globe source `c` |
| local TGN: ids verified by names | `tgn.py`, `reconcile_gates._tgn_evidence`: the names of the subject in `tgn/tgn_names.parquet` (all languages, historical forms, Latin) against the entry's names: compatible = accepted (reason = the name level), not = rejected `tgn_name_mismatch`, a subject the index does not hold = suggested `unverified`. Audit check 1 reads the accepted TGN ids in the same index with its own frozen matcher |
| local TGN: coordinates | an accepted TGN id gives a point when the register, Wikidata and GeoNames ids give none: `coord_source` `tgn` (globe source `t`), kind from the TGN place type |
| local TGN: names for the geocoder | `reconcile_geocode.add_tgn_names`: TGN names that GeoNames lacks join the name index as another name of the GeoNames feature of the same name within 8 km (inherits its population and type) or as a TGN-only candidate (`TGN <id>`); ranked like every candidate with the edition's regional prior; `gc_tgn` marks a result that rests on one |
| homonym gate recalibrated | `namegate.homonym_check`: the candidate (village of population < 100 against a larger namesake > 50 km away, as before) is rejected only when it lies more than `HOMONYM_ISOLATED_KM` = 30 km from every other placed place of the edition (`reconcile_gates.edition_context`); fewer than 5 placed places in the edition: suggested `homonym_unverified`; otherwise accepted |
| verify rule recalibrated | `verify_geocode.GN_MIN_P` = 0.8: an uncorroborated, non-major GeoNames-only match is taken back when the geocoder's certainty `gc_p` is below it (the v1.1 strata table is gone); the other rules are unchanged |
| both fitted on labels | `scripts/calibrate_rules.py`, `data/calibration_v2_1.json`: 70 % of the labels fit the threshold, 30 % (hash split, seed in the file) report the confusion matrices of the old and the new rule; `tests/test_namegate.py` keeps the constants equal to the fitted values |
| BDPN | thirteen cosmological and compass terms typed as places are `celestial` (out of the counts) |
| MoEML | not covered by the `MAP` rule (Medici Archive Project only): `common.FORBIDDEN_PATH_TOKENS` never matched it; its 878 harvested register files are read (`tests/test_map_of_early_modern_london.py`) |

`reconcile build2` takes `--tgn-dir` and `--casebooks-map`; `reconcile geocode index` takes `--tgn-dir`; `audit v2` takes `--tgn-dir`.
`reconcile_report.json` gets `excluded_mentions` (by first reason, kind, in period) and `casebooks_directory`; the by-source counts include `tgn` and `casebooks_directory`.
The TGN is under the Open Data Commons Attribution License 1.0: any release that shows a TGN-derived coordinate or name carries
*Getty Thesaurus of Geographic Names, The J. Paul Getty Trust*. `vocab.getty.edu` (robots.txt `Disallow: /`) is never queried.


## 18. v2.2: the sample check of v2.1 (2026-10-19: homonym overrides, document priors, generic nouns with ids, TGN checks, kinds, letters)

An independent sample check of v2.1 (persons, places, 207 files; `scratchpad/atlas/v21-sample/`) found the errors below. v2.2 is one clean chain (`slurm/v22_chain.sh`,
steps in `slurm/v22.sbatch`; `out_v2_2/`, `reconcile_v2_2/`, `geocode_v2_2/`, published to `RQ/derived/atlas/v2.2/`; v1 .. v2.1 and `RQ/data/raw` are never written).

| # | finding of the check | rule now | where |
|---|---|---|---|
| 1 | the recalibrated homonym gate re-accepted 13 ids that the manual review had judged wrong (Marignano, Boll, Gärnitz, Anhausen, Münden, Longueville, Neuburg, Kalkar, Mülheim am Rhein, Fenile, Brüggen, Jüchen, Landsberg) | **every `wrong` verdict of `manual_review.tsv` is a hard override** (`reject`; `suggest` where the gate had only suggested the id) and every `undecidable` verdict whose evidence says that the editor marks the identification as uncertain ("Identifizierung unsicher") is a `suggest`: 63 new rows in `data/id_overrides.tsv` (`scripts/make_hard_overrides.py`, `tests/test_hard_overrides.py`); Rohrbach of the Elector's ride is the section of Heidelberg (`coord_overrides.tsv`); the Anhalt diary's Alpen (variants *gebirge*) are not the town of Alpen (`V22-01`). **Gate**: a village whose larger namesake lies in the same region (`HOMONYM_REGION_KM` = 150) is accepted only when the edition's own places favour the village over the town (the town is itself more than 30 km from every other placed place); otherwise it is only `suggested` (`homonym_unverified`) | `namegate.homonym_check`, `manual_overrides` |
| 2 | name-based points far from the territory of their file: 169 Hearth Tax places in another county, London streets, inns and churches of MoEML and Old Bailey in Dorset, Somerset, Hertfordshire, Lyon, Savoie; `PLACE OUTSIDE OF LONDON` as one point | (a) **document regions**: a file of the Hearth Tax is a return of one county; its own headings (`placeName@county`, header title, city / borough) give the region (`regions.COUNTY_REGIONS`, 21 counties, ridings and cities), and a candidate must lie in the disc of one of the files the place occurs in; the document code is the file where one `doc_id` spans several files (Hearth Tax: 61 files, 5 `doc_id`; suriano, ckcc, leibniz ...). (b) **London prior**: MoEML (all name-based candidates) and Old Bailey (names that look like London streets, inns, churches, houses ...: `regions.london_like`) must lie in Greater London (`LONDON_BOX`) unless the feature is a country, a first-order region, a capital or a city of a million; the exonym table's cities are subject to the same priors. (c) **file consistency** (`reconcile_geocode.file_outliers`): a name-based point of 60 km radius or less more than 100 km from the median of a regional file (>= 8 placed places, 80 % within 150 km of their median) is taken back to a low-confidence candidate when it is an outlier in every file of the place. (d) MoEML `POOL1` (`PLACE OUTSIDE OF LONDON`) is a placeholder like `LLLL1`: its 84 mentions are keyed by their own surfaces. (e) exonym / override tables: *Hanse, Hansestädte* (Hanseatic League, area), *Alpen* (Alps, area, kind n), *Thermopylae* (also *Thermopyles*, *pas des Thermopyles*), *Windesore* = Windsor, *Savoy* = the Savoy of the Strand in the London editions (`exonyms.BY_COLL`), (f) the bare name of an edition whose register has exactly one qualified place of that name takes it (*Bergen* in the Descartes letters = *Bergen op Zoom*; `edition_qualified`, medium) | `regions.py`, `reconcile_geocode.py`, `exonyms.py`, MoEML extractor |
| 3 | common nouns that carry an id (alhier, Stadt, lande, Reich, pays, Schloß, this towne, Meer; Vater, Bruder, Kaiser, Gemahlin with QIDs) shown as names; plurals (Schwestern, Söhne, kinder, Provintzien) counted for the first referent only | **the mention stays linked** and is flagged `generic_with_id` (mentions column; `n_generic_with_id` and `generic_name` in the local tables, `generic_with_id` and `name_is_generic` in the global ones); **a common noun is never the display name**: `aggregate_global` prefers any real name, and a global place or person whose only edition names are nouns is named by its authority (Wikidata label, else the GeoNames or TGN name, else the Casebooks directory entry). **Several referents**: a mention whose `ref_raw` names two or more entries of the register (`#a #b`, `eats:1 eats:2`) is **excluded from the per-entity counts** (reason `multi_referent`, rows stay, `multi_referent_terms.tsv`); the alternative, one row per referent, would count one mention several times. Chosen because the atlas counts must stay sums of mentions (audit check 6) | `merge.flag_generic_with_id`, `merge.exclude_multi_referent`, `generic_terms.multi_referent`, `reconcile_build.aggregate_global` |
| 4 | TGN: *Cythnos* accepted for Sífnos at fuzzy 0.83; TGN points that contradict the subject's own hierarchy (*Aptera* is 500 km from its parent Khaniá; *Lusitania* has +6.3 for -6.3); *Bad Düben, Alaunwerk* took the Alaunwerk of Brandenburg | a fuzzy name relation counts for a TGN id from **0.90** (`TGN_FUZZY_MIN`); below it the id needs a GeoNames feature that carries both names, else it is only `suggested` (`tgn_fuzzy`). **Hierarchy versus coordinates**: the TGN point is not used when it lies farther from its preferred parent than 99.9 % of the TGN's own subjects of that parent type (`tgn.PARENT_MAX_KM`, computed over 2.9 million subjects; parents that are continents, nations, empires or the World have a centroid for a point and are not tested) or farther than max(3 x radius, 200 km) from the curated exonym table's point for the same name (`tgn.coord_issue`); the id stays accepted; `tgn_coord_check.tsv` lists them. **Head and qualifier**: for 'Head, Qualifier', a TGN match on the qualifier alone is rejected (`tgn_head_far`) when the head (GeoNames) lies more than 100 km (300 km for an area) from the TGN point. Bergen / Bergen op Zoom: see 2 (f) | `namegate`, `reconcile_gates._tgn_evidence`, `tgn.py` |
| 5 | Ballenstedt (a town of 8,064 inhabitants) was an area; 275 towns (7,602 mentions) of the same kind: Prague, Plzeň, Cambridge, Canterbury, Münster, Bonn | `kind_of_classes` read the whole class label: *municipality with town privileges in the **Czech Republic*** is a republic, *city of **United Kingdom*** a kingdom. Now the **head** of a label (before *of* / *in*) decides, labels that say nothing (*geographic location*) are ignored, and a GeoNames settlement of 1,000 inhabitants or more whose item is only a region or province without a polity class (Mantua = Province of Mantua, PPLA2) is the town | `reconcile_v2.kind_of_classes`, `polity_classes` |
| 6 | three kinds were too few | **five-way kind `k`** in `globe_places.json` and `core.json`: `c` cities, towns, villages, city quarters; `a` countries, regions, territories; `n` rivers, lakes, seas, islands, mountains, valleys, passes, capes, forests; `b` buildings, castles, monasteries, streets, squares, gates, bridges, sites; `u` other. From the Wikidata classes of the accepted item (the **head noun** of each English label votes), the GeoNames feature code (also of name-based matches), the TGN place type and the register type; ties go to the more specific kind, the atlas kind (city / area / other) decides the rest | `placekind.py`, `reconcile_v2.point_kind5` |

### 18.1 Letter network (`cards_globe`, `letters.json`)

`[[from_row, to_row, count, [10 quarter counts], [[slug_idx, count], ...]], ...]`, rows = indexes into `core.json["rows"]`, slug indexes = `core.json["slugs"]`. A letter is a
document with a canonical, non-excluded, in-period place mention of role `sent_from` and one of role `sent_to`, each resolved (local key or surface key of its edition) to exactly one
point of the globe. A letter whose end names two different points (dated in two places) or no point of the globe is not an edge; a letter inside one city is a self-loop. Quarters are
by the document year (letters of a wide date range have no quarter). Coverage per edition (`letters`, with a sender place, with an addressee place, both, ambiguous, both on the
globe) is in `build_report.json["letters"]` and `letters_coverage.tsv`. Editions that tag only the sender's place (cidtc, ferdinand_i, croala ...) have no edge: the addressee's
place is not inferred from the correspondent.

### 18.2 Details that the trial runs on the real tables forced

* **Register points that came from a wrong id.** Bullinger's register coordinates of Marignano, Boll, Gärnitz, Anhausen, Münden, Longueville and Neuburg are the GeoNames points of the wrong homonyms:
  rejecting the id leaves the point. `coord_overrides.tsv` takes `lat = -` (drops the register point, `reconcile_build.load_overrides`: `ov_drop`) or the corrected coordinates (Marignano = Melegnano,
  Boll bei Heiningen, Anhausen an der Brenz, Hann. Münden; GeoNames ids in the notes). The Religionsfrieden entries had no register point (placed by the id only): they go to the geocoder.
* **File consistency is for regional collections only** (`regions.FILE_CONSISTENCY_COLLECTIONS` = Hearth Tax). On all editions the rule took back 150 correct points of Basel's oath records and
  of the Swiss law sources (they name Italy, Lorraine and Valais from one file).
* **The exonym table's cities obey London and the file's region, not the collections' hard boxes** (Rome is Rome in the Lüneburg nuns' letters; Lyon's Inn is not Lyon). `bruggen` is no form of Bruges
  (Brüggen on the Lower Rhine took it).
* **London prior: no first-order region.** The parishes of Jersey are ADM1 in GeoNames (`St. Mary (Abchurch)` became Saint Mary of Jersey); only countries, capitals and cities of a million are exempt.
* **TGN parent check**: parents that are continents, nations, empires or the World have a centroid for a point and are not tested (the first trial flagged every country of Europe); the exonym check
  reads the entry's name only, not its variants (a variant *Batavis* of Passau is the Dutch in the exonym table).
* **Kinds**: `kind_of_classes` also stops reading *cultural heritage monument* as a cultural area, reads plurals (*Ukrainian historical regions*), knows satrapies and viceroyalties, and returns
  `other` only for a natural or built class (a *municipal part of the Czech Republic* is nothing: GeoNames decides). A register type `country` outranks a TGN *inhabited place* of the same name
  (Hessen, Grafschaft Schwarzburg). GeoNames ADM3 / ADM4 / ADM5 are communes (`c`), not areas. The name itself votes in `placekind.name_kind` (*Kloster Zinna* b, *The Thames* n,
  *Ballenstedt, Amt* a).
* **Audit check 5** reads the head of a Wikidata class label like the build (*borough in the United Kingdom* is not a "state / country class"); `_label_head` in `audit_v2.py`; the 'frozen'
  matchers of names (`audit_namecmp`, `audit_compat`) are untouched.
* **Display names**: a global place or person whose only names are common nouns and that has no authority name gets `label` = null, `name_is_generic` = True and an empty name on the globe; `cards_globe`
  never uses a noun as `name_en` or `original_name` (`Unnamed place` as the last resort, counted in `build_report.json`).

### 18.3 What the tests found

`tests/test_hard_overrides.py`, `test_regions_v22.py`, `test_v22_build.py` and additions to `test_namegate.py`, `test_reconcile_gates.py`, `test_reconcile_v2.py`, `test_merge.py`,
`test_cards_globe.py`, `test_map_of_early_modern_london.py` give one test per rule. The numbers of the rebuild are in `RQ/derived/atlas/v2.2/rapport_v2_2.md`.


## 19. v2.3: Latin names, historical polities, English homonyms, and the fixes of the independent check of v2.2 (2026-10-19)

v2.3 is v2.2 plus the paths below. One clean chain (`slurm/v23_chain.sh`, steps in `slurm/v23.sbatch`; `out_v2_3/`, `reconcile_v2_3/`, `geocode_v2_3/`, the globe shards from
`slurm/cards_globe_v23.sbatch`; published to `RQ/derived/atlas/v2.3/`; v1 .. v2.2 and `RQ/data/raw` are never written). `slurm/v23_reconcile_chain.sh` re-runs the reconcile half when only
the code of that half changed. `scripts/geocode_trace.py` (`slurm/v23.sbatch trace NAME...`) prints the paths a name takes through the geocoder; it is a development aid and runs on Slurm only.
Precision first: a path that adds placements which cannot be verified writes them as **low** (a suggestion in the `gc_*` columns, never on the globe).

### 19.1 Path A: Latin names (`coord_source` `name_latin`, globe source code `l` = "located by name")

Code: `latin_names.py` (lemmatiser, context), `latin_geo.py` (lexicons, gates), `Geocoder.latin` in `reconcile_geocode.py`. For an unplaced (or low-confidence) place in a **Latin context**
(the place's own mentions have a `lang`, 60 % or more of the mentions with a known language are Latin; else the edition's level: `latin_names.latin_context_level`, `latin` / `mixed`;
`non_latin` and `unknown` do nothing: Moree, Delfos), the name is de-inflected to nominative candidates (`-am -> -a`, `-um -> -us`, `-iae -> -ia` ...; `min_score` 0.4 in a Latin context, 0.6 in
a mixed one) and looked up in:

| source | what | family |
|---|---|---|
| Wikidata Latin labels and aliases | `wd_entities.parquet` of the reconcile directory (10 languages with `la`), the point of the atlas's own **placed** locals that carry the item (a hidden place of the hold-out cannot vouch for itself) | atlas |
| names of placed places of the Latin editions | `Geocoder.RL` (share of Latin >= 0.5), agreement of the placed namesakes | atlas |
| exonym table, polity table | `exonyms.lookup`, `polities.PLAIN` on the candidates | table |
| Orbis Latinus (Graesse-Benedict, 2nd ed. 1909, Internet Archive `bub_gb_oJt-AAAAMAAJ`, public domain; `data/orbis_latinus.tsv`, tiers A and B, ambiguous keys dropped) | Latin -> German-spelled modern name, type, country; the modern name is looked up in GeoNames with the class of the type and the country of the region text (`latin_geo.region_countries`); the index answers these names (`reconcile geocode index --orbis`) | orbis |
| Pleiades | not available locally: not used | |

Gates (`latin_geo.combine`): the class of the name (`placeclass`: celestial, ethnonym ... are never looked up; Latin ethnonym lemmas are skipped), the homonym rule (the hits must lie in
one place, within max(50 km, the larger radius): otherwise `latin_ambiguous`, nothing is placed), the regional priors of the edition (`row_keep`: hard boxes, London, file region), and the
strength of the rule. **high** = two independent families agree, rule score >= 0.8, Latin context; **medium** = one family with a strong rule in a Latin context, or two in a mixed one;
**low** = the rest; a rule below 0.6 (a Greek accusative plural read as Latin: Delfos -> Delfi) is a suggestion whatever the sources say. Further guards (`Geocoder.latin`):
the Orbis Latinus alone places only a settlement of 15,000 inhabitants or more (`ORBIS_MIN_POP`: 'Vianam' -> the village of Weinstetten is a suggestion); a known Latin homonym (`LATIN_HOMONYMS`:
Lugdunum = Lyon or Leiden, Traiectum, Augusta, Isara ...) needs 0.35 of support from the placed places of its own documents; one family alone is a suggestion when the documents name
other placed places and none is near (`LATIN_FAR_CTX`; Vallesia in Bullinger is Valais, not Wales) unless the place has 30 mentions in the atlas; a surface that is itself the name of a placed place
elsewhere ('Monaco', the principality, and the ablative of Monacum = Munich) is a suggestion; two places that carry the same Latin name (Rhenus = the Rhine or the Reno) are decided by the
edition's context (margin 0.5) or by prominence (20 times the mentions with a context margin of 0.1), never above medium; the hits of one Wikidata item are one place (a river has several points).
The data columns: `gc_source = name_latin`, `gc_tier = latin_<families>`, `gc_key` = the lemma, `gc_evidence` = form, lemma, rule, score, sources, `gc_qid` = the item.
`--eval-latin N` of `reconcile geocode run` hides N placed places of Latin context (and their items) and runs only this path: `holdout_latin_seed{1,2}.{tsv,json}`.

### 19.2 Path B: qualifiers and historical polities (`polities.py`)

`parse_qualified(name)`: 'Head, Qualifier' and 'Name (Other name)' -> head, bracketed alternative names, qualifier type (kingdom, duchy, county, landgraviate, margraviate, principality,
electorate, republic, `union` (Vereinigte), empire, bishopric, circle, confederation, hetmanate, region, lordship ...). **The qualifier sets the kind**: the entity is an area
(`place_kind = area`, `kind_source = qualifier`; name-based points are areas of the radius of the type, `QTYPE_RADIUS`), whatever the head is called elsewhere.
`POLITIES` is a small **reviewed** table (30 entries): label, Wikidata item verified in the local `wd_entities.parquet` (or none), point, radius and `basis` (`wd` = the item's P625,
`seat:<town>`, `centroid:<members>`, `table` = the exonym table's region); heads that are also something else (`Moldau` the river) need a polity qualifier. Order in `Geocoder.resolve`:
(1) `polity` (table on the whole name, head, alternatives; then the exonym table on the head of a qualified name; `name_exonym`, medium, area), generic lookup, then for what is still not
placed (2) `polity_gn` (head as an area, or a bracketed alternative name as any place, through GeoNames; medium at most), (3) `head_alt` ('Löbau (Lubawa)': head and alternative are resolved
**separately**; one confident answer wins, two that agree corroborate, two that disagree are decided by the edition's and the document's places (`ctx_at`, margin 0.8), else nothing),
(4) `regional_unique`, (5) `latin`.

### 19.3 Path C: English small homonyms (Hearth Tax, Casebooks)

`REGIONAL_UNIQUE` collections: a bare village name is placed only when **exactly one** settlement of that exact name lies in the region the document gives (`regional_region`): the county of the
Hearth Tax file (`row_regions`); for the Casebooks the 15 km round the directory's own suggestion (`cb_lat`, `cb_lon`, geocoding high / medium: a 'low'-rated match is a suggestion and a unique
namesake next to it is corroboration), else the 100 km round the centre of the places the collection has placed. Several namesakes in the region (distinct at `UNIQUE_CLUSTER_KM` = 3 km): nothing,
and a GeoNames guess among them is demoted to low (`regional_ambiguous`). A Casebooks guess more than 40 km from the directory's suggestion is only a candidate (`casebooks_conflict`).

### 19.4 The fixes of the check of v2.2

| # | finding | rule now | where |
|---|---|---|---|
| 1 | Old Bailey names in the provinces (Holburn, Petty-France, Gatehouse, Kingsland, Hermitage, Hamstead, Flushing, Green-street) | the London prior is a **jurisdiction prior for every Old Bailey name** (`LONDON_PRIOR['old_bailey'] = 'all'`): London plus Middlesex (`LONDON_BOX` minus the Hertfordshire corner `LONDON_CUT`); exempt: a country, a capital, a populated place of a million (NOT an administrative unit), and, for Old Bailey only, a populated place outside the British Isles ('Flushing, a Town in Holland': exonym Vlissingen) and a provincial town of 100,000 inhabitants (Bristol) | `regions.london_keep`, `Geocoder.row_keep` |
| 2 | MoEML parishes copied each other's point; `North` -> Département du Nord | a London church takes the point of a placed church only with the same dedication AND qualifier (`church_sig`); directions (`north south east west nord ...`) are deictics; the million-inhabitant exemption is for populated places only | `Geocoder.church_filter`, `placeclass.DEICTIC` |
| 3 | Homberg (bullinger:l219) at 50.64 N 8.11 E | `coord_overrides.tsv`: the Homburg of Sayn-Wittgenstein at Nümbrecht (GeoNames 2899448) | data |
| 4 | compound strings fell back to the province centroid or the point of Europe | a compound 'Container, Leaf' (container = country, exonym-table region, or a province / region that names no settlement: `area_cc`) resolves on the **leaf only** (`Forms.leaf_only`: no comma key, no middle part as fallback); the nearest container anchors the leaf with the radii of a region (`qual_anchor`, generous); unknown leaf: unplaced | `Geocoder.adjust`, `qual_anchor` |
| 5 | Casebooks directory points of low geocoding confidence shown as points | `cb_geo_conf = low` and radius < 20 km: not used (a suggestion, `cb_used` False); an area-level point (radius >= 20 km) stays an area; a name-based guess more than 25 km from such a matched entry is dropped too (`casebooks_directory_conflict`: the Charterhouse of London must not become that of Somerset) | `reconcile_v2.apply_casebooks` |
| 6 | display names from surface forms; `lsjbot Duisburg` | `name_en`: Wikidata label, exonym label, TGN, GeoNames, **the register name** (`register_name`), a candidate's name (propagation, Latin), Casebooks entry, only then a surface; bot labels are never names (`BOT_LABEL`); `Q31976906` is Duisburg (`DUPLICATE_ITEMS`) | `cards_globe` |
| 7 | duplicate polity points (France x3, England, Sweden, Poland; 505 pairs of the same name within 5 km) | (a) an exonym or polity result joins the cluster of its Wikidata item (`qid_exo`, `qid_cluster`; `EXONYM_QID` verified against the item's English label); (b) `point_unions` hands `merge_points` the pairs that are one point for another reason than the name: the same candidate (`coord_cand`), a candidate that is the GeoNames id of another place, a propagation and its source, an exonym and the item of the same label, the same display name within 5 km (an area within half its radius); a pair of different kinds (a canton and its capital) is never joined, except an administrative area named after its seat (`Zerbst` / `Anhalt-Zerbst, Fürstentum`, `Ballenstedt(-Hoym), Amt`) | `reconcile_v2.attach_exonym_qids`, `point_unions`, `reconcile_build.merge_points(extra=)` |
| - | Veluwe (a hamlet 45 km south), `heilige Land` (a village near Nijmegen) | exonym table: Veluwe (GeoNames AREA 2745666), Holy Land (area) | `exonyms.py` |
| - | Fryburg (1644, ckcc) = Freiburg im Breisgau; Ecluse near Arras | `coord_overrides.tsv`; `geocode_rejects.tsv` (no coordinate of the fort is available locally) | data |
| - | Solothurn (Bullinger canton, Q11929) on the city | the register point of an entry typed region / canton / province ... whose authority is an area too is replaced by the authority's centroid when they are 2-100 km apart (`reg_area_replaced`) | `reconcile_v2._places` |
| - | kinds: Veluwe c->a, Rorbach n->c, Hocht a->c, Lingen c->a, Green Street c->b, Ludgate church c->b | qualifier kind (Lingen, Grafschaft); no `-bach` rule (`placekind.name_kind`); 'Parish of St. X' is a building; `data/kind_overrides.tsv` for the individual points | `placekind`, `kind_overrides.tsv` |

### 19.5 Letters

`ratlas.cards_globe build` writes `letters.json`, `letters_docs.json` (the letters of each edge: title, date, year, sender, addressee, edition index, url) and `letters_coverage.tsv`;
`cards_globe_v23.sbatch` runs it on Hypatia, the `letters` subcommand rebuilds only these three files from the final tables (`tests/test_v23_build.py`).

### 19.6 Tests

`tests/test_v23_geocode.py` (Old Bailey prior, parishes, compounds, polities, head / alternative names, regional homonyms, Latin path and its hold-out), `tests/test_v23_build.py`
(Casebooks, areas, qualifier kind, exonym clusters, kind overrides, merged points, display names, letters), and changes to `test_regions_v22.py`, `test_reconcile_v2.py`.

### 19.7 Result of the build of 2026-10-20

In-period place mentions placed 461,375 (80.63 %) -> 474,076 (82.85 %); globe points 16,870 -> 14,798; same-name pairs within 5 km 677 -> 60; Latin names 3,832 placed mentions (629 local places); the nine audit checks pass; 488 tests pass on the Mac (11 skipped), the whole suite with the real-data tests passes on Hypatia. Numbers per edition and per path, losses and remaining issues: `RQ/derived/atlas/v2.3/rapport_v2_3.md`; the numbers are produced by `scripts/v23_numbers.py`.

## 20. v2.4: the fixes of the independent check of v2.3, one vocabulary of place types, names and variants (2026-10-20)

v2.4 is v2.3 plus the changes below (`slurm/v24_chain.sh`, steps in `slurm/v24.sbatch`; `out_v2_4/`, `reconcile_v2_4/`, `geocode_v2_4/`; the globe shards from `slurm/cards_globe_v24.sbatch`,
which now also writes `extra.json`; published to `RQ/derived/atlas/v2.4/`; v1 .. v2.3 and `RQ/data/raw` are never written). `slurm/v24_reconcile_chain.sh` re-runs the reconcile half when
only the code of that half changed (not when a collection module changed: `hearth_tax` and `h15corpus` are extracted again by the full chain).

### 20.1 Latin names of several places (`latin_homonyms.py`)

`Lugduni`, `Lugdunum`, `Lugdunenses`, `Lugduno`, `Lugdunensi` (about 290 mentions of the Dutch letters of `ckcc_epistolarium`, all on Lyon in v2.3, a majority meaning Leiden) were given to Lyon by
the exonym table (v2.2) and by the Latin path (v2.3: the guard `LATIN_HOMONYM_CTX` let Lyon through because the letters name other French places too). The bare Latin forms are no longer in the
exonym table; `latin_homonyms.FAMILIES` lists the Latin names that belong to several places (Lugdunum = Lyon or Leiden; Traiectum = Utrecht, Maastricht or Frankfurt (Oder); Augusta = Augsburg, Turin
or Trier; Hadria = the Adriatic Sea or the town of Adria) with, for each candidate, a **cue** (a regular expression on the passage: `Batavorum`, `Rhodani`, `ad Mosam`) and whether it **needs** a cue.
`Geocoder.resolve` asks the family first (`Geocoder.latin_homonym`, `latin_homonyms.decide`): a candidate named by the passages of the place's mentions is placed (medium; share of passages
`CUE_SHARE_MIN` or support of the documents' places `DOC_CTX_MIN`; a candidate that needs a cue (Lyon, Turin ...) needs `CUE_NEEDS_SHARE` = 80 % of the passages, or `CUE_SHARE_STRONG` = 50 % with the documents' support: the 'Augusta' of the Dutch letters is Turin in 11 passages, Augsburg in 9, so nothing); with no cue the documents' places decide (`DOC_CTX_MIN`, margin `DOC_CTX_MARGIN`) but only for a candidate that does not need
a cue (Leiden, Utrecht, Augsburg, the sea; **not** Lyon, Maastricht, Turin ...); cues for two candidates, or no support: unplaced (`why` = `latin_homonym_<reason>`). The decision is per local place
(a surface form in an edition): a form that is Leiden in some letters and Lyon in others is not placed. The passages come from `mention_aux` (`reconcile_geocode.load_mention_aux`, read from
`--mentions`). `Lugduni Batavorum`, `Lugduni Bat.`, `Lugduno Batavorum` are explicit forms of the exonym table (Leiden).

### 20.2 Polities, Latin names from one source, surfaces

* `polity_gn` never accepts a TGN-only candidate or a locality (`LCTY`) as the head of a polity qualifier (`POLITY_HEAD_FCODES`: a GeoNames ADM / RGN / AREA / TERR or a country); the check after the
  generic lookup is the same. `polities.POLITIES` has five reviewed entries more (Stolberg im Harz, Gaster (Landvogtei, St. Gallen), Oberstein (Idar-Oberstein), Hohenlohe, Falkenberg (Niemodlin));
  the keys are the whole qualified names, never the bare head. `Werdenberg, Grafschaft` of the Augsburg peace is rejected (`geocode_rejects.tsv`): the table does not pin it.
* Latin path: the Orbis Latinus alone (one family) places a name only with a strong rule (>= 0.8), no masculine adjective ending (`LATIN_ADJECTIVE_END`) and the support of the documents' places
  (`LATIN_HOMONYM_CTX`); otherwise a suggestion (`why` = `latin_orbis_alone`). Two families that are not a curated table are also a suggestion when the documents name other places and none is near
  (`LATIN_FAR_CTX`), unless the place is prominent. `latin_geo.NOT_PLACES` (Palatinus, Tiberina, Romuleae, Castalia ...) are never looked up. A place whose mentions' surface is not the form of its name
  (`surface_compatible`: the first three letters; `SURFACE_OK_MIN`) is not read by the Latin path (`why` = `latin_surface_mismatch`): the register standardises `Tagum` as `Fagum`.
* The explicit rejects of the check are in `data/geocode_rejects.tsv` (Coronia, AEgea, Becam, Nordovico, Palatinus x3, Fagum, Old Bailey New-street / St. Helens / Bow, Werdenberg).

### 20.3 Individual corrections

| case | rule |
|---|---|
| Lucena (Córdoba) | `head_alt`: a head that is a settlement within `LOCATOR_KM` of a larger bracketed place that locates it is the answer (`head_alt_head_locator`) |
| Kirkby (Hearth Tax) | `collections/hearth_tax.py`: the key of every heading is scoped to the county of its file (`surf:kirkby @lancashire`, `hearth_tax.place_key`, `file_county`) |
| Heath (Casebooks) | `coord_overrides.tsv`: the directory entry Heath and Reach, Leighton Buzzard, Bedfordshire |
| Italia, Milán (h15corpus) | `collections/h15corpus.py`: a name whose TEI geos contradict each other (more than `GEO_AGREE_KM`) gets no TEI point; the geocoder reads the leaf; a name whose geos agree keeps the geo of its own letter (no median any more) |
| Cross Keys Inn x3 | `Geocoder.CHURCH_TOKS` has `inn tavern alehouse`: the saint / qualifier rule applies to inns |
| Thorney, St. Pancras (Soper Lane) | `coord_overrides.tsv`: Westminster island; the City parish on Pancras Lane (MoEML's own register point of the lane) |
| Samothrace, Jettenbach Schloss | `kind_overrides.tsv` n, b |
| Hadriae | `latin_homonyms`: the Adriatic Sea (exonym table entry, kind n) unless the passage names the town |
| Hearth Tax county discs | `regions.in_disc`: a county disc lets a point through only if the point is, relative to the radii, not closer to another county of `COUNTY_REGIONS` (`rival_ok`, `COUNTY_FAMILY`) |
| exonym table reread | removed `vienne`, `indias`, `rutheni ruthenia`, `regiomontanus` (see the header of `_RAW`) |

### 20.4 Display data

* Continents: `reconcile_v2.CONTINENTS` / `apply_continents` give Europe, Asia, Africa and the Americas one point and one radius whatever the source (Wikidata's P625 of Europe was a district of
  Württemberg); `EXONYM_QID` has the four continents. The type `continent` is in `extra.json`; the page does not pin a continent.
* `casebooks_low_candidates.tsv` (next to the final tables): the matched Casebooks places whose directory point has a geocoding rated `low`; they stay unplaced, as in v2.3 (decision pending),
  with the entry, county, point and `london` / `suggested_radius_km` so that they can become areas of London level.
* `extra.json` (next to `core.json`, layout of `core.json` unchanged): `{"types": {code: [label_en, parent_code, coarse_kind]}, "rows": {"<row index>": [type_code, [[variant, lang, n], ... top 10],
  name_based_mentions]}}`; produced by `python -m ratlas.name_variants extra` at the end of `cards_globe_v24.sbatch`, with `global_places_names_types.tsv` (reference form, folded key, variants,
  type per global place) and `placetype_coverage.tsv`. Vocabulary, mapping rules, coverage and limits: `docs/NORMALISATION.md`.

### 20.5 Tests

`tests/test_v24.py` (one test per fix), `tests/test_placetypes.py`, `tests/test_name_variants.py`, and changes to `tests/test_v23_geocode.py` (the three v2.3 tests that encoded the behaviours
v2.4 changes: the Orbis alone, Lugduno in a Dutch cloud, the hold-out of the Latin path).

### 20.6 Result of the build of 2026-10-20

In-period place mentions placed 474,076 (82.85 %) -> 473,599 (82.76 %); globe points 14,798 -> 14,795; local places placed 28,629 -> 28,523 (the Hearth Tax has 7,421 local places instead of 7,142: the headings are scoped
to the county of their file); the nine audit checks pass (check 4: 11 reviewed exceptions, 3 of them the county-scoped keys of Hearth Tax headings that were already allowed under the old keys); `audit editions`: 207 files,
25 flagged, 262 places flagged (v2.3: 264); suite: 749 passed, 12 skipped on the Mac, exit code 0 on Hypatia with the real-data tests; `extra.json`: 14,795 rows, 54 types, 0 term of 20+ mentions without a reviewed
row. The numbers per edition and per case: `RQ/derived/atlas/v2.4/rapport_v2_4.md`, `v23_v24_numbers.json` (`scripts/v24_numbers.py`).
