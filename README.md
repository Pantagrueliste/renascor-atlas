# Renascor Atlas

Open database of the **persons and places tagged by editors** in the TEI editions harvested by
[Renascor](https://pantagrueliste.github.io/renascor) (1450-1700). v1 = editor-tagged entities only (no NER).
Places go on a globe first; the persons index comes later from the same tables.

Package `ratlas` (src layout, managed with `uv`, Python >= 3.11, 3.12 pinned in `.python-version`).

## Data model (two tables per collection slug)

* `mentions.parquet`: one row per tagged mention (persName, placeName, orgName, rs@type=..., name@type=...):
  collection, catalogue id, source path, document id and dates, kind (person/place/org), tag, role (mention,
  sender, addressee, sent_from, sent_to, author, signer...), zone (text, header, editorial_commentary, apparatus, footnote,
  regest, repository; v2) and part (txt/hdr), person_class and excluded (default atlas counts, v2), surface text, local key resolved
  against the edition's register, authority ids (GND, VIAF, Wikidata, GeoNames, TGN, ULAN, Pleiades, HLS),
  a <= 120 character context (T1/T2 only), language, licence, tier.
* `entities.parquet`: one row per register entry of an edition (edition-scoped): name, variants, coordinates,
  birth/death as given, authority ids, note type, number of mentions.

Exact schemas: `src/ratlas/model.py`. Contract for extraction modules: `docs/CONTRACT.md`.

## Layout

    src/ratlas/model.py        dataclasses + pyarrow schemas
    src/ratlas/common.py       shared helpers (scan access, parsing, authority ids, dates, scanner, registers, writer)
    src/ratlas/collections/    one module per collection (SLUG.py) + generic_tei.py (default extractor)
    src/ratlas/runner.py       prepare registers, extract, write out/<slug>/
    scripts/push_coll.sh       rsync the raw files of collections to Hypatia
    scripts/push_repo.sh       rsync the code to Hypatia + uv sync
    slurm/submit.sh            one Slurm array task per slug (debug partition, <= 8 cpus)
    tests/                     pytest

## Commands

    uv sync
    .venv/bin/python -I -m pytest                          # unit tests (tiny, run with nice -n 19 on the Mac)
    .venv/bin/python -I -m ratlas list
    .venv/bin/python -I -m ratlas inspect <coll-or-prefix>  # what the scan says about a collection
    .venv/bin/python -I -m ratlas extract <slug> [--limit N] [--out DIR] [--workers N]

## v2 (rebuild of 2026-10-17)

Re-harvested editions and registers (pinned SHA in `SOURCES.tsv`), refined zones, `person_class`, default exclusions, kind-mismatch
rule, stable keys, range dates: see `docs/CONTRACT.md` section 13. Rebuild on Hypatia:

    scripts/push_repo.sh                                       # code, scan.jsonl.gz
    rsync RQ/derived/atlas/v2/scan_extra.jsonl.gz hypatia:renascor-atlas-data/      # or: sbatch slurm/scan_extra.sbatch
    OUTNAME=out_v2 CONCURRENT=3 slurm/submit.sh $(ls src/ratlas/collections/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -v -e '^__init__$' -e '^generic_tei$')
    sbatch --dependency=afterok:<array job> --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/merge_v2.sbatch

## Reconcile v2 (2026-10-17): gates, arbitration by names, barriers

Authority ids are now gated (class, names in ten languages, `@cert`, chronology, homonyms; `docs/CONTRACT.md` section 14): an id that fails is
`suggested` or `rejected` and listed in `gate_log.tsv`. Counts are the atlas counts (no commentary, footnotes, repositories, characters ...).
Register points of countries are dropped (areas get a coarse point), conflicts are arbitrated by names and flagged over 100 km, the geocoder has
barriers (classes, deictics, exonym table, regional prior, register type). Each collection module states its licence and
tier (`LICENCE`/`TIER` constants), stamped on every row.

    .venv/bin/python -I -m ratlas reconcile ids --v1 RQ/derived/atlas/v2 --dest RQ/derived/atlas/reconcile_v2      # Mac
    .venv/bin/python -I -m ratlas reconcile authority fetch --ids .../reconcile_v2/ids.tsv --old-rdir .../reconcile --dest .../reconcile_v2   # Mac, network, nice
    sbatch slurm/gnscan.sbatch; sbatch slurm/build2.sbatch --dest $HOME/renascor-atlas-data/out_v2/reconcile/pass1     # Hypatia
    # then namecheck_v2, build2 (pass2), geocode_index_v2, geocode_run_v2, build2 (pre_verify), verify_v2, build2 (final): see CONTRACT 14
    .venv/bin/python -I scripts/compare_versions.py                                                                  # v1 / v1.1 / v2 numbers

### Manual review (2026-10-17)

The audit's doubtful cases were read by hand against the sources (`RQ/derived/atlas/v2/manual_review.tsv`: verdict correct / wrong /
undecidable, evidence, what v2 did, what it does now). The confirmed corrections are data and survive regeneration: `data/id_overrides.tsv`
(accept / reject / suggest an authority id of an entity, or create a Wikidata pair), `data/geocode_rejects.tsv` (name-based results taken back
by hand) and `data/coord_overrides.tsv`; copy the three to the reconcile directory before `build2` (`src/ratlas/manual_overrides.py`,
`docs/CONTRACT.md` section 15, `tests/test_manual_overrides.py`). `slurm/build2_manual.sbatch` rebuilds the final tables on a frozen copy of the inputs.

## v2.1 (final build of 2026-10-17)

One clean chain from scratch, written to `~/renascor-atlas-data/out_v2_1/` and published to `RQ/derived/atlas/v2.1/` (v1, v1.1 and v2 are never touched):
the manual-review overrides are part of the main build, tagged pronouns and common nouns are excluded by default (`generic_terms.py`), the Casebooks
directory places the PLACE ids it matched (`casebooks_directory`), the local Getty TGN verifies TGN ids by names, gives points (`tgn`) and names for the
geocoder, and the homonym gate and the verify rule are recalibrated on labels (`scripts/calibrate_rules.py`, `data/calibration_v2_1.json`).
`docs/CONTRACT.md` section 17.

    scripts/push_repo.sh                          # code to Hypatia
    ssh hypatia 'cd ~/renascor-atlas && slurm/v21_chain.sh'     # extract, merge, gnscan, build2 x4, namecheck, geocode, verify, audit, pytest (Slurm only)
    scripts/publish_v21.sh                        # copy to RQ/derived/atlas/v2.1 (+ version_comparison.json for v1, v1.1, v2, v2.1)

## Name-based geocoding (v1.1)

Places without coordinates get a separate, last-resort coordinate source (`coord_source` `name_propagation` or
`name_geonames`, with `coord_conf` high / medium / low, `radius_km`, the matched candidate and the evidence in the
`gc_*` columns of `places_local_geo.parquet`). Register, Wikidata and GeoNames-id coordinates are never overwritten;
only high / medium confidence fill `lat` / `lon`, enter `places_global.parquet` and the globe (`s` = `p` / `n`).

    sbatch slurm/geocode_index.sbatch                       # compact GeoNames name index (allCountries.zip already on Hypatia)
    sbatch slurm/geocode_run.sbatch --eval-n 1000 --eval-seeds 1,2   # geocode + precision self-check (hidden placed places)
    sbatch slurm/reconcile_build.sbatch --dest ~/renascor-atlas-data/out/reconcile_v1.1 \
        --geocode ~/renascor-atlas-data/geocode/geocode_places.parquet   # regenerate places_*, globe, reports

Code: `geotext.py` (surface cleaning, keys, Latin endings, spelling skeleton), `reconcile_geocode.py` (propagation,
GeoNames ranking with collection / document priors, corroboration of the two methods, hold-out evaluation),
`reconcile_build.py` (`--geocode`). Method and thresholds are described in the module docstring and in `TH`.

### Audit of the name-based coordinates (v1.1 verified)

The hold-out self-check overstates precision (the hidden places are the well-known ones). A read-through of ~800 newly placed
places against the TEI passages (stratified by collection, method and confidence) found 62 % to 99 % depending on the
stratum; strata below 85 % are taken back from the globe (they stay in `gc_*` as low-confidence candidates, with
`verify_rule` naming the rule). Rules, constants and the reasons are in `verify_geocode.py`.

    sbatch slurm/verify_sea.sbatch      # diagnostic: distance to the nearest populated place (GeoNames class P)
    sbatch slurm/verify_build.sbatch    # rules -> geocode_verify.tsv, then the reconcile build with --geocode-verify
                                        # (unaudited tables stay in out/reconcile_v1.1; audited ones in out/reconcile_v1.1_verified)


### Audit tests of v2 (the nine checks of the audit as blocking tests)

`docs/CONTRACT.md` section 16. The logic of every check is tested on tiny tables in `tests/test_audit_v2.py` (always); the checks on the real
tables run on Hypatia (`RATLAS_AUDIT_V2=1`, zero failures required; reviewed exceptions in `data/audit_v2_allow.tsv` with their reason):

    sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/audit_v2.sbatch          # CLI: ratlas audit v2 ... then the real-data pytest
    sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/audit_editions.sbatch     # fresh sample: 3 files + 10 top places per edition

## Rules (short)

* Mac: read, edit code, run unit tests and tiny checks only (`nice -n 19`, one process, < ~10 s). Extraction runs
  on Hypatia through Slurm, never on the login node.
* Never modify anything under `RQ/` except `RQ/derived/atlas/`; never touch the Renascor repo, the frozen DB,
  `text_archive_v1.0` or `RQ/corpus_private`. Open SQLite read-only. Use `python -I` whenever a script reads
  harvested files and never execute anything from the harvest.
* London Lives is counting-only (excluded everywhere: `common.FORBIDDEN_PATH_TOKENS`); Medici Archive Project:
  do not touch.
* Every row carries the collection's licence and tier (T1/T2/T3). Context snippets are stored only for T1/T2.
* Nothing is committed, pushed or published without an explicit decision. Network only where a stage allows it
  (User-Agent `Renascor-Atlas/0.1 (research; https://pantagrueliste.github.io/renascor)`, robots.txt, rate limits).

## v2.2 (sample check of v2.1, 2026-10-19)

An independent sample check of v2.1 found errors that v2.2 corrects in one clean chain (`slurm/v22_chain.sh`, output `~/renascor-atlas-data/out_v2_2/`, published to
`RQ/derived/atlas/v2.2/`; nothing committed, nothing published): the `wrong` verdicts of the manual review are hard overrides (`scripts/make_hard_overrides.py`) and the homonym gate
looks at the larger namesake of the region; the geocoder takes priors from the document itself (county of a Hearth Tax file, Greater London for MoEML and the London names of Old
Bailey, per-file consistency; `regions.py`); common nouns that carry an id stay linked but are flagged `generic_with_id` and are never a display name, mentions of several entities at
once leave the per-entity counts (`multi_referent`); the TGN is checked against its own hierarchy and names (fuzzy threshold 0.90, `tgn_coord_check.tsv`); towns of the Czech
Republic or the United Kingdom are no longer areas; the globe's kind is five-way `c a n b u` (`placekind.py`); `letters.json` is the letter network between the points of the globe
(`cards_globe.py`). Details and the numbers: `docs/CONTRACT.md` section 18.

    scripts/push_repo.sh
    slurm/v22_chain.sh                                   # extract .. final, the nine checks, the whole suite
    sbatch slurm/cards_globe_v22.sbatch                  # globe shards and letters.json (CARDS=$HOME/renascor-atlas-data/cards_v22, inputs in $CARDS/in)
    scripts/publish_v22.sh                               # Mac: rsync to RQ/derived/atlas/v2.2

## v2.3 (Latin names, historical polities, English homonyms, check of v2.2, 2026-10-19)

Built on v2.2 in one clean chain (`slurm/v23_chain.sh`, output `~/renascor-atlas-data/out_v2_3/`, published to `RQ/derived/atlas/v2.3/`; nothing committed, nothing published).
**Latin names** (`latin_names.py`, `latin_geo.py`, `data/orbis_latinus.tsv`; coordinate source `name_latin`, globe code `l` = located by name): an unplaced name in a Latin context is
de-inflected and looked up in the Wikidata Latin labels of the atlas's items, the Orbis Latinus, the exonym / polity tables and the Latin editions' placed names, with the homonym,
regional and strength-of-rule gates. **Qualifiers and polities** (`polities.py`): 'Name, Qualifier' and 'Name (Other name)' are split, the qualifier sets the kind (an area), a small
reviewed table of historical polities has Wikidata items and documented points. **English homonyms**: Hearth Tax and Casebooks villages are placed only when exactly one namesake lies in
the county of the file / next to the directory's suggestion. **Fixes of the independent check of v2.2**: a London-plus-Middlesex prior for every Old Bailey name, London churches by
dedication and qualifier, compound strings on the leaf, Casebooks points of low geocoding confidence not shown as points, display names from the register before surface forms,
duplicate polity points merged, individual corrections. Details: `docs/CONTRACT.md` section 19; numbers: `RQ/derived/atlas/v2.3/rapport_v2_3.md`.

    nice -n 19 rsync -a --exclude .venv --exclude .git --exclude __pycache__ ./ hypatia:renascor-atlas/       # (or scripts/push_repo.sh)
    slurm/v23_chain.sh                                   # extract .. final, the nine checks, the whole suite, the globe shards
    slurm/v23_reconcile_chain.sh                         # the reconcile half only (when only that code changed)
    sbatch slurm/v23.sbatch trace "Leidam" "Löbau (Lubawa)"   # development: the paths a name takes through the geocoder (Slurm only)
    scripts/publish_v23.sh                               # Mac: rsync to RQ/derived/atlas/v2.3
    .venv/bin/python -I scripts/v23_numbers.py           # Mac: v2.2 against v2.3

## Homonym audit (2026-10-23)

Bullinger's register gives Sodom the coordinates of Sodom, West Virginia, and the atlas trusted register and identifier coordinates as the editors' own: the same
mechanism put Emmaus in Pennsylvania, Troja in Prague, the Vesuvius on Svalbard (a wrong QID of the edition) and, through our name-based geocoder, Ladon in the Loiret.
`src/ratlas/curation.py` (tests: `tests/test_curation.py`) applies, at the end of `reconcile build2` and in the globe page build, three pieces of data:
`data/curation/homonym_fixes.json` (every place read in context: `move`, `unplace`, `not_a_place`, or `keep` for a real homonym), `data/curation/ancient_names.tsv`
(biblical, classical and mythical names and the region their point must lie in) and two guard rules (`ancient`, `far_homonym`) plus the `propagated` rule (a copy of a
corrected point goes too). A moved place stays in the Strict reading only when an identifier of the edition itself confirms the new point; otherwise its source is
`curated` (globe code `k`, Augmented only). The model layer has its own entries in `data/model_layer_overrides.json` (new action `split`: the mentions of some
collections leave a point that mixed two homonyms). Every build writes `homonym_check.tsv` (rule hits and per-edition outliers that no entry decides yet):

    .venv/bin/python -I -m ratlas curation check --atlas RQ/derived/atlas/v2.4 --model <model_layer_v2.json> --samename RQ/derived/atlas/reconcile_v2/gn_samename.parquet

Report: `RQ/derived/atlas/v2.4/audit_homonymes.md`.
