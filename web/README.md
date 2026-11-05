# First-release interface

This UI layer uses the existing v29 atlas, keeping its data, map renderer, search,
source layers, letter routes and evidence cards. It makes search the primary entry
point, groups secondary options under “Filters & display”, adds accessible date
selectors and a filter reset, and puts methodology in an expandable section.
The basemap also aligns inland-water cutouts with land polygons after date-line
unwrapping, fixing the Caspian Sea being filled as land.
Both the default editor layer and expanded layer remain available. Light and dark
system themes and narrow screens are supported.

Build into a **new, separate directory**; the source is never overwritten:

```sh
python3 scripts/build_release_ui.py /path/to/v29/build /tmp/atlas-release --inherited-assets /path/to/inherited_v24 --letters-docs /path/to/v2.4/letters_docs.json
python3 -m http.server 8880 --bind 127.0.0.1 --directory /tmp/atlas-release
```

Open http://127.0.0.1:8880. No extraction or reconciliation is needed.
The build copies the source assets and writes `release-report.json` with missing
inherited dependencies. Any `licences.tsv` found in the output is deleted and the
report records `licences_table_excluded` — the internal licence/audit table never
ships with the public site. Rebuild from the original v29 source after editing
`web/release.css` or `web/release.js`; choose a fresh output directory each time.

Validation:

```sh
nice -n 19 .venv/bin/python -I -m pytest tests/test_release_ui.py
node tests/test_map_geometry.mjs
CHROME=/path/to/chrome node scripts/check_release_ui.mjs http://127.0.0.1:8880
CHROME=/path/to/chrome node scripts/check_release_ui.mjs http://127.0.0.1:8880 --mobile
```

The browser checks exercise search and card opening, date selection, type filters,
reset, both source layers, letters, density, the methodology disclosure, and
horizontal overflow. They target v29's default count (334,021 mentions).

## Remaining release work

The v29 source build contains only `docs/64.json`; the earlier document and image
indexes are inherited from Hypatia's `~/renascor-atlas-data/cards_v24/out/`.
They have been recovered locally to
`RQ/derived/atlas/page_kit/inherited_v24/`. Pass this folder with
`--inherited-assets` to include them. Existing v29 shards take precedence.
Compatibility was checked against `v25/index_v23.html`, the v29 builder's base:
all 14,795 row names, coordinates and shard assignments agree; three authority IDs
were subsequently enriched in the page. The restored preview contains all 65
document shards, the image indexes and `letters/docs.json` (the v29 page fetches it
the first time a letter card opens; without it the card shows “The letters could not
be loaded.”). The letter keys of the v29 page were checked against that file: all
1,620 keys resolve.

T3 collections are published per the owner decision of 2026-11-04 (recorded in the
project's private licence table): catalogued facts with full source citation, non-profit
research; no context is stored for T3. Internal tier labels stay out of the visible
interface. The report marks the preview publication-ready once the inherited assets fill
the document/image gaps; open audit points (ND context questions, bdpn CC0 scope) remain
internal. Written permission for regensburger_wappen_portraetbuecher was requested from
the project by email on 2026-11-04.

The data pipeline, original build and source counts are unchanged. No commit, tag,
push or deployment is performed by this tooling.
