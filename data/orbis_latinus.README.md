# orbis_latinus.tsv

Latin place names with the German-spelled modern name, the type and the country of the entry, from **Graesse-Benedict, Orbis latinus, 2nd ed., Berlin 1909**
(Internet Archive item `bub_gb_oJt-AAAAMAAJ`, licence URL Public Domain Mark 1.0; the OCR `_djvu.txt` of the item, parsed by
`scratchpad/atlas/v23prep/parse_orbis_latinus.py`; raw files and `SOURCES.tsv` in `RQ/data/raw/originals_2026-10-19/graesse_orbis_latinus/`).

* Only tiers A (a direct entry, no repair) and B (a resolved cross-reference, a repaired OCR form, a form absent from the second scan) of the parser are kept, and only types
  the geocoder can use (settlements, rivers, regions, islands ...). Tier C (heuristic) rows are not in this file. Rows whose modern name is garbled by the OCR are dropped.
* A Latin key that leads to more than one modern name (`key_n_modern` > 1) is ignored by the geocoder (`latin_geo.OrbisLatinus.get`).
* The book gives no coordinates: the modern name is looked up in GeoNames (class from the type, country from `region_text`). Precision of the parse: about 90 % on a read-through of 60
  random rows before tiering; the geocoder treats the book as one source among several.
* **The OCR text itself must not be republished before the rights of the 1909 revision (Friedrich Benedict) are checked**; this table is a list of names and is internal to the build.
