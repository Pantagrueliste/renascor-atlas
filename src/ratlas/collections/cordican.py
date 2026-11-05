"""CORDICan — Corpus Documental de las Islas Canarias (Universidad de La Laguna, LexHis group), 1095 TEITOK documents.

What the editors tagged (scan of the 2026-11 harvest): persName and name in the headers (2,130 persName, 3,241 name,
mostly scribes/authors and repository places of the msDesc), placeName in the running text (1,307), dates in 1,094
of 1,095 files (<date when=...> in the header). Default GenericTEI extraction applies: header mentions zone 'hdr',
text mentions zone 'txt', document date from the header. No listPerson register files.

Licence: no licence or copyright statement on the corpus page or in the TEI headers (catalogue entry: Not stated);
LicenseRef-unclear, T3 per the LICENSE_TIERS convention for Renascor_ records without a statement (internal work,
no context snippets).
"""
from __future__ import annotations

SLUG = 'cordican'
COLLS = ['renascor_2026-11/cordican']
RENASCOR_IDS = ['cordican']
LICENCE = 'LicenseRef-unclear'
TIER = 'T3'
DESCRIPTION = 'Canary Islands documents (16th-19th c.): header persons/scribes and repository places, text placeNames'
OPTIONS = {}
