"""DraCor English drama corpus (753 plays from EEBO-TCP / EarlyPrint, 1485-1700).

Licence: CC-BY-NC-3.0, T2 (LICENSE_TIERS DraCorEnglish, to be confirmed; the TEI headers carry ``licence target=
creativecommons.org/licenses/by-nc/3.0/``; the underlying EEBO-TCP texts are CC0). Raw: ``dracor_eng/``.

What the editors tagged: the cast (``particDesc/listPerson`` in 468 plays: 10,780 characters, fictional or
mythological/historical dramatis personae, no authority ids, local ids ``eng000327-king-of-denmark``) and the author
(``titleStmt/author/persName`` + ``idno type=wikidata``, 753 plays / 201 authors). No places are tagged and the text has
no entity tags. Cast mentions are role ``cast`` (zone hdr), register entries ``note_type`` 'character'; a character is not
a historical person, so person indexes must filter on role/note_type.
Dates: standOff events written (457 plays) else print (296), source ``header``. The same texts exist in the EEBO-TCP
collections (EarlyPrint source): deduplicate before counting texts.

v2: re-harvest of 2026-10-17 (``git clone dracor-org/engdracor`` at da50c42, the TEI served by the DraCor API since
2026-07-21): 753 plays, 741 of them with a cast (17,480 entries; speaker-disambiguated ids such as
``eng000566-mahoone-s-man``), the 52 anonymous plays included. The cast is a list of fictional or historical
dramatis personae: person_class 'character', **excluded from the default counts**; the author (role ``author``,
Wikidata id from the sibling idno) stays a human. The corpus files are in ``dracor_eng_meta/`` and are not read.
"""
from __future__ import annotations

from .. import helpers_batch0 as H
from .. import helpers_v2 as V

SLUG = 'dracor_eng'
COLLS = V.pick(['dracor_eng'], [V.NEW + '/dracor_eng/'])
RENASCOR_IDS = ['dracor']
LICENCE = 'CC-BY-NC-3.0'
TIER = 'T2'
DESCRIPTION = 'English plays 1485-1700: cast lists and authors with Wikidata ids'
OPTIONS = {}
EXTRACTOR = H.DracorTEI()
