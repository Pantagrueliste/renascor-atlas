"""Rubens-online (Herzog August Bibliothek Wolfenbuettel): sources and documents on the life and work of Rubens.

307 archival documents (archivalien: contracts, payment books, letters about Rubens; Dutch, French, Latin, German
with German regests) and 46 excerpts of printed lives and sources (viten). Mentions are <rs type="person|place|
corporate|work" ref="#slug">: person, place and corporate (-> org) are kept; 'work' (Jaffe catalogue numbers) is not
a person/place/org and is skipped. The ref slugs (#rubens_peter_paul, #antwerpen, #michaelsorden_frankreich) point
to four register files declared as external entities in the DOCTYPE (diglib.hab.de/edoc/ed000083/register/*.xml);
these registers are NOT in the harvest and are not fetched (no network in this stage), so the slugs are opaque keys
'rubens_online:<slug>' (in_register False), shared by archivalien and viten. No authority ids, no coordinates.
<index><term type="TGN|PND"> blocks hold plain labels, no ids: not used.

v2: the four registers declared as external entities (``ed000083/register/register-person|place|corporate.xml``, HAB,
CC BY-SA) are harvested (``rubens_online-registers/``) and loaded by ``prepare()``: 99.4% of the mentions resolve; persons
carry PND/GND keys (482), places TGN ids (211, no coordinates), corporate bodies become orgs. ``register-work`` (Jaffe) and
``register-bibl`` are not entity registers. Slugs missing from the registers keep an opaque key.

Places inside msIdentifier (country / settlement of the archive) get role 'repository' and zone ``repository`` (excluded
from the default counts: Belgien 180, Antwerpen 171... were the top 'places' of the edition).
Dating: archivalien by msDesc/history/origin/origDate (header); viten by the year in the file name
(date_source 'filename', e.g. Bullart-Academie-des-Sciences-1682.xml), else the header (imprint year); 10 viten of
1708-1766 are outside the period (in_period False); 2 undated viten stay None.
Licence: LICENSE_TIERS puts it in T3 (CC-BY-SA applied by a library-wide rule, not stated on the edition; the
footer reads (c) HAB): row licence 'CC-BY-SA-4.0' with tier T3, so no context snippets are stored.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch1 as H
from .. import helpers_v2 as V

SLUG = 'rubens_online'
COLLS = ['renascor_2026-10/rubens-online']
RENASCOR_IDS = ['rubens-online']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T3'
DESCRIPTION = 'Rubens-online archival documents and printed lives: persons, places and corporate bodies'
OPTIONS = {'opaque_keys': True, 'type_map': {'corporate': 'org'}}


_REG = V.NEW + '/rubens_online-registers/ed000083/register/'


class Extractor(H.BaseTEI):
    def prepare(self, ctx):
        return V.load_registers(ctx, [_REG + 'register-person.xml', _REG + 'register-place.xml',
                                      _REG + 'register-corporate.xml'], fix=V.hab_person_fix)

    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.name != 'mets.xml']

    def doc_date(self, root, rec, ctx):
        if '/viten/' in rec.path:
            y = C.year_from_filename(rec.name, 1450, 1800)
            if y:
                return C.DocDate(y, y, 'filename')
        return super().doc_date(root, rec, ctx)


EXTRACTOR = Extractor()
