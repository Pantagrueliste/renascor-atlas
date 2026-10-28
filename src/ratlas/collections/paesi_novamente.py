"""Paesi novamente retrovati (Vicenza 1507, Italian) and Newe unbekanthe landte (Nuremberg 1508, German), HAB
Wolfenbuettel, transcription and annotation by Norbert Ankenbauer: two diplomatic TEI files.

Tagged: <rs type="person|place" ref="#slug"> in the running text (3,598 of 3,599 rs have a ref; the single rs type
'nom2' is no person or place and is not read). The slugs are readable, stable inside the edition and shared by both
files (#kozhikode, #portugal, #colombo_c, ...), but the files contain no register: ``opaque_keys`` gives
local_key = 'paesi_novamente:<slug>' (in_register False). No entities are produced; the place slugs mix cities, regions,
continents and seas (a type filter is needed before geocoding by name). Text is read diplomatically (orig / abbr kept,
reg / expan dropped), so surfaces are the 1507/1508 spellings.
v2 (issue 169): the editor's slug names the place ('kozhikode', 'cap_vert', 'kochi', 'malindi', 'rotes_meer') while the surface is the 1507/1508
spelling ('Calichuth', 'Capouerde', 'Cuchin'): ``prepare()`` builds one entity per slug whose NAME is the slug with underscores turned into
spaces and capitalised (the 1507/1508 spellings are its variants), ``note_type`` 'editor_slug' so that a geocoder reads the name from the
slug and not from the commonest surface ('Cathayo' = #china was put in Italy by its spelling).
Dating: imprint year in sourceDesc (1507, 1508; header).
Licence: LICENSE_TIERS and the catalogue say CC-BY-SA-3.0-DE (T1), but both TEI headers carry
availability status="restricted" with a HAB copyright link. As the contract asks for the stricter reading the module is
T3 'LicenseRef-HAB-restricted' (no context stored) until the discrepancy is cleared; flip LICENCE/TIER to
CC-BY-SA-3.0-DE / T1 if the HAB statement is confirmed.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .. import common as C
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'paesi_novamente'
COLLS = ['renascor_2026-10/paesi-novamente-retrovati-hab']
RENASCOR_IDS = ['paesi-novamente-retrovati-hab']
LICENCE = 'LicenseRef-HAB-restricted'
TIER = 'T3'
DESCRIPTION = 'Montalboddo voyage anthology, 1507 Italian and 1508 German: persons and places as slugs'
OPTIONS = {'opaque_keys': True}


def _humanize(slug: str) -> str:
    return ' '.join(w[:1].upper() + w[1:] for w in slug.replace('_', ' ').split())


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        forms: dict = defaultdict(Counter)
        kinds: dict = defaultdict(Counter)
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for h in C.scan_mentions(root, type_map=ctx.type_map):
                ref = (h.refs.get('ref') or '').strip().lstrip('#')
                if ref and ' ' not in ref and h.surface:
                    forms[ref][h.surface] += 1
                    kinds[ref][h.kind] += 1
        out = []
        for slug in sorted(forms):
            kind = kinds[slug].most_common(1)[0][0]
            ent = Entity(local_key=C.make_key(ctx.slug, slug), coll_slug=ctx.slug, kind=kind, name=_humanize(slug),
                         variants=[v for v, _ in forms[slug].most_common(6)], source_path='(editor slugs)',
                         renascor_id=ctx.renascor_id, note_type='editor_slug', licence=ctx.licence, tier=ctx.tier)
            ctx.shared.add(ent, [slug])
            out.append(ent)
        return out


EXTRACTOR = Extractor()
