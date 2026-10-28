"""Philipp Hainhofer, Reiseberichte und Sammlungsbeschreibungen 1594-1636 (annotated digital edition, HAB, heiDATA
DOI 10.11588/data/DVU14P): 16 TEI files, one per journey / collection description.

<rs type="person|place|org|work|literature" ref="psn:|plc:|org:|wrk:|lit:slug"> (28,624 tags, every one with a ref).
Persons, places and orgs are read; 'work' (artworks, Kunstkammer objects: 3,789) and 'literature' (123) are not
persons/places/orgs and are skipped (counted in the report, not in the tables).
v2 register: the LOD file of the same heiDATA record (``hainhofer-lod.xml``, v26 of 2026-07-20, harvested in
``hainhofer_reisen-registers/``, RDF/XML, CC0 per its header; the dataset is CC BY-SA 4.0) keys 3,502 persons, 1,036 places and
55 org groups by the slug of ``schema:mainEntityOfPage`` (.../register/personen/<slug>) and gives the label, GND (the
description's own URI), GeoNames / TGN / ULAN / RKD ``owl:sameAs`` links and birth / death years. ``prepare()`` loads it as the
shared register ('hainhofer_reisen:<slug>', aliases 'psn:<slug>' / 'plc:<slug>' / 'org:<slug>'): 88% of the mentions resolve.
The LOD file only holds entries with a GND record: the other slugs (saints, Christus, local people) stay opaque keys
(in_register False). ``register.xhtml`` has no slug ids and is not read. No coordinates in the register: places are
placed by GeoNames id or by name later.
Dating: the journey year in the file name (Dresden_1629.xml -> 1629), date_source 'filename'; the header only carries
the digital edition's own date. Editorial annotations (<note resp="#editor">) sit inside the text; their tags are zone
``footnote`` (v2) and excluded from the default counts.
Licence: LICENSE_TIERS CC-BY-SA-4.0 (heiDATA record; header agrees), T1.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_v2 as V
from ..helpers_batch3 import opaque_from_prefixed
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'hainhofer_reisen'
COLLS = ['renascor_2026-10/philipp-hainhofer-reiseberichte-sammlungsbeschreibungen-1594-1636-annotated-digital-edition']
RENASCOR_IDS = ['philipp-hainhofer-reiseberichte-sammlungsbeschreibungen-1594-1636-annotated-digital-edition']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Hainhofer journeys 1603-1636: rs person/place/org with psn:/plc:/org: slugs, register not in the harvest'
OPTIONS = {'filename_dates': True, 'default_years': (1594, 1636)}


_LOD = V.NEW + '/hainhofer_reisen-registers/hainhofer-lod.xml'
_RDF = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
_KIND = {'personen': ('person', 'psn'), 'orte': ('place', 'plc'), 'koerperschaftsregister': ('org', 'org')}


def load_lod(ctx) -> list:
    root = V.read_xml(_LOD)
    out: list = []
    if root is None:
        return out
    for d in root:
        if C.ln(d.tag) != 'Description':
            continue
        about = d.get(_RDF + 'about') or ''
        mep = next((C.norm(c.text) for c in d if C.ln(c.tag) == 'mainEntityOfPage' and c.text), '')
        m = re.search(r'/register/([a-z]+)/([^/?#]+)$', mep)
        if not m or m.group(1) not in _KIND:
            continue                      # artworks (objekte) and the dataset description
        kind, prefix = _KIND[m.group(1)]
        slug = m.group(2)
        label = next((C.norm(c.text) for c in d if C.ln(c.tag) == 'label' and c.text), slug)
        toks = [about] + [c.get(_RDF + 'resource') for c in d if C.ln(c.tag) == 'sameAs' and c.get(_RDF + 'resource')]
        auth = C.parse_auth([t for t in toks if t])
        by = {C.ln(c.tag): C.norm(c.text) for c in d if C.ln(c.tag) in ('birthDate', 'dateOfBirth', 'deathDate', 'dateOfDeath')
              and c.text}
        key = C.make_key(ctx.slug, slug)
        if key in ctx.shared.entities:
            key = C.make_key(ctx.slug, f'{prefix}:{slug}')
        ent = Entity(local_key=key, coll_slug=ctx.slug, kind=kind, name=label, source_path=_LOD,
                     renascor_id=ctx.renascor_id, auth=auth, note_type=kind,
                     birth=by.get('birthDate') or by.get('dateOfBirth'), death=by.get('deathDate') or by.get('dateOfDeath'),
                     licence=ctx.licence, tier=ctx.tier)
        ctx.shared.add(ent, [f'{prefix}:{slug}'])
        out.append(ent)
    return out


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        return load_lod(ctx)

    def adjust(self, m, hit):
        if not m.local_key:
            opaque_from_prefixed(m, SLUG, ['psn', 'plc', 'org', 'wrk', 'lit'])
        return m


EXTRACTOR = Extractor()
