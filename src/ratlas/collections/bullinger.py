"""Bullinger Digital: 13,114 one-letter TEI files (Zurich ICL, HTR text), correspondence of the Zurich reformers.

v2: re-harvest of 2026-10-17 (``git clone stazh/bullinger-korpus-tei`` at dc9fb52, 17 Aug 2026; 10,227 letters differ
from the June copy, mostly added <lb> markup and the HTR re-import; same 13,114 letters). The v1 path stays available
(RATLAS_HARVEST=legacy).

What is tagged: <persName ref="p####"> and <placeName ref="l####"> in the running text, and in the header
correspDesc/correspAction the sender and addressee (persName or orgName ref="og##pl###", an org group tied to a
place), plus the place of sending. The repository also holds the project's register (``data/index/`` of the same
commit, harvested as ``bullinger_register/``): persons.xml (6,043 persons: name variants, GND, Wikipedia, HLS), localities.xml
(3,077 places: settlement / district / country, coordinates, GeoNames), organizations.xml (91 org groups, 337 orgName
forms tied to places). ``prepare()`` loads them as a shared register: 'bullinger:p1283', 'bullinger:l803',
'bullinger:og5pl841' (same keys as v1, now in_register True with names, authority ids and coordinates). Refs missing from
the register (p19981) keep an opaque key (``opaque_keys``).
The editors' @cert of the identification (correspAction persName/placeName/orgName) is carried in ``cert``.
Zones: the regest of the letter (msDesc/msContents/summary, editors' German summary) is zone 'regest'; the sender,
addressee and place tags of correspAction are header; the text is 'text'. msItem/author (plain text) is not a mention.
Register entries noted 'Biblische Figur' / '...Mythologie' / 'Religiöse Figur' get note_type biblical / mythological.
Dating: correspAction[sent]/date (exact @when for 12,500 files, notBefore/notAfter for 614); the 2024 date in
publicationStmt is the edition's own and is ignored by the generic header logic. Default 1523-1575 if missing.
Licence: CC BY-NC-ND 3.0 CH (corpus README, LICENSE_TIERS BullingerLetters T2); facsimiles not included.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_v2 as V
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'bullinger'
COLLS = V.pick(['bullinger'], [V.NEW + '/bullinger/'])
RENASCOR_IDS = ['bullinger-digital']
LICENCE = 'CC-BY-NC-ND-3.0-CH'     # LICENSE_TIERS BullingerLetters (README Urheberrecht)
TIER = 'T2'
DESCRIPTION = 'Letters to and from Heinrich Bullinger 1523-1575 with the project register (persons, places, org groups)'
OPTIONS = {'opaque_keys': True, 'default_years': (1523, 1575)}

_REG = V.NEW + '/bullinger_register/data/index/'
_BASE = 'https://www.bullinger-digital.ch/'


def _txt(el) -> str:
    return C.surface_text(el) if el is not None else ''


def _auth_of(el) -> dict:
    a = C.empty_auth()
    for i in el:
        if C.ln(i.tag) != 'idno' or not (i.text or '').strip():
            continue
        sub = (i.get('subtype') or '').lower()
        t = (i.text or '').strip()
        if sub == 'gnd':
            C.parse_auth([t], into=a)
        elif sub == 'geonames':
            C.parse_auth([t], into=a)
        elif sub == 'wiki':
            C.parse_auth([t], into=a)
        elif sub == 'hls':
            m = re.search(r'/(\d{4,})', t)
            if m:
                a['hls'] = m.group(1)
    return a


def _person_note_type(el) -> str:
    notes = ' '.join(_txt(n) for n in el if C.ln(n.tag) == 'note')
    if re.search(r'mythologie|sagengestalt|g[oö]tt', notes, re.I):
        return 'mythological'
    if re.search(r'bibl', notes, re.I):
        return 'biblical'
    if re.search(r'komödienfigur|romanfigur', notes, re.I):
        return 'fictional'
    if 'Religiöse Gemeinschaft' in notes:
        return 'group'
    if 'Religiöse Figur' in notes:
        return 'biblical'
    return 'person'


def _pname(p) -> str:
    forename = _txt(C.child(p, 'forename'))
    surname = _txt(C.child(p, 'surname'))
    return C.norm(f'{forename} {surname}')


def load_bullinger_register(ctx: C.Context) -> list[Entity]:
    ents: list[Entity] = []
    # persons
    root = V.read_xml(_REG + 'persons.xml')
    for person in (root.iter() if root is not None else []):
        if C.ln(person.tag) != 'person' or not person.get(C.XML_ID):
            continue
        names = [c for c in person if C.ln(c.tag) == 'persName']
        main = next((c for c in names if c.get('type') == 'main'), names[0] if names else None)
        pid = person.get(C.XML_ID)
        ident = (main.get(C.XML_ID) if main is not None else None) or pid.lower()
        name = _pname(main) if main is not None else ident
        variants = []
        for c in names:
            if c is main:
                continue
            v = _pname(c)
            if v and v != name and v not in variants:
                variants.append(v)
        e = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='person', name=name or ident,
                   source_path=_REG + 'persons.xml', renascor_id=ctx.renascor_id, variants=variants[:12],
                   auth=_auth_of(person), note_type=_person_note_type(person), licence=ctx.licence, tier=ctx.tier)
        ctx.shared.add(e, [ident, pid])
        ents.append(e)
    # places
    root = V.read_xml(_REG + 'localities.xml')
    for place in (root.iter() if root is not None else []):
        if C.ln(place.tag) != 'place' or not place.get(C.XML_ID):
            continue
        ident = place.get(C.XML_ID)
        sett, dist, ctry = (_txt(C.child(place, t)) for t in ('settlement', 'district', 'country'))
        name = sett or dist or ctry or ident
        lat = lon = None
        for g in place.iter():
            if C.ln(g.tag) == 'geo':
                ll = C._parse_geo(_txt(g))
                if ll:
                    lat, lon = ll
        e = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='place', name=name,
                   source_path=_REG + 'localities.xml', renascor_id=ctx.renascor_id,
                   variants=[v for v in (f'{sett}, {dist}' if sett and dist else '',) if v], lat=lat, lon=lon,
                   coord_source='tei:geo' if lat is not None else None, auth=_auth_of(place),
                   note_type=('place' if sett else 'region' if dist else 'country'), licence=ctx.licence, tier=ctx.tier)
        ctx.shared.add(e, [ident])
        ents.append(e)
    # organisation groups: one entity per orgName form (the mentions cite og5pl841)
    root = V.read_xml(_REG + 'organizations.xml')
    for org in (root.iter() if root is not None else []):
        if C.ln(org.tag) != 'org' or not org.get(C.XML_ID):
            continue
        kind = (org.get('type') or '').replace('__', ' ')
        for on in org:
            if C.ln(on.tag) != 'orgName' or not on.get(C.XML_ID):
                continue
            ident = on.get(C.XML_ID)
            e = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='org', name=_txt(on) or ident,
                       source_path=_REG + 'organizations.xml', renascor_id=ctx.renascor_id, note_type=kind or 'org',
                       licence=ctx.licence, tier=ctx.tier)
            ctx.shared.add(e, [ident])
            ents.append(e)
    return ents


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        return load_bullinger_register(ctx)

    def adjust(self, m, hit):
        if not m.surface:               # empty correspAction tags: the neighbouring header text is no context
            m.context = None
        return m


EXTRACTOR = Extractor()
