"""REED Collections (Records of Early English Drama), English counties, publisher zip eng_dh/reed.zip.

Tagged: <rs ref="eats:ID" | ref="http://ereed.library.utoronto.ca/eats/entity/ID/" type="Person_*|Place_*|Collective_*|
Drama_*|Thing_*|..."> inside the 2,483 <text type="record"> documents of records/*.xml (10 files; the county
introductions outside the records are modern editorial text and are not read). One document = one record
(doc_id = the record's xml:id), dated from its head/date (@when-iso/@from-iso/@to-iso).
Kinds: Person_* -> person, Place_* -> place, Collective_Troupe / Collective_Guild(s)* -> org. All the other rs types
(Drama_*, Thing_*, CalendarDay_*, Entertain*, Event_*, CrimesMisdemeanors, Collective_Clergy / Office / Occupation(s) /
Title(s)) are categories, not entities, and are not read.
Register: tei/entities.xml (EATSML): Person / Place / troupe-guild entities with name variants, existence dates (birth /
death strings), subject identifiers (VIAF, Wikidata, ODNB, History of Parliament; internal Toronto URLs are dropped).
local_key = 'reed:<eats id>'. Coordinates: tei/geojson.xml, joined through the entity's ereed geomap place id
(subject identifier .../geomap/places/<pk>/ = feature pk); coord_source 'json:geojson.xml' ('json:geojson.xml~approx'
when the feature is flagged is_approximate). note_type = the EATS entity type, lower-cased (person_male, place_pa_settlement...).
Entities of other types (drama characters, things, events...) are not written.
v2: every record starts with a heading <head><rs Place_County>Berkshire</rs>, <rs Place_PA_Settlement>Reading</rs> date: these 3,874 of the
8,591 Place_* tags (45%) are the locality the record is filed under, not places of the transcribed text. They get role
``provenance`` and zone ``header`` and are excluded from the default counts (Hampshire 767, Suffolk 683, Berkshire 362...).
Licence: CC BY-NC-SA 4.0 (LICENSE.txt in the zip; README says CC BY-SA, stricter applied); LICENSE_TIERS REED_English T2.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from lxml import etree

from .. import common as C
from ..model import Entity, empty_auth
from .generic_tei import GenericTEI

SLUG = 'reed'
COLLS = ['eng_dh/reed']
RENASCOR_IDS = ['reed-english']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'REED English county records: EATS-tagged persons and places, with the eREED register and geomap coordinates'

_PERSON = ['Person_Clergy', 'Person_Commoner', 'Person_F_Street', 'Person_Family', 'Person_Female',
           'Person_GenderUnknown', 'Person_Gentry', 'Person_Male', 'Person_Nobility', 'Person_Royalty',
           'Person_StatusOther']
_PLACE = ['Place_BoroughParish', 'Place_Count', 'Place_Country', 'Place_County', 'Place_EA_Archdeaconry',
          'Place_EA_Diocese', 'Place_EcclesiasticalArea', 'Place_F_Arena', 'Place_F_Bridge', 'Place_F_Church',
          'Place_F_Gate', 'Place_F_GuildHall', 'Place_F_Hospital', 'Place_F_InnOfCourt', 'Place_F_OpenArea',
          'Place_F_Other', 'Place_F_PlaceOfPunishment', 'Place_F_Playhouse', 'Place_F_Property',
          'Place_F_ReligiousHouse', 'Place_F_Residence', 'Place_F_School', 'Place_F_Storage', 'Place_F_Street',
          'Place_F_TownHall', 'Place_F_VictuallingHouse', 'Place_F_WaterFeature', 'Place_Feature',
          'Place_PA_Liberty', 'Place_PA_Manor', 'Place_PA_Settlement', 'Place_PA_Ward', 'Place_PopulatedArea',
          'Place_ReligiousHouse']
_ORG = ['Collective_Troupe', 'Collective_GuildOccupational', 'Collective_GuildReligious',
        'Collective_GuildsOccupational', 'Collective_GuildsReligious']
TYPE_MAP = {**{t.lower(): 'person' for t in _PERSON}, **{t.lower(): 'place' for t in _PLACE},
            **{t.lower(): 'org' for t in _ORG}}
OPTIONS = {'type_map': TYPE_MAP}

TEI = '{http://www.tei-c.org/ns/1.0}'
EATS = '{http://eats.artefact.org.nz/ns/eatsml/}'
XMLID = '{http://www.w3.org/XML/1998/namespace}id'
_URL_REF = re.compile(r'ereed\.library\.utoronto\.ca/eats/entity/(\d+)', re.I)
_ENTITIES = 'Collections-master/tei/entities.xml'
_GEOJSON = 'Collections-master/tei/geojson.xml'


def _kind_of_types(types: list[str]):
    ts = [t for t in types if t]
    if any(t.startswith('Person_') for t in ts):
        return 'person'
    if any(t.startswith('Place_') for t in ts):
        return 'place'
    if any(t in _ORG for t in ts):
        return 'org'
    return None


def _auth_of_identifiers(urls: list[str]) -> dict:
    a = empty_auth()
    for u in urls:
        u = (u or '').strip()
        if not u:
            continue
        m = re.search(r'ref:odnb/(?:[\d.]+/odnb-[\d-]+-e-)?(\d+)', u) or re.search(r'oxforddnb\.com/view/article/(\d+)', u)
        if m:
            x = f'odnb:{m.group(1)}'
            if x not in a['other']:
                a['other'].append(x)
            continue
        if 'utoronto.ca' in u:           # internal eREED / REED / EMLoT / MASL pages: not authorities
            continue
        C.parse_auth([u], into=a)
    return a


def _date_parts(e):
    ex = e.find(f'{EATS}existences')
    if ex is None:
        return None, None
    start = end = None
    for dp in ex.iter(f'{EATS}date_part'):
        raw = (dp.findtext(f'{EATS}raw') or '').strip()
        if dp.get('type') == 'start' and start is None:
            start = raw or None
        elif dp.get('type') == 'end' and end is None:
            end = raw or None
    return start, end


def load_eats(ctx, archive: str):
    """Entities of tei/entities.xml with coordinates from tei/geojson.xml. -> [(Entity, aliases)]"""
    root = etree.fromstring(C.read_raw(f'{archive}!{_ENTITIES}'),
                            etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True))
    types = {e.get(XMLID): (e.findtext(f'{EATS}name') or '').strip() for e in root.find(f'{EATS}entity_types')}
    coords = {}
    try:
        g = etree.fromstring(C.read_raw(f'{archive}!{_GEOJSON}'),
                             etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True))
        for f in g.iter('feature'):
            try:
                d = json.loads(''.join(f.itertext()))
                lon, lat = d['geometry']['coordinates'][:2]
                coords[str(d['properties']['pk'])] = (float(lat), float(lon), bool(d['properties'].get('is_approximate')))
            except (ValueError, KeyError, TypeError):
                continue
    except (OSError, KeyError, etree.XMLSyntaxError):
        pass
    out = []
    for e in root.find(f'{EATS}entities'):
        et = e.find(f'{EATS}entity_types')
        tnames = [types.get(x.get('entity_type'), '') for x in et] if et is not None else []
        kind = _kind_of_types(tnames)
        if kind is None:
            continue
        eid = e.get('eats_id') or e.get(XMLID, '').replace('entity-', '')
        names = e.find(f'{EATS}names')
        forms = []
        pref = None
        if names is not None:
            for n in names:
                t = C.norm(n.findtext(f'{EATS}display_form') or '') or C.norm(n.findtext(f'{EATS}assembled_form') or '')
                if not t:
                    continue
                if n.get('is_preferred') == 'true' and pref is None:
                    pref = t
                forms.append(t)
        name = pref or (forms[0] if forms else f'entity-{eid}')
        variants = [t for i, t in enumerate(forms) if t != name and t not in forms[:i]]
        si = e.find(f'{EATS}subject_identifiers')
        urls = [s.text for s in si] if si is not None else []
        lat = lon = csrc = None
        if kind == 'place':
            for u in urls:
                m = re.search(r'/geomap/places/(\d+)', u or '')
                if m and m.group(1) in coords:
                    lat, lon, approx = coords[m.group(1)]
                    csrc = 'json:geojson.xml' + ('~approx' if approx else '')
                    break
        birth, death = _date_parts(e) if kind == 'person' else (None, None)
        ntype = next((t for t in tnames if t), '').lower() or None
        ent = Entity(local_key=C.make_key(ctx.slug, eid), coll_slug=ctx.slug, kind=kind, name=name,
                     source_path=f'{archive}!{_ENTITIES}', renascor_id=ctx.renascor_id, variants=variants,
                     lat=lat, lon=lon, coord_source=csrc, birth=birth, death=death,
                     auth=_auth_of_identifiers(urls), note_type=ntype, licence=ctx.licence, tier=ctx.tier)
        out.append((ent, [eid]))
    return out


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if '/tei/records/' in r.path]

    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        recs = [r for r in ctx.all_files]
        if not recs:
            return []
        out = []
        for ent, aliases in load_eats(ctx, recs[0].archive):
            ctx.shared.add(ent, aliases)
            out.append(ent)
        return out

    def extract_root(self, root, rec, ctx):
        out = []
        base_lang = C.doc_language(root)
        for text in root.iter(f'{TEI}text'):
            if text.get('type') != 'record':
                continue
            counters = Counter()
            hits = C.scan_mentions(text, type_map=ctx.type_map, want_context=ctx.want_context,
                                   doc_lang=base_lang, counters=counters)
            raw = []
            for h in hits:
                raw.append(h.refs.get('ref'))
                m = _URL_REF.search(h.refs.get('ref') or '')
                if m:
                    h.refs['ref'] = 'eats:' + m.group(1)
            date = self._record_date(text)
            ms = C.build_mentions(root, rec, ctx, date=date, hits=hits, adjust=self.adjust)
            did = text.get(XMLID) or rec.stem
            for m, rr in zip(ms, raw):
                m.doc_id = did
                m.ref_raw = rr
            out.extend(ms)
        return out, []

    def adjust(self, m, hit):
        if 'head' in hit.ancestors:                 # the record's heading: where the record is filed (county, town, date)
            m.role = 'provenance'
            m.zone_class = 'header'
        return m

    @staticmethod
    def _record_date(text):
        head = None
        for el in text.iter(f'{TEI}head'):
            head = el
            break
        ys = []
        if head is not None:
            for d in head.iter(f'{TEI}date'):
                r = C.date_range(d)
                if r:
                    ys.append(r)
        if not ys:
            return C.DocDate()
        return C.DocDate(min(r[0] for r in ys), max(r[1] for r in ys), 'header')


EXTRACTOR = Extractor()
