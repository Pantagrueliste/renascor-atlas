"""Letradas: women's writing in Spain (TEITOK, GITHE), 270 documents, ~1100-1898.

The text is tokenised and carries no entity tags. The entity information is in the teiHeader:
  * particDesc/person[@role=emisor|signer|scriptor|addressee]/persName: name strings, no ids;
  * msDesc/history/origin/origPlace (placeName, region, country, geo "lat lon") = place of writing, and an
    origPlace inside the addressee's person = place of the recipient (rarely filled).
Mentions (zone 'hdr'): one row per distinct person per document; emisor -> 'sender'; signer / scriptor with the same
name as an already listed person are not repeated, otherwise 'signer' and 'scribe' (non-standard role, documented
here); addressee -> 'addressee'; origPlace/placeName -> 'sent_from' (addressee's origPlace -> 'sent_to'). Empty
persName / placeName and the placeholders 's.l.' (sine loco) / 'desconocido' produce no row; a country-only origPlace ('Espana') is
not emitted (no place name, no coordinates).
Entities: places only, built from the origPlace elements (there is no register file): key
'letradas:<place>|<region>|<country>|<lat,lon rounded to 2 decimals>', coordinates from the file's own <geo>
(coord_source 'tei:geo'), so that different coordinates for one name (Zamora) stay distinct. Persons have no ids and
get no entity; local_key is None for persons.
Dating: msDesc origDate (@when or the 4-digit year in the text), date_source 'header'. Out-of-period documents are kept
and flagged by in_period False.
Licence: T3, all rights reserved ("(c) 2026 Letradas"; LICENSE_TIERS Renascor_letradas): internal use, no context.
"""
from __future__ import annotations

import re

from .. import common as C
from ..model import Entity, Mention, best_year, period_overlap
from .generic_tei import GenericTEI

SLUG = 'letradas'
COLLS = ['renascor_2026-10/letradas']
RENASCOR_IDS = ['letradas']
LICENCE = 'LicenseRef-AllRightsReserved'
TIER = 'T3'
DESCRIPTION = "Letradas: header-level senders, addressees and places of writing with coordinates (documents 1100-1898)"
OPTIONS = {}

_ROLE = {'emisor': 'sender', 'signer': 'signer', 'scriptor': 'scribe', 'addressee': 'addressee'}


_PLACEHOLDER = re.compile(r'^\s*(s\.?\s?l\.?|s\.?\s?n\.?|desconocid[oa]s?|ignorad[oa]|\?+|-+)\s*$', re.I)   # sine loco / unknown


def _txt(el, name):
    c = C.child(el, name) if el is not None else None
    return C.surface_text(c) if c is not None else ''


def _place(op, ctx, rec, doc_id):
    """(surface, Entity) from an origPlace element, or None if it has no place name."""
    name = _txt(op, 'placeName')
    if not name or _PLACEHOLDER.match(name):
        return None
    region, country = _txt(op, 'region'), _txt(op, 'country')
    g = C.child(op, 'geo')
    ll = C._parse_geo(C.surface_text(g)) if g is not None else None
    geo = f'{ll[0]:.2f},{ll[1]:.2f}' if ll else ''
    key = C.make_key(ctx.slug, f'{name}|{region}|{country}|{geo}')
    variants = [x for x in (f'{name}, {region}' if region else '',) if x]
    ent = Entity(local_key=key, coll_slug=ctx.slug, kind='place', name=name, source_path=rec.path,
                 renascor_id=ctx.renascor_id, variants=variants, lat=ll[0] if ll else None, lon=ll[1] if ll else None,
                 coord_source='tei:geo' if ll else None, auth=C.empty_auth(),
                 note_type='place', licence=ctx.licence, tier=ctx.tier)
    return name, ent


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    @staticmethod
    def _date(root):
        """origDate: @when is mostly an ISO year but 60 documents hold '[1603] enero 15' (year in brackets): any 4-digit
        year of @when / @notBefore / @notAfter / the text counts."""
        ys = []
        for od in root.iter('{*}origDate'):
            vals = [od.get(a) or '' for a in ('when', 'notBefore', 'notAfter', 'from', 'to')] + [C.surface_text(od)]
            for v in vals:
                ys += [int(y) for y in re.findall(r'(?<!\d)(1[0-9]{3})(?!\d)', v)]
        if ys:
            return C.DocDate(min(ys), max(ys), 'header')
        return C.header_dates(root)

    def extract_root(self, root, rec, ctx):
        date = self._date(root)
        doc_id = C.doc_id_of(root, rec)
        idno = next((C.norm(''.join(e.itertext())) for e in root.iter('{*}idno') if e.get('type') == 'corpus-num'), None)
        doc_id = idno or doc_id
        year, inp = best_year(date.year_from, date.year_to), period_overlap(date.year_from, date.year_to)
        ms: list[Mention] = []
        ents: dict[str, Entity] = {}

        def add(kind, tag, role, surface, key=None, ent=None):
            ms.append(Mention(coll_slug=ctx.slug, renascor_id=ctx.renascor_id, source_path=rec.path, doc_id=doc_id,
                              doc_year=year, doc_year_from=date.year_from, doc_year_to=date.year_to,
                              date_source=date.source, in_period=inp, kind=kind, tag=tag, role=role, zone='hdr',
                              surface=surface, ref_raw=None, local_key=key, in_register=True if key else None,
                              auth=C.empty_auth(), context=None, lang='es', licence=ctx.licence, tier=ctx.tier))
            if ent is not None:
                ents.setdefault(ent.local_key, ent)

        seen_names: set = set()
        for p in root.iter('{*}person'):
            if C.ln(p.getparent().tag) != 'particDesc':
                continue
            role = _ROLE.get((p.get('role') or '').strip().lower())
            pn = C.child(p, 'persName')
            name = C.surface_text(pn) if pn is not None else ''
            if role and name and not _PLACEHOLDER.match(name) and (role in ('sender', 'addressee') or name not in seen_names):
                add('person', 'persName', role, name)
                seen_names.add(name)
            op = C.child(p, 'origPlace')
            pl = _place(op, ctx, rec, doc_id) if op is not None else None
            if pl:
                add('place', 'placeName', 'sent_to', pl[0], pl[1].local_key, pl[1])
        for op in root.iter('{*}origPlace'):
            par = op.getparent()
            if C.ln(par.tag) == 'origin':
                pl = _place(op, ctx, rec, doc_id)
                if pl:
                    add('place', 'placeName', 'sent_from', pl[0], pl[1].local_key, pl[1])
        return ms, list(ents.values())


EXTRACTOR = Extractor()
