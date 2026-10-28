"""COSUIZA (GRAFILA, Universite de Lausanne): Spanish documents kept in Swiss archives, 123 TEITOK transcriptions.

Minimal headers, NO entity tags in the running text (tokens only). What is read (header only):
  * origin of the document: history/origin/origPlace (placeName / settlement, with region / country and a
    <geo>lat lon</geo>) and correspDesc/correspAction (settlement/placeName + geo), role 'sent_from' / 'sent_to';
    region and country tags are dropped when the same origPlace has a placeName/settlement (they would only add the
    containing province and Spain); placeholders ('Origen', geo 'Coordenadas') are dropped;
  * persons: msItem/editor/persName (the scribe of the original, role 'author'; the TEI element name 'editor' here is
    not a modern editor) and handDesc persName (role 'mention');
  * msIdentifier country / region / settlement is the REPOSITORY (Swiss archive): not an event place, dropped.
  * the template persons 'Scriptor / Autor / Firmante / Destinatario' of some headers are role labels, not people:
    no register is read;
  * Header geo is turned into a document-scoped entity 'cosuiza:<doc_id>#origin' (and #sent_from / #sent_to) with
    coord_source 'tei:geo' so that the coordinates are not lost (Mention has no lat/lon field).
  * dating: origDate in the header for 53 of 123 files (27 within 1450-1700); the other 70 cannot be dated (in_period
    None); the catalogue period 1483-1951 is NOT applied to them.
Licence: custom terms (headers say only 'Licencia CREATIVE COMMONS' without a variant); LICENSE_TIERS
Renascor_cosuiza: LicenseRef-Custom, T3 (recipe only): extracted for internal work, no context snippets.
"""
from __future__ import annotations

from .. import common as C
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'cosuiza'
COLLS = ['renascor_2026-10/cosuiza']
RENASCOR_IDS = ['cosuiza']
LICENCE = 'LicenseRef-Custom'
TIER = 'T3'
DESCRIPTION = 'Origin places (with coordinates) and scribes in the headers of 123 Spanish documents in Swiss archives'
OPTIONS = {}

_PLACEHOLDERS = {'origen', 'coordenadas', ''}


def _geo_of(el):
    """(lat, lon) from a <geo> inside the element or its parent origPlace/settlement."""
    for scope in (el, el.getparent()):
        if scope is None:
            continue
        for g in scope.iter():
            if C.ln(g.tag) == 'geo':
                ll = C._parse_geo(C.surface_text(g))
                if ll:
                    return ll
    return None


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):       # the 'listPerson' of some files is a template of role labels
        return False

    def extract_root(self, root, rec, ctx):
        # <editor> under msItem is the scribe / author of the original, not a modern editor
        for el in root.iter():
            if C.ln(el.tag) == 'editor' and el.getparent() is not None and C.ln(el.getparent().tag) == 'msItem':
                el.tag = el.tag.replace('editor', 'author')
        doc_id = C.doc_id_of(root, rec)
        ents: dict[str, Entity] = {}

        def adjust(m, hit):
            anc = hit.ancestors
            if 'msIdentifier' in anc:
                return None
            if m.kind == 'place':
                el = hit.el
                par = el.getparent()
                # the clean name: placeName child of a settlement, else the element text without geo
                name = C.surface_text(el)
                for c in el:
                    if C.ln(c.tag) == 'placeName':
                        name = C.surface_text(c)
                for c in el:
                    if C.ln(c.tag) == 'geo':
                        name = name.replace(C.surface_text(c), '').strip()
                m.surface = name
                if name.lower() in _PLACEHOLDERS:
                    return None
                where = None
                if par is not None and C.ln(par.tag) == 'origPlace':
                    siblings = [C.ln(s.tag) for s in par]
                    if m.tag in ('region', 'country') and ('placeName' in siblings or 'settlement' in siblings):
                        return None
                    where = 'origin'
                    m.role = 'sent_from'
                elif 'correspAction' in anc:
                    where = 'sent_from' if m.role == 'sent_from' else 'sent_to'
                if where:
                    ll = _geo_of(el)
                    if ll:
                        key = C.make_key(ctx.slug, where, doc_id)
                        if key not in ents:
                            ents[key] = Entity(local_key=key, coll_slug=ctx.slug, kind='place', name=name,
                                               source_path=rec.path, renascor_id=ctx.renascor_id, lat=ll[0],
                                               lon=ll[1], coord_source='tei:geo', note_type='document_' + where,
                                               licence=ctx.licence, tier=ctx.tier)
                        m.local_key, m.in_register = key, True
            return m

        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        ms = C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=adjust)
        return ms, list(ents.values())

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=False, doc_lang=C.doc_language(root))


EXTRACTOR = Extractor()
