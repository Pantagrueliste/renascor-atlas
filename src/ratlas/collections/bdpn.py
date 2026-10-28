"""Biblioteca Digital del Pensamiento Novohispano (BDPN, UNAM): astrological / cometary texts of New Spain, 1588-1698.

Licence: CC0-1.0, T1 (LICENSE_TIERS BDPN_Novohispano; the repository LICENSE is CC0 1.0). The catalogue entry says
CC-BY-NC-4.0 for the website; the TEI files carry no licence element, so the repository grant is used.
Raw: ``bdpn/`` (publisher GitHub repository).

What is special here
  * the texts are ``bdpn/archivosXMLDocumentos/*``: 10 distinct files. Two of them have NO ``.xml`` extension
    (``carta`` and ``DiscursoEthereologico``, ~830 tagged names) and are missing from the scan inventory: they are
    added to the file list by this module (FileRecord without scan counts); ``Copia de Discursos astronomicos
    Bonilla.xml`` is an identical copy (scan dup) and is skipped;
  * mentions are ``persName`` / ``placeName`` / ``orgName`` and ``name[@type=divPer]`` (divine persons: Dios) with
    ``@ref="catalogo_nombres_<kind>.xml#<id>"``; animals (``name[@type=animal]``), body terms and abbreviations are
    not persons, places or organisations and are not read;
  * the catalogues are flat, non-TEI XML (``<person id=A001 género procedencia>``, ``<place id>``,
    ``<organization id>``); ids are only unique within one catalogue, so entities get keys
    ``bdpn:person:A001`` / ``bdpn:place:A001`` / ``bdpn:org:A001`` and refs are resolved in this module
    (``adjust``), tolerating the typos of the edition (``$L004``, ``xmlV010``, ``#C00)``, ``lgares.xml``, ids
    without zero padding such as ``A35``);
  * the place catalogue mixes geography and cosmology (constellations, planets, celestial regions, zodiac signs):
    ``note_type`` is ``celestial`` for those (from the editors' comments in the catalogue and a keyword list),
    ``fictional`` / ``biblical`` where the catalogue says so, else ``place``. Persons with origin ``Mítico`` are
    ``mythological``; ``Dios`` and other divine persons ``divine``;
  * the catalogue file for persons has no extension (``catalogo_nombres_personas (1)``); the catalogue is not in the
    scan and is pushed with ``--extra``;
  * dates: header ``bibl/date`` when valid (``0000`` is ignored), else the ``date/@when`` values of the text
    (1450-1700, date_source ``text``), else a year in the title.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from lxml import etree

from .. import common as C
from .. import helpers_batch0 as H
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'bdpn'
COLLS = ['bdpn']
RENASCOR_IDS = ['bdpn-novohispano']
LICENCE = 'CC0-1.0'
TIER = 'T1'
DESCRIPTION = 'Novohispanic comet and almanac texts 1588-1698 with person/place/organisation catalogues'
OPTIONS = {'type_map': {'divper': 'person'}}

_DOC_DIR = 'bdpn/archivosXMLDocumentos/'
_EXTRA_DOCS = ['carta', 'DiscursoEthereologico']          # TEI files without .xml extension
_CATALOGS = {'person': 'bdpn/catalogo_nombres_personas (1)', 'place': 'bdpn/catalogo_nombres_lugares (1).xml',
             'org': 'bdpn/catalogo_nombres_organizaciones (1).xml'}
_ELEM = {'person': 'person', 'place': 'place', 'org': 'organization'}

_CELESTIAL = re.compile(
    r'constelaci|estrella|astro\b|astros\b|zodiac|planeta|orbita|órbita|\bpolo\b|hemisf|cielo|celeste|firmamento|'
    r'ecuador|equinoc|meridian|horizonte|r[eé]gion del (aire|fuego)|r[eé]gi[oó]n aetherea|[eé]ter\b|orto\b|ocaso|'
    r'apogeo|perigeo|nebulos|v[ií]a l[aá]ctea|universo|tr[oó]pico|signos|casas zodiacales|semic[ií]rculo|'
    r'paral[aá]xis|coluro|c[uú]mulo|atm[oó]sfera|zenit|cenit|\bplano\b|\bplanos\b|\borbe\b|orbes|\bmundo\b|'
    r'\bgrieta\b|parabola|parábola|hyperbola|elipse|declinaci|arcos? semi|altura meridiana|ascendente|'
    r'^(sol|luna|venus|tierra \(planeta\)|astros?|estrellas?|planetas?)$', re.I)
# v2.1: thirteen cosmological and compass terms the catalogue typed as places (the manual review of 2026-10-17: not places, never placed): out of the counts
_COSMOLOGICAL = re.compile(r'^(ecl[ií]ptica|c[ií]rculos horarios|azimutal|epiciclos?|cometas?|andr[oó]meda|septentrional|media regi[oó]n|oriente|'
                           r'occidente|norte|sur|poniente|levante|mediod[ií]a|mediodia)$', re.I)
_BIBLICAL = re.compile(r'b[ií]blic|valle de josaf', re.I)
_ID_RX = re.compile(r'([A-Z]{1,2})\s*0*(\d{1,4})\s*[\)\]]?\s*$')


def _kind_of_ref(ref: str):
    low = ref.lower()
    if 'pers' in low:
        return 'person'
    if 'lug' in low or 'lgar' in low:
        return 'place'
    if 'organ' in low:
        return 'org'
    return None


def _norm_id(raw: str):
    """'#L004', '$L004', 'xmlV010', 'C00)' -> letter + number (zero padded to 3), or None."""
    m = re.search(r'([A-Z]{1,2})(\d{1,4})\D*$', raw)
    if not m:
        return None
    return f'{m.group(1)}{int(m.group(2)):03d}'


_IDX: dict = {}      # (kind, id) -> Entity; filled by prepare() in the parent process, inherited by forked workers


def _catalog_entities(ctx, kind: str):
    path = _CATALOGS[kind]
    root = etree.fromstring(C.read_raw(path), etree.XMLParser(recover=True, remove_comments=False, huge_tree=True,
                                                               resolve_entities=False, no_network=True))
    out = []
    for el in root.iter(_ELEM[kind]):
        ident = el.get('id')
        if not ident or not ident.strip():
            continue
        ident = ident.strip()
        name = C.norm((el.text or '').split('\n')[0])
        if not name:
            continue
        nx = el.getnext()
        com = C.norm(nx.text) if nx is not None and nx.tag is etree.Comment else ''
        note = kind
        if kind == 'place':
            if _BIBLICAL.search(com):
                note = 'biblical'
            elif re.search(r'ficticio', com, re.I):
                note = 'fictional'
            elif _CELESTIAL.search(name) or _CELESTIAL.search(com) or _COSMOLOGICAL.match(name):
                note = 'celestial'
        elif kind == 'person':
            orig = (el.get('procedencia') or '').strip().lower()
            if orig.startswith('m') and orig.startswith(('mítico', 'mitico')):
                note = 'mythological'
            elif C.norm(name).lower() in ('dios', 'jesucristo'):
                note = 'divine'
        ent = Entity(local_key=f'{ctx.slug}:{kind}:{ident}', coll_slug=ctx.slug, kind=kind, name=name,
                     source_path=path, renascor_id=ctx.renascor_id, note_type=note, licence=ctx.licence,
                     tier=ctx.tier)
        out.append(ent)
    return out


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        recs = [r for r in super().iter_files(ctx, include_dups) if r.path.startswith(_DOC_DIR)]
        have = {r.path for r in recs}
        for n in _EXTRA_DOCS:
            p = _DOC_DIR + n
            if p not in have and (C.raw_root() / p).exists():
                recs.append(C.FileRecord(path=p, coll='bdpn'))
        return sorted(recs, key=lambda r: r.path)

    def prepare(self, ctx):
        _IDX.clear()
        ents = []
        for kind in ('person', 'place', 'org'):
            for e in _catalog_entities(ctx, kind):
                _IDX[(kind, e.local_key.rsplit(':', 1)[1])] = e
                ents.append(e)
        return ents

    def doc_date(self, root, rec, ctx):
        d = C.header_dates(root)
        if d.known:
            return d
        ys = []
        for el in root.iter():
            if C.ln(el.tag) == 'date' and el.get('when'):
                y = C._attr_year(el.get('when'))
                if y and 1450 <= y <= 1700:
                    ys.append(y)
        if ys:
            return C.DocDate(min(ys), max(ys), 'text')
        for el in root.iter():
            if C.ln(el.tag) == 'title':
                y = H.first_year(C.surface_text(el))
                if y and 1450 <= y <= 1700:
                    return C.DocDate(y, y, 'header')
        return C.DocDate()

    def adjust(self, m, hit):
        if m.zone == 'hdr':
            m.context = None
        raw = m.ref_raw or ''
        if raw:
            ident = _norm_id(raw)
            ent = None
            if ident:
                ent = _IDX.get((_kind_of_ref(raw) or m.kind, ident)) or _IDX.get((m.kind, ident))
            if ent is not None:
                m.local_key, m.in_register = ent.local_key, True
                m.auth = C.merge_auth(m.auth, ent.auth)
        return m


EXTRACTOR = Extractor()
