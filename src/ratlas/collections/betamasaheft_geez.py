"""Beta maṣāḥǝft (Hiob Ludolf Centre, Hamburg): five Ethiopian chronicles in Gǝʿǝz (Works repository).

Files: Chronicle of Susǝnyos, Chronicle of Galāwdewos, Edict of Galāwdewos, Chronicle of Zarʾa Yāʿqob, Tarikz.
Licence: the file headers say CC-BY-SA-4.0, the project website CC-BY-NC-SA-4.0; the stricter one is applied
(also the catalogue entry and the harvest's LICENCE.txt): CC-BY-NC-SA-4.0, T2.
Raw: ``originals_2026-10-15/github-lowresource-tei/BetaMasaheft_Geez/``.

What is special here
  * ``persName/@ref`` = ``PRS####Name`` (persons), ``ETH####Name`` (ethnic groups tagged as persName: kind ``org``,
    tag ``persName@ethnic``); ``placeName/@ref`` = ``LOC####Name`` (places), ``INS####`` (institutions such as
    monasteries: kind ``org``, tag ``placeName@institution``), ``ETH`` (kind ``org``) or direct ``wd:Q..`` /
    ``pleiades:...`` ids (28 places) which go to auth;
  * the BM id (prefix + digits, the name suffix is dropped: ``PRS9038Susenyos`` -> ``PRS9038``) is the stable key:
    ``betamasaheft_geez:PRS9038`` (in_register False: the Beta maṣāḥǝft authority files are not in the harvest) and
    ``bm:PRS9038`` in auth.other (resolvable at https://betamasaheft.eu/PRS9038);
  * ``roleName`` (titles such as "king") is not an entity and is ignored;
  * dates: ``creation`` in the header when present (Galāwdewos 1559-1560, Edict 1548); else the ``@when/@notBefore/
    @notAfter`` years that the editors put on tagged elements of the text (Susǝnyos 1587-1632ish, Zarʾa Yāʿqob
    1434-1468), date_source ``text``; Tarikz has neither: default_years 1527-1632 (reigns covered by the other
    chronicles), date_source ``collection``. These are composition/subject-period estimates, not manuscript dates.

v2:
  * the authority files (BetaMasaheft/Places at ff57906 and Persons at 802eb52, shallow clones, harvested in
    ``betamasaheft_geez-registers/``, CC BY-SA 4.0) are read for the ids the five chronicles use (402 LOC, 261 PRS): ``prepare()``
    indexes the file names, parses only those files and builds the entities ('betamasaheft_geez:PRS9038', in_register True):
    the normalised name, the Ge'ez-script names as variants, ``sameAs`` (wd:Q...), coordinates of the <geo> (8 places only),
    birth / death. The ids carry no Wikidata or GeoNames link for most entries: this adds Ge'ez labels more than coordinates;
  * dating by text: a tagged element's own @when / @notBefore / @notAfter, else the last dated element before it in reading
    order (the chronicles are told in order); the first dated element serves before it. This replaces the single min-max
    range of the whole chronicle (Susǝnyos: 1587-1620 -> 1603 for 4,403 mentions). A range wider than 20 years has no doc_year;
  * ethnic groups (ETH) and institutions (INS) keep kind org; ETH entries are class ``people``.
"""
from __future__ import annotations

import os
import re

from .. import common as C
from .. import helpers_batch0 as H
from .. import helpers_v2 as V
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'betamasaheft_geez'
COLLS = ['originals_2026-10-15/github-lowresource-tei/BetaMasaheft_Geez']
RENASCOR_IDS = ['betamasaheft-geez']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Five Ethiopian royal chronicles with Beta Masaheft person/place ids'
OPTIONS = {'default_years': (1527, 1632)}

_BM = re.compile(r'^(PRS|LOC|INS|ETH|WORK|LIT|ORG)(\d+)', re.I)
_BM_ANY = re.compile(rb'\b(PRS|LOC|INS|ETH)(\d+)')
_REGBASE = V.NEW + '/betamasaheft_geez-registers/'
_NAME_TYPES_PRIORITY = ('normalized', 'main', None)


def _bm_index() -> dict:
    """id ('PRS9038') -> raw-relative path of its authority file in the Places / Persons clones."""
    out = {}
    base = C.raw_root() / _REGBASE
    for sub in ('Places', 'Persons'):
        for dp, dn, fn in os.walk(base / sub):
            dn[:] = [d for d in dn if not d.startswith('.')]
            for f in fn:
                m = re.match(r'(PRS|LOC|INS|ETH)(\d+)', f)
                if m and f.endswith('.xml'):
                    out.setdefault(m.group(1) + m.group(2), os.path.relpath(os.path.join(dp, f), C.raw_root()))
    return out


def _bm_entity(ctx, ident: str, path: str):
    root = V.read_xml(path)
    if root is None:
        return None
    el = next((e for e in root.iter() if C.ln(e.tag) in ('person', 'personGrp', 'place', 'org') and C.ln(e.getparent().tag) in
               ('listPerson', 'listPlace', 'listOrg')), None)
    if el is None:
        return None
    kind = {'person': 'person', 'personGrp': 'person', 'place': 'place', 'org': 'org'}[C.ln(el.tag)]
    names = []
    for c in el:
        if C.ln(c.tag) in ('persName', 'placeName') and C.surface_text(c):
            names.append((c.get('type'), c.get(C.XML_LANG), C.surface_text(c)))
    pref = next((t for (t0, lg, t) in names if t0 == 'main' and lg in (None, 'en') and t), None)
    if pref is None:
        pref = next((t for (t0, lg, t) in names if t0 == 'normalized' and t), None)
    if pref is None:
        pref = next((t for (t0, lg, t) in names if lg in (None, 'en') and t), None)
    if pref is None and names:
        pref = names[0][2]
    name = C.norm(pref or ident)
    variants = list(dict.fromkeys(t for _ty, _lg, t in names if t and t != name))[:10]
    auth = C.parse_auth([el.get('sameAs') or ''])
    lat = lon = None
    for g in el.iter():
        if C.ln(g.tag) == 'geo':
            ll = C._parse_geo(C.surface_text(g))
            if ll:
                lat, lon = ll
                break
    prefix = ident[:3]
    ntype = {'PRS': 'person', 'LOC': 'place', 'INS': 'institution', 'ETH': 'ethnic'}[prefix]
    birth = death = None
    for c in el:
        if C.ln(c.tag) == 'birth':
            birth = c.get('when') or C.surface_text(c) or None
        elif C.ln(c.tag) == 'death':
            death = c.get('when') or C.surface_text(c) or None
    k = 'org' if prefix in ('INS', 'ETH') else kind
    ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind=k, name=name, variants=variants,
                 source_path=path, renascor_id=ctx.renascor_id, auth=auth, lat=lat, lon=lon,
                 coord_source='tei:geo' if lat is not None else None, birth=birth, death=death, note_type=ntype,
                 licence=ctx.licence, tier=ctx.tier)
    if prefix == 'ETH' or (C.ln(el.tag) == 'personGrp' and prefix == 'PRS'):
        ent.person_class = 'people'
    ent.auth['other'].append(f'bm:{ident}')
    return ent


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.path.endswith('.xml')]

    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        if not V.use_new() or not (C.raw_root() / _REGBASE).is_dir():
            return []
        wanted = set()
        for rec in ctx.all_files:
            for m in _BM_ANY.finditer(rec.read()):
                wanted.add(m.group(1).decode() + m.group(2).decode())
        index = _bm_index()
        out = []
        for ident in sorted(wanted):
            p = index.get(ident)
            if not p:
                continue
            ent = _bm_entity(ctx, ident, p)
            if ent is not None:
                ctx.shared.add(ent, [ident])
                out.append(ent)
        return out

    def extract_root(self, root, rec, ctx):
        # running date: the last dated element before each tagged element (reading order)
        self._run = {}
        self._keep = list(root.iter())          # lxml proxies are short-lived: keep them so that id(el) is stable
        cur = None
        pend = []
        for el in self._keep:
            n = C.ln(el.tag)
            if n in ('persName', 'placeName', 'date'):
                ys = [y for a in ('when', 'notBefore', 'notAfter') if (y := C._attr_year(el.get(a))) is not None
                      and 1400 <= y <= 1700]
                if ys:
                    cur = (min(ys), max(ys))
                    for p in pend:
                        self._run[p] = cur
                    pend = []
                elif n in ('persName', 'placeName'):
                    if cur is None:
                        pend.append(id(el))
                    else:
                        self._run[id(el)] = cur
                if ys:
                    self._run[id(el)] = cur
        return super().extract_root(root, rec, ctx)

    def doc_date(self, root, rec, ctx):
        # header creation (the generic header dates would also catch revision-free dates; keep to creation)
        for el in H.iter_local(root, 'creation'):
            rg = C.date_range(el, 1000, 1800)
            if rg:
                return C.DocDate(rg[0], rg[1], 'header')
        ys = []
        for el in root.iter():
            n = C.ln(el.tag)
            if n in ('persName', 'placeName'):
                for a in ('when', 'notBefore', 'notAfter'):
                    y = C._attr_year(el.get(a))
                    if y and 1400 <= y <= 1700:
                        ys.append(y)
        if ys:
            return C.DocDate(min(ys), max(ys), 'text')
        return C.DocDate(*ctx.default_years, 'collection')

    def adjust(self, m, hit):
        raw = m.ref_raw or ''
        tok = next((t for t in raw.split() if _BM.match(t)), None)
        if tok:
            mm = _BM.match(tok)
            bm_id = mm.group(1).upper() + mm.group(2)
            ent = self._ctx.shared.entities.get(f'betamasaheft_geez:{bm_id}') if getattr(self, '_ctx', None) else None
            if ent is not None:
                H.attach(m, ent)
            else:
                m.local_key, m.in_register = f'betamasaheft_geez:{bm_id}', False
            if f'bm:{bm_id}' not in m.auth['other']:
                m.auth['other'].append(f'bm:{bm_id}')
            prefix = mm.group(1).upper()
            if prefix == 'ETH':
                m.kind = 'org'
                m.tag = f'{hit.name}@ethnic'
            elif prefix == 'INS':
                m.kind = 'org'
                m.tag = f'{hit.name}@institution'
        if hit.el.get('role') == 'author' and m.kind == 'person':
            m.role = 'author'
        rg = getattr(self, '_run', {}).get(id(hit.el))
        if rg and m.date_source != 'header':        # a header creation date (Edict 1548, Galāwdewos) wins over the running date
            V.set_range(m, rg[0], rg[1], 'text')
        if m.zone == 'hdr':
            m.context = None
        return m

    def extract(self, rec, ctx):
        self._ctx = ctx
        return super().extract(rec, ctx)


EXTRACTOR = Extractor()
