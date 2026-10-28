"""HAB 'Selbstzeugnisse' edition sz2 (sz2.zip): travel diaries of the Brunswick-Wolfenbuettel dukes and duchess
(Christine Luise 1698/99, Ludwig Rudolph 1701-1724), TEI, with shared registers. Only Christine Luise's diary
(cod_guelf_211_1_extrav, 1698/99) is within 1450-1700 (most of it); the six other diary files are 1701-1724 and come
out with in_period False. (The single file diarium.xml of the same collection is module selbstzeugnisse_hab_diarium.)

Conventions of this edition:
  * <rs type="person|place" ref="#id" role="visited|mentioned|read"> in the running text; 'abbreviation' and 'bibl'
    are not entities; typo 'plac' is mapped. @role of the rs gives the non-standard roles 'visited' (places the writer
    went to, typos vistited/vistied/visisted included), 'read' and 'mention' ('mentioned', 'metioned'); refs are '#id'
    or a bare id;
  * shared registers in the zip (ids are edition-wide, shared keys '<slug>:<id>'): register/listPerson_s2.xml (901
    persons, 360 GND), tagebuecher/listPlace_s2.xml (443 places: GeoNames 266, GND 65; the register/ copy has 435, the
    union is used, the tagebuecher copy first) and register/listPlace_s2_lat_lng.xml (260 places with <geo>lat lon</geo>
    and GeoNames id: coordinates are joined by xml:id, else by GeoNames id, coord_source 'tei:geo:listPlace_s2_lat_lng');
    the tagebuecher copy of listPerson_s2.xml is a duplicate (dup=true, skipped);
  * itinerar_s2_tab.xml (derived table of the itinerary: 1,046 placeName elements without refs) repeats the visited
    stops of the diaries and is NOT read (double counting); introductions and other lists are not read;
  * dating: dates in the running text (<date when>), per diary, source 'text'. v2: each mention takes the date of the last
    <date> before it (isolated outliers such as the 1706 and 1724 tokens of the diary titled 'Tagebuch (1712)' are ignored:
    a date more than 5 years from the median of its neighbours is dropped); the diary's span is the fallback.
Licence: CC-BY-SA-4.0 (LICENSE_TIERS Renascor_selbstzeugnisse-hab, T1).
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch4 as H
from .. import helpers_v2 as V

SLUG = 'selbstzeugnisse_hab_sz2'
COLLS = ['renascor_2026-10/selbstzeugnisse-hab/files/sz2.zip']
RENASCOR_IDS = ['selbstzeugnisse-hab']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Ducal travel diaries 1698-1724 with persons/places registers (GND, GeoNames, coordinates)'
OPTIONS = {'opaque_keys': True, 'type_map': {'plac': 'place'}}

_REGISTERS = ('sz2/tagebuecher/listPlace_s2.xml', 'sz2/register/listPerson_s2.xml', 'sz2/register/listPlace_s2.xml')
_COORDS = 'sz2/register/listPlace_s2_lat_lng.xml'
_ROLE = {'visited': 'visited', 'vistited': 'visited', 'vistied': 'visited', 'visisted': 'visited',
         'visisted ': 'visited', 'mentioned': 'mention', 'metioned': 'mention', 'read': 'read'}


class Extractor(H.SharedRegisterTEI):
    def iter_files(self, ctx, include_dups=False):
        out = []
        for r in super().iter_files(ctx, include_dups):
            m = r.member or ''
            if m.startswith('sz2/tagebuecher/cod_guelf') or m in _REGISTERS or m == _COORDS:
                out.append(r)
        return out

    def is_register_candidate(self, rec):
        return (rec.member or '') in _REGISTERS

    def prepare(self, ctx):
        ctx.options['_register_paths'] = {r.path for r in ctx.all_files if (r.member or '') in _REGISTERS + (_COORDS,)}
        order = {m: i for i, m in enumerate(_REGISTERS)}
        ctx_files = ctx.all_files
        ctx.all_files = sorted((r for r in ctx_files if (r.member or '') in order), key=lambda r: order[r.member])
        try:
            ents = super().prepare(ctx)
        finally:
            ctx.all_files = ctx_files
        by_gn = {}
        for e in ents:
            if e.kind == 'place' and e.auth.get('geonames'):
                by_gn.setdefault(e.auth['geonames'], e)
        for rec in ctx_files:
            if rec.member != _COORDS:
                continue
            root = C.parse_xml(rec.read())
            for ent, aliases in C.parse_register(root, ctx, rec.path, doc_id=None)[0]:
                if ent.lat is None:
                    continue
                tgt = ctx.shared.entities.get(ent.local_key) or by_gn.get(ent.auth.get('geonames'))
                if tgt is not None and tgt.lat is None:
                    tgt.lat, tgt.lon = ent.lat, ent.lon
                    tgt.coord_source = 'tei:geo:listPlace_s2_lat_lng'
        return ents

    def extract(self, rec, ctx):
        if rec.path in ctx.options.get('_register_paths', ()):
            return [], []
        return super().extract(rec, ctx)

    def doc_date(self, root, rec, ctx):
        return H.text_dates_doc_date(root)

    def extract_root(self, root, rec, ctx):
        self._rd = V.RunningDate(root, 1500, 1750)
        return super().extract_root(root, rec, ctx)

    def adjust(self, m, hit):
        r = (hit.el.get('role') or '').strip().lower()
        if r in _ROLE:
            m.role = _ROLE[r]
        rd = getattr(self, '_rd', None)
        rg = rd.range_at(hit.el) if rd else None
        if rg:
            V.set_range(m, rg[0], rg[1], 'text')
        return m


EXTRACTOR = Extractor()
