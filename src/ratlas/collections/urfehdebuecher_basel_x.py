"""Urfehdebuch X of Basel (Ratsbuch O 10, 1563-1569), one TEI file with 621 entries (Uni Basel / ZIM Graz).

Conventions of this edition:
  * persName / placeName in the running text carry @ana with the role of the person or place in the case:
      persons: PersonTaeter -> offender, PersonMittaeter -> accomplice, PersonOpfer -> victim, PersonKlaeger ->
      plaintiff, PersonZeuge -> witness, PersonNotar -> notary, PersonAnwesende -> present, PersonHehler ->
      receiver, Schlagwort*Knecht (Ratsknecht, Stadtknecht, Oberster Knecht) -> court_official;
      places: OrtHerkunft -> origin, OrtWohnort -> residence, OrtTatort -> crime_scene, SchlagwortGefaengnis ->
      prison, SchlagwortWirtshaus -> inn.
    These roles are documented non-standard values; several @ana tokens give the first one of the lists above;
    anything else stays 'mention';
  * placeName/@ref='#Id' points to the listPlace of the same file (369 places, xml:id slug and a modern name,
    no GND/GeoNames/coordinates). The register is shared (keys '<slug>:<id>') since there is one file;
  * persName has no @ref and there is no person register (names are criminals, victims, notaries): local_key None;
  * dating: the book is 1563-1569 (sourceDesc/msContents/origDate). v2: each mention takes the date of the last
    <date> before it (613 of the 624 entries carry a date: 22 entries in 1563, 150 in 1565, 107 in 1566...); the one entry
    dated '1656' among 1566 is a typo and is ignored (a date more than 5 years from the median of its neighbours is dropped).
Licence: CC-BY-NC-SA-4.0 (LICENSE_TIERS, T2).
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'urfehdebuecher_basel_x'
COLLS = ['renascor_2026-10/urfehdebuecher-basel-x']
RENASCOR_IDS = ['urfehdebuecher-basel-x']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Basel oaths of truce 1563-1569: persons with their role in the case, places with a register'
OPTIONS = {}

_PERS_ROLES = [('personmitt', 'accomplice'), ('persontaeter', 'offender'), ('persontäter', 'offender'),
               ('personopfer', 'victim'), ('personkläger', 'plaintiff'), ('personklaeger', 'plaintiff'),
               ('personzeuge', 'witness'), ('personnotar', 'notary'), ('personanwesende', 'present'),
               ('personhehler', 'receiver'), ('schlagwortratsknecht', 'court_official'),
               ('schlagwortstadtknecht', 'court_official'), ('schlagwortoberstersknecht', 'court_official'),
               ('schlagwortobersterknecht', 'court_official')]
_PLACE_ROLES = [('ortherkunft', 'origin'), ('ortwohnort', 'residence'), ('orttatort', 'crime_scene'),
                ('schlagwortgefängnis', 'prison'), ('schlagwortgefaengnis', 'prison'),
                ('schlagwortwirtshaus', 'inn')]


def role_from_ana(ana: str, kind: str) -> str:
    toks = [t.lstrip('#').lower().replace('uf_', '') for t in (ana or '').split()]
    table = _PERS_ROLES if kind == 'person' else _PLACE_ROLES
    for key, role in table:
        if any(t.startswith(key) for t in toks):
            return role
    return 'mention'


class Extractor(GenericTEI):
    def prepare(self, ctx):
        out = []
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for ent, aliases in C.parse_register(root, ctx, rec.path, doc_id=None)[0]:
                ctx.shared.add(ent, aliases)
                out.append(ent)
        return out

    def extract_root(self, root, rec, ctx):
        self._rd = V.RunningDate(root, 1500, 1700)
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root))

    def adjust(self, m, hit):
        if m.role == 'mention':
            m.role = role_from_ana(hit.el.get('ana'), m.kind)
        rd = getattr(self, '_rd', None)
        rg = rd.range_at(hit.el) if rd else None
        if rg:
            V.set_range(m, rg[0], rg[1], 'text')
        return m


EXTRACTOR = Extractor()
