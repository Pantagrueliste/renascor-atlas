"""DISCO, Diachronic Spanish Sonnet Corpus, Golden Age slice (15th-17th centuries, author ids ending in 'g').

One TEI file per sonnet (tei/15th-17th/per-sonnet, 1,088 files); the entity information lives in the header (listPerson
with the author, birth / death place, century or year); the running text has no entity tags (the 54 <name> tags are
rhyme annotations like "[Ser]"). The per-author and all-periods files are concatenations of the same sonnets and are not
read (no double counting). Register: ancillary-files/prosopography.xml (477 Golden Age authors in the file; VIAF url
or bare VIAF number with @cert, birth / death place, birthDate / deathDate when known), read once by prepare().
Mentions (zone 'hdr', one set per sonnet file):
  * person: the author of the sonnet, role 'author', resolved to the author entity 'disco_sonnets:disco_001g'
    (VIAF in auth for 202 of the authors);
  * place: birth place and death place of the author (placeName under birth/location, death/location), roles
    'birth_place' and 'death_place' (non-standard, documented here); surface = settlement (as written, may contain a
    province in brackets) or the country when only the country is given. Entities for these places are built from the
    prosopography (key '<slug>:place:<settlement or country>'), without coordinates (none in the source).
Dating: the author's lifespan, not a sonnet date: birthDate / deathDate years when the header gives them (about 22% of
authors), otherwise the century of birth / death (e.g. century 17 -> 1601-1700), otherwise the folder range 1401-1700;
date_source 'header'. in_period therefore means "the author's life overlaps 1450-1700".
v2 (audit points 6 and 12):
  * the editors' ``@cert`` of a prosopography VIAF is honoured: only ``cert`` high (or no cert) goes to ``auth.viaf``; medium and low
    ones are kept as ``viaf-suggested:<id>`` in ``auth.other`` (31 of the 159 authors with a Wikidata id failed a check);
  * the rows are header metadata repeated in each of the 1,088 sonnets (Sevilla 232 = sonnets whose author was born there):
    the author, birth place and death place are emitted ONCE per author, with the first sonnet file in path order;
  * the date is the AUTHOR's lifespan (``date_source`` 'author_lifespan'), not a sonnet date: a range wider than 20 years has no
    doc_year (352 authors known by their century only), and must not be used for year histograms.
Licence: LICENSE file CC BY 4.0 (LICENSE_TIERS lists LicenseRef-CC-BY-unversioned), T1. Text from Biblioteca Cervantes
Virtual editions.
"""
from __future__ import annotations

import re

from .. import common as C
from ..model import Entity, Mention, best_year, period_overlap
from .generic_tei import GenericTEI

SLUG = 'disco_sonnets'
COLLS = ['queue5/disco-main/tei/15th-17th/per-sonnet', 'queue5/disco-main/ancillary-files/prosopography.xml']
RENASCOR_IDS = ['disco-spanishsonnets']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'DISCO Golden Age sonnets: authors (VIAF) with birth/death places, from headers and the prosopography'
OPTIONS = {'default_years': (1401, 1700)}

_PROSO = 'queue5/disco-main/ancillary-files/prosopography.xml'
_GOLDEN = re.compile(r'^disco_\d+g$')


def _t(el) -> str:
    return C.surface_text(el) if el is not None else ''


def _year(el):
    v = (el.get('content') or '').strip() if el is not None else ''
    return int(v) if re.fullmatch(r'\d{4}', v) else None


def _century(ev):
    for d in ev.iter('{*}date') if ev is not None else ():
        if d.get('type') == 'century':
            t = _t(d)
            if re.fullmatch(r'\d{1,2}', t):
                return int(t)
    return None


def _event_info(person, name):
    ev = C.child(person, name)
    if ev is None:
        return None, None, None
    year = next((_year(d) for d in ev.iter('{*}date') if d.get('property', '').endswith('Date')), None)
    pl = next((p for p in ev.iter('{*}placeName')), None)
    settlement = _t(next((s for s in pl.iter('{*}settlement')), None)) if pl is not None else ''
    country = _t(next((s for s in pl.iter('{*}country')), None)) if pl is not None else ''
    place = settlement or country or None
    return year, _century(ev), place


def lifespan(person, default):
    by, bc, _ = _event_info(person, 'birth')
    dy, dc, _ = _event_info(person, 'death')
    lo = by if by else ((bc - 1) * 100 + 1 if bc else None)
    hi = dy if dy else (dc * 100 if dc else None)
    if lo is None and hi is None:
        return default
    return (lo if lo is not None else hi, hi if hi is not None else lo)


def place_key(place: str) -> str:
    return C.make_key(SLUG, 'place:' + place)


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return rec.path == _PROSO

    def prepare(self, ctx):
        ents = []
        places: dict[str, Entity] = {}
        for rec in ctx.all_files:
            if rec.path != _PROSO:
                continue
            root = C.parse_xml(rec.read())
            for p in root.iter('{*}person'):
                ident = p.get(C.XML_ID) or ''
                if not _GOLDEN.match(ident):
                    continue
                names = {n.get('type'): n for n in p if C.ln(n.tag) == 'persName'}
                name = _t(names.get('source')) or C.name_text(names['full']) if names else ident
                variants = [v for v in (_t(names.get('short')), C.name_text(names['full']) if 'full' in names else '')
                            if v and v != name]
                toks, suggested = [], []
                for i in p:
                    if C.ln(i.tag) == 'idno':
                        v = (i.get('resource') or _t(i)).strip()
                        if v:
                            tok = v if 'viaf' in v else 'viaf:' + v
                            if (i.get('cert') or 'high').strip().lower() in ('high', ''):
                                toks.append(tok)
                            else:
                                suggested.append(tok)
                by, bc, bplace = _event_info(p, 'birth')
                dy, dc, dplace = _event_info(p, 'death')
                ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind='person', name=name,
                             source_path=rec.path, renascor_id=ctx.renascor_id, variants=variants,
                             birth=str(by) if by else (f'c{bc}' if bc else None),
                             death=str(dy) if dy else (f'c{dc}' if dc else None),
                             auth=C.parse_auth(toks), note_type='author', licence=ctx.licence, tier=ctx.tier)
                for tok in suggested:
                    sid = C.classify_token(tok)[1]
                    ent.auth['other'].append(f'viaf-suggested:{sid}')
                ctx.shared.add(ent, [ident])
                ents.append(ent)
                for pl in (bplace, dplace):
                    if pl and place_key(pl) not in places:
                        e = Entity(local_key=place_key(pl), coll_slug=ctx.slug, kind='place', name=pl, source_path=rec.path,
                                   renascor_id=ctx.renascor_id, note_type='place', licence=ctx.licence, tier=ctx.tier)
                        places[e.local_key] = e
                        ctx.shared.add(e, [e.local_key])
            ents.extend(places.values())
        ctx.options['_register_paths'] = {_PROSO}
        # the first sonnet (path order) of each author carries the author / birth / death rows
        first: dict[str, str] = {}
        for rec in ctx.all_files:
            if rec.path == _PROSO:
                continue
            m = re.search(rb'xml:id="(disco_\d+g)"', rec.read()[:6000])
            if m:
                first.setdefault(m.group(1).decode(), rec.path)
        ctx.options['_first_sonnet'] = first
        return ents

    def extract_root(self, root, rec, ctx):
        person = next((p for p in root.iter('{*}person')), None)
        if person is None:
            return [], []
        ident = person.get(C.XML_ID) or ''
        first = ctx.options.get('_first_sonnet')
        if first and first.get(ident, rec.path) != rec.path:
            return [], []                 # the author's rows are emitted with the first sonnet only
        lo, hi = lifespan(person, ctx.default_years)
        dsrc = 'author_lifespan' if (lo, hi) != tuple(ctx.default_years) else 'collection'
        year, inp = best_year(lo, hi), period_overlap(lo, hi)
        doc_id = C.doc_id_of(root, rec)
        ms = []

        def add(kind, tag, role, surface, key, ent):
            ms.append(Mention(coll_slug=ctx.slug, renascor_id=ctx.renascor_id, source_path=rec.path, doc_id=doc_id,
                              doc_year=year, doc_year_from=lo, doc_year_to=hi, date_source=dsrc, in_period=inp,
                              kind=kind, tag=tag, role=role, zone='hdr', surface=surface, ref_raw=ident or None,
                              local_key=key if ent is not None else None, in_register=True if ent is not None else None,
                              auth=C.merge_auth(C.empty_auth(), ent.auth) if ent is not None else C.empty_auth(),
                              context=None, lang='es', licence=ctx.licence, tier=ctx.tier))

        names = {n.get('type'): n for n in person if C.ln(n.tag) == 'persName'}
        surface = _t(names.get('source')) or _t(next((a for a in root.iter('{*}author')), None))
        ent = ctx.shared.lookup(ident)
        add('person', 'persName', 'author', surface, C.make_key(ctx.slug, ident), ent)
        for evname, role in (('birth', 'birth_place'), ('death', 'death_place')):
            _, _, place = _event_info(person, evname)
            if place:
                add('place', 'placeName', role, place, place_key(place), ctx.shared.lookup(place_key(place)))
        return ms, []


EXTRACTOR = Extractor()
