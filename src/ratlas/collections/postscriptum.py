"""P.S. Post Scriptum (CLUL, Lisboa): 1,881 private letters, Spain (ES1500, ES1600) and Portugal (PT1500, PT1600).

Licence: LicenseRef-unclear, tier T3 (LICENSE_TIERS: PostScriptum, recipe only, withdrawn from the archive on 2026-10-16;
PORTULAN CLARIN says CC BY-NC-ND 3.0, the TEI headers say 'Free download; CLUL is recognised as author by a set of ELRA
licences'). The stricter reading applies: facts only (ids, place strings, coordinates, years), no context snippets.

Mentions are the correspDesc of every letter:
  * <persName ref="pscdd:XYZn"> under correspAction[sent|received] -> role sender / addressee. The ids point to the
    external cdd.xml (not in the harvest); they are stable project-wide, so local_key = 'postscriptum:<XYZn>' with
    in_register False (no person entities: the names are only the inline strings);
  * <location><placeName>España, Guadalajara, Atienza</placeName><geo>lat lon</geo></location> -> role sent_from /
    sent_to. The hierarchical place string is the key: prepare() reads every distinct string and its <geo> (the most
    frequent non-empty value) and writes one place entity per string ('inline gazetteer', coord_source 'tei:geo', note
    'place'; NOT a separate register of the edition). Mentions get local_key 'postscriptum:<string>' (in_register
    True). Coordinates exist for about 70 % of the strings, mostly for senders;
  * untyped body <placeName> (509, datelines such as 'En Enberes') are kept as place mentions in zone txt without key;
  * msIdentifier/settlement|country (where the archive keeps the letter) are not mentions and are skipped.
Dating: correspAction[sent]/date@when (year level), all 1,881 letters (1500-1699).

v2:
  * the person register ``cdd.xml`` (CARDS demographic database, ps.clul.ul.pt/files/, harvested in ``postscriptum-registers/``,
    5,069 persons and institutions with name, sex, floruit...) is loaded by ``prepare()``: 'postscriptum:<XYZn>' now has a
    name (in_register True; 96% of the mentions resolve). Pseudo-names 'Anonimo<n>' are placeholders (excluded from the default
    counts); the register's licence stays T3;
  * places: inline hierarchical strings that share the same ``<geo>`` coordinates are ONE entity (Lisboa had 8 keys, Amsterdam 3);
    the most frequent string is the name and the others are variants; strings without coordinates are merged on their
    case / accent / punctuation-folded form;
  * 'S.l.' (sine loco), 's.n.' and empty strings are placeholders, not places (zone/exclusion by ``classify``).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .. import common as C
from .. import helpers_batch0 as H0
from .. import helpers_v2 as V
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'postscriptum'
COLLS = ['postscriptum']
RENASCOR_IDS = ['postscriptum']
LICENCE = 'LicenseRef-unclear'
TIER = 'T3'
DESCRIPTION = 'Post Scriptum letters 1500-1699: sender/addressee ids (pscdd), origin/destination places with geo'
OPTIONS = {}

_LOC = re.compile(rb'<placeName>([^<]*)</placeName>\s*(?:<geo>([^<]*)</geo>)?')
_CDD = V.NEW + '/postscriptum-registers/cdd.xml'


def _cdd_fix(ent, aliases, els):
    ent.note_type = 'person' if ent.kind == 'person' else ent.kind
    el = els.get(aliases[0]) if aliases else None
    if el is not None and el.get('sex') and ent.kind == 'person':
        ent.note_type = 'person'
    return ent, aliases


_PSCDD = re.compile(r'^(?:pscdd:)?([A-Za-z]+\d*)$')    # JR8, PO1; institutions have letters only (ST, AB)


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        geos = defaultdict(Counter)
        seen: dict[str, str] = {}
        for rec in ctx.all_files:
            for name, geo in _LOC.findall(rec.read()):
                name = C.norm(name.decode('utf8', 'replace'))
                if not name:
                    continue
                seen.setdefault(name, rec.path)
                g = C.norm((geo or b'').decode('utf8', 'replace'))
                if g:
                    geos[name][g] += 1
        # one entity per coordinate pair (else per folded string); the commonest string names it
        freq = Counter()
        for rec in ctx.all_files:
            for name, _geo in _LOC.findall(rec.read()):
                freq[C.norm(name.decode('utf8', 'replace'))] += 1
        groups: dict[str, list[str]] = defaultdict(list)
        coords: dict[str, tuple] = {}
        for name in sorted(seen):
            ll = C._parse_geo(geos[name].most_common(1)[0][0]) if geos[name] else None
            gk = f'geo:{ll[0]:.3f},{ll[1]:.3f}' if ll else 'str:' + H0.fold(name)
            groups[gk].append(name)
            if ll:
                coords[gk] = ll
        out = []
        for gk, names in sorted(groups.items()):
            names.sort(key=lambda n: (-freq[n], n))
            main = names[0]
            ent = Entity(local_key=f'{SLUG}:{main}', coll_slug=SLUG, kind='place', name=main, source_path=seen[main],
                         renascor_id=ctx.renascor_id, variants=names[1:], note_type='place', licence=ctx.licence,
                         tier=ctx.tier)
            if gk in coords:
                ent.lat, ent.lon = coords[gk]
                ent.coord_source = 'tei:geo'
            ctx.shared.add(ent, names)
            out.append(ent)
        out += V.load_registers(ctx, [_CDD], fix=_cdd_fix)
        return out

    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=False, doc_lang=C.doc_language(root),
                               counters=counters, skip_ancestors=('msIdentifier',))

    def extract_root(self, root, rec, ctx):
        hits = self.scan(root, rec, ctx, Counter())
        date = self.doc_date(root, rec, ctx)

        def adj(m, hit):
            if m.kind == 'person' and m.ref_raw:
                mm = _PSCDD.match(m.ref_raw.strip())
                if mm:
                    ent = ctx.shared.lookup(mm.group(1))
                    if ent is not None and ent.kind == 'person':
                        H0.attach(m, ent)
                    else:
                        m.local_key, m.in_register = f'{SLUG}:{mm.group(1)}', False
            elif m.kind == 'place' and m.zone == 'hdr' and not m.ref_raw:
                ent = ctx.shared.lookup(m.surface)
                if ent is not None:
                    m.local_key, m.in_register = ent.local_key, True
            return m

        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=adj), []


EXTRACTOR = Extractor()
