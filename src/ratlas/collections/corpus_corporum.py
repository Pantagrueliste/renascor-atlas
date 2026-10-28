"""Corpus Corporum (Univ. Zurich), 952 Latin texts, classical to neo-Latin (mostly CAMENA-derived verse).

Licence: LicenseRef-CorpusCorporum (non-commercial), tier T2 (LICENSE_TIERS: CorpusCorp).

Tagging: 47 files carry typed names in the running text, <name type=...> and <rs type=...> (5 files hold 90 %). Types:
person, place, org, myth, nation, religion (and a few populus). Mapping to the three atlas kinds:
  person  -> person ('name@person', 'rs@person')
  myth, religion -> person, tag 'name@myth' / 'name@religion' (gods, heroes, Maria, Jesus, saints: filter on the tag)
  nation, populus -> org, tag 'name@nation' (ethnonyms are neither persons nor places; filter on the tag)
  place -> place;  org -> org.
Untyped <name> (2,617, one file alone: 13407.xml, 921, mixes cited persons, places and the modern encoders) cannot be
classified and are skipped. No register and no ids: 99 % of the tags have none. Where there are some, they are kept in ref_raw and in auth
(VIAF in the odd form http://www.viaf.org/<id> is recognised; GeoNames and Pleiades ids on a few place tags of 21626.xml);
@key holds a name label ('Sinae', 'Trigault, Nicolas S.J.'), not an id, so local_key stays None. The normalised
name of <rs n="Vergilius"> is kept in ref_raw as 'n:Vergilius'.
Extra mentions: the header <author ref="viaf..."> (797 files) is emitted as a person mention with tag 'author', role
'author', zone hdr, because the VIAF id is an explicit editorial authority link (filter tag == 'author' to drop them).

Dating (no document date as such): creation/origDate in the header; else a year <= 1800 in sourceDesc (imprint of the
early modern source); else the highest year of the author's lifespan in titleStmt/author/date (death year proxy, as the
catalogue does; 'fl.c.1450' gives 1450); else any other header date. date_source is 'header' for all of them.
v2 (issues 24, 107, 108, 110): (a) a 3-digit ``creation`` 400 / 500 / 600 in a text by an author who lived 1400-1699 is the Quattrocento /
Cinquecento / Seicento (1400-1499...), not AD 400 (76 BibIt texts); (b) a hyphen before a lifespan year is a dash, not a minus
('Author-1548' is 1548); (c) a reference-edition imprint after 1700 (the Lipsius Sylloge 13999.xml, Leiden 1727; 1660, 1671...) is
not the date of the text when the author's death year is known and earlier: the author's death year is used instead; (d) trailing
punctuation is dropped from surfaces ('Leidam.' -> 'Leidam'). Tags keep their type: name@myth, name@religion are mythological /
biblical persons, name@nation / populus are peoples (person_class), all visible to the counts filters.
Antiquity is dated with years before 1000 (creation 400, author -19) so that it is dated and out of period.
"""
from __future__ import annotations

from lxml import etree

from .. import common as C
from .. import helpers_batch5 as H
from .generic_tei import GenericTEI

SLUG = 'corpus_corporum'
COLLS = ['corpuscorp']
RENASCOR_IDS = ['corpus-corporum']
LICENCE = 'LicenseRef-CorpusCorporum'
TIER = 'T2'
DESCRIPTION = 'Latin texts, typed names (person/place/myth/nation) in a few files; author VIAF in headers'
OPTIONS = {'type_map': {'myth': 'person', 'religion': 'person', 'nation': 'org', 'populus': 'org',
                        # typos / rare values of the source
                        'myth?': 'person', 'person?': 'person', 'plerson': 'person', 'religon': 'person',
                        'regio': 'place', 'river': 'place', 'locus': 'place'}}


class Extractor(GenericTEI):
    def scan(self, root, rec, ctx, counters):
        dl = C.doc_language(root)
        hits = C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context, doc_lang=dl,
                               counters=counters)
        hdr = H.first_header(root)
        if hdr is not None:
            for el in hdr.iter():
                if C.ln(el.tag) == 'author' and el.get('ref'):
                    name = H.own_text(el)
                    if name:
                        hits.append(H.make_hit(el, kind='person', tag='author', surface=name, zone='hdr',
                                               role='author', refs={'ref': el.get('ref').strip()}, lang=dl,
                                               context=name if ctx.want_context else None))
        return hits

    def adjust(self, m, hit):
        el = hit.el
        if not m.auth.get('viaf') and m.ref_raw:
            mm = H.VIAF_ANY.search(m.ref_raw)
            if mm:
                m.auth['viaf'] = mm.group(1)
        if not m.ref_raw and el.get('n'):
            m.ref_raw = 'n:' + C.norm(el.get('n'))
        return m

    def doc_date(self, root, rec, ctx):
        hdr = H.first_header(root)
        if hdr is None:
            return C.DocDate()
        creation, source, lifespan = [], [], []
        life_all = []
        stack = []
        for ev, el in etree.iterwalk(hdr, events=('start', 'end')):
            n = C.ln(el.tag)
            if n is None:
                continue
            if ev == 'end':
                stack.pop()
                continue
            stack.append(n)
            if n in ('date', 'creation', 'origDate'):
                rg = H.any_year_range(el)
                if not rg:
                    continue
                if {'creation', 'origDate'} & set(stack):
                    creation.append(rg)
                elif 'author' in stack and 'titleStmt' in stack:
                    ys = [abs(y) if 1400 <= abs(y) <= 1700 else y for y in H.years_in(C.surface_text(el))]
                    lifespan.append(max(ys or [abs(rg[1]) if 1400 <= abs(rg[1]) <= 1700 else rg[1]]))
                    life_all.extend(ys)
                elif 'sourceDesc' in stack and 1000 <= rg[0] <= 1800:
                    source.append(rg)
        modern_author = bool(life_all) and all(1400 <= y <= 1700 for y in life_all)
        if creation:
            lo, hi = min(r[0] for r in creation), max(r[1] for r in creation)
            if modern_author and lo in (400, 500, 600) and hi in (400, 500, 600):
                return C.DocDate(1000 + lo, 1000 + hi + 99, 'header')          # Quattrocento / Cinquecento / Seicento
            return C.DocDate(lo, hi, 'header')
        if source:
            lo, hi = min(r[0] for r in source), max(r[1] for r in source)
            if lo > 1700 and lifespan and max(lifespan) <= 1700:
                y = max(lifespan)
                return C.DocDate(y, y, 'header')                              # imprint of a later edition: author's death year
            return C.DocDate(lo, hi, 'header')
        if lifespan:
            y = max(lifespan)
            return C.DocDate(y, y, 'header')
        return super().doc_date(root, rec, ctx)


EXTRACTOR = Extractor()
