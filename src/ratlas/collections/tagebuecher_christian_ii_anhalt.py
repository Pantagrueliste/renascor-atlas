"""Diary of Prince Christian II of Anhalt-Bernburg (HAB Wolfenbuettel / Uni Freiburg), 338 monthly TEI files 1621-11
to 1656 (the 339th record, mets.xml, is not TEI text and is not read).

Conventions of this edition:
  * mentions are <rs type="person|place|org" ref="#id"> in the running text; rs of type symbol, abbreviation, bibl,
    term, translation are not entities and are skipped; typos ('peron', 'pers', 'plache', 'olace') are mapped;
    <rs type="#id"> (value and ref swapped, 3 tags) stays unmapped;
  * every monthly file embeds the subset of the register it needs in teiHeader/sourceDesc (listPerson / listPlace /
    listOrg, global slug ids such as '#anhalt-bernburg-harzgerode_friedrich', GND/VIAF urls on persName, GeoNames/GND
    urls on placeName). Ids are global slugs, so keys are shared ('<slug>:<id>') and prepare() merges the registers of
    all files (first one wins; 4 id conflicts across files are ignored). Dangling refs (about 3%) keep an opaque key
    '<slug>:<id>' (OPTIONS opaque_keys, in_register False);
  * places have GeoNames/GND ids but NO coordinates in the files (coordinates would need a GeoNames lookup);
  * <index indexName="place"><term ref="#id"> of each entry (the place where the prince was) is read as a place mention
    with tag 'index@place' and the non-standard role 'entry_place' (its text is the place name as given);
  * dating: header title date (<date when="1635-02"> in the title, Julian calendar as given), source 'header';
  * the diary is one person's: co-mentions are co-occurrences inside entries.
Licence: CC-BY-SA-3.0 as in the catalogue and LICENSE_TIERS (T1); the TEI headers say CC BY-SA 4.0
(creativecommons.org/licenses/by-sa/4.0): discrepancy to be reconciled by the owner.
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'tagebuecher_christian_ii_anhalt'
COLLS = ['renascor_2026-10/tagebuecher-christian-ii-anhalt']
RENASCOR_IDS = ['tagebuecher-christian-ii-anhalt']
LICENCE = 'CC-BY-SA-3.0'
TIER = 'T1'
DESCRIPTION = 'Diary of Christian II of Anhalt-Bernburg 1621-1656 with GND/GeoNames-linked persons, places, orgs'
OPTIONS = {'opaque_keys': True,
           'type_map': {'peron': 'person', 'pers': 'person', 'plache': 'place', 'olace': 'place'}}

_MONTH_FILE = re.compile(r'/\d{4}_\d{2}\.xml$')


def _header(root):
    for el in root.iter():
        if C.ln(el.tag) == 'teiHeader':
            return el
    return root


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if _MONTH_FILE.search(r.path)]

    def prepare(self, ctx):
        out = []
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for ent, aliases in C.parse_register(_header(root), ctx, rec.path, doc_id=None)[0]:
                if ent.local_key in ctx.shared.entities:
                    continue
                ctx.shared.add(ent, aliases)
                out.append(ent)
        return out

    def extract_root(self, root, rec, ctx):
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []

    def scan(self, root, rec, ctx, counters):
        hits = C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root))
        doc_lang = C.doc_language(root)
        for ix in root.iter():
            if C.ln(ix.tag) != 'index' or ix.get('indexName') != 'place':
                continue
            for t in ix:
                if C.ln(t.tag) == 'term' and (t.get('ref') or '').strip() not in ('', '#'):
                    hits.append(C.Hit(kind='place', tag='index@place', name='term', surface=C.surface_text(t),
                                      zone='txt', role='entry_place', refs={'ref': t.get('ref').strip()},
                                      lang=doc_lang, context=None, el=t, ancestors=('text', 'index')))
        return hits


EXTRACTOR = Extractor()
