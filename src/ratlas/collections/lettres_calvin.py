"""Lettres de Jean Calvin (Yanet Hernandez Pedraza, Univ. de Geneve / lettres-calvin on GitHub), 27 French letters
1538-1553 (src/lib/xml; the 27 Spanish translations in src/lib/xmles are not read).

Licence: CC-BY-SA-4.0, tier T1 (LICENSE_TIERS: Renascor_lettres-calvin; the legal notice of the site says 4.0, the TEI
header still says 3.0).

Mentions: persName / placeName / orgName / settlement with @ref = Wikidata url, @key = readable label ('Genève',
'Louis du Tillet') and @corresp = '#slug' (not resolvable: there is no register). They occur in correspDesc
(sender / addressee, sent_from / sent_to) and in the letter text, including Bonnet's 1854 commentary notes, which the
project encoded too (zone txt). No local key (the @corresp slugs are not stable; the Wikidata id in auth identifies the
entity); the label stays in ref_raw. persName type=collectivite ('Seigneurs de Geneve') is an 'org', <name
type=individu|collectivite> are mapped to person / org. Untyped <name> in sourceDesc (author, editor) are not mentions; the
sender Calvin has no @ref. Empty persName of correspContext (prev / next letter links) are skipped.
Coordinates: none in the files (Wikidata ids allow a later lookup).
Roles (v2): ``sent_from`` only for the place of sending (the correspAction[sent] place, or a dateline place that is not preceded by
'Envoyée à' / 'à' in the dateline); a dateline place introduced by 'Envoyée à' is ``sent_to``; places in the editorial notes
inside the dateline are plain mentions (the 27 letters have 27 true sent_from places; 72 more rows had been tagged).
Dating: correspAction[sent]/date@when, 27 of 27.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch5 as H
from .generic_tei import GenericTEI

SLUG = 'lettres_calvin'
COLLS = ['renascor_2026-10/lettres-calvin/files/lettres-calvin-main.tar.gz!lettres-calvin-main/src/lib/xml/']
RENASCOR_IDS = ['lettres-calvin']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Calvin French letters 1538-1553 with Wikidata ids on persons and places'
OPTIONS = {'type_map': {'individu': 'person', 'collectivité': 'org', 'collectivite': 'org'}}


class Extractor(GenericTEI):
    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root), counters=counters,
                               skip_ancestors=('correspContext', 'msIdentifier'))

    def adjust(self, m, hit):
        if m.kind == 'place' and m.role == 'sent_from' and m.zone == 'txt' and 'dateline' in hit.ancestors:
            par = hit.el.getparent()
            # text just before the tag inside its dateline
            prev = hit.el.getprevious()
            before = ((prev.tail or '') if prev is not None else (par.text or '' if par is not None else '')).lower()
            if 'note' in hit.ancestors:
                m.role = 'mention'
            elif re.search(r'envoy[ée]e?\s+(?:à|a)\s*$|(?:^|\s)à\s*$', before.strip()):
                m.role = 'sent_to'
        t = (hit.el.get('type') or '').strip().lower()
        if m.tag == 'persName' and t in ('collectivité', 'collectivite'):
            m.kind = 'org'
        if not m.surface and not m.ref_raw:
            return None
        return H.drop_header_context(m)


EXTRACTOR = Extractor()
