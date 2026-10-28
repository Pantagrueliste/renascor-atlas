"""Andreas Bodenstein von Karlstadt, Kritische Gesamtausgabe (HAB Wolfenbuettel, edoc ed000216): 74 transcripts of
writings and letters 1507-1518 (files/xml/NNN_transcript.xml; the METS file is not a text and is skipped).

Licence: LicenseRef-Custom (HAB terms, no republication), tier T3 (LICENSE_TIERS: Renascor_karlstadt-kritische-
gesamtausgabe). Extracted for internal work only; no context snippets.

Mentions: <rs type=person|place ref="#slug"> in the text (readable slugs: karlstadt_andreas, thomas_aquinas, wittenberg,
allerheiligenstift_wittenberg, also gods and muses: minerva, apollo), a few untyped / bare persName / placeName,
and the header <author ref="#karlstadt_andreas"> (tag 'author', role author).
The register of the edition (personenregister.xml, ortsregister.xml, xi:include'd from the header, edoc/ed000216/texts/
register) is not in the harvest: the slugs are opaque keys ('karlstadt_kritische_gesamtausgabe:thomas_aquinas',
in_register False), no entities, no authority ids. Ancient, mythological and patristic persons cannot be flagged from
the data (no register); they dominate the person counts.
v2: the register (personenregister.xml, ortsregister.xml, bibliography.xml; the xi:include paths ``texts/register/`` return 404, the files
live in ``ed000216/register/``, harvested in ``karlstadt_kritische_gesamtausgabe-registers/``) is loaded by ``prepare()``: 99.9%
of the mentions resolve; persons carry 379 GND ids, places 114 GeoNames ids; entries with a mythology / Bible note get
note_type mythological / biblical. File 071A is not served (404). Footnotes (note type 'footnote', 1,058 notes) and the critical
apparatus (note type 'crit_app') are the modern editors': zone ``footnote`` / ``apparatus``; marginal notes of the manuscripts
(note place='margin') stay in the text. HAB terms: private scholarly use only, tier T3.
Roles (v2, issue 77): a letter's title reads 'X an Y [place] [date]'. The header <author ref=#karlstadt_andreas> is the EDITION's principal,
not necessarily the writer: it is ``sender`` when his name stands before ' an ', ``addressee`` when it stands after it (Scheurl an
Karlstadt). The placeName of the title is the place of the addressee (``sent_to``; the audit suggested sent_from, but 'Scheurl an Karlstadt
Wittenberg' puts Wittenberg where the letter went). The other correspondent is plain text in the title, not a tag: not a mention.
Dating: the <date when> of the titleStmt title (the writing's date), else the collection span 1507-1518.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch5 as H
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'karlstadt_kritische_gesamtausgabe'
COLLS = ['renascor_2026-10/karlstadt-kritische-gesamtausgabe']
RENASCOR_IDS = ['karlstadt-kritische-gesamtausgabe']
LICENCE = 'LicenseRef-Custom'
TIER = 'T3'
DESCRIPTION = 'Karlstadt writings and letters 1507-1518, #slug references without register'
OPTIONS = {'opaque_keys': True, 'default_years': (1507, 1518)}


_REG = V.NEW + '/karlstadt_kritische_gesamtausgabe-registers/ed000216/register/'


class Extractor(GenericTEI):
    def prepare(self, ctx):
        return V.load_registers(ctx, [_REG + 'personenregister.xml', _REG + 'ortsregister.xml'], fix=V.hab_person_fix)

    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if not r.name.lower().startswith('mets')]

    def scan(self, root, rec, ctx, counters):
        dl = C.doc_language(root)
        hits = C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context, doc_lang=dl,
                               counters=counters)
        hits = hits + H.author_ref_hits(root, dl, ctx.want_context)
        title = next((t for t in root.iter() if C.ln(t.tag) == 'title' and not t.get('type')), None)
        if title is not None:
            full = C.surface_text(title)
            m = re.search(r'\san\s', full)
            if m:
                for h in hits:
                    if h.zone != 'hdr':
                        continue
                    if h.tag == 'author' and h.role == 'author':
                        surname = (h.surface.split() or [''])[-1].lower()
                        h.role = 'addressee' if surname and surname in full[m.end():].lower() else 'sender'
                    elif h.kind == 'place' and 'title' in h.ancestors and 'titleStmt' in h.ancestors:
                        h.role = 'sent_to'
        return hits


EXTRACTOR = Extractor()
