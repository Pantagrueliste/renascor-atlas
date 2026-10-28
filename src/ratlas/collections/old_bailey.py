"""Old Bailey Online (Sessions Papers + Ordinary's Accounts), the 1674-1700 slice.

The harvest holds the whole of OBO (1674-1913, 2,638 distinct XML files); only the files whose name year is
<= 1700 are read (307 files: 198 sessions papers and 109 Ordinary's Accounts), the rest is out of the period and
would be 99% of the tags. TEI P4 (<TEI.2>, no teiHeader, ISO-8859-1), ids in @id.

Tagged: <persName> (typed defendantName / victimName / jurorName / judiciaryName in the sessions, untyped for
witnesses and in the Ordinary's Accounts) with <interp type=surname|given|gender> children, and <placeName>
(untyped here). <rs> carries offence / verdict / punishment / occupation / alias / crimeDate categories: not
entities, never read. There is no register and no @ref: mentions have no local_key (the same person in two
sessions cannot be told from a homonym; merging is a later stage). The surface is the printed spelling.

v2 (issues 191-192): an UNTYPED <persName> without a surname <interp> (a child, or an interp whose @inst is the persName's id) is not
a name ('the prisoner', generic words) and is skipped (typed defendant / victim / juror / judge names are always kept);
each defendant / victim / witness gets its own opaque key 'old_bailey:<persName id>' (a trial-local id: the same man in two sessions
cannot be told from a homonym, so the 79 'John Smith' are no longer one person); nested tags are read innermost-first
('Michael Hair , and Mary his Wife' gives two persons, not one string).
Non-standard roles (documented here): defendant, victim, juror, judge (from the persName @type). Others 'mention'.
Dating: the year of the file name YYYYMMDD[x] (date_source 'filename'); the document is a session of that date.
Licence: CC BY-NC 4.0 (eng_dh/oldbailey/licence.txt); LICENSE_TIERS OldBaileyOnline T2.
"""
from __future__ import annotations

import re

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'old_bailey'
COLLS = ['eng_dh/oldbailey']
RENASCOR_IDS = ['old-bailey-online']
LICENCE = 'CC-BY-NC-4.0'
TIER = 'T2'
DESCRIPTION = 'Old Bailey Sessions Papers and Ordinarys Accounts 1674-1700, persons and places as tagged'
OPTIONS = {}

_NAME_YEAR = re.compile(r'^(?:OA)?(\d{4})(\d{4})?')
_ROLES = {'defendantname': 'defendant', 'victimname': 'victim', 'jurorname': 'juror', 'judiciaryname': 'judge'}


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        recs = super().iter_files(ctx, include_dups)
        out = []
        for r in recs:
            m = _NAME_YEAR.match(r.stem)
            if m and int(m.group(1)) <= 1700 and r.name.lower().endswith('.xml'):
                out.append(r)
        return out

    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        m = _NAME_YEAR.match(rec.stem)
        if m:
            y = int(m.group(1))
            return C.DocDate(y, y, 'filename')
        return super().doc_date(root, rec, ctx)

    def scan(self, root, rec, ctx, counters):
        hits = C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context, doc_lang=C.doc_language(root),
                               counters=counters, nested_inner=True)
        insts = {e.get('inst') for e in root.iter() if C.ln(e.tag) == 'interp' and e.get('type') == 'surname' and e.get('inst')}
        out = []
        for h in hits:
            if h.kind == 'person' and h.name == 'persName' and not h.el.get('type') and h.el.get('id') not in insts \
                    and not any(C.ln(c.tag) == 'interp' and (c.get('type') or '') == 'surname' for c in h.el.iter()):
                continue
            out.append(h)
        return out

    def adjust(self, m, hit):
        t = (hit.el.get('type') or '').lower()
        if hit.kind == 'person' and t in _ROLES:
            m.role = _ROLES[t]
        ident = hit.el.get('id')
        if hit.kind == 'person' and ident and not m.local_key:
            m.local_key, m.in_register = f'{SLUG}:{ident}', False
        return m


EXTRACTOR = Extractor()
