"""CORPUS17 - E-ditiones, 17th c. French texts (drama, letters, novels), github e-ditiones/CORPUS17.

The folder holds, per edition EDI_NNNN, a full file EDI_NNNN.xml (header + text) and derived pieces of the same text:
EDI_NNNN-1/-2/-3.xml (text levels, -3 is the tokenised version with <w>) and EDI_NNNN_header.xml. Only the full file is
read: the pieces repeat the same persName / placeName tags (the inventory counts 1,006 persName and 262 placeName
over all files, about twice the real number).

Tags: <persName>/<placeName> in the text, without @ref (a few persName carry ref="isni:..." -> auth.other).
Because the text tagging is thin, the bibliographic header is also read (editor-given authority ids):
titleStmt/author (ref isni:...) -> person, role 'author'; sourceDesc publisher (ref isni) -> person, role 'publisher';
sourceDesc pubPlace (ref geonames:2988507) -> place, role 'pubplace' (both roles are non-standard). Empty
publisher/pubPlace tags are ignored. Editors (ORCID) are modern and are not read.
v2 (issue 150): 489 of the 504 person mentions are dramatis personae of plays (Petit Jean, Isabelle, Chicanneau...) or characters of a
tragedy. In a play (a file with <sp> speeches or a <castList>) every body persName is person_class 'character' (excluded from the
default counts); the author and publisher of the header stay human. The unit of a play is the file, so a real person named in a
play is not told apart from the cast (no register, no ids to decide).
No register, no coordinates. Dating: bibl/date@when of the source print (header, e.g. 1639).
Licence: CC-BY-SA-2.0 (repo README / LICENSE_TIERS); the TEI headers say CC BY 4.0 (less strict); T1.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_batch1 as H

SLUG = 'corpus17_french'
COLLS = ['corpus17']
RENASCOR_IDS = ['corpus17-french']
LICENCE = 'CC-BY-SA-2.0'
TIER = 'T1'
DESCRIPTION = '17th-century French texts (E-ditiones): tagged persons/places plus author, publisher and place of publication'
OPTIONS = {}

_FULL = re.compile(r'^EDI_\d+\.xml$')


class Extractor(H.BaseTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if _FULL.match(r.name)]

    def extract_root(self, root, rec, ctx):
        self._drama = any(C.ln(e.tag) in ('sp', 'castList') for e in root.iter() if isinstance(e.tag, str))
        return super().extract_root(root, rec, ctx)

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is not None and m.kind == 'person' and m.zone == 'txt' and getattr(self, '_drama', False):
            m.person_class = 'character'
        return m

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        for el in root.iter():
            n = C.ln(el.tag)
            if n not in ('author', 'publisher', 'pubPlace') or not el.get('ref') or not C.surface_text(el):
                continue
            anc = H.ancestors_of(el)
            if 'teiHeader' not in anc:
                continue
            if n == 'author' and 'titleStmt' in anc:
                hits.append(H.make_hit(el, 'person', 'author', 'author'))
            elif n == 'publisher' and 'sourceDesc' in anc:
                hits.append(H.make_hit(el, 'person', 'publisher', 'publisher'))
            elif n == 'pubPlace' and 'sourceDesc' in anc:
                hits.append(H.make_hit(el, 'place', 'pubPlace', 'pubplace'))
        return hits


EXTRACTOR = Extractor()
