"""Die Internuntiatur des Johann Rudolf Schmid zum Schwarzenhorn (1649), DiPKo / GAMS (Univ. Salzburg and Graz).

Licence: CC-BY-4.0, tier T1 (LICENSE_TIERS: Renascor_dipko-schwarzenhorn-1649; stated in each file header).
31 files: 25 letters (l.*), 2 instructions (i, ni), 3 reports (fr, gb, rb).

Mentions: <rs type=person|place ref="#id"> in the running text (rs type=event are not persons or places and are ignored);
correspDesc persName (GND url in @ref) and placeName (GeoNames url); the persName xml:id=johsch in titleStmt/author.
The register of the local ids (GAMS) is not in the harvest, so '#ferd3', '#ofen' are opaque keys
('dipko_schwarzenhorn_1649:ferd3', in_register False, no entities). The correspDesc persons / places carry GND /
GeoNames in auth but no local id (the two vocabularies cannot be linked inside the files).
The rs in correspContext (links to the previous / next letter) only repeat the sender and recipient: skipped.
Dating: correspAction/date@when (30 files); rb.xml (final report, no correspDesc) falls back to the first dateline
date, else the collection span 1648-1649.
"""
from __future__ import annotations

from collections import Counter

from .. import common as C
from .. import helpers_batch5 as H
from .generic_tei import GenericTEI

SLUG = 'dipko_schwarzenhorn_1649'
COLLS = ['renascor_2026-10/dipko-schwarzenhorn-1649']
RENASCOR_IDS = ['dipko-schwarzenhorn-1649']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Letters and reports of the 1649 internuntiature to Constantinople; local #ids, GND/GeoNames in headers'
OPTIONS = {'opaque_keys': True, 'default_years': (1648, 1649)}


class Extractor(GenericTEI):
    def scan(self, root, rec, ctx, counters):
        return C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context,
                               doc_lang=C.doc_language(root), counters=counters,
                               skip_ancestors=('correspContext',))

    def adjust(self, m, hit):
        xid = hit.el.get(C.XML_ID)
        if xid and not m.local_key and m.kind == 'person':
            m.local_key, m.in_register = f'{SLUG}:{xid}', False
        return H.drop_header_context(m)


EXTRACTOR = Extractor()
