"""Briefverkehr der Stadt St. Gallen 1400-1650: Digitale Missivenedition (Stadtarchiv St. Gallen), 760 letters.

persName / orgName / placeName in the running text (and correspDesc sender / recipient / places) carry @ref with
edition ids: stadtasg-actors-N (persons and organisations in one space), stadtasg-places-N. No register is in the
harvest, so they are opaque keys 'st_galler_missiven:stadtasg-actors-N' (in_register False); no authority ids and
no coordinates anywhere in the files.

Roles: correspAction sender / addressee / sent_from / sent_to (generic); in the letter text the editors mark the
issuer and the recipient with @role: role="issuer" -> 'signer', role="recipient" -> 'addressee'.
Dating: correspAction[sent]/date (ISO, all 760 letters); 218 letters are before 1450 and 80 after 1640 or so;
those with a year outside 1450-1700 stay with in_period False.
v2 (issue 186): there is still no register file, but the letter headers (correspDesc) carry the NORMALISED modern form of each id
('Rüti Kloster', 'Hall in Tirol', 'Konstanz') while the text keeps the spellings of the scribe. ``prepare()`` builds a local register
from the forms: the entity name is the most frequent header surface of the id (else the most frequent text surface), the text
spellings are its variants, the kind is the majority kind of the tags; the key is unchanged ('st_galler_missiven:stadtasg-actors-N').
The 8 sub-letters (00401-01/-02, 00410-01/-02, 00485-01/-02, 00604-01/-02) listed by the site are not in the harvest (the harvest
script's id pattern accepted digits only) and were not re-fetched.
Licence: CC-BY-NC-SA-4.0 (LICENSE_TIERS, entry), T2.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .. import common as C
from .. import helpers_batch1 as H
from ..model import Entity

SLUG = 'st_galler_missiven'
COLLS = ['renascor_2026-10/st-galler-missiven']
RENASCOR_IDS = ['st-galler-missiven']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Letters of the city of St. Gallen 1417-1651: tagged actors and places with edition ids'
OPTIONS = {'opaque_keys': True}


_ID = re.compile(r'(stadtasg-(?:actors|places)-\d+)')


class Extractor(H.BaseTEI):
    def prepare(self, ctx):
        hdr, txt = defaultdict(Counter), defaultdict(Counter)
        kinds: dict = defaultdict(Counter)
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for h in C.scan_mentions(root, type_map=ctx.type_map, skip_ancestors=('msIdentifier',)):
                m = _ID.search(h.refs.get('ref') or '')
                if not m or not h.surface:
                    continue
                (hdr if h.zone == 'hdr' else txt)[m.group(1)][h.surface] += 1
                kinds[m.group(1)][h.kind] += 1
        out = []
        for ident in sorted(kinds):
            forms = hdr.get(ident) or txt.get(ident) or Counter()
            if not forms:
                continue
            name = forms.most_common(1)[0][0]
            others = Counter()
            others.update(txt.get(ident, {}))
            others.update(hdr.get(ident, {}))
            variants = [v for v, _ in others.most_common(9) if v != name][:8]
            kind = kinds[ident].most_common(1)[0][0]
            ent = Entity(local_key=C.make_key(ctx.slug, ident), coll_slug=ctx.slug, kind=kind, name=name, variants=variants,
                         source_path='(header forms)', renascor_id=ctx.renascor_id,
                         note_type='from_header_forms' if hdr.get(ident) else 'from_text_forms',
                         licence=ctx.licence, tier=ctx.tier)
            ctx.shared.add(ent, [ident])
            out.append(ent)
        return out

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is None:
            return None
        r = (hit.el.get('role') or '').strip().lower()
        if m.zone == 'txt' and m.kind != 'place':
            if r == 'issuer':
                m.role = 'signer'
            elif r == 'recipient':
                m.role = 'addressee'
        return m


EXTRACTOR = Extractor()
