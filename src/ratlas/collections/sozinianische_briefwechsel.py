"""Die sozinianischen Briefwechsel (Johannes a Lasco Bibliothek / Mainz Academy), 2,134 letters 1545-1697.

Licence: CC-BY-4.0, tier T1 (LICENSE_TIERS: Renascor_sozinianische-briefwechsel; 136 printed versions are CC BY-SA 3.0,
not distinguishable per file here: the licence stated in every file header, CC BY 4.0, is used).

File layout (LETTER_MAIN_*.xml): a teiCorpus wrapper holding (1) a standOff with listPerson / listPlace of the persons
and places of that letter, (2) the cover-letter TEI (correspDesc with sender / addressee and places, a regest in
profileDesc/abstract), (3) the reference-witness TEI with the transcription.

Handling:
  * the ids of the standOff registers are edition-wide (Boulliau, Lubieniecki_Stanislaw, ed_tp5_2hp_vdb, Hamburg...) and
    repeated in every letter that uses them, so prepare() reads only the <standOff> slice of every file and builds ONE
    shared register: local_key = 'sozinianische_briefwechsel:<xml:id>' (entries merged across letters: variants,
    authority ids, dates);
  * mentions: <rs type=person|place corresp=ID> in the regest (zone 'hdr': profileDesc/abstract, kept because they are
    editor-tagged; filter zone == 'txt' to avoid counting the regest and the transcription twice) and in the
    transcription (zone 'txt'); correspDesc persName/placeName with @key (roles sender / addressee, sent_from / sent_to).
    rs type=term|source|letter|subjectIndexTerm are not persons or places and are ignored;
  * correspondent keys absent from every register (Sozzini_Fausto ...) are kept as opaque keys (in_register False);
  * note_type 'ancient' for register persons born 'v. Chr.' (Aristoteles, Cicero...; the register also holds biblical and
    mythological figures it does not mark);
  * authority ids (GND, GeoNames, Wikidata, VIAF, ISNI, Deutsche Biographie) come from the register idno; no
    coordinates in the files (GeoNames ids allow a later lookup, not done here);
  * dating: correspAction[sent]/date@when of the cover letter, searched in the whole teiCorpus (the generic header
    reading only looks at the wrapper header, which has no dates).
"""
from __future__ import annotations

import re
from collections import Counter

from .. import common as C
from .. import helpers_batch5 as H
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'sozinianische_briefwechsel'
COLLS = ['originals_2026-10-15/web-editions-apis/sozinianische-briefwechsel/']
RENASCOR_IDS = ['sozinianische-briefwechsel']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Socinian correspondence 1545-1697: standOff registers with GND/GeoNames, rs/@corresp mentions'
OPTIONS = {'opaque_keys': True}


def _standoff_slice(data: bytes) -> bytes:
    i = data.find(b'<standOff')
    j = data.find(b'</standOff>')
    return data[i:j + len('</standOff>')] if i >= 0 and j > i else b''


def _surname_forename(root) -> dict:
    """xml:id -> 'Surname, Forename' for the preferred persName (type reg) of every register person."""
    out = {}
    for el in root.iter():
        if C.ln(el.tag) != 'person' or not el.get(C.XML_ID):
            continue
        for pn in el:
            if C.ln(pn.tag) == 'persName' and pn.get('type') == 'reg':
                sur = C.norm(' '.join(C.surface_text(c) for c in pn if C.ln(c.tag) == 'surname'))
                fore = C.norm(' '.join(C.surface_text(c) for c in pn if C.ln(c.tag) == 'forename'))
                if sur and fore:
                    out[el.get(C.XML_ID)] = f'{sur}, {fore}'
                break
    return out


def _merge(dst: Entity, src: Entity):
    for v in [src.name] + list(src.variants):
        if v and v != dst.name and v not in dst.variants:
            dst.variants.append(v)
    dst.auth = C.merge_auth(dst.auth, src.auth)
    for a in ('birth', 'death', 'note_type'):
        if not getattr(dst, a) and getattr(src, a):
            setattr(dst, a, getattr(src, a))


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        merged: dict[str, Entity] = {}
        aliases: dict[str, list] = {}
        for rec in ctx.all_files:
            sl = _standoff_slice(rec.read())
            if not sl:
                continue
            root = C.parse_xml(sl)
            if root is None:
                continue
            sn = _surname_forename(root)
            for ent, al in C.parse_register(root, ctx, rec.path, doc_id=None)[0]:
                if ent.kind == 'person' and al and al[0] in sn:
                    if sn[al[0]] != ent.name:
                        ent.variants = [v for v in ent.variants if v != sn[al[0]]]
                    ent.name = sn[al[0]]
                if ent.kind == 'person' and re.search(r'v\. ?Chr', ent.birth or ''):
                    ent.note_type = 'ancient'     # the register mixes in antique authors and figures (born BC)
                if ent.local_key in merged:
                    _merge(merged[ent.local_key], ent)
                else:
                    merged[ent.local_key] = ent
                    aliases[ent.local_key] = al
        for k, ent in merged.items():
            ctx.shared.add(ent, aliases[k])
        return list(merged.values())

    def doc_date(self, root, rec, ctx):
        sent, other = [], []
        for ca in root.iter():
            if C.ln(ca.tag) != 'correspAction':
                continue
            for d in ca.iter():
                if C.ln(d.tag) == 'date':
                    rg = C.date_range(d)
                    if rg:
                        ((sent if (ca.get('type') or '').startswith('sen') else other)).append(rg)
        rgs = sent or other
        if rgs:
            return C.DocDate(min(r[0] for r in rgs), max(r[1] for r in rgs), 'letter')
        return super().doc_date(root, rec, ctx)

    def adjust(self, m, hit):
        return H.drop_header_context(m)

    def extract_root(self, root, rec, ctx):
        hits = self.scan(root, rec, ctx, Counter())
        date = self.doc_date(root, rec, ctx)
        return C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust), []


EXTRACTOR = Extractor()
