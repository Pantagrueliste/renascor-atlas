"""Digital scholarly edition of Angelo Poliziano's marginal notes to the Epistle of Sappho to Phaon (EVT edition, 1 file).

Tagged: <persName ref="#Q7198"> (231 tags, about 92 with a ref) and <placeName ref="#Q190138"> (8) in the note
transcriptions; register in teiHeader/sourceDesc: listPerson (72 persons) and listPlace (3). Register ids are
Wikidata Q numbers for most entries (xml:id="Q7198"; the Wikidata link sits in persName/ref@target, which
common.parse_register does not read): this module reads it, and the xml:id when it has the Q shape, into auth.wikidata.
A few refs are bare ('Q868') and are matched to the register by their Wikidata id. Entries without a Q id are the
Sapphic names of the poem (Gyrinna, Atthis...) and characters of Lucian's dialogue (note_type 'fictional' when the
entry says 'personaggio'). All persons are classical figures; no coordinates (the three places have Wikidata ids only).
Dating: creation/date 1477-1481 (header).
Licence: CC BY-NC-SA 4.0 (TEI availability); LICENSE_TIERS T2.
"""
from __future__ import annotations

import re
from collections import Counter

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'poliziano_sappho'
COLLS = ['originals_2026-10-15/oct-binary/oct__digital-scholarly-edition-of-angelo-poliziano-s-marginal-notes-to-the-epistle-of-sappho-to-phaon']
RENASCOR_IDS = ['digital-scholarly-edition-of-angelo-poliziano-s-marginal-notes-to-the-epistle-of-sappho-to-phaon']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Poliziano marginalia to Ovid, Sappho to Phaon: classical persons with Wikidata ids'
OPTIONS = {}

_Q = re.compile(r'^Q\d+$')


class Extractor(GenericTEI):
    def adjust(self, m, hit):
        # holding library of the witness (msIdentifier/settlement) and contributors of the digitised copies listed in
        # the bibliography (respStmt/persName 'Robarts - University of Toronto') are not mentions
        if 'msIdentifier' in hit.ancestors or 'respStmt' in hit.ancestors:
            return None
        return m

    def extract_root(self, root, rec, ctx):
        doc_id = C.doc_id_of(root, rec)
        local = C.Register()
        ents = []
        for ent, aliases in C.parse_register(root, ctx, rec.path, doc_id=doc_id)[0]:
            ident = ent.local_key.rsplit('#', 1)[-1]
            els = [e for e in root.iter() if e.get(C.XML_ID) == ident]
            el = els[0] if els else None
            toks = []
            if _Q.match(ident):
                toks.append(ident)
            if el is not None:
                toks += [d.get('target') for d in el.iter() if C.ln(d.tag) in ('ref', 'ptr') and d.get('target')]
                if ent.kind == 'person' and 'personaggio' in C.surface_text(el):
                    ent.note_type = ent.note_type or 'fictional'
            C.parse_auth(toks, into=ent.auth)
            if ent.auth.get('wikidata'):
                aliases = list(aliases) + [ent.auth['wikidata']]
            local.add(ent, aliases)
            ents.append(ent)
        hits = self.scan(root, rec, ctx, Counter())
        date = self.doc_date(root, rec, ctx)
        ms = C.build_mentions(root, rec, ctx, local=local, date=date, hits=hits, adjust=self.adjust)
        for m in ms:                      # bare ref="Q868": the id is Wikidata's and also the register id
            if m.local_key is None and m.auth.get('wikidata'):
                e = local.lookup(m.auth['wikidata'])
                if e is not None and e.kind == m.kind:
                    m.local_key, m.in_register = e.local_key, True
                    m.auth = C.merge_auth(m.auth, e.auth)
        return ms, ents


EXTRACTOR = Extractor()
