"""FreEM copy of the BVH (Bibliotheques Virtuelles Humanistes) texts: FreEMmax_OA/0_source/BVH, 53 non-duplicate files.

48 of the 101 files are byte-identical to the tei-fr BVH folder (flagged dup by the inventory, not read here); the 53
others are 47 FRAD archival acts (Arrets du Parlement de Bordeaux 1562-66, Archives de la Gironde) and 6 other BVH texts
(Montaigne Essais, Songe de Poliphile, Theatre des divers cerveaux...). Renascor entries: freem, epistemon.

Tags: persName 2,907, placeName ~1,350, orgName ~200 in the text, without @ref; 5,217 UNTYPED <name> (4 files:
tables of cited authors and names in Rabelais-like prose), whose kind (person, place, work...) is not given and cannot
be guessed: they are skipped, except the few that point (key / ref / corresp) to a person / place / org declared in the
file's own listPerson / listPlace / listOrg: those are read with the kind of the register entry (tag 'name').
Header: sourceDesc biblStruct author persName -> role 'author'; respStmt people (editors) are never read.
File-local registers are resolved as 'freem_bvh:<doc_id>#<id>'. No authority ids and no coordinates.
Dating: imprint / first date of the header; FRAD acts all 1562-1566. v2 (issue 120): only the original's imprint / msDesc history is read
(the same rule as epistemon_bvh: B372615206_4023_tei.xml, printed 1600, got 1600-1996 because a 1996 bibliographic reference in
sourceDesc was read as the end of the range).
Licence: all 53 headers state CC BY-NC-SA 4.0 (BVH notice; LICENSE_TIERS BVH_Epistemon_Print), T2. (The FreEM label
'cc-by-nc-nd' that applies to other FreEM headers is not found in these files.)
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch1 as H

SLUG = 'freem_bvh'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/FreEMmax_OA/0_source/BVH']
RENASCOR_IDS = ['freem', 'epistemon']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'BVH texts as copied in FreEM (Montaigne, Poliphile, 47 Bordeaux legal acts): tagged persons, places, orgs'
OPTIONS = {}

_LIST_KIND = {'person': 'person', 'place': 'place', 'org': 'org'}


class Extractor(H.BaseTEI):
    def doc_date(self, root, rec, ctx):
        from .epistemon_bvh import Extractor as _Epistemon
        return _Epistemon.doc_date(self, root, rec, ctx)

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        kinds = {}
        for el in root.iter():
            k = _LIST_KIND.get(C.ln(el.tag))
            xid = el.get(C.XML_ID)
            if k and xid:
                kinds[xid] = k
        if not kinds:
            return hits
        seen = {id(h.el) for h in hits}
        for el in root.iter():
            if C.ln(el.tag) != 'name' or id(el) in seen or el.get('type'):
                continue
            anc = H.ancestors_of(el)
            if 'teiHeader' in anc:
                continue
            refs = H.refs_of(el)
            kind = None
            for v in refs.values():
                for tok in v.split():
                    sch, ident = C.classify_token(tok)
                    if sch == 'local' and ident in kinds:
                        kind = kinds[ident]
            if kind:
                hits.append(H.make_hit(el, kind, 'name', 'mention', zone='txt', refs=refs))
        return hits


EXTRACTOR = Extractor()
