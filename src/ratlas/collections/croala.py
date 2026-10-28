"""CroALa - Croatiae auctores Latini (Zagreb), 573 TEI texts (700-2000) of authors from Croatia; folder croala/txts/.

Entity tagging is sparse in the running text (persName / placeName / orgName / rs / typed name, 4,127 tags in 128
files) but the profile header is rich: per file the author (with a Wikidata url), the correspondents of a letter
(particDesc/person role sender|recipient) and the places of writing and destination (settingDesc/place
ana origin|destination). There is no register file.

Handling:
  * header author: the <persName> inside titleStmt/author gets the @ref of the <author> (Wikidata url), role 'author';
  * particDesc persons are mentions (not register entries): role 'sender', 'addressee' (role recipient), else
    'mention'; settingDesc places: role 'sent_from' (origin) / 'sent_to' (destination); the unreferenced
    creation/placeName is dropped when an origin place exists (it repeats it), else it is 'sent_from';
  * refs: Wikidata / VIAF urls and ids as usual; a BARE NUMBER on a place is a GeoNames id (project convention,
    Zagreb = 3186886 -> auth.geonames); a bare number on a person is *probably* a VIAF id but the data does not say,
    so it is stored as 'num:<id>' in auth.other and never in auth.viaf; project ids (vranc01, brod01) and '#slug' refs
    (Flacius letters: #matthias_flacius) are opaque keys 'croala:<id>' (in_register False). No coordinates;
  * respStmt / editor / publicationStmt / revisionDesc people (modern editors) are never read;
  * the two templates and croala-wordcounts.xml are not read. Untyped <name> (1,308, of which 918 in one poem) are
    skipped because their kind is unknown.
  * dating: profileDesc/creation/date (all files). Medieval and post-1700 texts are kept with in_period False.
v2 (issue 223): the variant names of one author with the same @ref give one 'author' row each (99 documents have 2-5, 54 extra rows with the
same Wikidata id: Ciriaco d'Ancona, Kyriacus Anconitanus, Ceriaco Pizzicolli): they are collapsed to ONE author row per document and
reference. 'Anonymus' and 'Auctores varii' are placeholders (excluded from the default counts). Continents get no coordinates here
(geocoding is another stage).
Licence: CC-BY-4.0 (LICENSE_TIERS), T1.
"""
from __future__ import annotations


from .. import common as C
from .. import helpers_batch1 as H

SLUG = 'croala'
COLLS = ['croala']
RENASCOR_IDS = ['croala']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'Croatian Latin authors: header authors, correspondents and places (Wikidata/VIAF) plus sparse text tags'
OPTIONS = {'opaque_keys': True}

_PLACEHOLDERS = {'unknown', 'ignotus', 'ignota', 'ignotum', 'nepoznato', 'n/a', '?'}
_PARTIC_ROLE = {'sender': 'sender', 'recipient': 'addressee', 'addressee': 'addressee', 'author': 'author'}


class Extractor(H.BaseTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.path.startswith('croala/txts/')]

    def is_register_file(self, rec, ctx):
        return False

    def doc_date(self, root, rec, ctx):
        """profileDesc/creation/date (the date of writing); the generic header rule would also pick up the author's
        life dates (<author><date>1504-1573</date>)."""
        rgs = []
        for el in root.iter():
            if C.ln(el.tag) == 'creation':
                for d in el.iter():
                    if C.ln(d.tag) == 'date':
                        rg = C.date_range(d)
                        if rg:
                            rgs.append(rg)
        if rgs:
            return C.DocDate(min(r[0] for r in rgs), max(r[1] for r in rgs), 'header')
        return super().doc_date(root, rec, ctx)

    def extract_root(self, root, rec, ctx):
        doc_id = C.doc_id_of(root, rec)
        hits = self.scan(root, rec, ctx, None)
        date = self.doc_date(root, rec, ctx)
        ms = C.build_mentions(root, rec, ctx, local=None, date=date, hits=hits, adjust=self.adjust)
        seen, out = set(), []
        for m in ms:
            if m.role == 'author' and m.kind == 'person':
                k = (m.ref_raw or m.surface)
                if k in seen:
                    continue                      # variant names of the same author (same @ref)
                seen.add(k)
            out.append(m)
        return out, []

    def scan(self, root, rec, ctx, counters):
        from collections import Counter
        hits = super().scan(root, rec, ctx, counters or Counter())
        # author: the ref sits on <author>, the name in the child persName
        for h in hits:
            if h.role == 'author' and h.kind == 'person' and not h.refs:
                par = h.el.getparent()
                while par is not None and C.ln(par.tag) != 'author':
                    par = par.getparent()
                if par is not None:
                    h.refs = H.refs_of(par)
        extra = []
        origin = False
        for el in root.iter():
            n = C.ln(el.tag)
            if n == 'person' and 'particDesc' in H.ancestors_of(el):
                pn = C.child(el, 'persName')
                if pn is None or not C.surface_text(pn):
                    continue
                role = _PARTIC_ROLE.get((el.get('role') or '').strip().lower(), 'mention')
                extra.append(H.make_hit(pn, 'person', 'persName', role, refs=H.refs_of(pn, el)))
            elif n == 'place' and 'settingDesc' in H.ancestors_of(el):
                pl = C.child(el, 'placeName')
                if pl is None or not C.surface_text(pl):
                    continue
                ana = (el.get('ana') or '').lower()
                role = 'sent_from' if 'origin' in ana else 'sent_to' if 'destination' in ana else 'mention'
                origin = origin or role == 'sent_from'
                extra.append(H.make_hit(pl, 'place', 'placeName', role, refs=H.refs_of(pl, el)))
        out = []
        extra = [h for h in extra if h.surface.strip().lower() not in _PLACEHOLDERS]
        for h in hits:
            if h.name == 'placeName' and h.zone == 'hdr' and 'creation' in h.ancestors:
                if origin:
                    continue
                h.role = 'sent_from'
            out.append(h)
        return out + extra

    def adjust(self, m, hit):
        m = super().adjust(m, hit)
        if m is None:
            return None
        if m.local_key and m.in_register is False and len(m.local_key.split(':', 1)[1]) < 2:
            m.local_key, m.in_register = None, None      # '#d', '#s': not usable ids
        for tok in (m.ref_raw or '').split():
            sch, ident = C.classify_token(tok)
            if sch == 'local':
                if H.bare_number_as_authority(m, ident, m.kind):
                    continue
                if m.local_key is None and len(ident) > 1:
                    H.set_opaque(m, SLUG, ident)
        return m


EXTRACTOR = Extractor()
