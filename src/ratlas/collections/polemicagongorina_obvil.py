"""Polemica Gongorina (OBVIL, Sorbonne Universite): 32 TEI texts of the polemic around Gongora's Soledades, 1613-1662.

Licence: LicenseRef-CC-BY-NC-ND-3.0-FR, tier T2 (LICENSE_TIERS: PolemicaGongorina_OBVIL; each header names
creativecommons.org/licenses/by-nc-nd/3.0/fr). ND and NC: used for internal work, flagged by the tier.
(1638_segunda-lecciones exists only as PDF besides its XML; the .pdf is not read.)

Mentions: <persName type=authority|polemist key="Gongora, Luis de"> and <name type=authority|polemista> in the text.
No places are tagged. The type separates the persons of the polemic ('polemist': Gongora, Lope, Jauregui...) from the cited
authorities (Aristoteles, Virgilio, Horacio...): the tag keeps it ('persName@authority', 'name@polemista'...), and
the register-less key (the editor's normalised name) becomes an opaque local_key
('polemicagongorina_obvil:Gongora, Luis de', in_register False). Name tags have no key. No authority ids, no entities.
The header <author key="..."> (except 'Anonimo') is emitted as a mention with tag 'author', role author.
Dating: profileDesc/creation/date@when (year of the text), 32 of 32.
v2: the 34 self-closing tags <persName type=authority key="Propercio" unit="quote"/> mark 'a quotation from X follows', not a
mention of X in the text: role ``quote_source`` (excluded from the default counts); the address forms 'vuesa / vuestra merced'
tagged as polemists are not names and are dropped. An unkeyed <name> ('don Luis' 212, 'Faría' 138, 'Góngora' 81, 'Lope' 24: 51% of the rows have no key) takes
the key of the keyed mention of the SAME document whose key, surface, surname or 'don' + forename equals it, when that is unique, else
of the whole collection (a form that has a single key in the 32 texts: 'Virgilio'); keys written 'Surname, Forename' are not otherwise normalised.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch0 as H0
from .. import helpers_batch5 as H
from .generic_tei import GenericTEI

SLUG = 'polemicagongorina_obvil'
COLLS = ['polemos/']
RENASCOR_IDS = ['polemicagongorina-obvil']
LICENCE = 'LicenseRef-CC-BY-NC-ND-3.0-FR'
TIER = 'T2'
DESCRIPTION = 'Gongorine polemic 1613-1662: persName/name typed authority vs polemist, name-string keys'
OPTIONS = {'note_zone': {'@place:bottom': 'footnote'},      # OBVIL / Teinte: the notes at the foot are the modern editors'
           'type_map': {'authority': 'person', 'polemista': 'person', 'polemist': 'person'},
           'filename_dates': True}

_ANON = {'anónimo', 'anonimo', 'anonymous', 'anon.'}
_ADDRESS = {'vuesa merced', 'vuestra merced', 'vuesa', 'vuestra', 'vuesamerced', 'vuestramerced', 'vuesastedes', 'vuestras mercedes'}


class Extractor(GenericTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.name.lower().endswith('.xml')]

    def scan(self, root, rec, ctx, counters):
        dl = C.doc_language(root)
        hits = C.scan_mentions(root, type_map=ctx.type_map, want_context=ctx.want_context, doc_lang=dl,
                               counters=counters)
        for h in H.author_ref_hits(root, dl, ctx.want_context):
            if h.surface.lower().strip('. ') not in _ANON:
                hits.append(h)
        return hits

    @staticmethod
    def _forms_of(root) -> dict:
        forms: dict = {}
        for el in root.iter():
            if C.ln(el.tag) in ('persName', 'name') and C.norm(el.get('key')) and C.norm(el.get('key')).lower() not in _ANON:
                key = C.norm(el.get('key'))
                sur, _, fore = key.partition(',')
                cands = {key, C.surface_text(el), sur, fore.split(' de ')[0].strip() and 'don ' + fore.split(' de ')[0].strip()}
                for c in cands:
                    f = H0.fold(c or '')
                    if f and len(f) > 3:
                        forms.setdefault(f, set()).add(key)
        return forms

    def prepare(self, ctx):
        allf: dict = {}
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is not None:
                for f, ks in self._forms_of(root).items():
                    allf.setdefault(f, set()).update(ks)
        ctx.options['_global_forms'] = {f: next(iter(k)) for f, k in allf.items() if len(k) == 1}
        return []

    def extract_root(self, root, rec, ctx):
        # surface forms of the keyed persons of this document -> key (only when unambiguous)
        forms = self._forms_of(root)
        self._doc_forms = {f: next(iter(k)) for f, k in forms.items() if len(k) == 1}
        self._global_forms = ctx.options.get('_global_forms', {})
        return super().extract_root(root, rec, ctx)

    def adjust(self, m, hit):
        if m.surface.lower().strip(' .,') in _ADDRESS:
            return None
        if not m.surface and (hit.el.get('unit') or '').lower() == 'quote':
            m.role = 'quote_source'
        t = (hit.el.get('type') or '').strip().lower()
        if t and m.tag == 'persName':
            m.tag = f'persName@{t}'
        key = C.norm(hit.el.get('key'))
        if key and m.kind == 'person' and key.lower() not in _ANON:
            m.local_key, m.in_register = f'{SLUG}:{key}', False
        elif not key and m.kind == 'person' and m.surface and m.role != 'quote_source':
            f = H0.fold(m.surface)
            k = getattr(self, '_doc_forms', {}).get(f) or getattr(self, '_global_forms', {}).get(f)
            if k:
                m.local_key, m.in_register = f'{SLUG}:{k}', False
        return m


EXTRACTOR = Extractor()
