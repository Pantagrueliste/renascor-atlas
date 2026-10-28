"""Ferdinand I., Familienkorrespondenz bis 1526 (vol. 1, Univ. Salzburg), 262 letter files A001-A261.

Tags: <persName ref="nognd" key="Surname, Forename, role"> (ref="nognd" = explicitly no GND; the key is the editor's
normalised name), <placeName ref="GeoNamesID" key="Wien"> (a bare numeric @ref is a GeoNames id), correspDesc with
sender / recipient / sender place / date. There is no register file.
Special handling:
  * bare numeric place @ref -> auth.geonames, local_key '<slug>:gn<id>' (opaque, in_register False);
  * persons: local_key '<slug>:<key>' from the descriptive @key (opaque, in_register False; header persons have
    no key and stay unresolved);
  * sender place of correspDesc has no id: it gets the GeoNames id of the text tag with the same name key when
    that mapping is unique in the volume (built in prepare());
  * the header <author> duplicates the sender (identical in 262/262 letters) and the biblFull author is the
    edition's principal (Ferdinand I.): both dropped; sender / addressee / sent_from come from correspAction;
  * <index><term> topical index terms are not typed entity tags and are not read;
  * v2 (issue 159): the correspondents of the correspDesc are bare first names with ref=nognd and no key ('Ferdinand' 260, 'Karl' 131,
    'Margareta' 79, 'Maria' 23, 'Ludwig II.'); the text never tags the principal persons under a key. They are tied by hand to
    five persons of the family: Ferdinand I (key 'ferdinand_i:hdr:ferdinand_i'), Charles V ('...:hdr:karl_v'), Maria of Hungary
    ('...:hdr:maria_von_ungarn'), and, for Margaret of Austria and Louis II, the keys the text uses ('Margarete von Österreich, ...',
    'Ludwig II., König von Böhmen und Ungarn'); editorial parentheticals are cut from the surface
    ('Karl (Instruktion für Henry de Hemricourt...)' -> 'Karl');
  * dating: correspAction date (letter); the 9 letters without a date fall back to the volume range 1514-1526
    ('collection').
Licence: CC-BY-3.0 (TEI header and LICENSE_TIERS), T1.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .. import common as C
from .generic_tei import GenericTEI

SLUG = 'ferdinand_i'
COLLS = ['originals_2026-10-15/oct-binary/oct__ferdinand-i-familienkorrespondenz']
RENASCOR_IDS = ['ferdinand-i-familienkorrespondenz']
LICENCE = 'CC-BY-3.0'
TIER = 'T1'
DESCRIPTION = 'Family correspondence of Ferdinand I 1514-1526: persons by name key, places by GeoNames id'
OPTIONS = {'default_years': (1514, 1526)}

_NUM = re.compile(r'^\d{3,9}$')
_PAREN = re.compile(r'\s*\([^)]*\)?\s*$')
# header first name -> key of the person (by hand; see the module docstring)
_HDR_PERSONS = {
    'ferdinand': 'hdr:ferdinand_i', 'karl': 'hdr:karl_v', 'maria': 'hdr:maria_von_ungarn',
    'margareta': 'Margarete von Österreich, Prinzessin von Asturien, Herzogin von Savoyen, Statthalterin der Niederlande',
    'margarete': 'Margarete von Österreich, Prinzessin von Asturien, Herzogin von Savoyen, Statthalterin der Niederlande',
    'ludwig ii.': 'Ludwig II., König von Böhmen und Ungarn',
}


def _name_key(el) -> str:
    return C.norm(el.get('key') or '')


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return False

    def prepare(self, ctx):
        # key -> geonames id for the places tagged in the transcriptions (unique mappings only)
        m = defaultdict(set)
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for el in root.iter():
                if C.ln(el.tag) == 'placeName' and _NUM.match((el.get('ref') or '').strip()) and el.get('key'):
                    m[C.norm(el.get('key'))].add(el.get('ref').strip())
        ctx.options['_gn_by_name'] = {k: next(iter(v)) for k, v in m.items() if len(v) == 1}
        return super().prepare(ctx)

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        out = []
        for h in hits:
            if h.zone == 'hdr' and ('biblFull' in h.ancestors or h.role == 'author'):
                continue
            out.append(h)
        return out

    def adjust(self, m, hit):
        el = hit.el
        ref = (el.get('ref') or '').strip()
        key = _name_key(el)
        if m.kind == 'place':
            gn = ref if _NUM.match(ref) else None
            if gn is None and m.zone == 'hdr':
                gn = ctx_map.get(C.norm(m.surface))
            if gn:
                m.auth['geonames'] = gn
                m.local_key, m.in_register = f'{SLUG}:gn{gn}', False
        elif m.kind == 'person' and key:
            m.local_key, m.in_register = f'{SLUG}:{key}', False
        elif m.kind == 'person' and m.zone == 'hdr' and m.role in ('sender', 'addressee'):
            clean = _PAREN.sub('', m.surface).strip()
            k = _HDR_PERSONS.get(clean.lower())
            if k:
                m.surface = clean
                m.local_key, m.in_register = f'{SLUG}:{k}', False
        return m

    def extract_root(self, root, rec, ctx):
        global ctx_map
        ctx_map = ctx.options.get('_gn_by_name', {})
        return super().extract_root(root, rec, ctx)


ctx_map: dict = {}
EXTRACTOR = Extractor()
