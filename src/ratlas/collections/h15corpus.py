"""H15corpus (Historia15, Universidad de Sevilla): 402 Spanish letters and documents of the 15th-16th c., 1417-1600.

Files are not in the TEI namespace. The text is a lemmatised token stream (<tok pos="NP00000">): proper nouns there
come from automatic POS tagging, NOT from editors, so the text is not read. The editor-given entities are in the
header: profileDesc/correspDesc/correspAction[@type=sent|received] with
  <persName><name>Bernardo Abarca de Bolea</name><gender/><socecStatus/></persName> (sender / addressee) and
  <placeName type="origin|destination"><location>Italia, Milan</location><geo>45.466944, 9.19</geo></placeName>,
plus sourceDesc/history/origin/origPlace/placeName (place of writing, repeated).
Handling:
  * persName surface = the <name> child (not the gender / status labels); roles sender / addressee from correspAction;
  * places: correspAction placeName -> role sent_from (origin) / sent_to (destination), surface = <location> text;
    the origPlace placeName is dropped when a correspAction origin with text exists (same place), else sent_from;
    the region / country tags of origPlace and of msIdentifier are qualifiers and are dropped; the msIdentifier
    settlement (archive location) gets role 'repository'; 4 files repeat the whole header three times (ner- ids): only
    the first teiHeader is read;
  * coordinates: when the correspAction place has a <geo> (decimal 'lat, lon' or DMS N45°27'51.37" E9°11'22.24") an
    Entity is written ('h15corpus:<doc_id>#origin' / '#destination', coord_source 'tei:geo') and the mention resolves
    to it (in_register True). These entities are document-scoped (one per letter), identity across letters is a
    later stage. Places without geo (about 1/3) are mentions only;
  * v2 (issue 233): the same place name carries different <geo> in different letters (24 contradictory cases: Naples, Milan, Barcelona);
    ``prepare()`` collected every (name, geo) of the 402 headers and an entity took the MEDIAN latitude and longitude of its name when
    the name had several distinct values (coord_source 'tei:geo~median'). The msIdentifier settlement
    (Madrid 288, Bilbao 70...) is role/zone 'repository' and excluded from the default counts (43% of the place rows);
  * v2.4 (check of v2.3): the median of the contradictory geos of 'Italia, Milán' (Milan in six letters, a point near Riva del Garda in eight others) was a point that no letter
    carries (45.78 N 10.86 E, Trentino), 14 mentions and 6 letters wrong. Now: a name whose geos agree (all within ``GEO_AGREE_KM``) keeps the geo of its own letter ('tei:geo'); a
    name whose geos contradict each other gets NO coordinate from the TEI (the entry is not written, the mention is a plain name) and the geocoder decides by the name (the
    leaf 'Milán' of 'Italia, Milán' is Milan);
  * no authority ids anywhere; particDesc scribes (role scriptor) are not read;
  * dating: origDate@when under sourceDesc/history (header).
Licence: no statement found anywhere (LICENSE_TIERS: LicenseRef-unclear, T3): no context snippets are stored.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch1 as H
from ..model import Entity

SLUG = 'h15corpus'
COLLS = ['renascor_2026-10/h15corpus']
RENASCOR_IDS = ['h15corpus']
LICENCE = 'LicenseRef-unclear'
TIER = 'T3'
DESCRIPTION = 'Historia15 letters: sender, addressee, places of writing and destination with header coordinates'
OPTIONS = {}

_QUAL = {'region', 'country', 'settlement', 'district', 'bloc'}


def _child_text(el, name):
    c = C.child(el, name)
    return C.norm(''.join(c.itertext())) if c is not None else ''


GEO_AGREE_KM = 30.0


def contradictory(pts: dict) -> set:
    """Names ({name: [(lat, lon)]}) whose geos lie more than ``GEO_AGREE_KM`` apart in different letters."""
    import math
    out = set()
    for nm, ll in pts.items():
        u = sorted(set(ll))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                (a, b), (c, d) = u[i], u[j]
                x = math.sin(math.radians(c - a) / 2) ** 2 + math.cos(math.radians(a)) * math.cos(math.radians(c)) * math.sin(math.radians(d - b) / 2) ** 2
                if 2 * 6371.0 * math.asin(math.sqrt(min(1.0, x))) > GEO_AGREE_KM:
                    out.add(nm)
        # (a name with a handful of values is checked pairwise: at most a few hundred pairs)
    return out


class Extractor(H.BaseTEI):
    def iter_files(self, ctx, include_dups=False):
        return [r for r in super().iter_files(ctx, include_dups) if r.name.endswith('.xml')]

    def prepare(self, ctx):
        from collections import defaultdict
        pts = defaultdict(list)
        for rec in ctx.all_files:
            root = C.parse_xml(rec.read())
            if root is None:
                continue
            for el in root.iter():
                if C.ln(el.tag) != 'placeName' or el.get('type') not in ('origin', 'destination'):
                    continue
                geo = C.child(el, 'geo')
                ll = H.parse_coord(C.norm(''.join(geo.itertext()))) if geo is not None else None
                nm = (_child_text(el, 'location').strip(' ,') or C.norm(el.text or '')).casefold()
                if ll and nm:
                    pts[nm].append(ll)
        ctx.options['_geo_contra'] = contradictory(pts)
        return []

    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        has_origin = any(h.kind == 'place' and 'correspAction' in h.ancestors and h.el.get('type') == 'origin'
                         and _child_text(h.el, 'location') for h in hits)
        hdrs = [e for e in root.iter() if C.ln(e.tag) == 'teiHeader']
        hdr0 = hdrs[0] if hdrs else None
        out = []
        for h in hits:
            anc = h.ancestors
            if hdr0 is not None and len(hdrs) > 1 and h.zone == 'hdr' and \
                    not any(p is hdr0 for p in h.el.iterancestors()):
                continue                                  # a few files repeat the header (ner- copies)
            if h.name in _QUAL and ('origPlace' in anc or 'correspAction' in anc):
                continue
            if h.name in ('region', 'country') and 'msIdentifier' in anc:
                continue                                  # qualifiers of the archive settlement
            if h.name == 'persName' and 'correspAction' in anc:
                nm = _child_text(h.el, 'name')
                if nm:
                    h.surface = nm
            elif h.name == 'placeName' and 'correspAction' in anc:
                loc = _child_text(h.el, 'location')
                own = C.norm(h.el.text or '')
                h.surface = loc.strip(' ,') or own
                t = (h.el.get('type') or '').lower()
                h.role = 'sent_to' if t == 'destination' else 'sent_from' if t == 'origin' else h.role
            elif h.name == 'placeName' and 'origPlace' in anc:
                if has_origin:
                    continue
                h.role = 'sent_from'
            out.append(h)
        return out

    def extract_root(self, root, rec, ctx):
        mentions, _ = super().extract_root(root, rec, ctx)
        doc_id = C.doc_id_of(root, rec)
        ents = []
        seen = set()
        for el in root.iter():
            if C.ln(el.tag) != 'placeName' or el.get('type') not in ('origin', 'destination'):
                continue
            geo = C.child(el, 'geo')
            ll = H.parse_coord(C.norm(''.join(geo.itertext()))) if geo is not None else None
            if ll is None:
                continue
            ident = el.get('type')
            if ident in seen:
                continue
            loc = _child_text(el, 'location').strip(' ,')
            csrc = 'tei:geo'
            if (loc or C.norm(el.text or '')).casefold() in ctx.options.get('_geo_contra', ()):
                continue                                   # the TEI contradicts itself for this name: no register point (nor key), the geocoder reads the name
            seen.add(ident)
            ents.append(Entity(local_key=C.make_key(SLUG, ident, doc_id), coll_slug=SLUG, kind='place',
                               name=loc or C.norm(el.text or ''), source_path=rec.path, renascor_id=ctx.renascor_id,
                               lat=ll[0], lon=ll[1], coord_source=csrc, note_type='place',
                               licence=ctx.licence, tier=ctx.tier))
        for m in mentions:
            if m.kind == 'place' and m.role in ('sent_from', 'sent_to') and m.zone == 'hdr':
                ident = 'origin' if m.role == 'sent_from' else 'destination'
                if ident in seen and C.make_key(SLUG, ident, doc_id) not in (None,):
                    m.local_key, m.in_register = C.make_key(SLUG, ident, doc_id), True
        return mentions, ents


EXTRACTOR = Extractor()
