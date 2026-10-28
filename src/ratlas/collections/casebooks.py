"""The Casebooks Project: medical consultation records of Simon Forman and Richard Napier, 1596-1634 (79,881 cases).

Licence: CC-BY-NC-SA-4.0 (``license.txt`` of the repository), T2 (LICENSE_TIERS lists Casebooks as T2, recipe only: not
held in a tarball). Raw: ``eng_dh/casebooks-data-master/{cases,master-lists}`` (the ``eng_dh/casebooks.zip`` members are
duplicates and are skipped).

What the editors tagged, and how it is read
  * persons are NOT tagged in the running text; they are the case-local ``particDesc/listPerson/person`` entries of each
    case header (patient, querent, relatives, practitioners; 292,354 entries) and the ``cb:consultation`` block in
    ``settingDesc`` that gives their role: ``cb:subject`` -> role ``subject`` (the patient), ``cb:querent`` ->
    ``querent``, ``cb:astrologer`` -> ``astrologer`` (Forman / Napier), ``cb:consultant`` -> ``consultant``, ``cb:object``
    -> ``object``; the nonstandard roles are documented here. v2: ONE mention (zone hdr, tag ``person@particDesc``) per
    (case, person): a person with several roles (Napier is both astrologer and consultant, the querent is usually the
    subject too) keeps the first of the priority order astrologer, consultant, subject, querent, object and the tag lists all of
    them (``person@particDesc+consultant``); this removes the 152,681 duplicated rows (45% of the 341,324 person rows);
    role ``mention`` for the others (relatives, third parties). Persons with an empty name stay (class human, surface '');
  * cross-case identity: ``master-lists/names-*.xml`` (PERSON####, ORG#, PERSONGRP#) list the same people with
    ``link/@sameAs="CASE6924#hr"`` to the case-local ids. A case-local person that is linked takes the master key
    ``casebooks:PERSON103`` (one entity for the whole corpus, 15,893 persons occur in several cases); an unlinked one gets
    a file-local entity ``casebooks:CASE30579#wc``. There are no authority ids (no GND / VIAF / Wikidata in the data);
  * places: ``rs[@type=address]`` (residence strings such as "of Stantone", 61,683) with ``@key="PLACE815"`` (59,755), and
    ``placeName`` (1,383). The PLACE#### keys have no register in the repository (no name, no coordinates): opaque keys
    ``casebooks:PLACE815`` (in_register False). ``rs sameAs="#address1"`` repeats the key of the referenced address;
  * ``rs[@type=date|time]`` is not an entity; the repository ``settlement`` / ``country`` in ``msIdentifier``
    (Oxford, UK; 160k rows) are not mentions and are dropped;
  * the transcription holds only the 'question' section of each consultation; the Latin and English layers of the
    Renascor catalogue entry are not separate files here;
  * dates: ``settingDesc/p/date/@when`` (header, 78,806 cases, 1587-1662); the 1,075 undated cases take the collection range
    1596-1634 (date_source ``collection``).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .. import common as C
from .. import helpers_batch0 as H
from ..model import Entity
from .generic_tei import GenericTEI

SLUG = 'casebooks'
COLLS = ['eng_dh/casebooks-data-master']
RENASCOR_IDS = ['casebooks']
LICENCE = 'CC-BY-NC-SA-4.0'
TIER = 'T2'
DESCRIPTION = 'Forman and Napier casebooks 1596-1634: patients, querents, practitioners and residence keys'
OPTIONS = {'default_years': (1596, 1634), 'opaque_keys': True, 'type_map': {'address': 'place'}}

_ROLE_TAGS = {'subject': 'subject', 'querent': 'querent', 'astrologer': 'astrologer', 'consultant': 'consultant',
              'object': 'object'}
_REG_TAGS = {'person': 'person', 'personGrp': 'person', 'org': 'org'}
_ROLE_PRIORITY = ['astrologer', 'consultant', 'subject', 'querent', 'object', 'mention']


def cb_name(pn) -> str:
    """Display name of a persName / orgName in reading order: prefix title, forename, name, surname, generation,
    other titles. The master lists write surname before forename; the case headers the other way round."""
    if pn is None:
        return ''
    if C.ln(pn.tag) == 'orgName':
        return C.surface_text(pn)
    parts = {'pre': [], 'fore': [], 'add': [], 'name': [], 'link': [], 'sur': [], 'gen': [], 'post': []}
    for c in pn:
        n = C.ln(c.tag)
        txt = C.surface_text(c)
        if not txt:
            continue
        if n == 'roleName':
            parts['pre' if c.get('type') in (None, 'prefix') else 'post'].append(txt)
        elif n == 'forename':
            parts['fore'].append(txt)
        elif n == 'addName':
            parts['add'].append(txt)
        elif n == 'name':
            parts['name'].append(txt)
        elif n == 'nameLink':
            parts['link'].append(txt)
        elif n == 'surname':
            parts['sur'].append(txt)
        elif n == 'genName':
            parts['gen'].append(txt)
    out = [x for k in ('pre', 'fore', 'add', 'name', 'link', 'sur', 'gen', 'post') for x in parts[k]]
    return C.norm(' '.join(out)) if out else C.surface_text(pn)


class Extractor(GenericTEI):
    # ---- master lists -------------------------------------------------------------------------------------------
    def is_register_file(self, rec, ctx):
        # only master-lists/names-*.xml are registers; a case with >= 5 header persons and no tagged text is not
        return '/master-lists/names-' in rec.path and not rec.path.endswith('names-master.xml')

    def parse_register_file(self, rec, ctx, root):
        ents = super().parse_register_file(rec, ctx, root)
        links, names = {}, {}
        for el in root.iter():
            if C.ln(el.tag) in _REG_TAGS and el.get(C.XML_ID):
                links[el.get(C.XML_ID)] = [lk.get('sameAs') for lk in el if C.ln(lk.tag) == 'link' and lk.get('sameAs')]
                names[el.get(C.XML_ID)] = next((c for c in el if C.ln(c.tag) in ('persName', 'orgName')), None)
        out = []
        for ent, aliases in ents:
            ident = ent.local_key.split(':', 1)[1]
            ent.name = cb_name(names.get(ident))          # '' for anonymous entries (no name in the list)
            ent.note_type = 'group' if ident.startswith('PERSONGRP') else ent.kind
            out.append((ent, list(aliases) + links.get(ident, [])))
        return out

    # ---- cases --------------------------------------------------------------------------------------------------
    def scan(self, root, rec, ctx, counters):
        hits = super().scan(root, rec, ctx, counters)
        keys = {e.get(C.XML_ID): e.get('key') for e in root.iter() if e.get(C.XML_ID) and e.get('key')}
        out = []
        for h in hits:
            if 'msIdentifier' in h.ancestors:
                continue                       # the repository (Oxford, UK) is not a mention
            sa = h.refs.get('sameAs')
            if sa and 'key' not in h.refs:
                k = keys.get(sa.lstrip('#'))
                h.refs = {'key': k} if k else {}
            out.append(h)
        return out

    def _person_hits(self, root, doc_id, ctx):
        roles = defaultdict(list)
        for el in root.iter():
            r = _ROLE_TAGS.get(C.ln(el.tag)) if C.ln(el.tag) in _ROLE_TAGS and el.get('ref') else None
            if r:
                for tok in el.get('ref').split():
                    ident = tok.lstrip('#')
                    if r not in roles[ident]:
                        roles[ident].append(r)
        local = C.Register()
        ents: list[Entity] = []
        hits: list[C.Hit] = []
        for el in root.iter():
            n = C.ln(el.tag)
            if n not in _REG_TAGS or not el.get(C.XML_ID):
                continue
            if not any(C.ln(a.tag) in ('listPerson', 'listOrg', 'particDesc') for a in el.iterancestors()):
                continue
            ident = el.get(C.XML_ID)
            kind = _REG_TAGS[n]
            pn = next((c for c in el if C.ln(c.tag) in ('persName', 'orgName')), None)
            name = cb_name(pn)
            mkey = ctx.shared.index.get(f'{doc_id}#{ident}')
            ent = ctx.shared.entities.get(mkey) if mkey else None
            if ent is None:
                ent = Entity(local_key=C.make_key(ctx.slug, ident, doc_id), coll_slug=ctx.slug, kind=kind,
                             name=name, source_path='', renascor_id=ctx.renascor_id, note_type='person',
                             licence=ctx.licence, tier=ctx.tier)
                ents.append(ent)
            local.add(ent, [ident])
            rl = roles.get(ident) or ['mention']
            role = min(rl, key=lambda r: _ROLE_PRIORITY.index(r) if r in _ROLE_PRIORITY else 99)
            others = [r for r in rl if r != role]
            hits.append(C.Hit(kind=kind, tag=f'{n}@particDesc' + ''.join('+' + r for r in others), name=n, surface=name,
                              zone='hdr', role=role, refs={'corresp': '#' + ident}, lang='en', context=None, el=el,
                              ancestors=('TEI', 'teiHeader', 'profileDesc', 'particDesc', 'listPerson')))
        return local, ents, hits

    def extract_root(self, root, rec, ctx):
        doc_id = C.doc_id_of(root, rec)
        local, ents, phits = self._person_hits(root, doc_id, ctx)
        for e in ents:
            e.source_path = rec.path
        hits = self.scan(root, rec, ctx, Counter()) + phits
        date = self.doc_date(root, rec, ctx)
        ms = C.build_mentions(root, rec, ctx, local=local, date=date, hits=hits, adjust=self.adjust)
        return ms, ents

    def doc_date(self, root, rec, ctx):
        # only the consultation date(s) of settingDesc: birth dates of persons (particDesc) must not widen the range
        rgs = []
        for sd in H.iter_local(root, 'settingDesc'):
            for el in sd.iter():
                if C.ln(el.tag) == 'date':
                    rg = C.date_range(el, 1400, 1750)
                    if rg:
                        rgs.append(rg)
        if rgs:
            return C.DocDate(min(r[0] for r in rgs), max(r[1] for r in rgs), 'header')
        return C.DocDate(*ctx.default_years, 'collection')

    def adjust(self, m, hit):
        if m.zone == 'hdr':
            m.context = None
        return m


EXTRACTOR = Extractor()
