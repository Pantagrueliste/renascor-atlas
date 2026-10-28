"""v2 classification of mentions: refined zone, class of the thing mentioned, default exclusion from the counts.

Audit points 7 and 8 (rapport_audit_v1.md, section 2).  Everything here is derived from what the editions say
themselves: the ``note/@type`` and ``@resp`` of the element that contains the tag, apparatus elements, the header
abstract (regest), ``msIdentifier`` (repository), the ``@type`` of an ``rs``/``name``, the ``note_type`` of the register
entry and the cast list.  No name lists decide who is a god: only the lists of *types* below.

Entry points
  * ``zone_class_of_hit(hit, opts)``   refined zone of a scanner hit (called by ``common.build_mentions``)
  * ``class_of_note_type(nt, kind)``   class of a register entry from its ``note_type``
  * ``class_of_tag(tag)``              class from the tag label (``name@myth``, ``rs@biblical``...)
  * ``is_placeholder(surface)``        S.l., Pl0, UNKNOWN, Anonimo-NN, N. N., ...
  * ``finalize(m, ent)``               fills zone_class / person_class / excluded / excluded_reason (called by the Writer)
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from . import generic_terms as GT
from .model import AUTH_KEYS, EXCLUDED_ROLES, EXCLUDED_ZONES, Entity, Mention

# ----------------------------------------------------------------------------------------------------------------------
# zones
# ----------------------------------------------------------------------------------------------------------------------
# note/@type values written by the editors (lower-case).  Authorial or original notes (marginalia, glosses) are text.
APPARATUS_NOTE_TYPES = {'crit_app', 'critapp', 'apparatus', 'app', 'textcrit', 'textkritik', 'variant', 'variants',
                        'lesart', 'lesarten', 'critical', 'critical_apparatus', 'textual', 'textual_note',
                        'variae_lectiones'}
FOOTNOTE_NOTE_TYPES = {'footnote', 'footnotes', 'fussnote', 'fußnote', 'fn', 'endnote', 'sachanmerkung', 'foot',
                       'editorial_footnote', 'note1', 'notel', 'bottom'}
COMMENTARY_NOTE_TYPES = {'editorial', 'editor', 'commentary', 'comment', 'kommentar', 'annotation', 'anmerkung',
                         'editorial_note', 'editorial-note', 'editors_note', 'editor_note', 'editorialnote',
                         'explanatory', 'explanation', 'gloss_editorial', 'erläuterung', 'erlaeuterung', 'sachkommentar',
                         'editorial_comment', 'notes', 'translation_note', 'historical', 'historical_note',
                         'introduction', 'intro', 'einleitung'}
ORIGINAL_NOTE_TYPES = {'marginal', 'margin', 'marginalia', 'authorial', 'author', 'original', 'gloss', 'inline',
                       'outer_margin', 'outer-margin', 'inner_margin', 'margin_left', 'margin_right', 'marginale',
                       'interlinear', 'sidenote', 'randnotiz', 'handwritten', 'manuscript'}
_EDITOR_RESP = re.compile(r'editor|herausgeb|\bed\b|\bed\.|redakt|bearbeit|annotat', re.I)
APPARATUS_ELEMENTS = {'rdg', 'witDetail', 'rdgGrp'}      # <lem> is the base reading, i.e. the text itself
REGEST_ELEMENTS = {'abstract', 'argument', 'summary'}      # header regest / summary of the edition


def _note_zone(el, note_zone: dict) -> Optional[str]:
    t = (el.get('type') or '').strip().lower()
    sub = (el.get('subtype') or '').strip().lower()
    for key in (t, sub):
        if key and key in note_zone:
            return note_zone[key]
    for key in (t, sub):
        if not key:
            continue
        if key in ORIGINAL_NOTE_TYPES:
            return 'text'
        if key in APPARATUS_NOTE_TYPES:
            return 'apparatus'
        if key in FOOTNOTE_NOTE_TYPES:
            return 'footnote'
        if key in COMMENTARY_NOTE_TYPES:
            return 'editorial_commentary'
    # a footnote placed at the foot of the page is NOT evidence of a modern editor (DTA transcriptions keep the printed original's footnotes):
    # only an edition that says so (OPTIONS['note_zone'] = {'@place:bottom': 'footnote'}) has its place-only notes excluded
    pl = (el.get('place') or '').strip().lower()
    if pl and ('@place:' + pl) in note_zone:
        return note_zone['@place:' + pl]
    resp = (el.get('resp') or '').strip()
    if resp and _EDITOR_RESP.search(resp):
        return 'footnote'
    return None


def zone_class_of_hit(h, opts: Optional[dict] = None) -> Optional[str]:
    """Refined zone of a scanner Hit, or None when the default applies (text / header).

    ``opts`` (``ctx.options``) may carry per-edition rules:
      ``note_zone``  {note type: zone}      overrides the generic reading of ``note/@type``; '@place:bottom' keys do the same for
                                            ``note/@place`` (place-only notes are text unless an edition says they are editorial)
      ``div_zone``   {div type: zone}       sub-divisions of an entry that are modern commentary (Hainhofer Stammbuch)
      ``zone_fn``    callable(hit) -> zone  last word
    """
    opts = opts or {}
    fn = opts.get('zone_fn')
    if fn is not None:
        z = fn(h)
        if z:
            return z
    if h.role == 'repository' or 'msIdentifier' in h.ancestors:
        return 'repository'
    note_zone = opts.get('note_zone') or {}
    div_zone = opts.get('div_zone') or {}
    hdr = h.zone == 'hdr'
    p = h.el.getparent() if h.el is not None else None
    while p is not None:
        tag = p.tag
        n = tag.rsplit('}', 1)[-1] if isinstance(tag, str) else None
        if n in APPARATUS_ELEMENTS:
            return 'apparatus'
        if n == 'note':
            z = _note_zone(p, note_zone)
            if z == 'text':
                return None
            if z:
                return z
        elif n in REGEST_ELEMENTS and hdr:
            return 'regest'
        elif n == 'div' and div_zone:
            z = div_zone.get((p.get('type') or '').strip())
            if z:
                return z
        p = p.getparent()
    return None


# ----------------------------------------------------------------------------------------------------------------------
# classes
# ----------------------------------------------------------------------------------------------------------------------
_MYTH = {'mythological', 'mythical', 'mythic', 'myth', 'legendary', 'deity', 'divine', 'god', 'goddess', 'nomen sacrum',
         'nomen_sacrum', 'demon', 'angel', 'spirit', 'personified', 'personification', 'allegorical', 'divinity',
         'mythologique', 'mythologie', 'mythisch', 'mythologisch', 'persm', 'person:deity', 'allegorie',
         'mythological_figure', 'literary_or_mythical', 'mytho'}
_CHAR = {'fictional', 'character', 'fiction', 'fictive', 'fictitious', 'dramatis_personae', 'dramatis', 'persona',
         'persf', 'personnage', 'cast', 'castitem', 'fictional_character', 'literary', 'imaginary'}
_BIBLE = {'biblical', 'bible', 'biblique', 'biblisch', 'religion', 'religious', 'prophet', 'apostle',
          'patriarch', 'christ', 'trinity', 'scriptural'}
_DYN = {'dynasty', 'dynastic', 'family', 'house', 'dynastie', 'geschlecht', 'lineage', 'noble_family', 'clan'}
_ROLE = {'role', 'office', 'title', 'function', 'fonction', 'amt', 'position', 'rolename', 'honorific'}
_CELEST = {'celestial', 'astronomical', 'astral', 'cosmological', 'planet', 'zodiac', 'star', 'constellation'}
_PEOPLE = {'nation', 'people', 'peoples', 'populus', 'ethnic', 'ethnonym', 'ethnicity', 'tribe', 'gens', 'nationality',
           'ethnonyme', 'volk', 'ethnie', 'ethnos'}


def _norm_type(t: Optional[str]) -> str:
    return (t or '').strip().lower().replace('-', '_')


def class_of_note_type(nt: Optional[str], kind: str) -> Optional[str]:
    """Class of a register entry from its ``note_type``. Persons default to 'human'; places and orgs to None."""
    t = _norm_type(nt)
    # 'person:deity' style values: look at the last part too
    parts = {t, t.split(':')[-1], t.split(':')[0]} if t else set()
    parts = {p for p in parts if p}
    if parts & _CELEST:
        return 'celestial'
    if parts & _PEOPLE:
        return 'people'
    if kind == 'person':
        if parts & _CHAR:
            return 'character'
        if parts & _MYTH:
            return 'mythological'
        if parts & _BIBLE:
            return 'biblical'
        if parts & _DYN:
            return 'dynasty'
        if parts & _ROLE:
            return 'role'
        return 'human'
    if kind == 'org':
        if parts & _DYN:
            return 'dynasty'
        return None
    return None


def class_of_tag(tag: Optional[str], kind: str) -> Optional[str]:
    """Class from the tag label: ``name@myth`` -> mythological, ``rs@biblical`` -> biblical, ``name@nation`` -> people."""
    if not tag or '@' not in tag:
        return None
    return class_of_note_type(tag.split('@', 1)[1], kind) if kind != 'person' else _person_tag_class(tag)


def _person_tag_class(tag: str) -> Optional[str]:
    t = _norm_type(tag.split('@', 1)[1])
    c = class_of_note_type(t, 'person')
    return None if c == 'human' else c


# ----------------------------------------------------------------------------------------------------------------------
# placeholders and empty surfaces
# ----------------------------------------------------------------------------------------------------------------------
_PLACEHOLDER = re.compile(
    r'''^(?:
        s\.?\s?l\.?                        # sine loco
      | s\.?\s?n\.?                        # sine nomine
      | n\.?\s?n\.?                        # nomen nescio
      | pl0+                               # Pl0 (empty place id)
      | unknown | unbekannt\w* (?:\s+\w+)? | ignot[oa] | inconnu[e]? | onbekend\w*
      | an[oó]nim[oa]s?[\s_\-]*\d*         # Anonimo-12, Anónimo 3
      | anonymus | anonymous | anonyme | anon\.?
      | auctores\s+varii
      | \?+ | -+ | \.+ | …+
    )$''', re.I | re.X)


def _has_letter(s: str) -> bool:
    return any(unicodedata.category(c).startswith('L') for c in s)


def is_placeholder(surface: Optional[str]) -> bool:
    s = (surface or '').strip()
    if not s:
        return False
    return bool(_PLACEHOLDER.match(s))


# ----------------------------------------------------------------------------------------------------------------------
# finalisation
# ----------------------------------------------------------------------------------------------------------------------
def entity_class(e: Entity) -> Optional[str]:
    return e.person_class or class_of_note_type(e.note_type, e.kind)


def _has_authority(m: Mention) -> bool:
    a = m.auth or {}
    return bool(any(a.get(k) for k in AUTH_KEYS) or a.get('other'))


def finalize(m: Mention, ent: Optional[Entity] = None) -> Mention:
    """Fill zone_class, person_class, excluded and excluded_reason of a mention (idempotent)."""
    # refined zone
    if not m.zone_class:
        if m.role == 'repository':
            m.zone_class = 'repository'
        else:
            m.zone_class = 'header' if m.zone == 'hdr' else 'text'
    # class of the thing mentioned
    if not m.person_class:
        c: Optional[str] = None
        if m.kind == 'person' and m.role == 'cast':
            c = 'character'
        if c is None and ent is not None and ent.kind == m.kind:
            ec = entity_class(ent)
            if ec and ec != 'human':
                c = ec
        if c is None:
            c = class_of_tag(m.tag, m.kind)
        if c is None and m.kind == 'person':
            c = 'human'
        m.person_class = c
    reasons = [r for r in (m.excluded_reason or '').split(';') if r]        # reasons set by the module (duplicate_letter...)
    if m.zone_class in EXCLUDED_ZONES:
        reasons.append(m.zone_class)
    if m.role in EXCLUDED_ROLES:
        reasons.append(m.role)
    if m.person_class == 'character':
        reasons.append('character')
    elif m.person_class == 'celestial':
        reasons.append('celestial')
    s = m.surface or ''
    if is_placeholder(s):
        reasons.append('placeholder')
    elif not _has_letter(s) and not m.ref_raw and not m.local_key:
        reasons.append('empty_surface')
    g = GT.generic_kind(m.kind, s)
    if not reasons and (g == 'pronoun' or (g == 'common_noun' and not _has_authority(m))):
        reasons.append(GT.REASON)             # v2.1: tagged pronouns and common nouns are not names (an authority id on a common noun keeps it)
    reasons = list(dict.fromkeys(reasons))
    m.excluded = bool(reasons)
    m.excluded_reason = ';'.join(reasons) or None
    return m
