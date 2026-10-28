"""Row models and pyarrow schemas of the Renascor Atlas (the shared contract).

Two tables per collection slug:

* ``mentions.parquet``  one row per editor-tagged mention (persName, placeName, rs@type=..., ...)
* ``entities.parquet``  one row per register entry (edition-scoped)

Do not change field names or types here: every extraction module and the later merge stage depend on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Optional

import pyarrow as pa

PERIOD = (1450, 1700)

KINDS = ('person', 'place', 'org')
ZONES = ('txt', 'hdr')   # structural zone as seen by the extractors; the published ``part`` column (``zone`` is refined, v2)
# v2 refined zone (published as ``zone``): where the editor's tag sits
ZONE_CLASSES = ('text', 'header', 'editorial_commentary', 'apparatus', 'footnote', 'regest', 'repository')
# mentions of these zones are not the text of the period: excluded from the default counts
EXCLUDED_ZONES = ('editorial_commentary', 'apparatus', 'footnote', 'regest', 'repository')
# roles whose rows are not mentions of the text (record heading locality of REED, empty quotation markers): excluded by default
EXCLUDED_ROLES = ('provenance', 'quote_source')
# v2 class of the thing mentioned (published as ``person_class``; null = ordinary place or organisation)
PERSON_CLASSES = ('human', 'character', 'mythological', 'biblical', 'dynasty', 'role', 'metonym', 'celestial', 'people')
DATE_SOURCES = ('header', 'filename', 'collection', 'letter', 'text', 'title', 'imprint_century', 'author_lifespan')  # 'text' = date from the running text (diaries, itineraries)
WIDE_RANGE = 20          # a document date range wider than this (years) has no single year (doc_year is null)
# Standard roles. Modules may add others (document them in the module docstring).
ROLES = ('mention', 'author', 'sender', 'addressee', 'sent_from', 'sent_to', 'signer', 'subject', 'cast',
         # added at integration (used by several modules, documented in their docstrings):
         'repository', 'scribe', 'artist', 'birth_place', 'death_place', 'written_at', 'provenance', 'quote_source')
TIERS = ('T1', 'T2', 'T3')
AUTH_KEYS = ('gnd', 'viaf', 'wikidata', 'geonames', 'tgn', 'ulan', 'pleiades', 'hls')  # + 'other'

# --- auth ---------------------------------------------------------------------------------------------------------
AUTH_TYPE = pa.struct([
    ('gnd', pa.string()), ('viaf', pa.string()), ('wikidata', pa.string()), ('geonames', pa.string()),
    ('tgn', pa.string()), ('ulan', pa.string()), ('pleiades', pa.string()), ('hls', pa.string()),
    ('other', pa.list_(pa.string())),   # 'scheme:id' strings, e.g. 'isni:0000000121251078', 'loc:n79021164'
])


def empty_auth() -> dict:
    return {k: None for k in AUTH_KEYS} | {'other': []}


def auth_to_arrow(a: Optional[dict]) -> dict:
    out = empty_auth()
    if a:
        for k in AUTH_KEYS:
            out[k] = a.get(k) or None
        out['other'] = list(a.get('other') or [])
    return out


def auth_is_empty(a: Optional[dict]) -> bool:
    return not a or not (any(a.get(k) for k in AUTH_KEYS) or a.get('other'))


# --- Mention ------------------------------------------------------------------------------------------------------
@dataclass
class Mention:
    coll_slug: str
    source_path: str                  # relative to RQ/data/raw, '!' separates archive and member
    doc_id: str                       # root @xml:id, else file stem (without extension)
    kind: str                         # 'person' | 'place' | 'org'
    tag: str                          # 'persName', 'placeName', 'rs@place', 'name@person', ...
    surface: str                      # whitespace-normalised text of the tag
    renascor_id: Optional[str] = None  # catalogue id(s) of the edition; several joined by ';'
    doc_year: Optional[int] = None    # best single year (exact year, else midpoint of the range)
    doc_year_from: Optional[int] = None
    doc_year_to: Optional[int] = None
    date_source: Optional[str] = None  # 'header' | 'filename' | 'collection' | 'letter' | 'text' (None if undated)
    in_period: Optional[bool] = None   # [from,to] overlaps 1450-1700; None = undated
    role: str = 'mention'
    zone: str = 'txt'                 # 'txt' | 'hdr'
    ref_raw: Optional[str] = None     # raw @ref/@key/@corresp/@sameAs value(s), space-joined
    local_key: Optional[str] = None   # '<slug>:<id>' once resolved against the register (else None)
    in_register: Optional[bool] = None  # True if local_key points to an Entity present in entities.parquet
    auth: dict = field(default_factory=empty_auth)  # ids on the tag itself + ids of the resolved Entity
    context: Optional[str] = None     # <=120 chars; ONLY for tier T1/T2
    lang: Optional[str] = None        # nearest xml:lang (or document language)
    licence: Optional[str] = None
    tier: Optional[str] = None
    # --- v2 -------------------------------------------------------------------------------------------------------
    zone_class: Optional[str] = None   # refined zone, see ZONE_CLASSES (None -> derived by classify.finalize)
    person_class: Optional[str] = None  # see PERSON_CLASSES (None -> derived by classify.finalize)
    excluded: Optional[bool] = None    # True = left out of the default atlas counts (derived by classify.finalize)
    excluded_reason: Optional[str] = None  # ';'-joined reasons: zone name, 'character', 'placeholder', 'empty_surface'
    cert: Optional[str] = None         # editor's @cert of the identification (high | medium | low), when given

    def to_row(self) -> dict:
        d = asdict(self)
        d['part'] = d['zone']                 # structural zone (txt | hdr) keeps its own column
        d['zone'] = d.pop('zone_class')       # ``zone`` is the refined zone
        d['auth'] = auth_to_arrow(self.auth)
        return d


MENTION_SCHEMA = pa.schema([
    ('coll_slug', pa.string()),
    ('renascor_id', pa.string()),
    ('source_path', pa.string()),
    ('doc_id', pa.string()),
    ('doc_year', pa.int32()),
    ('doc_year_from', pa.int32()),
    ('doc_year_to', pa.int32()),
    ('date_source', pa.string()),
    ('in_period', pa.bool_()),
    ('kind', pa.string()),
    ('tag', pa.string()),
    ('role', pa.string()),
    ('zone', pa.string()),            # v2: refined (text, header, editorial_commentary, apparatus, footnote, regest, repository)
    ('part', pa.string()),            # v2: structural zone, txt | hdr (what ``zone`` was in v1)
    ('person_class', pa.string()),
    ('excluded', pa.bool_()),
    ('excluded_reason', pa.string()),
    ('cert', pa.string()),
    ('surface', pa.string()),
    ('ref_raw', pa.string()),
    ('local_key', pa.string()),
    ('in_register', pa.bool_()),
    ('auth', AUTH_TYPE),
    ('context', pa.string()),
    ('lang', pa.string()),
    ('licence', pa.string()),
    ('tier', pa.string()),
])


# --- Entity -------------------------------------------------------------------------------------------------------
@dataclass
class Entity:
    local_key: str                    # '<slug>:<xml:id>' (shared register) or '<slug>:<doc_id>#<xml:id>' (in-file register)
    coll_slug: str
    kind: str                         # 'person' | 'place' | 'org'
    name: str
    source_path: str                  # file the register entry comes from
    renascor_id: Optional[str] = None
    variants: list = field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None
    coord_source: Optional[str] = None  # 'tei:geo' | 'json:<file>' | 'geonames' (resolved later) | ...
    birth: Optional[str] = None       # as given in the source (string)
    death: Optional[str] = None
    auth: dict = field(default_factory=empty_auth)
    note_type: Optional[str] = None   # 'person' | 'place' | 'mythological' | 'biblical' | 'author_cited' | 'fictional' ...
    licence: Optional[str] = None
    tier: Optional[str] = None
    n_mentions: Optional[int] = None  # filled by the writer at close()
    person_class: Optional[str] = None  # v2: class of the register entry (see PERSON_CLASSES); None -> derived from note_type

    def to_row(self) -> dict:
        d = asdict(self)
        d['auth'] = auth_to_arrow(self.auth)
        return d


ENTITY_SCHEMA = pa.schema([
    ('local_key', pa.string()),
    ('coll_slug', pa.string()),
    ('renascor_id', pa.string()),
    ('kind', pa.string()),
    ('name', pa.string()),
    ('variants', pa.list_(pa.string())),
    ('lat', pa.float64()),
    ('lon', pa.float64()),
    ('coord_source', pa.string()),
    ('birth', pa.string()),
    ('death', pa.string()),
    ('auth', AUTH_TYPE),
    ('note_type', pa.string()),
    ('source_path', pa.string()),
    ('licence', pa.string()),
    ('tier', pa.string()),
    ('n_mentions', pa.int64()),
    ('person_class', pa.string()),
])

MENTION_FIELDS = [f.name for f in fields(Mention)]
ENTITY_FIELDS = [f.name for f in fields(Entity)]


def period_overlap(y_from: Optional[int], y_to: Optional[int]) -> Optional[bool]:
    """Is the document in 1450-1700?  True / False / None (undated or too wide to tell).

    A range that lies inside the period is True, one that lies outside is False. A range that straddles a boundary
    is True when it is narrow (<= WIDE_RANGE years, as in v1) and None when it is wider: we do not know where in
    the range the text sits (v2, audit point 12).
    """
    if y_from is None and y_to is None:
        return None
    a = y_from if y_from is not None else y_to
    b = y_to if y_to is not None else y_from
    if a > b:
        a, b = b, a
    if a > PERIOD[1] or b < PERIOD[0]:
        return False
    if a >= PERIOD[0] and b <= PERIOD[1]:
        return True
    return True if b - a <= WIDE_RANGE else None


def best_year(y_from: Optional[int], y_to: Optional[int]) -> Optional[int]:
    """The exact year, or the midpoint of a narrow range; None for a range wider than WIDE_RANGE years (v2)."""
    if y_from is None and y_to is None:
        return None
    a = y_from if y_from is not None else y_to
    b = y_to if y_to is not None else y_from
    if a > b:
        a, b = b, a
    if a == b:
        return a
    return (a + b) // 2 if b - a <= WIDE_RANGE else None
