"""PCEEC2, Parsed Corpus of Early English Correspondence 2 (github beatrice57/pceec2): NOT TEI.

84 Penn-treebank files (data/parsed/*.psd, ~4,970 letters) whose every sentence repeats a per-letter record
(METADATA (AUTHOR NAME:SEX:_:DOB:AGE) (RECIPIENT NAME:SEX:_:DOB:AGE) (LETTER ID:period:YEAR:autograph:other)), plus
the sociolinguistic table data/aif_2022.csv (author / recipient, titles, DOB, relation). Facts-only: the letter text is
not touched. There are no place tags (the editorial ED line of some letters mentions a place in free text: not read) and
no authority ids.
Mentions (zone 'hdr', one pair per letter): AUTHOR -> role 'sender', RECIPIENT -> role 'addressee'; surface = the
normalised name with underscores as spaces ('WILLIAM ALLEN'), ref_raw = the raw name key ('WILLIAM_ALLEN', which also
carries editorial marks like '[N.TALBOT]' or '&'), local_key '<slug>:<key>' resolved against the entities of the table.
Entities: one per distinct author / recipient name key of aif_2022.csv (name, birth = DOB string incl. '?').
Dating: the year of the LETTER record (date_source 'letter'); a decade '1570S' is the range 1570-1579 (v2); a letter without
year is undated (None).
v2: re-harvest of 2026-10-17 (pceec2 main at 99b7012): the 105,373 (AUTHOR)(RECIPIENT)(LETTER) records and aif_2022.csv are
identical to the June copy; the new SHA is the only change.
Not extracted: pepys_4200.txt (plain-text diary of the neighbouring catalogue entry 'pepys-diary': no tags).
Licence: LICENSE_TIERS PCEEC2 LicenseRef-Custom, T3 (redistribution not permitted): no context, internal use only.
Files: not in scan.jsonl (not XML); to ship them use
  scripts/push_coll.sh --extra 'eng_dh/pceec2-main/data/aif_2022.csv' --extra 'eng_dh/pceec2-main/data/parsed/*.psd' eng_dh/pceec2-main
"""
from __future__ import annotations

import csv
import io
import re

from .. import common as C
from .. import helpers_v2 as V
from ..model import Entity, Mention, best_year, period_overlap

SLUG = 'pceec2'
COLLS = V.pick(['eng_dh/pceec2-main'], [V.NEW + '/pceec2/pceec2-main'])
RENASCOR_IDS = ['pceec2']
LICENCE = 'LicenseRef-Custom'
TIER = 'T3'
DESCRIPTION = 'PCEEC2 letters: sender and addressee of ~4,970 letters from the treebank metadata (facts only)'
OPTIONS = {}

_BASE = V.pick('eng_dh/pceec2-main', V.NEW + '/pceec2/pceec2-main')
_DIR = _BASE + '/data/parsed/'
_CSV = _BASE + '/data/aif_2022.csv'
_REC = re.compile(r'\(AUTHOR (\S+)\)\s*\(RECIPIENT (\S+)\)\s*\(LETTER (\S+?)\)')


def iter_files(ctx, include_dups=False):
    import os
    root = C.raw_root() / _DIR
    names = sorted(n for n in os.listdir(root) if n.endswith('.psd')) if root.is_dir() else []
    return [C.FileRecord(path=_DIR + n, coll='eng_dh') for n in names]


def _surface(key: str) -> str:
    return re.sub(r'\s+', ' ', key.replace('_', ' ')).strip()


def prepare(ctx):
    rows = csv.DictReader(io.StringIO(C.read_raw(_CSV).decode('utf-8-sig')))
    ents = {}
    for r in rows:
        for nm, dob in ((r.get('Author name'), r.get('Author DOB')), (r.get('Recipient'), r.get('Recipient DOB'))):
            nm = (nm or '').strip()
            if not nm:
                continue
            key = C.make_key(ctx.slug, nm)
            e = ents.get(key)
            if e is None:
                e = Entity(local_key=key, coll_slug=ctx.slug, kind='person', name=_surface(nm), source_path=_CSV,
                           renascor_id=ctx.renascor_id, birth=(dob or '').strip() or None, note_type='person',
                           licence=ctx.licence, tier=ctx.tier)
                ents[key] = e
            elif e.birth is None and (dob or '').strip():
                e.birth = dob.strip()
    for e in ents.values():
        ctx.shared.add(e, [e.local_key.split(':', 1)[1]])
    return list(ents.values())


def _split(field: str):
    name, *rest = field.split(':')
    return name, rest


def extract(rec, ctx):
    text = rec.read().decode('latin-1')
    seen, ms = set(), []
    for m in _REC.finditer(text):
        a, r, letter = m.group(1), m.group(2), m.group(3)
        lid, *lrest = letter.split(':')
        if lid in seen:
            continue
        seen.add(lid)
        ym = re.fullmatch(r'(\d{4})(S?)\??', lrest[1], re.I) if len(lrest) > 1 else None
        y = int(ym.group(1)) if ym else None
        y2 = y + 9 if (ym and ym.group(2)) else y        # '1570S' = the decade 1570-1579 (v2)
        for fld, role in ((a, 'sender'), (r, 'addressee')):
            name, _ = _split(fld)
            if not name or name in ('_', 'UNKNOWN'):
                continue
            ent = ctx.shared.lookup(name)
            ms.append(Mention(
                coll_slug=ctx.slug, renascor_id=ctx.renascor_id, source_path=rec.path, doc_id=lid,
                doc_year=best_year(y, y2), doc_year_from=y, doc_year_to=y2, date_source='letter' if y else None,
                in_period=period_overlap(y, y2), kind='person', tag='METADATA', role=role, zone='hdr',
                surface=_surface(name), ref_raw=name, local_key=ent.local_key if ent else None,
                in_register=True if ent else None, auth=C.empty_auth(), context=None, lang='en',
                licence=ctx.licence, tier=ctx.tier))
    return ms, []
