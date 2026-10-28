"""Shared helpers to turn the public NER datasets into our JSONL format (LOC / PER only).

A record (one segment = a sentence, a line group or a chunk of <= MAX_TOKENS tokens):
  id, source (dataset name), subset (corpus / edition inside the dataset), split (train | dev | test | all | ...), split_suggested (for the unsplit sets),
  lang (ISO 639-1/3), year (int or None), licence, licence_flag (open | nc | sa | nc-sa | gpl | unstated | synthetic), quality (gold | silver | synthetic),
  tokens, ner_tags (ints into LABELS: O B-LOC I-LOC B-PER I-PER, same list as ratlas.ner.splits), ner_tags_loc, ner_tags_per,
  text (tokens joined with their original spacing when known, else one space), token_starts (character offset of each token in text),
  spans ([{start, end, label}] in characters), loc_valid / per_valid (False when the dataset does not annotate that label for this record, e.g. a
  Pile-NER passage that was only asked about persons), n_loc, n_per, dropped (labels of the source that were mapped to O, with counts).

Mapping of the source labels: place -> LOC (LOC, GPE, GEO, loc.*, LOC.*, location, NP-Spatial, B-LOC_NAME ...), person -> PER (PER, PERS, PRS, PERSON,
pers, person, NP-Person ...); everything else (ORG, MISC, GRP = peoples, NORP, dates, works ...) becomes O and is counted in ``dropped``.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional

LABELS = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']
LOC_LABELS = ['O', 'B-LOC', 'I-LOC']
PER_LABELS = ['O', 'B-PER', 'I-PER']
MAX_TOKENS = 200


# ---------------------------------------------------------------------------------------------------------------------
# label mapping and BIO
# ---------------------------------------------------------------------------------------------------------------------
def map_label(raw: str, loc: Iterable[str], per: Iterable[str]) -> Optional[str]:
    """'LOC' / 'PER' / None for a source type name (case-insensitive, exact or ``prefix.`` match: 'loc' matches 'loc.adm.town')."""
    r = raw.strip().lower()
    for names, out in ((loc, 'LOC'), (per, 'PER')):
        for n in names:
            n = n.lower()
            if r == n or r.startswith(n + '.'):
                return out
    return None


def split_tag(tag: str) -> tuple[str, str]:
    """('B' | 'I' | 'O', type) for 'B-LOC', 'I-per', 'LOC-B' (suffix style), 'O', '0', '_' ; anything else is a bare type with an implicit 'I'."""
    t = tag.strip()
    if t in ('O', '0', '_', '', '-', 'o'):
        return 'O', ''
    m = re.match(r'^([BIES])[-_](.+)$', t, re.I)
    if m:
        p = m.group(1).upper()
        return ('B' if p in ('B', 'S') else 'I'), m.group(2)
    m = re.match(r'^(.+)[-_]([BIES])$', t, re.I)
    if m and m.group(2).isupper():
        p = m.group(2)
        return ('B' if p in ('B', 'S') else 'I'), m.group(1)
    return 'I', t


def bio_spans(tags: list[str], loc: Iterable[str], per: Iterable[str], dropped: Optional[Counter] = None) -> list[tuple[int, int, str]]:
    """Token spans (start, end exclusive, 'LOC' | 'PER') from BIO tags, tolerant of IOB1 (an 'I-' after 'O' or after another type opens a span).
    Types that are neither place nor person are counted in ``dropped`` (one per source span) and ignored."""
    loc, per = list(loc), list(per)
    spans: list[tuple[int, int, str]] = []
    cur_type, cur_raw, start = None, None, 0   # cur_type = 'LOC' | 'PER' | '' (dropped type) | None (outside)

    def close(end: int):
        nonlocal cur_type, cur_raw
        if cur_type:
            spans.append((start, end, cur_type))
        elif cur_type == '' and dropped is not None:
            dropped[cur_raw] += 1
        cur_type, cur_raw = None, None

    for i, t in enumerate(tags):
        p, ty = split_tag(t)
        if p == 'O':
            close(i)
            continue
        if p == 'B' or cur_raw is None or ty.lower() != cur_raw.lower():
            close(i)
            start, cur_raw = i, ty
            cur_type = map_label(ty, loc, per) or ''
    close(len(tags))
    return spans


def ids_from_spans(n: int, spans: list[tuple[int, int, str]]) -> tuple[list[int], list[int], list[int], list[tuple[int, int, str]]]:
    """(joint ids, LOC-only ids, PER-only ids, accepted spans) for n tokens.  Overlapping spans: the earlier / longer one wins, the other is dropped."""
    joint, locs, pers = [0] * n, [0] * n, [0] * n
    used = [False] * n
    acc = []
    for s, e, lab in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        if e <= s or any(used[s:e]):
            continue
        acc.append((s, e, lab))
        for k in range(s, e):
            used[k] = True
        b, i = (1, 2) if lab == 'LOC' else (3, 4)
        arr = locs if lab == 'LOC' else pers
        for k in range(s, e):
            joint[k] = b if k == s else i
            arr[k] = (1 if k == s else 2)
    return joint, locs, pers, acc


# ---------------------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------------------
def h32(s: str) -> float:
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) / 2 ** 32


def suggested_split(key: str) -> str:
    """Deterministic 80 / 10 / 10 split for the datasets that come without one (key = a document / block id)."""
    x = h32('ext-v1|' + key)
    return 'train' if x < 0.8 else ('dev' if x < 0.9 else 'test')


def build_text(tokens: list[str], no_space_after: Optional[list[bool]] = None) -> tuple[str, list[int]]:
    parts, starts, pos = [], [], 0
    for i, t in enumerate(tokens):
        starts.append(pos)
        parts.append(t)
        pos += len(t)
        if i < len(tokens) - 1 and not (no_space_after and no_space_after[i]):
            parts.append(' ')
            pos += 1
    return ''.join(parts), starts


def make_record(*, rid: str, source: str, subset: str, split: str, lang: str, tokens: list[str], bio: Optional[list[str]] = None,
                tok_spans: Optional[list[tuple[int, int, str]]] = None, no_space_after: Optional[list[bool]] = None,
                loc_types: Iterable[str] = ('LOC',), per_types: Iterable[str] = ('PER',), meta: Optional[dict] = None,
                loc_valid: bool = True, per_valid: bool = True) -> dict:
    """One record from tokens and either source BIO tags (``bio``) or already mapped token spans (``tok_spans``)."""
    dropped: Counter = Counter()
    if tok_spans is None:
        tok_spans = bio_spans(bio or ['O'] * len(tokens), loc_types, per_types, dropped)
    n = len(tokens)
    # the LOC view and the PER view are resolved separately (a place inside a person name stays a place in the LOC view: nested layers such as the
    # medieval charters); the joint view resolves the overlaps across types (the earlier / longer span wins)
    joint, _, _, acc_joint = ids_from_spans(n, tok_spans)
    _, locs, _, acc_loc = ids_from_spans(n, [x for x in tok_spans if x[2] == 'LOC'])
    _, _, pers, acc_per = ids_from_spans(n, [x for x in tok_spans if x[2] == 'PER'])
    acc = sorted(set(acc_loc) | set(acc_per))
    text, starts = build_text(tokens, no_space_after)
    cspans = [{'start': starts[s], 'end': starts[e - 1] + len(tokens[e - 1]), 'label': l} for s, e, l in acc]
    if not loc_valid:
        locs = [0] * n
    if not per_valid:
        pers = [0] * n
    rec = {'id': rid, 'source': source, 'subset': subset, 'split': split, 'lang': lang}
    rec.update(meta or {})
    rec.update({'tokens': tokens, 'ner_tags': joint, 'ner_tags_loc': locs, 'ner_tags_per': pers, 'text': text, 'token_starts': starts, 'spans': cspans,
                'loc_valid': loc_valid, 'per_valid': per_valid, 'n_loc': sum(1 for s in cspans if s['label'] == 'LOC'),
                'n_per': sum(1 for s in cspans if s['label'] == 'PER'), 'dropped': dict(dropped)})
    return rec


def chunk(n: int, tags: Optional[list[str]] = None, stops: Optional[list[bool]] = None, limit: int = MAX_TOKENS) -> list[tuple[int, int]]:
    """Cut a run of n tokens into pieces of <= limit tokens, preferably after a sentence stop and never inside an entity (no cut before an 'I-')."""
    out, a = [], 0
    while n - a > limit:
        b = a + limit
        cut = None
        for k in range(b, a + limit // 2, -1):
            inside = tags is not None and k < n and split_tag(tags[k])[0] == 'I'
            if stops is not None and stops[k - 1] and not inside:
                cut = k
                break
        if cut is None:
            for k in range(b, a + limit // 2, -1):
                if not (tags is not None and k < n and split_tag(tags[k])[0] == 'I'):
                    cut = k
                    break
        cut = cut or b
        out.append((a, cut))
        a = cut
    out.append((a, n))
    return out


class Out:
    """Writes <out>/<name>/<split>.jsonl and keeps the counters of the dataset (tokens, records, spans, per split / subset / lang)."""

    def __init__(self, root: Path, name: str, licence: str, flag: str, quality: str = 'gold'):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.licence, self.flag, self.quality = name, licence, flag, quality
        self.fh: dict[str, object] = {}
        self.count: Counter = Counter()
        self.dropped: Counter = Counter()
        self.seen: set[str] = set()

    def write(self, rec: dict, licence: Optional[str] = None, flag: Optional[str] = None, quality: Optional[str] = None):
        sp = rec['split']
        rec['licence'] = licence or self.licence
        rec['licence_flag'] = flag or self.flag
        rec['quality'] = quality or rec.get('quality') or self.quality
        if sp not in self.fh:
            self.fh[sp] = open(self.dir / f'{sp}.jsonl', 'w', encoding='utf-8')
        self.fh[sp].write(json.dumps(rec, ensure_ascii=False) + '\n')
        key = (sp, rec['subset'], rec['lang'], rec['licence_flag'], rec['quality'])
        self.count[key + ('records',)] += 1
        self.count[key + ('tokens',)] += len(rec['tokens'])
        self.count[key + ('LOC',)] += rec['n_loc']
        self.count[key + ('PER',)] += rec['n_per']
        if not rec['loc_valid']:
            self.count[key + ('loc_invalid_records',)] += 1
        if not rec['per_valid']:
            self.count[key + ('per_invalid_records',)] += 1
        for k, v in rec['dropped'].items():
            self.dropped[k] += v

    def close(self) -> list[dict]:
        for f in self.fh.values():
            f.close()
        rows: dict[tuple, dict] = {}
        for k, v in self.count.items():
            sp, sub, lang, flag, q, field = k
            r = rows.setdefault((sp, sub, lang, flag, q), {'dataset': self.name, 'split': sp, 'subset': sub, 'lang': lang, 'licence_flag': flag,
                                                         'quality': q, 'records': 0, 'tokens': 0, 'LOC': 0, 'PER': 0, 'loc_invalid_records': 0,
                                                         'per_invalid_records': 0})
            r[field] = v
        out = sorted(rows.values(), key=lambda r: (r['split'], r['subset'], r['lang']))
        (self.dir / 'counts.json').write_text(json.dumps({'rows': out, 'dropped_source_labels': dict(self.dropped.most_common())}, indent=1, ensure_ascii=False))
        return out
