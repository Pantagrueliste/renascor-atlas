"""NER v2, clean test set (``test_v2``): the held-out v1 test passages rebuilt from the TEI with the layout repairs of ``tei.Options``.

Why: in the v1 test, 1063 of the 1066 passages of ``elan_crenum`` carry "detached initials" ("f rance", "p aris", "ſ ur"). They are not drop capitals in the
source: ``tei.py`` turned the whitespace between ``<orig>`` and ``<reg>`` inside a ``<choice>`` (pretty-printed XML) into a word break. ``hainhofer_reisen``
and ``kaiserin_eleonora_magdalena`` split words at ``-<lb/>`` ("Straßbur- g"). Both are repaired here by two options of ``tei.Options`` (default off):

  * ``repair_ws``: whitespace-only text between the children of ``choice`` / ``app`` / ``subst`` is not a word break;
  * ``dehyph``: a hyphen at the end of a line (before ``lb`` / ``pb`` / ``cb``) joins the two halves, the hyphen is dropped before a lower-case letter
    and kept (without space) before a capital.

The passages keep the regions of v1 exactly: every v1 passage is located in the repaired text of its document (the non-space, non-hyphen characters
are identical) and cut there again, so that passage ``i`` of a document is the same stretch of text in v1 and v2. A v2 passage whose text is unchanged
keeps its v1 id; a repaired one gets the id ``<v1 id>#r``. The v1 id is always in ``v1_id``.

    python -m ratlas.ner.ner2_testset rebuild SLUG --raw DATA_V1/raw/passages/SLUG.jsonl.gz --out OUT/passages/SLUG.jsonl.gz [--repair-ws] [--dehyph] [--workers 8]
    python -m ratlas.ner.ner2_testset assemble --v1-test DATA_V1/test.jsonl --passages OUT/passages --out OUT/test_v2.jsonl --stats OUT/test_v2_stats.json

``rebuild`` needs the raw files of the edition (``RATLAS_RAW``, ``RATLAS_SCAN``, as for ``ratlas.ner.extract``). Without ``--repair-ws`` and ``--dehyph``
it must give back the v1 texts exactly (control).
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .. import common as C
from . import editions as E
from .tei import BRK, HYPHENS, MIN_WORDS, Options, document_passages

_STATE: dict = {}
_DETACHED = re.compile(r'(?<![\w\'’])[^\W\d_](?= [^\W\d_]{2,})', re.U)     # a one-letter word followed by a word of >= 2 letters


# ---------------------------------------------------------------------------------------------------------------------
# alignment of the v1 passages in the new buffer
# ---------------------------------------------------------------------------------------------------------------------
def norm_stream(s: str) -> tuple[str, list[int]]:
    """``s`` without white space, break markers and hyphens, and the position in ``s`` of each kept character."""
    chars, pos = [], []
    for i, ch in enumerate(s):
        if ch.isspace() or ch == BRK or ch in HYPHENS:
            continue
        chars.append(ch)
        pos.append(i)
    return ''.join(chars), pos


class Aligner:
    """``group_fn`` for ``tei.document_passages``: cut the new buffer at the places where the v1 passages of the document start and end."""

    def __init__(self, v1_texts: list[str]):
        self.v1 = [norm_stream(t)[0] for t in v1_texts]
        self.kept: list[int] = []       # index of the v1 passages that come out of document_passages (>= MIN_WORDS words)
        self.failed: list[int] = []
        self.gaps: list[int] = []

    def __call__(self, buf: str, units: list) -> list:
        nb, pos = norm_stream(buf)
        found: list[Optional[tuple[int, int]]] = []
        ptr = 0
        for k, n1 in enumerate(self.v1):
            m = nb.find(n1, ptr) if n1 else -1
            if m < 0 and len(n1) > 120:
                head = nb.find(n1[:120], ptr)
                if head >= 0 and nb[head:head + len(n1)] == n1:
                    m = head
            if m < 0:
                self.failed.append(k)
                found.append(None)
                continue
            self.gaps.append(m - ptr)
            found.append((m, m + len(n1)))
            ptr = m + len(n1)
        groups: list = []
        self.kept = []
        for k, f in enumerate(found):
            if f is None:
                groups.append([])
                continue
            a = pos[f[0]]
            b = pos[f[1] - 1] + 1                   # just after the last character of the passage, and the hyphens that close it
            while b < len(buf) and buf[b] in HYPHENS:
                b += 1
            g = []
            for s, e in units:
                if e <= a or s >= b:
                    continue
                part = buf[s:e]
                first = s + len(part) - len(part.lstrip())
                last = e - (len(part) - len(part.rstrip()))
                # a unit that the region covers entirely is kept as it is (white space included, as in v1); a cut unit is trimmed at the cut
                g.append((s if a <= first else a, e if b >= last else b))
            g = [(s, e) for s, e in g if buf[s:e].strip()]
            groups.append(g)
            if sum(len(buf[s:e].split()) for s, e in g) >= MIN_WORDS:
                self.kept.append(k)
        return groups


# ---------------------------------------------------------------------------------------------------------------------
# rebuild one edition
# ---------------------------------------------------------------------------------------------------------------------
def _work(arg):
    path, data, v1 = arg
    al = Aligner([p['text'] for p in v1])
    try:
        r = document_passages(data, opts=_STATE['opts'], group_fn=al)
    except Exception as e:      # noqa: BLE001  one bad file must not stop the run
        return path, {'error': f'{type(e).__name__}: {e}', 'passages': {}, 'failed': list(range(len(v1))), 'gaps': []}
    out = {}
    for k, p in zip(al.kept, r['passages']):
        out[k] = p
    return path, {'error': r['error'], 'passages': out, 'failed': al.failed, 'gaps': al.gaps}


def _feed(recs, by_path):
    for r in C.iter_with_data(recs):
        d = r.data
        r.data = None
        yield r.path, d, by_path[r.path]


def rebuild(slug: str, raw: Path, out: Path, meta_dir: Optional[Path], repair_ws: bool, dehyph: bool, workers: int, ids: Optional[set] = None) -> dict:
    t0 = time.time()
    meta = E.load_meta(Path(meta_dir) / 'edition_meta.tsv' if meta_dir else None)
    ed = E.get_edition(slug, meta)
    by_path: dict = defaultdict(list)
    with gzip.open(raw, 'rt', encoding='utf-8') as fh:
        for line in fh:
            p = json.loads(line)
            if '#' in p['path']:
                raise SystemExit(f'{slug}: pseudo-XML chunks are not supported ({p["path"]})')
            by_path[p['path']].append(p)
    for v in by_path.values():
        v.sort(key=lambda p: p['i'])
        if [p['i'] for p in v] != list(range(len(v))):
            raise SystemExit(f'{slug}: passage indices of a document are not 0..n-1')
    files = [r for r in ed.files() if r.path in by_path]
    missing = sorted(set(by_path) - {r.path for r in files})
    _STATE['opts'] = dataclasses.replace(ed.options, repair_ws=repair_ws, dehyph=dehyph)
    results: dict = {}
    if workers > 1:
        pool = mp.get_context('fork').Pool(min(workers, 24))
        it = pool.imap_unordered(_work, _feed(files, by_path), chunksize=2)
    else:
        pool = None
        it = (_work(a) for a in _feed(files, by_path))
    try:
        for path, r in it:
            results[path] = r
    finally:
        if pool:
            pool.close()
            pool.join()
    stats = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
        for path in sorted(by_path):
            v1 = by_path[path]
            r = results.get(path)
            if r is None or r.get('error'):
                stats['docs_error'] += 1
                stats['passages_lost'] += len(v1)
                continue
            stats['docs'] += 1
            stats['align_failed'] += len(r['failed'])
            for k, p1 in enumerate(v1):
                p = r['passages'].get(k)
                if p is None:
                    stats['passages_lost'] += 1
                    continue
                same = p['text'] == p1['text']
                stats['passages'] += 1
                stats['text_same' if same else 'text_changed'] += 1
                rec = {'id': p1['id'], 'path': path, 'i': k, 'text': p['text'], 'spans': p['spans'], 'nested': p['nested'], 'ignored': p['ignored'],
                       'orgs': p['orgs'], 'words': p['words'], 'flags': p['flags'], 'dropped_span': p['dropped_span'], 'changed': not same,
                       'v1_words': p1['words']}
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(out)
    stats = dict(stats)
    stats.update({'slug': slug, 'repair_ws': repair_ws, 'dehyph': dehyph, 'files_missing': missing[:10], 'n_files_missing': len(missing),
                  'max_gap': max([g for r in results.values() for g in r.get('gaps', [])] or [0]), 'seconds': round(time.time() - t0, 1)})
    out.with_suffix('').with_suffix('.stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats), file=sys.stderr, flush=True)
    return stats


# ---------------------------------------------------------------------------------------------------------------------
# assemble test_v2
# ---------------------------------------------------------------------------------------------------------------------
def _surface(text: str, s: dict) -> str:
    return text[s['start']:s['end']]


def _norm(s: str) -> str:
    return norm_stream(s)[0]


_LEGIT_ONE_LETTER = frozenset('àÀyYaAeEoOiIuU')


def detached(text: str) -> int:
    """One-letter words followed by a word of >= 2 letters: the symptom of a word cut after its first letter ("f rance", "ſ ur").
    A, à, y, e, o, i, u are real words in several of the languages and are not counted, so this is a lower bound."""
    return sum(1 for m in _DETACHED.finditer(text) if m.group() not in _LEGIT_ONE_LETTER)


def assemble(v1_test: Path, passages_dir: Path, out: Path, stats_path: Optional[Path], loc_out: Optional[Path] = None) -> dict:
    from .splits import tokens_and_tags
    new: dict = {}
    for f in sorted(Path(passages_dir).glob('*.jsonl.gz')):
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                d = json.loads(line)
                new[d['id']] = d
    st: dict = defaultdict(Counter)
    tmp = out.with_suffix(out.suffix + '.tmp')
    n_out = 0
    with open(v1_test, encoding='utf-8') as fin, open(tmp, 'w', encoding='utf-8') as fout:
        for line in fin:
            r = json.loads(line)
            ed = r['edition']
            s = st[ed]
            s['passages_v1'] += 1
            p = new.get(r['id'])
            loc1 = [x for x in r['spans'] if x['label'] == 'LOC']
            per1 = [x for x in r['spans'] if x['label'] == 'PER']
            s['loc_gold_v1'] += len(loc1)
            s['per_gold_v1'] += len(per1)
            s['detached_v1'] += detached(r['text'])
            if p is None:
                s['passages_lost'] += 1
                continue
            loc2 = [x for x in p['spans'] if x['label'] == 'LOC']
            per2 = [x for x in p['spans'] if x['label'] == 'PER']
            changed = p['text'] != r['text']
            s['passages_v2'] += 1
            s['passages_changed'] += changed
            s['detached_v2'] += detached(p['text'])
            s['flag_ws_repair'] += 'ws_repair' in p['flags']
            s['flag_dehyph'] += 'dehyph' in p['flags']
            # gold spans: same number and same letters (white space and hyphens aside) -> the span is mapped; a repaired surface is "changed"
            for lab, a1, a2 in (('loc', loc1, loc2), ('per', per1, per2)):
                c1 = Counter(_norm(_surface(r['text'], x)) for x in a1)
                c2 = Counter(_norm(_surface(p['text'], x)) for x in a2)
                s[f'{lab}_gold_v2'] += len(a2)
                s[f'{lab}_gold_removed'] += sum((c1 - c2).values())      # e.g. a generic term ("Königreich", "seigneur") that the repair lets the generic-term filter see
                s[f'{lab}_gold_added'] += sum((c2 - c1).values())
                if len(a1) == len(a2):
                    s[f'{lab}_gold_changed'] += sum(1 for x, y in zip(sorted(a1, key=lambda z: z['start']), sorted(a2, key=lambda z: z['start']))
                                                    if _surface(r['text'], x) != _surface(p['text'], y))
            toks, tags, loc, per, offs = tokens_and_tags(p['text'], p['spans'])
            s['tokens_v1'] += len(r['tokens'])
            s['tokens_v2'] += len(toks)
            rec = dict(r)
            rec.update({'id': r['id'] + '#r' if changed else r['id'], 'v1_id': r['id'], 'split': 'test_v2', 'changed': changed,
                        'repair_flags': sorted(set(p['flags']) & {'ws_repair', 'dehyph'}),
                        'text': p['text'], 'spans': [{'start': x['start'], 'end': x['end'], 'label': x['label'], 'tag': x['tag']} for x in p['spans']],
                        'tokens': toks, 'token_starts': offs, 'ner_tags': tags, 'ner_tags_loc': loc, 'ner_tags_per': per,
                        'flags': p['flags'], 'words': p['words']})
            fout.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
            n_out += 1
    tmp.replace(out)
    if loc_out:         # the passages that count for LOC: loc_valid, in the editions that tag places
        with_loc = {ed for ed, c in st.items() if c['loc_gold_v1'] > 0}
        n_loc = 0
        with open(out, encoding='utf-8') as fin, open(loc_out, 'w', encoding='utf-8') as fout:
            for line in fin:
                d = json.loads(line)
                if d['loc_valid'] and d['edition'] in with_loc:
                    fout.write(line)
                    n_loc += 1
        st['_loc_file']['passages'] = n_loc
    res = {ed: dict(c) for ed, c in st.items()}
    res['_total'] = dict(sum((c for k, c in st.items() if not k.startswith('_')), Counter()))
    res['_n_written'] = n_out
    if stats_path:
        Path(stats_path).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas.ner.ner2_testset')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('rebuild')
    r.add_argument('slug')
    r.add_argument('--raw', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--meta', type=Path, default=None)
    r.add_argument('--repair-ws', action='store_true')
    r.add_argument('--dehyph', action='store_true')
    r.add_argument('--workers', type=int, default=1)
    a_ = sub.add_parser('assemble')
    a_.add_argument('--v1-test', type=Path, required=True)
    a_.add_argument('--passages', type=Path, required=True)
    a_.add_argument('--out', type=Path, required=True)
    a_.add_argument('--stats', type=Path, default=None)
    a_.add_argument('--loc-out', type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'rebuild':
        rebuild(a.slug, a.raw, a.out, a.meta, a.repair_ws, a.dehyph, a.workers)
    else:
        res = assemble(a.v1_test, a.passages, a.out, a.stats, a.loc_out)
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
