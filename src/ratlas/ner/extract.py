"""Slurm job: the passages of one edition.

    python -m ratlas.ner.extract SLUG --out ~/renascor-atlas-data/ner/data_v1/raw --meta ~/renascor-atlas-data/ner/meta
           [--workers 8] [--budget-tagged 8000000] [--budget-untagged 2000000] [--limit N] [--max-year 1800]

Reads the raw files of the edition (``RATLAS_RAW``, ``RATLAS_SCAN`` as for ``ratlas extract``), parses the sampled documents with
``ner.tei`` and writes

    <out>/passages/<slug>.jsonl.gz   one passage per line (see ``ner.tei.document_passages``) + slug, doc, path, year, language, licence
    <out>/summary/<slug>.json        counts of the edition (files, words, sampled share, spans, skipped parts, languages...)

Deterministic: the documents are taken in the order of the SHA-1 of their path until the word budget is reached (tagged documents first
budget, untagged documents a second, smaller budget: they only serve the completeness measure), the output is sorted by path.
The 15% cap, the splits and the completeness measure are later steps (``ner.complete``, ``ner.splits``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from .. import common as C
from . import editions as E
from .langs import norm_lang
from .tei import Options, doc_hash, document_chunks

_STATE: dict = {}


def _work(arg):
    path, data = arg
    try:
        r = document_chunks(data, opts=_STATE['opts'])
    except Exception as e:      # noqa: BLE001  one bad file must not stop the run
        return path, {'passages': [], 'doc_lang': None, 'words': 0, 'stats': {}, 'error': f'{type(e).__name__}: {e}',
                      'xml_id': None, 'hdr_year': None}
    return path, r


def _feed(recs):
    for r in C.iter_with_data(recs):
        d = r.data
        r.data = None
        yield r.path, d


def load_doc_meta(meta_dir: Optional[Path], slug: str) -> dict:
    out: dict = {}
    p = Path(meta_dir) / 'doc_meta.tsv.gz' if meta_dir else None
    if p and p.exists():
        with gzip.open(p, 'rt', encoding='utf-8', newline='') as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                if row['slug'] == slug:
                    y = row['doc_year']
                    out[row['source_path']] = (int(float(y)) if y else None, row['lang'] or None)
    return out


def edition_language(meta: dict) -> Optional[str]:
    """The language of the edition when the catalogue names one."""
    names = [x for x in (meta.get('languages') or '').split(';') if x]
    codes = {norm_lang(x.strip().lower()) for x in names}
    codes.discard(None)
    return next(iter(codes)) if len(codes) == 1 else None


def select_files(files: list, budget_tagged: int, budget_untagged: int, known_years: dict, max_year: Optional[int]) -> tuple[list, dict]:
    tagged, untagged = [], []
    dropped_year = 0
    for r in files:
        y = known_years.get(r.path, (None, None))[0]
        if max_year and y and y > max_year:
            dropped_year += 1
            continue
        (tagged if r.count(*E.TAGGED_KEYS) > 0 else untagged).append(r)

    def take(rs, budget):
        out, w = [], 0
        for r in sorted(rs, key=lambda r: doc_hash(r.path)):
            if w >= budget:
                break
            out.append(r)
            w += r.words
        return out

    sel_t, sel_u = take(tagged, budget_tagged), take(untagged, budget_untagged)
    info = {'files_total': len(files), 'words_total': sum(r.words for r in files),
            'files_tagged_total': len(tagged), 'words_tagged_total': sum(r.words for r in tagged),
            'files_untagged_total': len(untagged), 'words_untagged_total': sum(r.words for r in untagged),
            'files_dropped_year': dropped_year, 'files_sampled_tagged': len(sel_t), 'files_sampled_untagged': len(sel_u),
            'tagged_paths': sorted(r.path for r in sel_t)}
    return sorted(sel_t + sel_u, key=lambda r: r.path), info


def resolve_lang(p: dict, doc_lang: Optional[str], mention_lang: Optional[str], edition_lang: Optional[str]) -> tuple[Optional[str], str]:
    """xml:lang of the passage > language of the document > language of its mentions > stop-word guess > the edition's only language."""
    if p.get('lang'):
        return p['lang'], 'xml:lang'
    if doc_lang:
        return doc_lang, 'doc'
    if mention_lang:
        return mention_lang, 'mentions'
    if p.get('lang_guess'):
        return p['lang_guess'], 'guess'
    if edition_lang:
        return edition_lang, 'edition'
    return None, 'none'


def run(slug: str, out: Path, meta_dir: Optional[Path], workers: int, budget_tagged: int, budget_untagged: int,
        limit: Optional[int], max_year: Optional[int], verbose: bool = True) -> dict:
    t0 = time.time()
    meta = E.load_meta(Path(meta_dir) / 'edition_meta.tsv' if meta_dir else None)
    ed = E.get_edition(slug, meta)
    files = ed.files()
    dm = load_doc_meta(meta_dir, slug)
    if not any(r.count(*E.TAGGED_KEYS) > 0 for r in files):
        budget_untagged = 0            # an edition without any tag in its text has nothing to learn from nor to measure
    sel, info = select_files(files, budget_tagged, budget_untagged, dm, max_year)
    if limit:
        sel = sel[:limit]
    tagged_paths = set(info.pop('tagged_paths'))
    ed_lang = edition_language(ed.meta)
    default_lang = E.EXTRA.get(slug, {}).get('default_lang')
    if verbose:
        print(f'[{slug}] {info["files_total"]} files, {len(sel)} sampled ({info["files_sampled_tagged"]} tagged), '
              f'{time.time() - t0:.1f}s', file=sys.stderr, flush=True)
    _STATE['opts'] = ed.options
    by_path = {r.path: r for r in sel}
    results: dict = {}
    if workers > 1:
        pool = mp.get_context('fork').Pool(min(workers, 24))
        it = pool.imap_unordered(_work, _feed(sel), chunksize=2)
    else:
        pool = None
        it = (_work((r.path, r.data)) for r in C.iter_with_data(sel))
    n = 0
    try:
        for path, r in it:
            results[path] = r
            n += 1
            if verbose and n % 2000 == 0:
                print(f'[{slug}] {n}/{len(sel)} files, {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    (out / 'passages').mkdir(parents=True, exist_ok=True)
    (out / 'summary').mkdir(parents=True, exist_ok=True)
    skipped: Counter = Counter()
    seen: Counter = Counter()
    lang_src: Counter = Counter()
    lang_n: Counter = Counter()
    n_pass = n_span = n_docs_span = n_err = n_dropped_year = n_conflict = 0
    words = 0
    errors = []
    tmp = out / 'passages' / f'{slug}.jsonl.gz.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
        for path in sorted(results):
            r = results[path]
            if r['error']:
                n_err += 1
                errors.append((path, r['error']))
                continue
            skipped.update(r['stats'].get('skipped', {}))
            seen.update(r['stats'].get('seen', {}))
            y, ml = dm.get(path, (None, None))
            year = y if y is not None else r.get('hdr_year')
            ysrc = 'mentions' if y is not None else ('header' if year is not None else None)
            if max_year and year and year > max_year:
                n_dropped_year += 1
                continue
            doc_spans = 0
            h = hashlib.sha1(path.encode('utf-8')).hexdigest()[:10]
            recs = []
            for i, p in enumerate(r['passages']):
                if max_year and y is None and p.get('hdr_year') and p['hdr_year'] > max_year:
                    continue
                lang, src = resolve_lang(p, p.get('doc_lang') or r['doc_lang'], ml, ed_lang)
                if p.get('lang_guess') and lang and p['lang_guess'] != lang:
                    n_conflict += 1
                    p['flags'].append('lang_conflict')
                chunk = p.get('chunk')
                rec = {'id': f'{slug}/{h}/{i}', 'slug': slug, 'path': path + ('#' + chunk if chunk else ''), 'doc_xml_id': r['xml_id'], 'i': i,
                       'doc_tagged': path in tagged_paths, 'year': year, 'year_src': ysrc, 'lang': lang, 'lang_src': src,
                       'doc_lang': p.get('doc_lang') or r['doc_lang'], 'mention_lang': ml, 'licence': ed.licence, 'tier': ed.tier}
                rec.update({k: v for k, v in p.items() if k not in ('lang', 'lang_src', 'chunk', 'doc_lang', 'hdr_year')})
                if chunk and y is None and p.get('hdr_year'):
                    rec['year'], rec['year_src'] = p['hdr_year'], 'header'
                recs.append(rec)
            # a passage without any language evidence (a list of names, a heading) takes the main language of its own document,
            # else the edition's default language (EXTRA 'default_lang': an edition known to be monolingual whose passages are lists of names)
            dl = Counter(x['lang'] for x in recs if x['lang'] and x['lang_src'] != 'edition')
            doc_main = dl.most_common(1)[0][0] if dl else None
            for rec in recs:
                if rec['lang'] is None:
                    if doc_main:
                        rec['lang'], rec['lang_src'] = doc_main, 'doc_majority'
                    elif default_lang:
                        rec['lang'], rec['lang_src'] = default_lang, 'edition_default'
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
                n_pass += 1
                words += rec['words']
                doc_spans += len(rec['spans'])
                lang_src[rec['lang_src']] += 1
                lang_n[rec['lang'] or 'None'] += 1
            n_span += doc_spans
            n_docs_span += 1 if doc_spans else 0
    tmp.replace(out / 'passages' / f'{slug}.jsonl.gz')
    summ = {'slug': slug, 'licence': ed.licence, 'tier': ed.tier, **info, 'files_parsed': len(results), 'parse_errors': n_err,
            'docs_dropped_year_after_parse': n_dropped_year, 'docs_with_spans': n_docs_span, 'passages': n_pass,
            'words': words, 'spans': n_span, 'skipped_parts': dict(skipped), 'seen': dict(seen.most_common(60)), 'lang_src': dict(lang_src),
            'lang': dict(lang_n.most_common()), 'lang_conflicts': n_conflict, 'errors': errors[:10],
            'edition_language': ed_lang, 'max_year': max_year, 'budget_tagged': budget_tagged, 'budget_untagged': budget_untagged,
            'seconds': round(time.time() - t0, 1)}
    (out / 'summary' / f'{slug}.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding='utf-8')
    if verbose:
        print(json.dumps({k: summ[k] for k in ('slug', 'files_parsed', 'docs_with_spans', 'passages', 'words', 'spans', 'seconds')}),
              file=sys.stderr, flush=True)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas.ner.extract')
    ap.add_argument('slug')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--meta', type=Path, default=None)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--budget-tagged', type=int, default=8_000_000)
    ap.add_argument('--budget-untagged', type=int, default=2_000_000)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--max-year', type=int, default=1800)
    a = ap.parse_args(argv)
    run(a.slug, a.out, a.meta, a.workers, a.budget_tagged, a.budget_untagged, a.limit, a.max_year)
    return 0


if __name__ == '__main__':
    sys.exit(main())
