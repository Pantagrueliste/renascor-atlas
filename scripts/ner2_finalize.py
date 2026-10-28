"""Write the files of reference set v2 (gold2) from the candidate passages: passages.jsonl (no predictions, no annotation), one batch file per language
(id, lang, text, tokens only), the seed's agent annotation kept apart, and a manifest with the checks.

  * the 30 seed passages keep their id, text and tokens; date, author and source_url are added from the source files when they can be read;
  * passages are shuffled inside each language (fixed seed) so that annotators do not see the sources in blocks;
  * every invariant of the brief is checked and written to the manifest (language counts, <= 2 passages per document, <= 15 % per author, 1450-1700,
    120-200 words for the new passages, >= 8 capitalised words, <= 10 % digits, no duplicate id).

    python scripts/ner2_finalize.py --cand cand.jsonl --seed-file gold_seed_v1.jsonl --old-ta SCRATCH/ta --out-dir .../gold2
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ratlas.ner.ner2_passages import CAP_RE, tokenize  # noqa: E402

LANG_ORDER = ['en', 'la', 'it', 'es', 'fr', 'de']
TARGET = {'en': 45, 'la': 45, 'it': 60, 'es': 35, 'fr': 35, 'de': 30}


def year_of(s):
    m = re.search(r'\b(1[4-7]\d\d)\b', str(s or ''))
    return int(m.group(1)) if m else None


def seed_meta(r: dict, old_ta: Path) -> dict:
    """date / author / source_url of a seed passage, read from the archive copies used in the v1 audit."""
    doc, coll = r['doc'], r['collection']
    out = {'author': None, 'date': None, 'year': None, 'date_precision': None, 'source_url': None}
    if coll.startswith('EEBO-TCP'):
        for pat, base in (('eebo/eebotcp_fill/xml', 'eebo'), ('eebo/eebo_phase2/xml', 'eebo2'), ('latin/latineebo/xml', 'eebo')):
            f = old_ta / pat / f'{doc}.xml'
            if f.exists():
                root = etree.parse(str(f)).getroot()
                xp = lambda q: ''.join(root.xpath(q)).strip()
                yr = year_of(xp('string(//*[local-name()="editionStmt"]//*[local-name()="date"])')) or year_of(
                    xp('string(//*[local-name()="sourceDesc"]//*[local-name()="publicationStmt"]/*[local-name()="date"])'))
                out.update(author=re.sub(r'\s+', ' ', xp('string(//*[local-name()="titleStmt"]/*[local-name()="author"])')).strip(' .,') or f'anon:{doc}',
                           date=str(yr) if yr else None, year=yr, date_precision='edition', source_url=f'https://quod.lib.umich.edu/e/{base}/{doc}.0001.001')
                return out
    elif coll == 'Wikisource-IT-Renaissance':
        f = old_ta / 'ws_it' / 'ws_it_renaissance' / f'{doc}.txt'
        if f.exists():
            raw = f.read_text(encoding='utf-8', errors='replace')
            g = lambda tag: (re.search(rf'<dc:{tag}[^>]*>\s*(.*?)\s*</dc:{tag}>', raw) or [None, ''])[1]
            ident = (re.search(r'title=([^&<\s]+)', raw) or [None, doc])[1]
            out.update(author=g('creator') or f'anon:{doc}', date=g('date'), year=year_of(g('date')), date_precision='wikisource_metadata',
                       source_url=f'https://it.wikisource.org/wiki/{ident}')
            return out
    elif coll == 'Galileo-Favaro-Wikisource':
        f = old_ta / 'galileo' / 'Galileo_Opere_Favaro' / 'Galileo_Opere_Favaro.jsonl'
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            if d['filename'] == doc:
                yr = year_of(d.get('year'))
                out.update(author='Galileo Galilei', date=str(yr) if yr else None, year=yr, date_precision='work',
                           source_url='https://it.wikisource.org/wiki/Autore:Galileo_Galilei')
                return out
    return out


# Wikisource metadata dates that are edition dates (outside the period or empty): the work date, set by hand
SEED_DATE_OVERRIDE = {'seed/it/Il_secondo_libro_delle_lettere__Aretino_': ('1542', 1542), 'seed/it/Le_vite_de__pi__eccellenti_pittori__scultori_e_architettori__1568__Accademici_de': ('1568', 1568)}


def check_passage(r: dict, seed: bool) -> list[str]:
    errs = []
    n = len(r['text'].split())
    if not seed and not (120 <= n <= 200):
        errs.append(f'words={n}')
    if not (1450 <= (r.get('year') or 0) <= 1700) and not seed:
        errs.append(f'year={r.get("year")}')
    if sum(c.isdigit() for c in r['text']) / max(1, len(r['text'])) > 0.10:
        errs.append('digits')
    if len(CAP_RE.findall(' ' + r['text'])) < 8:
        errs.append('caps')
    toks, starts = tokenize(r['text'])
    if toks != r['tokens'] or starts != r['token_starts']:
        errs.append('tokens')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True)
    ap.add_argument('--seed-file', required=True)
    ap.add_argument('--old-ta', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=20261021)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    old_ta = Path(a.old_ta)
    cand = [json.loads(l) for l in open(a.cand, encoding='utf-8') if l.strip()]
    seeds = {json.loads(l)['id']: json.loads(l) for l in open(a.seed_file, encoding='utf-8') if l.strip()}
    recs = []
    for r in cand:
        if r.get('seed'):
            s = seeds[r['id']]
            assert s['text'] == r['text'] and s['tokens'] == r['tokens'], r['id']
            m = seed_meta(s, old_ta)
            if s['id'] in SEED_DATE_OVERRIDE:
                m['date'], m['year'] = SEED_DATE_OVERRIDE[s['id']]
                m['date_precision'] = 'work'
            rec = {'id': s['id'], 'lang': s['lang'], 'collection': s['collection'], 'doc': s['doc'], 'title': s['title'], 'author': m['author'], 'date': m['date'],
                   'year': m['year'], 'date_precision': m['date_precision'], 'source_url': m['source_url'], 'licence_note': s['licence_note'], 'seed': True,
                   'text': s['text'], 'tokens': s['tokens'], 'token_starts': s['token_starts'], 'dropcap_repairs': []}
        else:
            rec = {k: r[k] for k in ('id', 'lang', 'collection', 'doc', 'title', 'author', 'date', 'year', 'date_precision', 'source_url', 'licence_note')}
            rec.update(seed=False, text=r['text'], tokens=r['tokens'], token_starts=r['token_starts'], dropcap_repairs=r['dropcap_repairs'])
        recs.append(rec)
    rng = random.Random(a.seed)
    ordered = []
    for lang in LANG_ORDER:
        L = [r for r in recs if r['lang'] == lang]
        L.sort(key=lambda r: r['id'])
        rng.shuffle(L)
        ordered += L
    assert len(ordered) == len(recs)

    errs = {}
    for r in ordered:
        e = check_passage(r, r['seed'])
        if e:
            errs[r['id']] = e
    ids = [r['id'] for r in ordered]
    manifest = {'n_passages': len(ordered), 'n_seed': sum(r['seed'] for r in ordered), 'duplicate_ids': len(ids) - len(set(ids)), 'invariant_errors': errs, 'by_lang': {}}
    for lang in LANG_ORDER:
        L = [r for r in ordered if r['lang'] == lang]
        docs = Counter(r['doc'] for r in L)
        auth = Counter((r['author'] or r['doc']) for r in L if not str(r['author'] or '').startswith('anon:'))
        manifest['by_lang'][lang] = {
            'n': len(L), 'target': TARGET[lang], 'n_seed': sum(r['seed'] for r in L), 'n_docs': len(docs), 'max_per_doc': max(docs.values()),
            'top_author': auth.most_common(1)[0] if auth else None, 'top_author_share': round(auth.most_common(1)[0][1] / len(L), 3) if auth else 0,
            'collections': dict(Counter(r['collection'] for r in L)), 'periods': dict(sorted(Counter(((r['year'] or 0) // 50) * 50 for r in L).items())),
            'tokens': sum(len(r['tokens']) for r in L), 'dropcap_repairs': sum(len(r['dropcap_repairs']) for r in L)}
    with open(out / 'passages.jsonl', 'w', encoding='utf-8') as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for lang in LANG_ORDER:
        with open(out / f'batch_{lang}.jsonl', 'w', encoding='utf-8') as f:
            for r in ordered:
                if r['lang'] == lang:
                    f.write(json.dumps({'id': r['id'], 'lang': r['lang'], 'text': r['text'], 'tokens': r['tokens']}, ensure_ascii=False) + '\n')
    with open(out / 'seed_agent_annotation_v1.jsonl', 'w', encoding='utf-8') as f:
        for sid, s in seeds.items():
            f.write(json.dumps({'id': sid, 'spans': s['spans'], 'annotation': s['annotation']}, ensure_ascii=False) + '\n')
    (out / 'gold2_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
