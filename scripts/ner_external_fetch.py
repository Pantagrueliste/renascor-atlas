"""Download the public NER datasets (official sources only) into ~/renascor-atlas-data/ner/external/<name>/raw and write SOURCES.tsv.

    python3 -I scripts/ner_external_fetch.py [--root DIR] [--only name1,name2] [--list]

Runs on a Hypatia compute node (Slurm: slurm/ner_external_fetch.sbatch). Standard library only.
Sources: GitHub (tarball of a pinned commit from codeload.github.com), the Hugging Face Hub (files of a pinned revision via /resolve/), Zenodo (files
of a record, md5 checked). Before every host a robots.txt check is made; a path that robots.txt disallows is skipped and recorded.
The data are UNTRUSTED: each dataset has its own directory, archives are extracted with tarfile filter='data' / zipfile with a path check, nothing is run.

SOURCES.tsv (per dataset and merged in <root>/SOURCES.tsv): name, url, retrieved, revision, licence, licence_flag, sha256, bytes, robots, note.
licence_flag: open (CC0 / CC BY / MIT / Apache / public domain), nc (non-commercial), sa (share-alike), nc-sa, gpl (AGPL / GPL), unstated, synthetic (silver
or LLM labels).  A later published model must drop everything that is not "open" (see external_survey.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = 'renascor-atlas-research/0.1 (private research; no contact)'

# WikiANN languages kept for conversion (all are downloaded: 142 MB). European languages, Latin, Greek and a few historical-press languages.
WIKIANN_LANGS = ['la', 'en', 'fr', 'de', 'it', 'es', 'nl', 'pt', 'ca', 'pl', 'cs', 'sk', 'sl', 'hr', 'sr', 'hu', 'ro', 'fi', 'sv', 'da', 'no', 'nn', 'is',
                 'et', 'lv', 'lt', 'ru', 'uk', 'bg', 'el', 'eu', 'ga', 'cy', 'lb', 'gl', 'oc', 'br', 'af', 'sq', 'tr', 'he', 'yi', 'sco', 'fy', 'li', 'vls',
                 'nds', 'als', 'bar', 'hsb', 'csb', 'be', 'mk', 'ast', 'an', 'sh']

SOURCES = {
    # ---- historical ----------------------------------------------------------------------------------------------------------------------
    'hipe2022': dict(kind='github', repo='hipe-eval/HIPE-2022-data', ref='147f5bc3c7fb7e5c6b024a9ffd6503cd019fb9ea',
                     licence='CC BY-NC-SA 4.0 (hipe2020, letemps, topres19th); CC BY 4.0 (newseye, sonar, ajmc)', flag='nc-sa',
                     note='v2.1 files incl. unmasked test sets; per-dataset licences in documentation/README-*.md'),
    'europeana': dict(kind='github', repo='EuropeanaNewspapers/ner-corpora', ref=None, licence='CC0 1.0', flag='open',
                      note='Neudecker 2016, nl (kb), fr (bnf), de (lft, onb, sbb), 1840-1930 newspapers, OCR'),
    'reichsanzeiger': dict(kind='github', repo='UB-Mannheim/reichsanzeiger-nlp', ref=None, licence='CC0 1.0 (GitHub licence field; LICENSE file)', flag='open',
                           note='curated UIMA XMI, 19th c. German newspaper, work in progress'),
    'kind': dict(kind='github', repo='dhfbk/KIND', ref=None, licence='CC BY-NC 4.0 (wikinews, fiction, moro); CC BY-NC-SA 4.0 (De Gasperi)', flag='nc-sa',
                 note='Italian; De Gasperi 1901-1954 and Moro are the historical chapters; Moro is silver'),
    'geoedda': dict(kind='hf', hf_type='datasets', repo='GEODE/GeoEDdA-NER', files=['train.jsonl', 'val.jsonl', 'test.jsonl', 'README.md'],
                    licence='CC BY-NC 4.0', flag='nc', note='18th c. French Encyclopedie; NP-Spatial -> LOC, NP-Person -> PER'),
    'globalise': dict(kind='hf', hf_type='datasets', repo='globalise/globalise_NER_token_classification_dataset',
                      files=['train.csv', 'validation.csv', 'test.csv', 'README.md'], licence='MIT', flag='open',
                      note='early modern Dutch, VOC letters 17th-18th c., 15 fine labels'),
    'earlymodern_german': dict(kind='github', repo='O-leChuck/NER-eval-corpus_early-modern-high-German', ref=None, licence='none stated in the repository', flag='unstated',
                               note='evaluation corpus, 16th-17th c. ego-documents, 24k tokens; use for TEST ONLY, do not redistribute'),
    'sdhk_old_swedish': dict(kind='github', repo='phenningsson/sdhk-ner-old-swedish', ref=None, licence='CC BY 4.0 (data), GPL-3.0 (code)', flag='open',
                             note='Old Swedish charters 1380-1382, silver by projection (training); expert_gold_test is gold'),
    'voices_funeral': dict(kind='github', repo='RYFoR/VOICES-FuneralEntries-NER', ref=None, licence='CC BY 4.0 (README badge)', flag='open',
                           note='early modern Irish funeral entries, English, HTR; small'),
    'medieval_charters': dict(kind='zenodo', record='6463699', licence='CC BY 4.0', flag='open', files={'Multilingual_NER_medieval_charters.zip': 'md5:d4eb6cb69d5a9d374d6b60b69fd157eb'},
                              revision='zenodo 6463699 v1 2022-04-15 doi 10.5281/zenodo.6463699',
                              note='Torres Aguilar 2022: Latin, French, Spanish charters 10th-15th c.; one 15 GB zip with datasets, embeddings, models'),
    # ---- Latin ---------------------------------------------------------------------------------------------------------------------------
    'latin_herodotos': dict(kind='github', repo='Herodotos-Project/Herodotos-Project-Latin-NER-Tagger-Annotation', ref=None, licence='AGPL-3.0', flag='gpl',
                            note='Erdmann 2016; GEO / PRS / GRP (peoples, dropped); Caesar, Pliny, Ovid; train/test as in the paper'),
    'latin_lasla': dict(kind='github', repo='NER-AncientLanguages/Ner-Latin-RANLP', ref=None, licence='CC BY-NC-SA (data), GPL (code)', flag='nc-sa',
                        note='gold: Cicero Phil. 1, Juvenal 1-3, Tacitus Hist. 1; silver folder is not converted'),
    'latincy_ner': dict(kind='github', repo='latincy/latincy-ner', ref=None, licence='CC BY-NC-SA 4.0', flag='nc-sa',
                        note='LatinCy curated NER, primer album (Ritchie, Sonnenschein)'),
    # ---- modern / aggregate --------------------------------------------------------------------------------------------------------------
    'wikineural': dict(kind='hf', hf_type='datasets', repo='Babelscape/wikineural', files=['README.md'], patterns=['data/'],
                       licence='CC BY-NC-SA 4.0', flag='nc-sa', note='silver (Wikipedia + BERT/WordNet, Tedeschi 2021); de en es fr it nl pl pt ru'),
    'multinerd': dict(kind='hf', hf_type='datasets', repo='Babelscape/multinerd', files=['README.md'], patterns=['train/', 'val/', 'test/'],
                      licence='CC BY-NC-SA 4.0', flag='nc-sa', note='silver (Wikipedia/Wikipedia-derived, Tedeschi 2022), 15 fine types, 10 languages'),
    'wikiann': dict(kind='hf', hf_type='datasets', repo='unimelb-nlp/wikiann', files=['README.md'], patterns=[f'{l}/' for l in WIKIANN_LANGS],
                    licence='unknown on the Hub card; PAN-X / WikiANN (Pan 2017) is distributed with a non-commercial note; xtreme card says Apache-2.0',
                    flag='unstated', note='silver (Wikipedia links), PER / LOC / ORG, short titles-like sentences'),
    'pile_ner': dict(kind='hf', hf_type='datasets', repo='Universal-NER/Pile-NER-type', files=['train.json', 'README.md'],
                     licence='none on the card; labels made by ChatGPT (OpenAI terms) over Pile passages', flag='synthetic',
                     note='GLiNER training data; conversation format, spans are recovered by string search'),
    'pile_mistral': dict(kind='hf', hf_type='datasets', repo='urchade/pile-mistral-v0.1', files=['data.json', 'README.md'], licence='Apache-2.0 (card); labels by Mistral-7B over Pile passages',
                         flag='synthetic', note='training data of urchade/gliner_multi-v2.1; GLiNER training format, positive spans only'),
    'gliner_multi_synth': dict(kind='hf', hf_type='datasets', repo='knowledgator/gliner-multilingual-synthetic', files=['european_dataset.json'],
                               licence='none on the card', flag='synthetic', note='training data of knowledgator/gliner-x-*; European languages, positive spans only, no language field'),
    'nuner': dict(kind='hf', hf_type='datasets', repo='numind/NuNER', files=['README.md', 'data/full-00001-of-00001.csv'], licence='MIT (card); labels by an LLM over C4 / Pile',
                  flag='synthetic', note='training data of numind/NuNER_Zero; English; 3.2 M free-text type descriptions ("individual person", "city in texas"): fetched, NOT converted, mapping to LOC / PER would drop most persons'),
    'hist_pile_ner': dict(kind='hf', hf_type='datasets', repo='emanuelaboros/Hist-Pile-NER-Type', files=['README.md', 'spans/train.parquet', 'hipe_iob/statistics.json'],
                          patterns=['hipe_iob/train-'], licence='none on the card (text from biglam/europeana_newspapers, CC0; labels by gpt-5-mini)',
                          flag='synthetic', note='silver; 12 languages incl. el et fi fr hr pl ru sr sv uk yi'),
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


_robots: dict[str, tuple[list[tuple[str, bool]], float] | None] = {}
_last: dict[str, float] = {}


def _parse_robots(text: str) -> tuple[list[tuple[str, bool]], float]:
    """Rules of the 'User-agent: *' group as (pattern, allow) and its Crawl-delay.  Google semantics: longest matching pattern wins, Allow wins a tie
    (urllib.robotparser applies the first match and has no wildcards, which wrongly blocks Zenodo's explicit 'Allow: /api/records/*/files')."""
    rules, delay, group_star, in_rules = [], 0.0, False, False
    for line in text.splitlines():
        line = line.split('#')[0].strip()
        if ':' not in line:
            continue
        k, v = (x.strip() for x in line.split(':', 1))
        k = k.lower()
        if k == 'user-agent':
            if in_rules:
                group_star, in_rules = False, False
            group_star = group_star or v == '*'
        elif group_star:
            in_rules = True
            if k in ('allow', 'disallow') and v:
                rules.append((v, k == 'allow'))
            elif k == 'crawl-delay':
                try:
                    delay = float(v)
                except ValueError:
                    pass
    return rules, delay


def _match(pat: str, path: str) -> bool:
    import re
    rx = re.escape(pat).replace(r'\*', '.*')
    if rx.endswith(r'\$'):
        rx = rx[:-2] + '$'
    return re.match(rx, path) is not None


def robots_ok(url: str) -> str:
    """'ok', 'no-robots' (none published = allowed), or 'DISALLOWED'.  Also sleeps the host's Crawl-delay between two requests."""
    u = urllib.parse.urlsplit(url)
    host = u.netloc
    if host not in _robots:
        try:
            req = urllib.request.Request(f'https://{host}/robots.txt', headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                _robots[host] = _parse_robots(r.read().decode('utf-8', 'replace'))
        except Exception:  # noqa: BLE001
            _robots[host] = None
    rb = _robots[host]
    if rb is None:
        return 'no-robots'
    rules, delay = rb
    if delay and time.time() - _last.get(host, 0) < delay:
        time.sleep(delay)
    _last[host] = time.time()
    path = u.path + (('?' + u.query) if u.query else '')
    best = None
    for pat, allow in rules:
        if _match(pat, path) and (best is None or len(pat) > len(best[0]) or (len(pat) == len(best[0]) and allow)):
            best = (pat, allow)
    return 'ok' if best is None or best[1] else 'DISALLOWED'


def http_json(url: str):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url: str, dest: Path, tries: int = 4) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            h = hashlib.sha256()
            n = 0
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    n += len(b)
            return h.hexdigest(), n
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (k + 1))
    raise RuntimeError(f'download failed {url}: {err}')


def safe_extract_tar(tar: Path, out: Path) -> int:
    n = 0
    with tarfile.open(tar) as t:
        t.extractall(out, filter='data')
        n = len(t.getmembers())
    return n


def hf_tree(hf_type: str, repo: str, rev: str) -> list[dict]:
    out, url = [], f'https://huggingface.co/api/{hf_type}/{repo}/tree/{rev}?recursive=true'
    return [x for x in http_json(url) if x['type'] == 'file']


def fetch_one(name: str, spec: dict, root: Path) -> list[list[str]]:
    base = root / name
    raw = base / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []

    def rec(url, sha, size, rev, robots, note=''):
        rows.append([name, url, now(), rev, spec['licence'], spec['flag'], sha, str(size), robots, note])

    kind = spec['kind']
    if kind == 'github':
        repo = spec['repo']
        meta = http_json(f'https://api.github.com/repos/{repo}')
        ref = spec.get('ref') or http_json(f'https://api.github.com/repos/{repo}/commits/{meta["default_branch"]}')['sha']
        url = f'https://codeload.github.com/{repo}/tar.gz/{ref}'
        rb = robots_ok(url)
        if rb == 'DISALLOWED':
            rec(url, '', 0, ref, rb, 'skipped')
            return rows
        tarp = base / f'{repo.split("/")[1]}-{ref[:10]}.tar.gz'
        sha, size = download(url, tarp)
        n = safe_extract_tar(tarp, raw)
        rec(url, sha, size, ref, rb, f'{spec["note"]} | github licence field: {(meta.get("license") or {}).get("spdx_id")} | {n} members')
    elif kind == 'hf':
        repo, t = spec['repo'], spec['hf_type']
        rev = http_json(f'https://huggingface.co/api/{t}/{repo}')['sha']
        tree = hf_tree(t, repo, rev)
        want = []
        for x in tree:
            p = x['path']
            if p in spec.get('files', []) or any(p.startswith(pre) for pre in spec.get('patterns', [])):
                want.append(x)
        for x in want:
            url = f'https://huggingface.co/{t}/{repo}/resolve/{rev}/{urllib.parse.quote(x["path"])}'
            rb = robots_ok(url)
            if rb == 'DISALLOWED':
                rec(url, '', 0, rev, rb, 'skipped')
                continue
            sha, size = download(url, raw / x['path'])
            rec(url, sha, size, rev, rb, spec['note'] if x['path'] == want[0]['path'] else '')
    elif kind == 'zenodo':
        # robots.txt of zenodo.org disallows /api except /api/records/*/files: the record metadata is NOT queried, name and md5 are in the spec
        for key, checksum in spec['files'].items():
            url = f'https://zenodo.org/api/records/{spec["record"]}/files/{urllib.parse.quote(key)}/content'
            rb = robots_ok(url)
            if rb == 'DISALLOWED':
                rec(url, '', 0, spec['revision'], rb, 'skipped')
                continue
            sha, size = download(url, raw / key)
            algo, _, want = checksum.partition(':')
            h = hashlib.new(algo)
            with open(raw / key, 'rb') as fh:
                for b in iter(lambda: fh.read(1 << 22), b''):
                    h.update(b)
            ok = 'md5 ok' if h.hexdigest() == want else f'CHECKSUM MISMATCH {h.hexdigest()} != {want}'
            rec(url, sha, size, spec['revision'], rb, f'{spec["note"]} | {ok}')
    else:
        raise ValueError(kind)
    hdr = ['name', 'url', 'retrieved', 'revision', 'licence', 'licence_flag', 'sha256', 'bytes', 'robots', 'note']
    with open(base / 'SOURCES.tsv', 'w') as f:
        f.write('\t'.join(hdr) + '\n')
        for r in rows:
            f.write('\t'.join(c.replace('\t', ' ').replace('\n', ' ') for c in r) + '\n')
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path.home() / 'renascor-atlas-data/ner/external'))
    ap.add_argument('--only', default='')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, v in SOURCES.items():
            print(k, v['kind'], v['flag'], v['licence'][:60], sep='\t')
        return 0
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    names = [n for n in a.only.split(',') if n] or list(SOURCES)
    failed = []
    for n in names:
        print(f'[{now()}] {n}', flush=True)
        try:
            rows = fetch_one(n, SOURCES[n], root)
            print(f'   {len(rows)} file(s), {sum(int(r[7]) for r in rows) / 1e6:.1f} MB', flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f'   FAILED {n}: {e}', file=sys.stderr, flush=True)
    # merge
    hdr = ['name', 'url', 'retrieved', 'revision', 'licence', 'licence_flag', 'sha256', 'bytes', 'robots', 'note']
    with open(root / 'SOURCES.tsv', 'w') as out:
        out.write('\t'.join(hdr) + '\n')
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / 'SOURCES.tsv'
            if f.exists():
                out.write(''.join(f.read_text().splitlines(True)[1:]))
    print('failed:', failed)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
