"""Place cards, part 2: free-licensed Commons thumbnails for the places with a P18 image (network; Slurm on Hypatia).

    python -m ratlas.cards_images --wd wd_places.parquet --work DIR --img DIR --tsv wd_images.tsv
                                  [--top 2500] [--width 320] [--quality 70] [--reencode-only]

1. Take the ``--top`` places with a P18 and in-period mentions, most mentioned first.
2. Commons API (``prop=imageinfo&iiprop=url|extmetadata|size|mime&iiurlwidth=W``, 50 titles per call, 1 request/s,
   cached): thumbnail URL, artist (HTML stripped), licence short name / URL, attribution required, file page.
3. Keep only free licences (public domain, CC0, CC BY, CC BY-SA, GFDL, FAL, OGL ...). Never non-free, NC, ND or unknown.
4. Download the thumbnail only (sequential, 1 request/s, back-off on 429), keep the bytes in ``--work/raw``, write a
   WebP (quality ``--quality``) to ``--img/<QID>.webp`` (JPEG when Pillow is missing).
5. ``--tsv``: qid, file, local, bytes, artist, licence, licence_url, attribution_required, file_page (+ a
   ``<tsv>.skipped.tsv`` with the reason for every place left out).

Resumable: Commons answers are cached, raw thumbnails and finished images are not fetched twice.
``--reencode-only`` rebuilds the images from ``--work/raw`` (e.g. at a smaller ``--width``) without any network.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .wikidata import UA, WD

COMMONS = 'https://commons.wikimedia.org/w/api.php'
BATCH = 50
FREE_RE = re.compile(
    r'^(public domain|pd\b|pd-|cc0|cc[- ]by(?![- ]?(nc|nd))|cc[- ]sa|gfdl|gnu free|free art|fal\b|ogl|open government|'
    r'attribution|copyrighted free use|no restrictions|licence ouverte|licen[cs]e ouverte|etalab|bild-pd|'
    r'cc[- ]zero|dedicated to the public domain|text is available under)', re.I)
NONFREE_RE = re.compile(r'(\bnc\b|non[- ]?commercial|\bnd\b|no[- ]?deriv|fair use|non[- ]?free|all rights reserved|'
                        r'unknown|permission|\bcopyright\b)', re.I)
TAG_RE = re.compile(r'<[^>]+>')


def clean(s, n=300):
    if not s:
        return ''
    s = html.unescape(TAG_RE.sub(' ', s))
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:n]


def is_free(meta: dict) -> tuple[bool, str]:
    """(free?, reason). ``meta`` = extmetadata reduced to plain strings."""
    short = clean(meta.get('LicenseShortName', ''))
    if (meta.get('NonFree') or '').lower() == 'true':
        return False, 'nonfree'
    if not short:
        return False, 'no licence'
    if NONFREE_RE.search(short):
        return False, f'licence {short}'
    if not FREE_RE.search(short):
        return False, f'licence {short}'
    return True, ''


def get_commons(wd: WD, titles: list[str]) -> dict:
    """Cached imageinfo call. Returns {'pages': {title: info}}, titles resolved through normalisation + redirects."""
    params = {'action': 'query', 'prop': 'imageinfo', 'iiprop': 'url|extmetadata|size|mime',
              'iiurlwidth': str(wd.width), 'redirects': '1', 'titles': '|'.join(titles), 'format': 'json'}
    key = urllib.parse.urlencode(sorted(params.items()))
    p = wd._path('commons', key)
    got = wd._get_cached(p)
    if got is not None:
        return got
    req = urllib.request.Request(COMMONS + '?' + key, headers={'User-Agent': UA, 'Accept': 'application/json'})
    resp = wd._fetch(req)
    q = resp.get('query', {})
    back = {}
    for lst in (q.get('normalized', []), q.get('redirects', [])):
        for m in lst:
            back[m['from']] = m['to']
    pages = {}
    for pg in (q.get('pages') or {}).values():
        ii = (pg.get('imageinfo') or [None])[0]
        if not ii:
            pages[pg.get('title')] = None
            continue
        em = {k: (v.get('value') if isinstance(v, dict) else v) for k, v in (ii.get('extmetadata') or {}).items()
              if k in ('Artist', 'Credit', 'LicenseShortName', 'LicenseUrl', 'AttributionRequired', 'NonFree',
                       'Copyrighted', 'UsageTerms')}
        pages[pg['title']] = {'thumburl': ii.get('thumburl'), 'thumbwidth': ii.get('thumbwidth'), 'url': ii.get('url'),
                              'descriptionurl': ii.get('descriptionurl'), 'size': ii.get('size'), 'mime': ii.get('mime'),
                              'width': ii.get('width'), 'meta': em}
    out = {'pages': pages, 'map': back}
    wd._put(p, key, out)
    return out


def resolve(resp: dict, title: str):
    t = title
    for _ in range(4):
        if t in resp['pages']:
            return resp['pages'][t]
        if t in resp['map']:
            t = resp['map'][t]
        else:
            break
    return resp['pages'].get(t)


class Downloader:
    def __init__(self, min_interval=1.1, max_tries=7, log=print):
        self.min_interval, self.max_tries, self.log = min_interval, max_tries, log
        self._last = 0.0

    def get(self, url: str) -> bytes:
        delay = 5.0
        for attempt in range(self.max_tries):
            w = self.min_interval - (time.time() - self._last)
            if w > 0:
                time.sleep(w)
            self._last = time.time()
            try:
                req = urllib.request.Request(url, headers={'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=90) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get('Retry-After') if e.headers else None
                    try:
                        wt = max(float(ra), delay) if ra else delay
                    except ValueError:
                        wt = delay
                    self.log(f'  HTTP {e.code}, retry in {wt:.0f}s')
                    time.sleep(min(wt, 300))
                    delay = min(delay * 2, 300)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                self.log(f'  network error {e!r}, retry in {delay:.0f}s')
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise RuntimeError('too many failed attempts: ' + url)


def encode(raw: bytes, width: int, quality: int):
    """-> (bytes, ext). WebP through Pillow, else the raw bytes as .jpg."""
    try:
        from PIL import Image
    except ImportError:
        return raw, 'jpg'
    im = Image.open(io.BytesIO(raw))
    im.load()
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
    im = im.convert('RGBA' if im.mode in ('RGBA', 'LA', 'P') and 'transparency' in im.info or im.mode in ('RGBA', 'LA') else 'RGB')
    buf = io.BytesIO()
    im.save(buf, 'WEBP', quality=quality, method=6)
    return buf.getvalue(), 'webp'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--wd', required=True, type=Path)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--img', required=True, type=Path)
    ap.add_argument('--tsv', required=True, type=Path)
    ap.add_argument('--cache', type=Path, help='cache root (default: --work/cache)')
    ap.add_argument('--top', type=int, default=2500)
    ap.add_argument('--width', type=int, default=320)
    ap.add_argument('--quality', type=int, default=70)
    ap.add_argument('--reencode-only', action='store_true')
    ap.add_argument('--src-width', type=int, default=320, help='width of the cached metadata/raw thumbnails (reencode)')
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    rows = pq.read_table(a.wd, columns=['qid', 'image', 'n_in_period']).to_pylist()
    cand = [r for r in rows if r['image'] and (r['n_in_period'] or 0) > 0]
    cand.sort(key=lambda r: (-r['n_in_period'], int(r['qid'][1:])))
    cand = cand[:a.top]
    print(f'{sum(1 for r in rows if r["image"])} places with P18, {len(cand)} selected', flush=True)
    raw_dir = a.work / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    a.img.mkdir(parents=True, exist_ok=True)
    meta_path = a.work / f'meta_w{a.src_width if a.reencode_only else a.width}.json'
    wd = WD(a.cache or (a.work / 'cache'), min_interval=1.1)
    wd.width = a.width

    # -- Commons metadata ----------------------------------------------------------------------------------------
    info = {}
    if a.reencode_only:
        info = json.loads(meta_path.read_text('utf-8'))
    else:
        titles = {r['qid']: 'File:' + r['image'].replace('_', ' ') for r in cand}
        qids = [r['qid'] for r in cand]
        for i in range(0, len(qids), BATCH):
            ch = qids[i:i + BATCH]
            resp = get_commons(wd, [titles[q] for q in ch])
            for q in ch:
                info[q] = {'title': titles[q], 'info': resolve(resp, titles[q])}
            print(f'  commons {min(i + BATCH, len(qids))}/{len(qids)} (net {wd.n_net}, cached {wd.n_cached})', flush=True)
        meta_path.write_text(json.dumps(info, ensure_ascii=False), 'utf-8')

    # -- licence filter + download ---------------------------------------------------------------------------------
    dl = Downloader()
    kept, skipped = [], []
    done = 0
    for r in cand:
        q = r['qid']
        it = info.get(q)
        if not it or not it['info']:
            skipped.append((q, r['image'], 'no imageinfo'))
            continue
        ii, meta = it['info'], it['info']['meta']
        ok, why = is_free(meta)
        if not ok:
            skipped.append((q, r['image'], why))
            continue
        mime = ii.get('mime') or ''
        if not mime.startswith('image/') or not ii.get('thumburl'):
            skipped.append((q, r['image'], f'mime {mime}'))
            continue
        rawp = raw_dir / f'{q}.bin'
        if not rawp.exists():
            if a.reencode_only:
                skipped.append((q, r['image'], 'no raw'))
                continue
            try:
                rawp.write_bytes(dl.get(ii['thumburl']))
            except Exception as e:  # keep going; the reason is reported
                skipped.append((q, r['image'], f'download {e}'))
                continue
        try:
            data, ext = encode(rawp.read_bytes(), a.width, a.quality)
        except Exception as e:
            skipped.append((q, r['image'], f'decode {e!r}'[:80]))
            continue
        for old in a.img.glob(f'{q}.*'):
            old.unlink()
        out = a.img / f'{q}.{ext}'
        out.write_bytes(data)
        kept.append({'qid': q, 'file': it['title'][5:], 'local': out.name, 'bytes': len(data),
                     'artist': clean(meta.get('Artist', '')) or clean(meta.get('Credit', '')),
                     'licence': clean(meta.get('LicenseShortName', ''), 80),
                     'licence_url': clean(meta.get('LicenseUrl', ''), 300),
                     'attribution_required': (meta.get('AttributionRequired') or '').lower(),
                     'file_page': ii.get('descriptionurl') or ''})
        done += 1
        if done % 100 == 0:
            print(f'  {done} images done', flush=True)

    cols = ['qid', 'file', 'local', 'bytes', 'artist', 'licence', 'licence_url', 'attribution_required', 'file_page']
    with open(a.tsv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(cols)
        for k in kept:
            w.writerow([str(k[c]).replace('\t', ' ').replace('\n', ' ') for c in cols])
    sk = a.tsv.with_suffix('.skipped.tsv')
    with open(sk, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['qid', 'file', 'reason'])
        w.writerows(skipped)
    from collections import Counter
    reasons = Counter(re.sub(r'[:].*', '', s[2])[:30] for s in skipped)
    lic = Counter(k['licence'] for k in kept)
    total = sum(k['bytes'] for k in kept)
    print(json.dumps({'selected': len(cand), 'kept': len(kept), 'skipped': len(skipped), 'total_bytes': total,
                      'mean_bytes': total // max(1, len(kept)), 'width': a.width,
                      'skip_reasons': reasons.most_common(15), 'licences': lic.most_common(15)}, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
