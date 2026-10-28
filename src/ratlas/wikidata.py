"""Polite, cached Wikidata client (stdlib only).

* WDQS  https://query.wikidata.org/sparql   (POST, SPARQL JSON)
* API   https://www.wikidata.org/w/api.php  (GET, JSON)

Rules (stage "reconcile"): plain User-Agent without an e-mail address, at most one request per second, exponential
back-off on 429/5xx (Retry-After honoured), every response cached on disk (key = sha1 of the request), so reruns are
free and offline. Nothing else is fetched.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

UA = 'Renascor-Atlas/0.1 (research; https://pantagrueliste.github.io/renascor)'
WDQS = 'https://query.wikidata.org/sparql'
API = 'https://www.wikidata.org/w/api.php'
DEFAULT_CACHE = Path('/Users/clem/Projects/Papers/RQ/derived/atlas/cache/wikidata')


class WD:
    def __init__(self, cache: Path = DEFAULT_CACHE, min_interval: float = 1.1, max_tries: int = 7,
                 offline: bool = False, log=print):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_tries = max_tries
        self.offline = offline
        self.log = log
        self._last = 0.0
        self.n_net = 0
        self.n_cached = 0

    # -- cache ------------------------------------------------------------------------------------------------------
    def _path(self, kind: str, key: str) -> Path:
        h = hashlib.sha1(key.encode('utf-8')).hexdigest()
        d = self.cache / kind / h[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f'{h}.json'

    def _get_cached(self, p: Path):
        if p.exists():
            try:
                self.n_cached += 1
                return json.loads(p.read_text('utf-8'))['response']
            except Exception:
                self.n_cached -= 1
        return None

    def _put(self, p: Path, key: str, resp):
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps({'request': key, 'response': resp}, ensure_ascii=False), 'utf-8')
        tmp.replace(p)

    # -- network ----------------------------------------------------------------------------------------------------
    def _fetch(self, req: urllib.request.Request):
        delay = 5.0
        for attempt in range(self.max_tries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    self.n_net += 1
                    return json.loads(r.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get('Retry-After') if e.headers else None
                    try:
                        w = max(float(ra), delay) if ra else delay
                    except ValueError:
                        w = delay
                    self.log(f'  HTTP {e.code}, retry in {w:.0f}s (attempt {attempt + 1})')
                    time.sleep(min(w, 300))
                    delay = min(delay * 2, 300)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                self.log(f'  network error {e!r}, retry in {delay:.0f}s (attempt {attempt + 1})')
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise RuntimeError('too many failed attempts')

    def sparql(self, query: str) -> dict:
        p = self._path('sparql', query)
        got = self._get_cached(p)
        if got is not None:
            return got
        if self.offline:
            raise RuntimeError('offline and not cached')
        data = urllib.parse.urlencode({'query': query, 'format': 'json'}).encode()
        req = urllib.request.Request(WDQS, data=data, headers={
            'User-Agent': UA, 'Accept': 'application/sparql-results+json',
            'Content-Type': 'application/x-www-form-urlencoded'})
        resp = self._fetch(req)
        self._put(p, query, resp)
        return resp

    def api(self, params: dict, reduce=None) -> dict:
        """GET the API; ``reduce`` (a pure function) shrinks the response BEFORE it is cached (full claims are large)."""
        params = dict(sorted(params.items())) | {'format': 'json'}
        key = urllib.parse.urlencode(params)
        p = self._path('api', key)
        got = self._get_cached(p)
        if got is not None:
            return got
        if self.offline:
            raise RuntimeError('offline and not cached')
        req = urllib.request.Request(API + '?' + key, headers={'User-Agent': UA, 'Accept': 'application/json'})
        resp = self._fetch(req)
        if 'error' in resp:
            raise RuntimeError(f"API error: {resp['error'].get('info', resp['error'])}")
        if reduce is not None:
            resp = reduce(resp)
        self._put(p, key, resp)
        return resp


def bindings(resp: dict) -> list[dict]:
    """SPARQL JSON -> list of {var: value-string}."""
    out = []
    for b in resp.get('results', {}).get('bindings', []):
        out.append({k: v.get('value') for k, v in b.items()})
    return out


def qid_of(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    t = uri.rsplit('/', 1)[-1]
    return t if t[:1] in 'QP' and t[1:].isdigit() else None
