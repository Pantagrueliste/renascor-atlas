"""Scan of the re-harvested folders (originals_2026-10-17): same records as the study scan.jsonl.gz, same schema.

Copy of RQ/docs/etudes_2026-10-16/scan.py with sub-folder selection. Usage:
    python scan_extra.py <raw_root> <out.jsonl> <subdir> [<subdir> ...]
The ``coll`` of a record is ``originals_2026-10-17/<folder>``. ``dup`` is computed inside this scan only.
"""
import hashlib, io, json, os, re, sys, tarfile, zipfile, gzip
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from lxml import etree

ROOT = sys.argv[1]            # raw root (RQ/data/raw or ~/renascor-atlas-data/raw)
OUT = sys.argv[2]             # output .jsonl (gzip it afterwards)
SUBDIRS = sys.argv[3:]        # sub-folders of ROOT to scan, e.g. originals_2026-10-17/cavriana
WORKERS = int(os.environ.get('SCAN_WORKERS', '8'))
MAX_BYTES = 400 * 1024 * 1024

ENT = {'persName', 'placeName', 'orgName', 'geogName', 'settlement', 'country', 'region', 'district',
       'bloc', 'name', 'rs', 'roleName', 'forename', 'surname', 'addName'}
ENT_REFD = {'persName', 'placeName', 'orgName', 'geogName', 'settlement', 'name', 'rs', 'country', 'region'}
OTHER = {'date', 'term', 'bibl', 'quote', 'cit', 'w', 'choice', 'reg', 'orig', 'expan', 'abbr', 'corr', 'sic',
         'correspDesc', 'correspAction', 'note', 'pb', 'lb', 'graphic', 'facsimile', 's', 'lg', 'l', 'sp',
         'speaker', 'letter', 'opener', 'closer', 'dateline', 'salute', 'signed', 'title', 'idno',
         'person', 'place', 'org', 'event', 'listPerson', 'listPlace', 'listOrg', 'listBibl', 'listEvent',
         'standOff', 'geo', 'location', 'birth', 'death', 'occupation', 'measure', 'num', 'foreign'}
REG_CONTAINERS = {'standOff', 'listPerson', 'listPlace', 'listOrg', 'listEvent', 'listBibl', 'particDesc', 'settingDesc'}

AUTH = [
    ('gnd', re.compile(r'd-nb\.info/gnd|\bgnd[:_/]|/gnd/|lobid\.org/gnd', re.I)),
    ('viaf', re.compile(r'viaf', re.I)),
    ('wikidata', re.compile(r'wikidata|^Q\d+$|\bwd:Q', re.I)),
    ('geonames', re.compile(r'geonames', re.I)),
    ('tgn', re.compile(r'getty\.edu/(page/)?tgn|\btgn[:_]', re.I)),
    ('ulan', re.compile(r'getty\.edu/(page/)?ulan|\bulan[:_]', re.I)),
    ('pleiades', re.compile(r'pleiades', re.I)),
    ('loc', re.compile(r'id\.loc\.gov', re.I)),
    ('isni', re.compile(r'isni', re.I)),
    ('dbpedia', re.compile(r'dbpedia', re.I)),
    ('wikipedia', re.compile(r'wikipedia\.org', re.I)),
    ('cerl', re.compile(r'cerl\.org', re.I)),
    ('bnf', re.compile(r'data\.bnf|catalogue\.bnf|ark:/12148', re.I)),
    ('deutsche-biographie', re.compile(r'deutsche-biographie', re.I)),
    ('hls', re.compile(r'hls-dhs-dss', re.I)),
    ('whg', re.compile(r'whgazetteer', re.I)),
]


def classify(v):
    out = []
    for tok in v.split():
        k = None
        for name, rx in AUTH:
            if rx.search(tok):
                k = name
                break
        if k is None:
            if tok.startswith('#'):
                k = 'local#'
            elif tok.startswith('http'):
                m = re.match(r'https?://([^/]+)', tok)
                k = 'http:' + (m.group(1).lower() if m else '?')
            elif ':' in tok:
                k = 'prefix:' + tok.split(':', 1)[0][:20].lower()
            elif re.search(r'\.xml', tok):
                k = 'file'
            else:
                k = 'bare'
        out.append(k)
    return out


def ln(tag):
    if not isinstance(tag, str):
        return None
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def scan_bytes(data):
    r = {'bytes': len(data)}
    c = Counter()          # zone|tag counts
    refs = Counter()       # tag|attr|class counts
    types = Counter()      # tag|@type values for name/rs/date
    idno = Counter()       # idno/@type inside person/place/org
    langs = Counter()
    dates = []
    words = 0
    geo = 0
    ent_text_with_ref = 0
    ent_text = 0
    root = None
    stack = []             # zone markers
    try:
        ctx = etree.iterparse(io.BytesIO(data), events=('start', 'end'), recover=True, huge_tree=True,
                              resolve_entities=False, no_network=True, load_dtd=False)
        hdr = txt = reg = 0
        in_reg_entity = 0
        for ev, el in ctx:
            t = ln(el.tag)
            if t is None:
                continue
            if ev == 'start':
                if root is None:
                    root = t
                    r['ns'] = el.tag.split('}')[0][1:] if '}' in el.tag else ''
                if t == 'teiHeader':
                    hdr += 1
                elif t == 'text' and not hdr:
                    txt += 1
                if t in REG_CONTAINERS:
                    reg += 1
                zone = 'reg' if reg else ('hdr' if hdr else ('txt' if txt else 'other'))
                stack.append(t)
                if t in ENT or t in OTHER:
                    c[zone + '|' + t] += 1
                if t in ENT_REFD:
                    has = False
                    for a in ('ref', 'key', 'corresp', 'sameAs'):
                        v = el.get(a)
                        if v:
                            has = True
                            for k in classify(v):
                                refs[f'{zone}|{t}|{a}|{k}'] += 1
                    if zone == 'txt':
                        ent_text += 1
                        ent_text_with_ref += has
                    ty = el.get('type')
                    if ty and t in ('name', 'rs', 'placeName', 'persName', 'orgName'):
                        types[f'{t}|{ty[:30]}'] += 1
                if t == 'date' and zone == 'txt':
                    if el.get('when') or el.get('from') or el.get('notBefore'):
                        c['txt|date@when'] += 1
                lang = el.get('{http://www.w3.org/XML/1998/namespace}lang')
                if t in ('text', 'body', 'TEI', 'language') and lang:
                    langs[lang[:12]] += 1
                if t == 'language' and el.get('ident'):
                    langs[el.get('ident')[:12]] += 1
                if t == 'idno' and stack[-2:-1] and stack[-2] in ('person', 'place', 'org', 'persName', 'placeName'):
                    idno[(el.get('type') or 'none')[:20].lower()] += 1
                if t in ('geo', 'location') and reg:
                    geo += 1
            else:  # end
                if zone_txt(stack) and el.text and t not in ('note',):
                    words += len(el.text.split())
                if el.tail and len(stack) > 1 and zone_txt(stack[:-1]):
                    words += len(el.tail.split())
                if hdr and t in ('creation', 'docDate', 'origDate', 'date') and len(dates) < 6:
                    parent = stack[-2] if len(stack) > 1 else ''
                    if t != 'date' or parent in ('creation', 'correspAction', 'sourceDesc', 'bibl', 'imprint', 'publicationStmt', 'biblStruct', 'origin', 'history'):
                        d = el.get('when') or el.get('notBefore') or el.get('from') or (''.join(el.itertext()) or '')[:40].strip()
                        if d:
                            dates.append(f'{parent}/{t}:{d}'[:70])
                if t == 'teiHeader':
                    hdr -= 1
                elif t == 'text' and txt:
                    txt -= 1
                if t in REG_CONTAINERS:
                    reg -= 1
                stack.pop()
                el.clear(keep_tail=True)
    except Exception as e:  # noqa
        r['error'] = str(e)[:120]
    r['root'] = root
    r['c'] = dict(c)
    r['refs'] = dict(refs)
    r['types'] = dict(types.most_common(20))
    r['idno'] = dict(idno)
    r['langs'] = dict(langs.most_common(5))
    r['dates'] = dates
    r['words'] = words
    r['geo'] = geo
    r['ent_text'] = ent_text
    r['ent_text_ref'] = ent_text_with_ref
    return r


_TXT_STACK_TAGS = {'text'}


def zone_txt(stack):
    # inside <text> and not inside teiHeader or a register
    if 'teiHeader' in stack:
        return False
    if any(s in REG_CONTAINERS for s in stack):
        return False
    return 'text' in stack


def collection_of(rel):
    parts = rel.split('/')
    if parts[0] in ('renascor_2026-10', 'originals_2026-10-15', 'originals_2026-10-17') and len(parts) > 2:
        return parts[0] + '/' + parts[1]
    return parts[0] if len(parts) > 1 else '_top'


def is_xml_name(n):
    n = n.lower()
    return n.endswith('.xml') or n.endswith('.tei')


def work(path):
    """Return a list of records for one file (several for an archive)."""
    rel = os.path.relpath(path, ROOT)
    coll = collection_of(rel)
    out = []
    low = path.lower()
    try:
        if is_xml_name(low):
            if os.path.getsize(path) > MAX_BYTES:
                return [{'path': rel, 'coll': coll, 'skipped': 'too big', 'bytes': os.path.getsize(path)}]
            data = open(path, 'rb').read()
            out.append(_rec(rel, coll, data))
        elif low.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                for i in z.infolist():
                    if is_xml_name(i.filename) and i.file_size < MAX_BYTES and not i.filename.split('/')[-1].startswith('._'):
                        out.append(_rec(rel + '!' + i.filename, coll, z.read(i)))
        elif low.endswith('.tar.gz') or low.endswith('.tgz') or low.endswith('.tar'):
            with tarfile.open(path) as tf:
                for m in tf:
                    if m.isfile() and is_xml_name(m.name) and m.size < MAX_BYTES and not m.name.split('/')[-1].startswith('._'):
                        out.append(_rec(rel + '!' + m.name, coll, tf.extractfile(m).read()))
        elif low.endswith('.xml.gz'):
            out.append(_rec(rel, coll, gzip.open(path).read()))
    except Exception as e:
        out.append({'path': rel, 'coll': coll, 'error': 'open: ' + str(e)[:120]})
    return out


def _rec(rel, coll, data):
    h = hashlib.sha1(data).hexdigest()
    r = scan_bytes(data)
    r['path'] = rel
    r['coll'] = coll
    r['sha1'] = h
    return r


def main():
    files = []
    for sd in SUBDIRS:
      for dp, dn, fn in os.walk(os.path.join(ROOT, sd)):
        dn[:] = [d for d in dn if d not in ('.git', '__pycache__', 'x') and not d.startswith('._')]
        for f in fn:
            if f.startswith('._'):
                continue
            lf = f.lower()
            if is_xml_name(lf) or lf.endswith(('.zip', '.tar.gz', '.tgz', '.tar', '.xml.gz')):
                files.append(os.path.join(dp, f))
    # big archives first so they do not finish last
    files.sort(key=lambda p: -os.path.getsize(p) if not is_xml_name(p) else 0)
    print('files to scan', len(files), flush=True)
    seen = set()
    n = 0
    with open(OUT, 'w') as fo, ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, p) for p in files]
        for f in as_completed(futs):
            for r in f.result():
                sha = r.get('sha1')
                if sha:
                    r['dup'] = sha in seen
                    seen.add(sha)
                fo.write(json.dumps(r, ensure_ascii=False) + '\n')
                n += 1
                if n % 20000 == 0:
                    print('records', n, flush=True)
    print('done records', n, 'distinct', len(seen), flush=True)


if __name__ == '__main__':
    main()
