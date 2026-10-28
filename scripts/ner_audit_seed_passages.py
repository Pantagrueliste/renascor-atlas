"""Draw the passages of the seed gold set (audit, task 2) from texts of the text archive that carry NO place markup.

Sources (read-only, extracted by the caller from RQ/release/text_archive_v1.0 into a scratch directory):
  English   EEBO-TCP (eebotcp_fill, eebo_phase2)       TEI without any placeName / persName
  Latin     EEBO-TCP Latin (latineebo)                 same
  Italian   Italian Wikisource Renaissance texts (nas__ws_it_renaissance) and Galileo (Favaro edition, Wikisource)
The draw is random with a fixed seed and uses no model output. One passage per document; 100-170 words, starting at a sentence start and ending at a
sentence end; a passage is refused when it has fewer than 8 capitalised words that are not sentence-initial or when more than 10% of its characters are digits.

    python scripts/ner_audit_seed_passages.py --root SCRATCH/ta --out seed_passages.jsonl
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path

from lxml import etree

TEI = '{http://www.tei-c.org/ns/1.0}'
WORD = re.compile(r'\w+|[^\w\s]', re.U)
SENT = re.compile(r'(?<=[.!?:;])\s+(?=[A-ZÀ-ÝÆŒ"“«(])')
POETRY = re.compile(r'canti|rime|sonett|canzon|gerusalemme|aminta|orlando|stanze|ecloga|capitolo|amorosa|caccia|dubbj|cantate|fior_|pianta|chi_|datemi|come_nell|gi__vidi|ove_son|poi_che|ligiadro|decennali|alcesto|angelica|clizia|commedie|favola_di_belfagor', re.I)


def xml_text(path: Path) -> tuple[str, str]:
    tree = etree.parse(str(path))
    root = tree.getroot()
    title = ''.join(root.xpath('string(//*[local-name()="titleStmt"]/*[local-name()="title"])')).strip()
    body = root.find(f'.//{TEI}body')
    if body is None:
        body = root.find('.//body')
    out = []

    def walk(el, buf):
        tag = etree.QName(el).localname if isinstance(el.tag, str) else None
        if tag in ('note', 'figure', 'pb', 'fw', 'gap', 'teiHeader', 'del', 'corr', 'reg', 'expan', 'supplied', 'rdg'):
            return           # diplomatic reading: the printed form (sic / orig / abbr) is kept, editorial corrections and additions are not
        if tag == 'g' and (el.get('ref') or '').endswith('EOLhyphen'):
            return           # end-of-line hyphen: the two halves of the word are joined (the tail is added by the caller)
        if tag in ('p', 'l', 'lg', 'head', 'item', 'salute', 'dateline', 'closer', 'signed', 'trailer', 'label', 'speaker', 'stage') and buf and not buf[-1].endswith('\n'):
            buf.append('\n')
        if el.text:
            buf.append(el.text)
        for c in el:
            walk(c, buf)
            if c.tail:
                buf.append(c.tail)
        if tag in ('p', 'lg', 'head', 'item'):
            buf.append('\n')

    buf: list[str] = []
    walk(body, buf)
    txt = ''.join(buf).replace('∣', '')            # TCP line-break mark inside words
    txt = re.sub(r'[ \t\r\f\v]+', ' ', txt)
    txt = re.sub(r'\n\s*', '\n', txt)
    return title, txt.strip()


def wikisource_text(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'<dc:title>\s*(.*?)\s*</dc:title>', raw)
    title = m.group(1) if m else path.stem
    i = raw.find('Informazioni sulla fonte del testo')
    body = raw[i + len('Informazioni sulla fonte del testo'):] if i >= 0 else raw
    body = html.unescape(re.sub(r'&#(\d+);', lambda mm: chr(int(mm.group(1))), body))
    body = re.sub(r'\[\s*\d+\s*\]', ' ', body)
    return title, re.sub(r'\s+', ' ', body).strip()


def pick_passage(text: str, rng: random.Random, lo=100, hi=170, tries=80):
    starts = [0] + [m.end() for m in SENT.finditer(text)]
    for _ in range(tries):
        s = rng.choice(starts)
        end = s
        n = 0
        # extend by sentences until >= lo words
        j = starts.index(s)
        k = j
        while k < len(starts):
            nxt = starts[k + 1] if k + 1 < len(starts) else len(text)
            seg = text[s:nxt].strip()
            n = len(seg.split())
            end = nxt
            k += 1
            if n >= lo:
                break
        seg = text[s:end].strip()
        n = len(seg.split())
        if n < lo or n > hi:
            continue
        digits = sum(c.isdigit() for c in seg) / max(1, len(seg))
        caps = len(re.findall(r'(?<![.!?:;"\n])\s([A-ZÀ-Ý][a-zà-ÿſ]{2,})', ' ' + seg))
        if digits > 0.1 or caps < 8:
            continue
        return s, seg
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--n-en', type=int, default=11)
    ap.add_argument('--n-la', type=int, default=10)
    ap.add_argument('--n-it-ws', type=int, default=6)
    ap.add_argument('--n-it-gal', type=int, default=3)
    a = ap.parse_args()
    root = Path(a.root)
    rng = random.Random(a.seed)
    recs = []

    def add(lang, collection, doc, title, text):
        got = pick_passage(text, rng)
        if not got:
            print('no passage in', doc)
            return False
        s, seg = got
        seg = re.sub(r'\s+', ' ', seg)
        ms = list(WORD.finditer(seg))
        recs.append({'id': f'seed/{lang}/{doc}', 'lang': lang, 'collection': collection, 'doc': doc, 'title': title[:140], 'offset_in_doc': s, 'text': seg,
                     'tokens': [m.group() for m in ms], 'token_starts': [m.start() for m in ms]})
        return True

    en_files = sorted(root.glob('eebo/*/xml/*.xml'))
    la_files = sorted(root.glob('latin/*/xml/*.xml'))
    for lang, files, n in (('en', en_files, a.n_en), ('la', la_files, a.n_la)):
        pool = files[:]
        rng.shuffle(pool)
        got = 0
        for f in pool:
            if got >= n:
                break
            title, txt = xml_text(f)
            if len(txt.split()) < 500:
                continue
            got += add(lang, 'EEBO-TCP' + ('-Latin' if lang == 'la' else ''), f.stem, title, txt)
    ws = [p for p in sorted((root / 'ws_it' / 'ws_it_renaissance').glob('*.txt')) if p.stat().st_size >= 8000 and not POETRY.search(p.stem)]
    rng.shuffle(ws)
    got = 0
    for p in ws:
        if got >= a.n_it_ws:
            break
        title, txt = wikisource_text(p)
        if len(txt.split()) < 800:
            continue
        got += add('it', 'Wikisource-IT-Renaissance', p.stem, title, txt)
    gal = []
    for line in open(root / 'galileo' / 'Galileo_Opere_Favaro' / 'Galileo_Opere_Favaro.jsonl', encoding='utf-8'):
        d = json.loads(line)
        if (d.get('language') or 'it').lower().startswith('it'):
            gal.append(d)
    rng.shuffle(gal)
    got = 0
    for d in gal:
        if got >= a.n_it_gal:
            break
        got += add('it', 'Galileo-Favaro-Wikisource', d['filename'], d['filename'], re.sub(r'\s+', ' ', d['raw']))
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    from collections import Counter
    print(len(recs), Counter(r['lang'] for r in recs), sum(len(r['tokens']) for r in recs), 'tokens')


if __name__ == '__main__':
    main()
