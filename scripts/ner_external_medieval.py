"""Converter of the medieval charters NER corpus (Torres Aguilar 2022, Zenodo 6463699, CC BY 4.0): Latin, French and Spanish charters, 10th-15th c.

Read straight from the 15 GB zip (only datasets/*.txt, 56 MB, is opened; the embeddings and models in the zip are not touched).
Format, one token per line, tab separated:  token  PERS-tag  LOC-tag  language  charter-id   (two independent BIO layers: a place can sit inside a person name)
  datasets/train_set_all_17_03_2022.txt, val_set_..., test_set_...   the corpus splits (by charter)
  datasets/Nesle_Eslonza.txt                                          an out-of-domain set (Nesle censier, French; Eslonza cartulary, Spanish): split test_ood
The LOC layer follows the guidelines of the corpus: it also covers adjectives derived from a place (e.g. "Cabilonensis"); PERS covers the person names.
Used from scripts/ner_external_convert.py (conv_medieval) with the emit helpers passed in.
"""
from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Callable

from ratlas.ner import external as E

LANG = {'lat': 'la', 'latin': 'la', 'french': 'fr', 'fre': 'fr', 'spanish': 'es', 'spa': 'es'}
FILES = [('train_set_all_17_03_2022.txt', 'train'), ('val_set_all_17_03_2022.txt', 'dev'), ('test_set_all_17_03_2022.txt', 'test'),
         ('Nesle_Eslonza.txt', 'test_ood')]
MAXTOK = 120


def subset_of(doc: str, split: str = '') -> str:
    if split == 'test_ood':          # Nesle censier (French) and the Eslonza cartulary (Spanish; charters numbered in roman numerals)
        return 'NESLE' if doc.upper().startswith('NESLE') else 'ESLONZA'
    m = re.match(r'[A-Za-z]+', doc)
    return (m.group(0) if m else 'x')[:12]


def charters(zf: zipfile.ZipFile, name: str):
    """(doc, lang, tokens, pers_tags, loc_tags) per charter, in file order; a blank line or a change of charter id ends a charter."""
    cur = None
    toks, pt, lt, lang = [], [], [], 'la'
    with zf.open(name) as raw:
        for line in io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline=''):
            line = line.rstrip('\r\n')
            if not line.strip():
                continue
            f = line.split('\t')
            if len(f) < 5:
                continue
            if cur is not None and f[4] != cur and toks:
                yield cur, lang, toks, pt, lt
                toks, pt, lt = [], [], []
            cur = f[4]
            lang = LANG.get(f[3].strip().lower(), f[3].strip().lower())
            toks.append(f[0])
            pt.append(f[1])
            lt.append(f[2])
    if toks:
        yield cur, lang, toks, pt, lt


def convert(root: Path, out: E.Out, emit: Callable = None, spans_to_bio: Callable = None) -> None:
    zf = zipfile.ZipFile(Path(root) / 'medieval_charters/raw/Multilingual_NER_medieval_charters.zip')
    names = {Path(i.filename).name: i.filename for i in zf.infolist() if i.filename.startswith('datasets/') and i.filename.endswith('.txt')}
    langs = Counter()
    for fname, split in FILES:
        for doc, lang, toks, pt, lt in charters(zf, names[fname]):
            langs[lang] += len(toks)
            # cut points: never before a continuation tag of either layer, preferably after a full stop
            cont = ['I-X' if (pt[k].startswith('I-') or lt[k].startswith('I-')) else 'O' for k in range(len(toks))]
            stops = [t in ('.', ';', ':') for t in toks]
            for k, (a, b) in enumerate(E.chunk(len(toks), cont, stops, MAXTOK)):
                sp = E.bio_spans(pt[a:b], [], ['PERS', 'PER', 'PERSON'], out.dropped) + E.bio_spans(lt[a:b], ['LOC', 'GEO'], [], out.dropped)
                rec = E.make_record(rid=f'charters:{split}:{doc}#{k}', source='medieval_charters', subset=subset_of(doc, split), split=split, lang=lang, tokens=toks[a:b],
                                    tok_spans=sp, meta={'doc': doc})
                out.write(rec)
