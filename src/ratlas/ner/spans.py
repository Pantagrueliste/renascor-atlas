"""Strict span metrics for LOC / PER (exact token boundaries, same as seqeval ``mode='strict', scheme=IOB2`` for one label at a time).

A task has tags 0 = O, 1 = B, 2 = I (the ``ner_tags_loc`` / ``ner_tags_per`` fields of the data). Spans are (start, end) token indices, end exclusive.
Counts (tp, fp, fn) are kept per group so that micro scores can be recomputed for any grouping (overall, language, edition) and macro scores can be averaged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

TASKS = ('LOC', 'PER')


def tags_to_spans(tags: Iterable[int]) -> list[tuple[int, int]]:
    """Spans of a 0/1/2 (O/B/I) sequence. An I that does not continue a B/I run is invalid and ignored together with the I's that follow it (strict mode)."""
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    n = 0
    for i, t in enumerate(tags):
        n = i + 1
        if t == 1:
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == 2:
            pass                      # continues the open span, or is a stray I (start is None): ignored
        else:
            if start is not None:
                spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, n))
    return spans


def spans_to_tags(n: int, spans: Iterable[tuple[int, int]]) -> list[int]:
    out = [0] * n
    for s, e in spans:
        if 0 <= s < e <= n:
            out[s] = 1
            for k in range(s + 1, e):
                out[k] = 2
    return out


def char_to_token_spans(tok_starts: list[int], tok_lens: list[int], char_spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Token spans for character spans: every token that overlaps the character span is inside. Empty or non-overlapping spans are dropped; overlaps are kept as is."""
    out = []
    n = len(tok_starts)
    for cs, ce in char_spans:
        idx = [i for i in range(n) if tok_starts[i] < ce and tok_starts[i] + tok_lens[i] > cs]
        if idx:
            out.append((idx[0], idx[-1] + 1))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'n_gold': tp + fn, 'n_pred': tp + fp, 'p': round(p, 4), 'r': round(r, 4), 'f1': round(f, 4)}


class Counts:
    """tp / fp / fn per task and per group. Groups are strings: 'all', 'lang=fr', 'edition=reed', ..."""

    def __init__(self):
        self.c: dict[str, dict[str, list[int]]] = {t: defaultdict(lambda: [0, 0, 0]) for t in TASKS}

    def add(self, task: str, groups: Iterable[str], gold: set, pred: set) -> None:
        tp = len(gold & pred)
        fp = len(pred) - tp
        fn = len(gold) - tp
        for g in groups:
            row = self.c[task][g]
            row[0] += tp
            row[1] += fp
            row[2] += fn

    def merge(self, other: 'Counts') -> None:
        for t in TASKS:
            for g, (a, b, c) in other.c[t].items():
                row = self.c[t][g]
                row[0] += a
                row[1] += b
                row[2] += c

    def report(self, min_gold: int = 1) -> dict:
        """{task: {'overall': prf, 'by_lang': {lang: prf}, 'by_edition': {ed: prf}, 'macro_edition_f1': x, 'macro_lang_f1': x}}"""
        out = {}
        for t in TASKS:
            rows = self.c[t]
            if not rows:
                continue
            d = {'overall': prf(*rows['all']) if 'all' in rows else prf(0, 0, 0), 'by_lang': {}, 'by_edition': {}}
            for g, v in sorted(rows.items()):
                if g.startswith('lang='):
                    d['by_lang'][g[5:]] = prf(*v)
                elif g.startswith('edition='):
                    d['by_edition'][g[8:]] = prf(*v)
            for key, name in (('by_edition', 'macro_edition_f1'), ('by_lang', 'macro_lang_f1')):
                fs = [x['f1'] for x in d[key].values() if x['n_gold'] >= min_gold]
                d[name] = round(sum(fs) / len(fs), 4) if fs else None
            out[t] = d
        return out


def record_groups(rec: dict) -> list[str]:
    lang = rec.get('lang') or 'unk'
    ed = rec.get('edition') or (f"{rec.get('source', '?')}/{rec.get('subset', '?')}/{lang}")
    return ['all', f'lang={lang}', f'edition={ed}']
