"""One TEI document -> passages of <= ~200 words with character spans labelled LOC / PER (deterministic, lxml).

What is read
  * the primary text only: every outermost ``<body>`` of the document (``front`` and ``back`` are not read). Never read: ``teiHeader``,
    registers and lists (``listPlace``, ``listPerson``, ``listOrg``, ``standOff``...), ``msIdentifier``, regests (``abstract``), the
    apparatus (``rdg``, ``witDetail``), running heads and catchwords (``fw``), figures, index markers, and the divisions the editors
    mark as commentary / translation / regest / register (``DIV_SKIP`` and the ``div_zone`` of the edition).
  * ``<note>``: only the original ones (marginalia, glosses, interlinear...: ``classify.ORIGINAL_NOTE_TYPES`` or a marginal ``@place``).
    Untyped notes, footnotes and editorial notes are left out: in most editions they are the modern editor's commentary.

The reading (ONE reading, ``READING = 'diplomatic'``): the text as the source prints / writes it, because the editions the model will be
used on (untagged ones) are mostly diplomatic transcriptions.
  * ``<choice>``: ``orig`` / ``abbr`` / ``sic`` win over ``reg`` / ``expan`` / ``corr`` (the first diplomatic child is kept, the other
    children are dropped; a ``choice`` without any of them keeps its first child);
  * ``<subst>``: ``add`` is kept, ``del`` is dropped (also a lone ``del``, and ``surplus``);
  * ``<app>``: ``lem`` is kept, ``rdg`` is dropped (without a ``lem`` the first ``rdg`` is kept);
  * ``<supplied>`` (text added by the editor) and ``<ex>`` (editorial expansion of an abbreviation) are dropped;
  * ``<lb break="no"/>`` joins the two halves of a word without a space (a printed hyphen stays); any other ``lb`` is a space.
  The passage keeps a list of ``flags`` for what happened in it (``choice``, ``subst``, ``app``, ``ex_dropped``, ``supplied_dropped``,
  ``del_dropped``, ``join``), so that a later user can filter on the editorial interventions.

Spans
  * LOC: ``placeName settlement country region district geogName bloc`` and ``name`` / ``rs`` whose ``@type`` (``@role``) maps to a place
    (``common.kind_of``, with the ``type_map`` of the edition); PER: ``persName`` and ``name`` / ``rs`` typed as a person. ``orgName`` is not
    a label: its text stays ``O`` and a place or person tagged inside it keeps its own label.
  * flat spans: of nested tags the OUTERMOST wins (``Johannes von <placeName>Zürich</placeName>`` is one PER; the inner tag is
    recorded in ``nested``). Exception: a LOC (PER) around several LOC (PER) children with only punctuation between them
    (``<placeName><settlement>Oxford</settlement>, <country>England</country></placeName>``) is split into its children.
  * tagged pronouns / common nouns (``generic_terms``) and placeholders (``S.l.``) are not names: they are listed in ``ignored`` and stay ``O``.
  * a span longer than ``MAX_SPAN_WORDS`` is dropped and the passage is marked ``dropped_span`` (it is not used for training).
"""
from __future__ import annotations

import bisect
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .. import classify
from .. import common as C
from .. import generic_terms as GT
from . import READING  # noqa: F401  (documented above)
from .langs import guess_lang, norm_lang

MAX_WORDS = 200
MIN_WORDS = 5
MAX_SPAN_WORDS = 15
BRK = '\x1e'            # block break marker in the working buffer

# ---------------------------------------------------------------------------------------------------------------------
# what is read
# ---------------------------------------------------------------------------------------------------------------------
SKIP_ALWAYS = frozenset('''teiHeader fw figure graphic facsimile sourceDoc index listPlace listPerson listOrg listEvent listRelation
standOff rdg witDetail rdgGrp abstract msIdentifier del surplus supplied ex respStmt ptr noteGrp encodingDesc revisionDesc
listPrefixDef'''.split())
DIPLO = ('orig', 'abbr', 'sic')
INLINE_SKIPS = frozenset(('choice', 'app', 'ex', 'supplied', 'del', 'surplus'))
DIV_SKIP = frozenset('''commentary kommentar commentaire commentario translation traduction traduccion traducción uebersetzung übersetzung
traduzione vertaling regest regesta regeste regesto abstract summary resume résumé apparatus critical_apparatus apparat
register index glossary glossar editorial introduction einleitung intro bibliography bibliographie literatur literaturverzeichnis
footnotes endnotes notes toc contents inhaltsverzeichnis editorial_note editors_note sachkommentar erlaeuterung erläuterung
textkritik lesarten provenance remarks metadata'''.split())
_SKIP_TYPED = frozenset(('seg', 'div', 'div1', 'div2', 'div3', 'div4', 'div5', 'ab', 'p', 'floatingText', 'list', 'table', 'head', 'lg', 'quote'))
MARGINAL_PLACES = frozenset(('margin', 'marginal', 'margin-left', 'margin-right', 'left', 'right', 'inner-margin', 'outer-margin',
                             'inline', 'interlinear', 'supralinear', 'sublinear', 'above', 'over', 'interlinear_above', 'margin_left',
                             'margin_right', 'top-margin', 'bottom-margin', 'sidenote'))
BLOCK = frozenset('''p div div1 div2 div3 div4 div5 head l lg ab item list row cell table sp speaker stage titlePart byline docTitle
dateline opener closer salute signed postscript trailer argument epigraph castItem castList docAuthor docDate docImprint entry sense
floatingText text front back body bibl biblStruct listBibl label note lem address addrLine'''.split())
# a tagged span that crosses a break is re-joined with a space (see document_passages).

STRUCT_WS = frozenset(('choice', 'app', 'subst'))     # containers whose whitespace-only text / tails are layout (Options.repair_ws)
HYPHENS = '-\u00ad\u2010\u2011\u2012\u2e17\u2e40'      # hyphen-like characters closing a line (Options.dehyph)

_WS_RE = re.compile(r'\s+|\S+')
_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TRAIL_RE = re.compile(r'[\s,;:··]+$')
_LEAD_RE = re.compile(r'^[\s,;:··]+')
_LETTER = re.compile(r'[^\W\d_]', re.UNICODE)
_SENT_END = re.compile(r'[.!?;:…]["»”’)\]]*$')


def tokenize(text: str) -> list[tuple[int, int]]:
    """(start, end) of the tokens: words (letters, digits, underscore) and single punctuation marks."""
    return [(m.start(), m.end()) for m in _TOK_RE.finditer(text)]


# ---------------------------------------------------------------------------------------------------------------------
# options of an edition
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Options:
    type_map: dict = field(default_factory=dict)       # extra rs/name @type -> kind (person / place / org)
    note_zone: dict = field(default_factory=dict)      # {note type: zone} as in classify (only 'text' notes are read)
    div_zone: dict = field(default_factory=dict)       # {div type: zone}; a zone other than 'text' skips the division
    strip_lead: tuple = ()                             # lower-case words stripped from the start of a LOC span ('of', 'at' for rs@address)
    skip_ancestors: tuple = ()                         # extra element names never read
    keep_supplied: bool = False
    # NER v2 repairs, all off by default (the v1 passages are built without them)
    repair_ws: bool = False        # whitespace-only text between the children of choice / app / subst is layout, not a word break
    dehyph: bool = False           # 'leidet-<lb/>en': end-of-line hyphen joined (hyphen dropped before a lower-case letter, kept before a capital)

    @classmethod
    def from_context(cls, options: Optional[dict], type_map: Optional[dict] = None, extra: Optional[dict] = None) -> 'Options':
        o = dict(options or {})
        e = dict(extra or {})
        return cls(type_map={**(type_map or {}), **(e.get('type_map') or {})},
                   note_zone=dict(o.get('note_zone') or {}), div_zone=dict(o.get('div_zone') or {}),
                   strip_lead=tuple(e.get('strip_lead') or ()), skip_ancestors=tuple(e.get('skip_ancestors') or ()),
                   keep_supplied=bool(e.get('keep_supplied', False)))


@dataclass
class _Span:
    kind: str           # loc | per | org
    tag: str
    start: Optional[int]
    end: Optional[int] = None
    parent: int = -1    # index of the enclosing candidate
    has_ref: bool = False
    children: list = field(default_factory=list)


class _Walker:
    def __init__(self, opts: Options):
        self.o = opts
        self.chunks: list[str] = []
        self.n = 0
        self.last = None            # None (start) | ' ' | BRK | other
        self.pending = False
        self.nospace = False
        self.spans: list[_Span] = []
        self.open: list[int] = []   # indices of open candidate spans
        self.lang_stack: list[Optional[str]] = [None]
        self.runs: list[tuple[int, Optional[str]]] = []
        self.marks: list[tuple[int, str]] = []
        self.skipped = Counter()
        self.seen = Counter()       # kind of divisions and notes met (kept or skipped), for the review of the rules
        self._choice_pick: dict = {}
        self.hyph_at: Optional[int] = None      # buffer length right after a line-end hyphen (Options.dehyph)
        self.soft_ws = False                    # layout white space was dropped just before the next word (Options.repair_ws)

    # ---- buffer ----------------------------------------------------------------------------------------------------
    def add_tail(self, el) -> None:
        """The tail text of ``el``; with ``Options.repair_ws`` a whitespace-only tail inside choice / app / subst is dropped."""
        t = el.tail
        if t and self.o.repair_ws and not t.strip():
            p = el.getparent()
            if p is not None and C.ln(p.tag) in STRUCT_WS:
                self.marks.append((self.n, 'ws_repair'))
                self.soft_ws = True
                return
        self.add_text(t)

    def _dehyph(self, tok: str) -> None:
        """First word after a line break that follows a hyphen: join the halves (see Options.dehyph)."""
        h, self.hyph_at = self.hyph_at, None
        if self.n != h or not self.chunks or self.last not in HYPHENS or not self.chunks[-1].endswith(self.last):
            return
        prev = self.chunks[-1]
        if len(prev) < 2 or not (prev[-2].isalnum() or prev[-2] == '_'):
            return
        c0 = tok[0]
        if c0.islower():
            self.chunks[-1] = prev[:-1]
            self.n -= 1
            self.last = prev[-2]
            for sp in self.spans:
                if sp.end == h:
                    sp.end = self.n
            self.marks.append((self.n, 'dehyph'))
        elif c0.isupper():
            self.marks.append((self.n, 'dehyph'))
        else:
            return
        self.pending = False

    def add_text(self, s: Optional[str]) -> None:
        if not s:
            return
        for m in _WS_RE.finditer(s):
            tok = m.group()
            if tok[0].isspace():
                if self.nospace:
                    continue
                if self.last not in (None, ' ', BRK):
                    self.pending = True
                continue
            if self.soft_ws:        # the dropped layout space stays only between an abbreviation point and a word ('ser.' <expan/> 'item')
                self.soft_ws = False
                if self.last is not None and self.last in '.:;!?' and tok[0].isalnum() and not self.nospace:
                    self.pending = True
            if self.hyph_at is not None:
                self._dehyph(tok)
            if self.pending and self.last not in (None, BRK):
                self.chunks.append(' ')
                self.n += 1
            self.pending = False
            self.nospace = False
            for i in self.open:
                if self.spans[i].start is None:
                    self.spans[i].start = self.n
            lg = self.lang_stack[-1]
            if not self.runs or self.runs[-1][1] != lg:
                self.runs.append((self.n, lg))
            self.chunks.append(tok)
            self.n += len(tok)
            self.last = tok[-1]

    def space(self) -> None:
        if self.last not in (None, ' ', BRK) and not self.nospace:
            self.pending = True

    def block(self) -> None:
        if self.last not in (None, BRK):
            self.chunks.append(BRK)
            self.n += 1
            self.last = BRK
        self.pending = False
        self.nospace = False

    # ---- decisions ---------------------------------------------------------------------------------------------------
    def _keep_child(self, el, n: str, parent) -> bool:
        pn = C.ln(parent.tag) if parent is not None else None
        if pn == 'choice':
            pick = self._choice_pick.get(id(parent))
            if pick is None:
                kids = [k for k in parent if isinstance(k.tag, str)]
                pick = next((k for name in DIPLO for k in kids if C.ln(k.tag) == name and _has_text(k)), None)
                if pick is None:     # a diplomatic child without text (a glyph, <g ref="#crossedp"/>): the editorial reading is the only readable one
                    pick = next((k for k in kids if _has_text(k)), None)
                    if pick is None and kids:
                        pick = kids[0]
                    if pick is not None and C.ln(pick.tag) not in DIPLO:
                        self.marks.append((self.n, 'choice_fallback'))
                self._choice_pick[id(parent)] = pick
            if pick is not el:
                self.skipped['choice_alt'] += 1
                return False
            return True
        if pn == 'app' and n == 'rdg':
            has_lem = any(C.ln(k.tag) == 'lem' for k in parent)
            first_rdg = next((k for k in parent if C.ln(k.tag) == 'rdg'), None)
            return (not has_lem) and first_rdg is el
        return True

    def skip_reason(self, el, n: str) -> Optional[str]:
        parent = el.getparent()
        if n == 'rdg':
            return None if self._keep_child(el, n, parent) else 'app'
        if n in SKIP_ALWAYS and not (n == 'supplied' and self.o.keep_supplied):
            return n
        if n in self.o.skip_ancestors:
            return n
        if n == 'note':
            z = classify._note_zone(el, self.o.note_zone)
            pl = (el.get('place') or '').strip().lower()
            key = f'note@type={el.get("type") or ""}@place={pl}'
            if z == 'text' or (z is None and pl in MARGINAL_PLACES):
                self.seen['kept ' + key] += 1
                return None
            self.seen['skipped ' + key] += 1
            return 'note'
        if n in _SKIP_TYPED or n.startswith('div'):
            ty, sub = (el.get('type') or '').strip().lower(), (el.get('subtype') or '').strip().lower()
            decls = (el.get('decls') or '').lower()
            ident = (el.get(C.XML_ID) or '').lower()
            why = None
            for key in (ty, sub):
                if not key:
                    continue
                zz = self.o.div_zone.get(key) or self.o.div_zone.get(el.get('type') or '')
                if zz and zz != 'text':
                    why = 'div_zone'
                elif key in DIV_SKIP:
                    why = 'div_type'
            if why is None and (n.startswith('div') or n in ('ab', 'floatingText', 'seg', 'p')) and (
                    any(t in decls for t in ('translation', 'commentary', 'regest')) or ident.startswith(('translation', 'commentary'))):
                why = 'div_decls'
            if n.startswith('div') or n in ('ab', 'floatingText'):
                self.seen[('skipped ' if why else 'kept ') + f'{n}@type={ty}@decls={decls[:30]}'] += 1
            if why:
                return why
        if not self._keep_child(el, n, parent):
            return 'choice'
        return None

    def candidate(self, el, n: str) -> Optional[tuple[str, str]]:
        kind, label = C.kind_of(n, el, self.o.type_map)
        if kind is None:
            return None
        return {'person': 'per', 'place': 'loc', 'org': 'org'}[kind], label

    # ---- walk ------------------------------------------------------------------------------------------------------
    def walk(self, body) -> None:
        stack: list = [(body, False, None)]
        while stack:
            el, done, idx = stack.pop()
            if done:
                self.exit(el, idx, el is body)
                continue
            tag = el.tag
            n = C.ln(tag) if isinstance(tag, str) else None
            if n is None:           # entity reference / comment left by the parser
                if el is not body:
                    self.add_tail(el)
                continue
            why = None if el is body else self.skip_reason(el, n)
            if why:
                self.skipped[why] += 1
                if why in ('ex', 'supplied', 'del', 'surplus', 'choice', 'app'):
                    self.marks.append((self.n, {'ex': 'ex_dropped', 'supplied': 'supplied_dropped', 'del': 'del_dropped',
                                                'surplus': 'del_dropped', 'choice': 'choice', 'app': 'app'}[why]))
                if why in ('div_zone', 'div_type', 'div_decls'):
                    self.block()
                elif why not in INLINE_SKIPS:
                    self.space()
                self.add_tail(el)
                continue
            idx = self.enter(el, n)
            stack.append((el, True, idx))
            for c in reversed(list(el)):
                stack.append((c, False, None))

    def enter(self, el, n: str) -> Optional[int]:
        self.lang_stack.append(el.get(C.XML_LANG) or self.lang_stack[-1])
        if n in BLOCK:
            self.block()
        elif n == 'lb':
            if (el.get('break') or '').lower() == 'no':
                self.pending = False
                self.nospace = True
                self.marks.append((self.n, 'join'))
            else:
                self.space()
        elif n in ('pb', 'cb'):
            if (el.get('break') or '').lower() == 'no':
                self.pending = False
                self.nospace = True
            else:
                self.space()
        elif n in ('subst', 'choice', 'app'):
            self.marks.append((self.n, n))
        if self.o.dehyph and n in ('lb', 'pb', 'cb') and self.last is not None and self.last in HYPHENS and self.chunks \
                and self.chunks[-1].endswith(self.last):
            self.hyph_at = self.n
        idx = None
        cand = self.candidate(el, n)
        if cand is not None:
            kind, label = cand
            refs = any((el.get(a) or '').strip() not in ('', '#') for a in C.REF_ATTRS)
            self.spans.append(_Span(kind=kind, tag=label, start=None, parent=self.open[-1] if self.open else -1, has_ref=refs))
            idx = len(self.spans) - 1
            if self.spans[idx].parent >= 0:
                self.spans[self.spans[idx].parent].children.append(idx)
            self.open.append(idx)
        if self.o.repair_ws and n in STRUCT_WS and el.text and not el.text.strip():
            self.marks.append((self.n, 'ws_repair'))
            self.soft_ws = True
        else:
            self.add_text(el.text)
        return idx

    def exit(self, el, idx: Optional[int], is_root: bool) -> None:
        n = C.ln(el.tag)
        if idx is not None:
            self.open.pop()
            self.spans[idx].end = self.n
        if n == 'choice':
            self._choice_pick.pop(id(el), None)
        if n in BLOCK:
            self.block()
        self.lang_stack.pop()
        if not is_root:
            self.add_tail(el)

    def text(self) -> str:
        return ''.join(self.chunks)


# ---------------------------------------------------------------------------------------------------------------------
# spans: flat labelled spans from the tree of candidates
# ---------------------------------------------------------------------------------------------------------------------
def _has_text(el) -> bool:
    return any(t.strip() for t in el.itertext())


def _trim(buf: str, s: int, e: int) -> tuple[int, int]:
    while s < e and (buf[s].isspace() or buf[s] == BRK):
        s += 1
    while e > s and (buf[e - 1].isspace() or buf[e - 1] == BRK):
        e -= 1
    seg = buf[s:e]
    m = _TRAIL_RE.search(seg)
    if m:
        e = s + m.start()
    m = _LEAD_RE.match(buf[s:e])
    if m:
        s += m.end()
    return s, e


def _nearest_labelled(spans: list[_Span], idx: int) -> list[int]:
    """Nearest loc / per descendants of a candidate (organisations are looked through)."""
    out: list[int] = []
    for c in spans[idx].children:
        if spans[c].kind == 'org':
            out.extend(_nearest_labelled(spans, c))
        else:
            out.append(c)
    return out


def _all_labelled(spans: list[_Span], idx: int) -> list[int]:
    out: list[int] = []
    for c in spans[idx].children:
        if spans[c].kind != 'org' and spans[c].start is not None:
            out.append(c)
        out.extend(_all_labelled(spans, c))
    return out


def resolve_spans(buf: str, spans: list[_Span], opts: Options) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """-> (effective, nested, ignored, orgs); every item {'start','end','label'|'kind','tag'}; offsets in ``buf``."""
    for sp in spans:
        if sp.start is not None and sp.end is not None:
            sp.start, sp.end = _trim(buf, sp.start, sp.end)
            if sp.end <= sp.start or not _LETTER.search(buf[sp.start:sp.end]):
                sp.start = None
    eff: list[dict] = []
    nested: list[dict] = []
    ignored: list[dict] = []
    orgs: list[dict] = []

    def emit(i: int) -> None:
        sp = spans[i]
        eff.append({'start': sp.start, 'end': sp.end, 'label': sp.kind.upper(), 'tag': sp.tag, 'has_ref': sp.has_ref, 'i': i})

    def resolve(i: int) -> None:
        sp = spans[i]
        if sp.start is None:             # empty span: look at the children
            for c in sp.children:
                resolve(c)
            return
        if sp.kind == 'org':
            orgs.append({'start': sp.start, 'end': sp.end, 'tag': sp.tag})
            for c in sp.children:
                resolve(c)
            return
        kids = [c for c in _nearest_labelled(spans, i) if spans[c].start is not None]
        if not kids:
            emit(i)
            return
        same = all(spans[c].kind == sp.kind for c in kids)
        gap_ok = False
        if same:
            pos = sp.start
            gap = []
            for c in sorted(kids, key=lambda k: spans[k].start):
                gap.append(buf[pos:spans[c].start])
                pos = max(pos, spans[c].end)
            gap.append(buf[pos:sp.end])
            gap_ok = not _LETTER.search(''.join(gap))
        if same and gap_ok:
            for c in kids:
                resolve(c)
        else:
            emit(i)
            for c in _all_labelled(spans, i):
                nested.append({'start': spans[c].start, 'end': spans[c].end, 'label': spans[c].kind.upper(), 'tag': spans[c].tag})

    roots = [i for i, sp in enumerate(spans) if sp.parent < 0]
    for r in roots:
        resolve(r)
    # strip function words ('of', 'at') in front of a place (rs@address: 'of Stantone')
    out: list[dict] = []
    for e in eff:
        if e['label'] == 'LOC' and opts.strip_lead:
            seg = buf[e['start']:e['end']]
            while True:
                m = re.match(r'(\S+)\s+(?=\S)', seg)
                if m and m.group(1).lower() in opts.strip_lead:
                    e['start'] += m.end()
                    seg = buf[e['start']:e['end']]
                else:
                    break
        text = buf[e['start']:e['end']]
        kind = 'person' if e['label'] == 'PER' else 'place'
        if classify.is_placeholder(text) or GT.generic_kind(kind, text):
            ignored.append({'start': e['start'], 'end': e['end'], 'label': e['label'], 'tag': e['tag'],
                            'why': 'placeholder' if classify.is_placeholder(text) else 'generic'})
        else:
            out.append(e)
    out.sort(key=lambda d: (d['start'], d['end']))
    return out, nested, ignored, orgs


# ---------------------------------------------------------------------------------------------------------------------
# passages
# ---------------------------------------------------------------------------------------------------------------------
def _units(buf: str) -> list[tuple[int, int]]:
    out = []
    pos = 0
    for part in buf.split(BRK):
        s, e = pos, pos + len(part)
        pos = e + 1
        if part.strip():
            out.append((s, e))
    return out


def _words(buf: str, s: int, e: int) -> list[tuple[int, int]]:
    return [(s + m.start(), s + m.end()) for m in re.finditer(r'\S+', buf[s:e])]


def _chunks(buf: str, s: int, e: int, inside: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cut a long unit into chunks of <= MAX_WORDS words, at a sentence end when possible, never inside a span."""
    ws = _words(buf, s, e)
    if len(ws) <= MAX_WORDS:
        return [(s, e)]
    ins = sorted(inside)

    def cut_ok(pos: int) -> bool:
        return not any(a < pos < b for a, b in ins)

    out = []
    i = 0
    while i < len(ws):
        j = min(i + MAX_WORDS, len(ws))
        if j < len(ws):
            best = None
            for k in range(j - 1, i + MAX_WORDS // 2 - 1, -1):
                if _SENT_END.search(buf[ws[k][0]:ws[k][1]]) and cut_ok(ws[k][1]):
                    best = k + 1
                    break
            if best is None:
                for k in range(j - 1, i, -1):
                    if cut_ok(ws[k - 1][1]) and cut_ok(ws[k][0]):
                        best = k
                        break
            j = best or j
        out.append((ws[i][0], ws[j - 1][1]))
        i = j
    return out


def document_passages(data: bytes, *, opts: Optional[Options] = None, max_words: int = MAX_WORDS, group_fn=None) -> dict:
    """Parse one TEI file (bytes). -> {'passages': [...], 'doc_lang': str|None, 'words': int, 'stats': {...}, 'error': str|None}.

    A passage: {'text', 'spans': [{'start','end','label','tag','has_ref'}], 'nested': [...], 'ignored': [...], 'orgs': [[s, e]],
    'lang', 'lang_src', 'lang_guess', 'langs': {lang: share}, 'words', 'flags': [...], 'dropped_span': bool}; also 'xml_id', 'hdr_year'.
    Offsets are characters of the passage text. Passages without any text are not produced.
    ``group_fn(buf, units)`` (optional, default: the 200-word grouping above) returns the passages as lists of ``(start, end)`` pieces of ``buf``.
    """
    opts = opts or Options()
    root = C.parse_xml(data)
    if root is None:
        return {'passages': [], 'doc_lang': None, 'words': 0, 'stats': {}, 'error': 'unparseable', 'xml_id': None, 'hdr_year': None}
    doc_lang = norm_lang(C.doc_language(root))
    bodies = []
    for el in root.iter():
        if C.ln(el.tag) == 'body':
            anc = {C.ln(a.tag) for a in el.iterancestors()}
            if anc & {'teiHeader', 'standOff', 'facsimile', 'body', 'sourceDoc'}:
                continue
            bodies.append(el)
    if not bodies and C.ln(root.tag) not in ('TEI', 'TEI.2', 'teiCorpus', 'teiHeader'):
        bodies = [root]
    w = _Walker(opts)
    for b in bodies:
        w.block()
        inherited = next((a.get(C.XML_LANG) for a in b.iterancestors() if a.get(C.XML_LANG)), None)
        w.lang_stack = [inherited]
        w.walk(b)
    buf = w.text()
    eff, nested, ignored, orgs = resolve_spans(buf, w.spans, opts)
    # a break inside a tagged span is a space
    all_ranges = [(d['start'], d['end']) for d in eff + nested + ignored + orgs]
    if BRK in buf and all_ranges:
        cb = list(buf)
        for a, b in all_ranges:
            for k in range(a, b):
                if cb[k] == BRK:
                    cb[k] = ' '
        buf = ''.join(cb)
    stats = {'skipped': dict(w.skipped), 'spans_loc': sum(1 for d in eff if d['label'] == 'LOC'),
             'spans_per': sum(1 for d in eff if d['label'] == 'PER'), 'ignored': len(ignored), 'nested': len(nested),
             'bodies': len(bodies), 'seen': dict(w.seen)}
    # passages
    units = _units(buf)
    inside = sorted((d['start'], d['end']) for d in eff + nested + ignored + orgs)
    # a span of more than MAX_SPAN_WORDS words is an error of the markup: the passage that holds it is marked
    bad_ranges = sorted((d['start'], d['end']) for d in eff if len(buf[d['start']:d['end']].split()) > MAX_SPAN_WORDS)
    eff = [d for d in eff if (d['end'] - d['start'] <= 1) or len(buf[d['start']:d['end']].split()) <= MAX_SPAN_WORDS]
    cuts: list[tuple[int, int]] = []
    for s, e in units:
        cuts.extend(_chunks(buf, s, e, inside))
    groups: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    cw = 0
    for s, e in cuts:
        nw = len(buf[s:e].split())
        if cur and cw + nw > max_words:
            groups.append(cur)
            cur, cw = [], 0
        cur.append((s, e))
        cw += nw
    if cur:
        groups.append(cur)
    if group_fn is not None:        # NER v2: the caller decides where the passages start and end (same regions as an earlier extraction)
        groups = group_fn(buf, units)
    marks = w.marks
    mark_pos = [m[0] for m in marks]
    runs = w.runs
    run_starts = [r[0] for r in runs]
    lists = {}
    for name, items in (('spans', eff), ('nested', nested), ('ignored', ignored), ('orgs', orgs)):
        items = sorted(items, key=lambda d: d['start'])
        lists[name] = (items, [d['start'] for d in items])
    bad_starts = [b[0] for b in bad_ranges]
    out = []
    for g in groups:
        text_parts = []
        offs = []                       # (unit start, unit end, offset in the passage)
        pos = 0
        for s, e in g:
            offs.append((s, e, pos))
            text_parts.append(buf[s:e])
            pos += (e - s) + 1
        text = '\n'.join(text_parts)
        nwords = len(text.split())
        if nwords < MIN_WORDS:
            continue
        lo, hi = g[0][0], g[-1][1]

        ostarts = [o[0] for o in offs]

        def pick(name):
            items, starts = lists[name]
            res = []
            k = bisect.bisect_left(starts, lo)
            while k < len(items) and items[k]['start'] < hi:
                d = items[k]
                k += 1
                u = bisect.bisect_right(ostarts, d['start']) - 1
                if u >= 0 and d['end'] <= offs[u][1]:
                    a, b, p = offs[u]
                    res.append({kk: v for kk, v in d.items() if kk != 'i'} | {'start': d['start'] - a + p, 'end': d['end'] - a + p})
            return res

        sp, nd, ig, og = pick('spans'), pick('nested'), pick('ignored'), pick('orgs')
        og = [[x['start'], x['end']] for x in og]
        kb = bisect.bisect_left(bad_starts, lo)
        dropped = kb < len(bad_starts) and bad_starts[kb] < hi
        # languages: characters per explicit xml:lang in the passage
        langs: Counter = Counter()
        for s, e in g:
            k = bisect.bisect_right(run_starts, s) - 1
            pos = s
            while pos < e:
                nxt = runs[k + 1][0] if k + 1 < len(runs) else e
                seg_end = min(e, nxt)
                if k >= 0 and runs[k][1]:
                    langs[norm_lang(runs[k][1]) or runs[k][1]] += seg_end - pos
                pos = seg_end
                k += 1
                if k >= len(runs):
                    break
        tot = sum(langs.values())
        lang = lang_src = None
        if tot >= 0.5 * len(text) and langs:
            lang, lang_src = langs.most_common(1)[0][0], 'xml:lang'
        flags = sorted({marks[m][1] for m in range(bisect.bisect_left(mark_pos, lo), bisect.bisect_right(mark_pos, hi))})
        if len([l for l, c in langs.items() if c >= 0.2 * max(tot, 1)]) > 1:
            flags.append('mixed_lang')
        if dropped:
            flags.append('dropped_span')
        out.append({'text': text, 'spans': sp, 'nested': nd, 'ignored': ig, 'orgs': og, 'lang': lang, 'lang_src': lang_src,
                    'langs': {k: round(v / tot, 3) for k, v in langs.items()} if tot else {},
                    'lang_guess': guess_lang(text), 'words': nwords, 'flags': flags, 'dropped_span': dropped})
    try:
        dd = C.header_dates(root)
        hdr_year = (dd.year_from + dd.year_to) // 2 if dd.year_from is not None and dd.year_to is not None else (dd.year_from or dd.year_to)
    except Exception:       # noqa: BLE001  a date is only a hint
        hdr_year = None
    return {'passages': out, 'doc_lang': doc_lang, 'words': sum(p['words'] for p in out), 'stats': stats, 'error': None,
            'xml_id': root.get(C.XML_ID), 'hdr_year': hdr_year}


_CHUNK_RE = re.compile(rb'(?m)^### (\S+)[ \t]*\r?\n')


def split_chunks(data: bytes) -> list[tuple[Optional[str], bytes]]:
    """Pseudo-XML corpora (CKCC .txt files): one TEI-like chunk per letter after a line '### <id>.xml'. Other files are one chunk."""
    ms = list(_CHUNK_RE.finditer(data))
    if len(ms) < 2 and not (ms and ms[0].start() == 0):
        return [(None, data)]
    out = []
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(data)
        out.append((m.group(1).decode('utf-8', 'replace'), data[m.end():end]))
    return out


def document_chunks(data: bytes, *, opts: Optional[Options] = None) -> dict:
    """``document_passages`` for a file that may hold several documents (see ``split_chunks``); passages get ``chunk``, ``doc_lang``, ``hdr_year``."""
    chunks = split_chunks(data)
    if len(chunks) == 1 and chunks[0][0] is None:
        return document_passages(data, opts=opts)
    merged = {'passages': [], 'doc_lang': None, 'words': 0, 'stats': {'skipped': Counter(), 'seen': Counter()}, 'error': None, 'xml_id': None,
              'hdr_year': None}
    errs = 0
    for name, chunk in chunks:
        r = document_passages(chunk, opts=opts)
        if r['error']:
            errs += 1
            continue
        for p in r['passages']:
            p['chunk'], p['doc_lang'], p['hdr_year'] = name, r['doc_lang'], r['hdr_year']
            merged['passages'].append(p)
        merged['words'] += r['words']
        merged['stats']['skipped'].update(r['stats'].get('skipped', {}))
        merged['stats']['seen'].update(r['stats'].get('seen', {}))
    if errs == len(chunks):
        merged['error'] = 'unparseable'
    return merged


def doc_hash(path: str) -> str:
    """Stable pseudo-random key of a document (used for sampling and splits)."""
    return hashlib.sha1(path.encode('utf-8')).hexdigest()
