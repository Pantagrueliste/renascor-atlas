"""Place names of the atlas normalised with the Dictionnaire topographique de la France (owner decision 4 of 2026-10-28).

Source: the École des chartes' DicoTopo (github.com/chartes/dico-topo, XML volumes in ``data/``; the dicotopo.cths.fr API is down). The articles
were parsed and matched to the points of the atlas in ``RQ/derived/atlas/dicotopo`` (``scripts/match.py``); the result that the pages use is
``data/curation/dicotopo_names.tsv``, one line per matched point:

* ``point_id`` -- ``row:N`` (row N of D.rows of the page build after the homonym audit) or ``ml2:N`` (a model point; ``model_key`` gives its ``prev``
  list, which finds the point again when the layer is rebuilt and the ids move);
* ``name`` -- the display name; ``headword`` -- the DicoTopo headword (vedette) as printed, case restored; ``official_name_fr`` -- today's official
  name (Wikidata) when the headword is the name the commune bore when its volume was printed;
* ``article_id`` (``P...`` or ``DTnn-nnnnn``), ``insee``, ``dep``, ``qid`` -- kept so that the names can be traced back to the dictionary;
* ``method`` (qid > headword > old_form > fuzzy), ``matched_form``, ``distance_km``, ``notes``;
* ``extra_forms`` -- dated old forms of the article, ``Form (year)`` separated by `` | ``: written forms that recognise variant spellings.

Display name rule (``choose_label``). The interface stays in English, so a French place that has a real English exonym keeps it (Brittany, Burgundy,
Normandy: Wikidata's English label differs from its French one). Otherwise the name is the DicoTopo headword in its modern French form with the
typographic case restored (``SAINT-GERMAIN-EN-LAYE`` -> ``Saint-Germain-en-Laye``). Two corrections made on reading the matches: a headword
that has lost its accents or its article (Nimes, Pont-a-Mousson, Le Câtelet) is replaced by the official name, and so is a headword that is
an older name than today's (Châlons-sur-Marne, Alais, Vassy), except where today's name is a merged commune (Moret-sur-Loing, Villegagnon...).
``POLICY = 'headword'`` gives the headword untouched instead.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'curation'
FILE = DATA / 'dicotopo_names.tsv'
POLICY = 'official_when_older'        # or 'headword': the DicoTopo headword as printed, whatever today's official name
SMALL = {'de', 'du', 'des', 'la', 'le', 'les', 'sur', 'sous', 'en', 'et', 'aux', 'au', 'd', 'l', 'lès', 'lez', 'à'}


def fold(s: str) -> str:
    """Lower case, no accents, letters and digits only (Saint-Père = saintpere)."""
    s = unicodedata.normalize('NFKD', (s or '').lower())
    return re.sub(r'[^a-z0-9]', '', ''.join(c for c in s if not unicodedata.combining(c)))


def fold_noart(s: str) -> str:
    """``fold`` without a leading article (Le Catelet = Catelet)."""
    f = fold(s)
    for a in ('le', 'la', 'les', 'l'):
        if f.startswith(a) and len(f) > len(a) + 3:
            return f[len(a):]
    return f


def restore_case(s: str) -> str:
    """All-capitals vedette -> typographic case: SAINT-GERMAIN-EN-LAYE -> Saint-Germain-en-Laye."""
    if not (s.isupper() and len(s) > 3):
        return s
    out = []
    for i, w in enumerate(re.split(r"([\s\-’']+)", s.lower())):
        out.append(w if (i > 0 and w in SMALL) else w[:1].upper() + w[1:])
    return ''.join(out)


def english_exonym(en_label: str | None, fr_label: str | None) -> str | None:
    """Wikidata's English label when it is a real exonym: present, different from the French label after folding, no comma (a qualifier)."""
    if not en_label or not fr_label or ',' in en_label:
        return None
    return None if fold(en_label) == fold(fr_label) else en_label


def choose_label(headword: str, official: str | None = None, en_label: str | None = None, fr_label: str | None = None,
                 headword_older: bool = False, keep_headword: bool = False, policy: str | None = None) -> tuple[str, str]:
    """-> (display name, rule). rule = english_exonym | dicotopo_headword | official_name.

    ``headword_older``: the headword is the name the commune bore when its volume was printed (Châlons-sur-Marne, today Châlons-en-Champagne).
    ``keep_headword``: today's official name is a merged commune, the headword is the right historical place (Moret-sur-Loing)."""
    ex = english_exonym(en_label, fr_label)
    if ex:
        return ex, 'english_exonym'
    head = restore_case(headword)
    if (policy or POLICY) == 'headword' or not official or keep_headword:
        return head, 'dicotopo_headword'
    if head != official and (headword_older or fold(head) == fold(official) or fold_noart(head) == fold_noart(official)):
        return official, 'official_name'          # an older name, or accents or article lost in the printed headword
    return head, 'dicotopo_headword'


def parse_forms(s: str) -> list[tuple[str, int | None]]:
    """'Roem (1280) | Rotomagus (350)' -> [('Roem', 1280), ('Rotomagus', 350)]."""
    out = []
    for f in (s or '').split(' | '):
        f = f.strip()
        if not f:
            continue
        m = re.match(r'^(.*?)\s*\((\d{3,4})\)$', f)
        out.append((m.group(1), int(m.group(2))) if m else (f, None))
    return out


def load(path=None, policy: str | None = None) -> pd.DataFrame:
    """The matched points. Columns of the file, plus ``label`` (the display name under ``policy``) and ``forms`` (parsed ``extra_forms``)."""
    d = pd.read_csv(path or FILE, sep='\t', keep_default_na=False, dtype=str)
    if d['point_id'].duplicated().any():
        raise ValueError('dicotopo_names: duplicate point_id')
    pol = policy or POLICY
    d['label'] = d['name'] if pol != 'headword' else d['headword']
    d['forms'] = d['extra_forms'].map(parse_forms)
    d['prev'] = d['model_key'].map(lambda s: json.loads(s) if s else None)
    return d
