"""CAMENA Poemata: Neo-Latin verse anthologies (Pareus, Adam, Chytraeus, Stigel...), 88 TEI.2 files, 1525-1700.

Licence: CC-BY-SA-4.0, T1 (LICENSE_TIERS CAMENA group; ATTRIBUTION.txt: CC BY-SA 4.0). Raw:
``originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/poemata/``.

What the editors tagged: ``<name type="...">`` in the running text (and in titles / bibliographies), no ``@ref``, no
register. The scanner maps the types as follows (all are mentions, none resolves to a register):
  * ``person`` -> kind person (includes the Christ forms ``Christe``, ``Christi``, tagged person by the editors);
  * ``myth`` (Phoebus, Musae, Iuppiter...) -> kind person, tag ``name@myth`` (mythological figures);
  * ``religion`` (Deus, Christus, Samson, Tobias, Susanna...) -> kind person, tag ``name@religion`` (divine and
    biblical figures);
  * ``place`` / ``river`` -> kind place;
  * ``nation`` (Germani, Gallia, Nemetum...) -> kind org, tag ``name@nation`` (peoples, sometimes region names; kept
    apart from places and persons);
  * ``org`` -> kind org; the 5 ``type="?"`` tags and untyped ``name`` (modern markup credits in the header) are not read.
Names are inflected Latin forms (``Haidelbergae``); no normalisation is attempted (the atlas needs a linking pass).
Dating: imprint year in ``sourceDesc/bibl`` (header); files without a year stay undated.
v2 (issues 249-250): the class of a tag comes from its type (myth -> mythological, religion -> biblical, nation -> people: ``person_class``);
the 90 title-statement names that repeat in each of the 18 chunk files of Pareus' *Musae fugitivae* (header persons under titleStmt /
sourceDesc) are excluded from the default counts (``excluded_reason`` 'chunk_header'): a file is a chunk of a book, not a work.
Only 40 of the 88 harvested files (of 949 POEMATA files in the repository) carry any tag: the edition is a sample.
"""
from __future__ import annotations

from .. import common as C
from .. import helpers_batch0 as H
from .generic_tei import GenericTEI

SLUG = 'camena_poemata'
COLLS = ['originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/poemata']
RENASCOR_IDS = ['camena']
LICENCE = 'CC-BY-SA-4.0'
TIER = 'T1'
DESCRIPTION = 'Neo-Latin verse anthologies: typed names (persons, places, myth, religion, nations)'
OPTIONS = {'type_map': {'myth': 'person', 'religion': 'person', 'nation': 'org', 'river': 'place'}}


class Extractor(GenericTEI):
    def doc_date(self, root, rec, ctx):
        d = C.header_dates(root)
        return d if d.known else H.sourcedesc_year(root)

    def adjust(self, m, hit):
        if m.zone == 'hdr':
            m.context = None
            if m.kind == 'person' and ({'titleStmt', 'sourceDesc'} & set(hit.ancestors)):
                m.excluded_reason = 'chunk_header'
        return m


EXTRACTOR = Extractor()
