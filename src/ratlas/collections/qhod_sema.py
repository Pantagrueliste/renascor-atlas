"""QhoD, sub-project 'sema': the 16 letters of Selim II to Maximilian II and Mustafa Pasa (1567-1574), CC-BY 4.0.

Source: the GitHub clone of qhod-edition-data (qhod/sema/Selim_II_Urkunde_*.xml) with the shared register
qhod/qhod/QhoDstandOff.xml (persons, places, orgs of all QhoD sub-projects, APIS/MPR). The other sub-projects
(vipa, kuse, vrancic-data) are not covered by this module; kuse and vrancic-data are not in the harvested clone.
Mentions: header correspAction (sender / addressee persName with GND, sent_from / sent_to placeName with GeoNames),
the regest in profileDesc/abstract (<rs type=person|place ref="#mprNNNN">, zone hdr: it is the editors' German summary)
and the rs of the text. titleStmt/author 'Selim II.' (no ref) is role author; the same person is the sender.
Refs '#mprNNNN' resolve against the shared register (local_key 'qhod_sema:mprNNNN'), whose entries carry GND,
GeoNames, Wikidata ids and, for places, <location>/<geo> coordinates. Entities: only the register entries cited by
the sema letters are written (prepare() filters), the others belong to the other sub-projects. The APIS/MPR url ids
are replaced by one 'mpr:<n>' per entity in auth.other so that they do not inflate the authority counts.
Dating: correspAction[sent]/date (@notBefore/@notAfter, Hijri calendar attribute ignored), date_source 'letter'.
Licence: CC-BY-4.0 (TEI availability, Zenodo 14449603 and README); LICENSE_TIERS QHOD T1.

v2: re-harvest of 2026-10-17 from the Zenodo v3 archive (2.0, 13 Dec 2024), which is newer than the GitHub clone: the shared
register grew from 270 to 847 persons and from 166 to 624 places, and two sub-projects in the period were missing:
``kuse`` (Hanns Jakob Kurtz von Senftenau, 1623/24, 76 files) and ``vrancic-data`` (Faust Vrancic, 1553-1568, the
``published/documents-*`` files; the foreword and introduction files are modern commentary and are not read). They use the
same ``<rs type=person|place ref="#mprNNNN">`` convention and the same register. ``vipa`` (1719/20, out of period) is not taken.
"""
from __future__ import annotations

import re

from .. import common as C
from .. import helpers_v2 as V
from .generic_tei import GenericTEI

SLUG = 'qhod_sema'
COLLS = V.pick(['qhod/sema', 'qhod/qhod/QhoDstandOff.xml'],
               [V.NEW + '/qhod_sema/sema/Selim_II_Urkunde', V.NEW + '/qhod_sema/qhod/QhoDstandOff.xml',
                V.NEW + '/qhod_sema/kuse/', V.NEW + '/qhod_sema/vrancic-data/published/documents'])
RENASCOR_IDS = ['qhod']
LICENCE = 'CC-BY-4.0'
TIER = 'T1'
DESCRIPTION = 'QhoD / SeMa: letters of Selim II 1567-1574 with the APIS/MPR register (GND, GeoNames, coordinates)'
OPTIONS = {}

_MPR = re.compile(r'#(mpr\d+)')


def _mpr_ids(ent):
    return ent.local_key.split(':', 1)[1]


class Extractor(GenericTEI):
    def is_register_file(self, rec, ctx):
        return rec.path.endswith('QhoDstandOff.xml')

    def prepare(self, ctx):
        ents = super().prepare(ctx)
        cited = set()
        for rec in ctx.all_files:
            if rec.path.endswith('QhoDstandOff.xml'):
                continue
            cited.update(_MPR.findall(rec.read().decode('utf8', 'replace')))
        out = []
        for e in ents:
            if _mpr_ids(e) not in cited:
                continue
            others = [x for x in e.auth['other'] if not x.startswith('url:https://mpr.acdh.oeaw.ac.at')
                      and not x.startswith('url:http://mpr.acdh.oeaw.ac.at')]
            mpr = _mpr_ids(e)[3:]
            e.auth['other'] = others + [f'mpr:{mpr}']
            out.append(e)
        return out


EXTRACTOR = Extractor()
