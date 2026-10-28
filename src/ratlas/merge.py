"""Integration stage: merge the per-collection outputs into ``out/all`` and build the derived tables.

    python -m ratlas merge [--out-root DIR] [--dest DIR] [--no-overlaps]

Reads ``<out_root>/<slug>/{mentions,entities}.parquet`` + ``stats.json`` and writes to ``<dest>`` (default
``<out_root>/all``):

* ``mentions.parquet``  all rows, + ``canonical`` (False = the same source counted by another collection / module, see
  ``overlap_note``) + ``overlap_note``. Totals are always computed over ``canonical`` rows.
* ``entities.parquet``  all register entries, + ``n_mentions_canonical``.
* ``collections.tsv``   one row per collection (counts, shares, inventory comparison).
* ``deviations.tsv``    collections whose tagged text differs from the scan inventory by more than 10 %.
* ``places_local.parquet``, ``persons_local.parquet``, ``orgs_local.parquet``  edition-scoped distinct entities.
* ``duplicate_sources.tsv``, ``harvest_dup_pairs.tsv``, ``text_overlap_pairs.tsv``, ``uncovered_colls.tsv``, ``totals.json``.

Nothing here reads harvested XML: only the parquet outputs and scan.jsonl.gz.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import common as C
from . import generic_terms as GEN
from .model import AUTH_KEYS, ENTITY_SCHEMA, MENTION_SCHEMA, PERSON_CLASSES, ZONE_CLASSES

Q0, QN = 1450, 10                     # ten quarter-centuries 1450-1474 ... 1675-1700
QCOLS = [f'q{Q0 + 25 * i}' for i in range(QN)]
LETTER_ROLES = ('sender', 'addressee', 'sent_from', 'sent_to')
ROLE_COLS = ('mention', 'author', 'sender', 'addressee', 'sent_from', 'sent_to', 'signer', 'cast')

MENTION_OUT = MENTION_SCHEMA.append(pa.field('canonical', pa.bool_())).append(pa.field('overlap_note', pa.string())) \
    .append(pa.field('generic_with_id', pa.bool_()))      # v2.2
ENTITY_OUT = ENTITY_SCHEMA.append(pa.field('n_mentions_canonical', pa.int64()))

# Text-level overlaps (different bytes, same text) decided after reading text_overlap_pairs.tsv:
# {loser_slug: (winner_slug, reason)}; every row of the loser is non-canonical.
TEXT_OVERLAP_LOSERS: dict[str, tuple[str, str]] = {}

# Same text, different bytes, identical file names (checked by hand): (loser, winner, why). The loser's rows for files
# whose basename also exists in the winner are non-canonical.
BASENAME_OVERLAPS: list[tuple[str, str, str]] = [
    ('freem_bvh', 'epistemon_bvh', 'BVH Epistemon texts in the FreEMmax_OA copy and in tei-fr/Projects/BVH (richer tagging wins)'),
]

# Reasons for deviations from the inventory (slug -> text); filled from the batch reports and this run.
DEVIATION_REASONS: dict[str, str] = {
    'bdpn': 'two TEI files missing from the scan (carta, DiscursoEthereologico; no .xml extension) are read by the module (+1,120); the Lunario file is now read too',
    'betamasaheft_geez': 'ethnic groups (persName@ethnic, 637) and institutions (placeName@institution, 36) are re-kinded to org by design',
    'biblioteca_italiana': 'only 2 of 989 files carry typed rs (name/luogo, not in the generic type map); the 8,563 untyped <name> of the other files cannot be classified without NER',
    'camena_poemata': 'name@myth/religion mapped to person and name@nation to org (types outside the generic map); counts match the study',
    'camena_thesaurus': 'name@myth/religion mapped to person (types outside the generic map); counts match the study',
    'casebooks': 'persons exist only in case headers (one mention per person and role from cb:consultation: 245,580 nonstandard-role rows); places 63,011 vs 63,066 in the scan (-0.09%, batch0), a count the scan keeps outside txt/hdr',
    'celt': 'name@deity / nomen sacrum read as person, person:collective as org (types outside the generic map)',
    'cidtc': 'header persName under titleStmt/title duplicates correspAction and is skipped; text +2% because plural keys give one row per id',
    'ckcc_epistolarium': 'adds 20,000 letters of structured sender/addressee/place from interpGrp (header rows, not tags) and 11,019 tags of the 7 .txt corpora; persName that resolve to an org entry become org',
    'corpus_corporum': 'name@myth/religion -> person, nation/populus -> org (types outside the generic map); header author with VIAF ref read as person (794 files); 1,460 untyped <name> skipped',
    'cosuiza': 'header orgName = repository, dropped by design',
    'croala': 'header persName/orgName are author life dates, bibliography, respStmt and publishers (creation/date is used for dating); text matches (1,809 vs 1,814)',
    'devonshire_ms': 'untyped <name ref> read as person (363); 199 header settlement tags (archive) dropped',
    'dibilit_german': 'only the 47 early-modern texts of 2,084 files are read; header persName/orgName are the modern editors and publishers, the author is taken once from titleStmt',
    'dipko_schwarzenhorn_1649': 'header orgName = repository/publisher, dropped by design',
    'disco_sonnets': 'header-zone author and birth/death place rows (1,088 + 1,330) built from the sonnet metadata and prosopography.xml, not from persName/placeName tags the inventory counts',
    'dracor_eng': 'cast list (castItem, 11,145) and header authors; the inventory only counts header persName',
    'dracor_ita': 'cast list (castItem, 1,008) and header authors; "(Anonimo)" authors dropped',
    'dta_1600': 'header persName/orgName = editors, publishers and the sourceDesc copy of the author; text matches (6,252 vs 6,255)',
    'emmo_folger': 'untyped <name> read as person (602)',
    'epistemon_bvh': 'untyped <name> (29,949) read as person, allegorie as person; text key prefixes carry the fictional/mythical class',
    'ferdinand_i': 'header author duplicates the sender and biblFull author is the principal (dropped); header orgName "nognd" dropped',
    'freem_bvh': 'header persName = modern editors; 5,217 untyped <name> skipped; 5 files (934 mentions) also exist in epistemon_bvh (flagged non-canonical)',
    'freem_other': 'Descartes salutations: 234 addressee names kept of 364 untyped <name>; header names are modern editors',
    'fruchtbringende_briefe': 'header orgName = respStmt (modern project), dropped',
    'h15corpus': 'text is machine-tagged (POS) and not read; header orgName = repository; empty/placeholder header places dropped',
    'hearth_tax': 'header orgName = repository/publisher, dropped; header places include repository (role repository)',
    'koenigsfelden': 'header persName = respStmt/seriesStmt (modern editors); text matches exactly (16,167)',
    'leibniz_reihe_viii': 'mentions are <index> markers (Namensregister/Ortsregister) which the tag inventory does not count',
    'letradas': 'persons/places taken from correspAction; the text is tokenised with no entity tags; archive orgName dropped',
    'lettres_calvin': 'small counts; text 614 vs 622 in colls.json (-1.3%); Bonnet commentary notes kept',
    'luene_nonnenbriefe': 'places: empty <rs type=place/> placeholders and the repository settlement are dropped; text matches colls.json (978 vs 977)',
    'magdeburger_zenturien_methode_hab': 'sender/addressee taken from the rs persons of the msDesc title; +10.6% vs estimate',
    'map_of_early_modern_london': 'untyped <name ref="mol:..."> read as person (23,014) and <ref target="mol:..."> as place (20,751): tags the generic inventory does not count',
    'pceec2': 'non-TEI .psd treebank (not in the scan): sender/addressee from the LETTER records, 84 files',
    'polemicagongorina_obvil': 'name@authority / name@polemista (1,152) read as person on top of persName@key (2,620); 26 header authors',
    'qhod_sema': 'header persName in the scan are respStmt/editors; correspAction persons are read (txt 175 vs 179)',
    'reed': 'rs type values are EATS categories (outside the generic type map); read through the EATS register in entities.xml (17,495 mentions in 2,483 records; introductions skipped)',
    'religionsfrieden_quellenedition': 'estimate for typed rs is approximate; module counts match the study (2,479 persons in period); header orgName dropped',
    'rubens_online': 'header placeName of archivalia are the repository (role repository, 526) which the text inventory does not count',
    'sandrart': 'header orgName = publisher, dropped; text matches exactly (42,066 / 13,832)',
    'sozinianische_briefwechsel': 'header persName/orgName = publisher (Johannes a Lasco Bibliothek), respStmt; text: rs type term/source/letter ignored',
    'ssrq_swisslaw': 'header persName/orgName = respStmt/editor/seriesStmt (modern editors); text matches exactly (34,788 / 12,484)',
    'st_galler_missiven': 'header persName = revisionDesc/change (modern); text matches exactly',
    'suriano': 'sender/addressee/sent_from/sent_to/repository come from the untyped header <name>/<settlement> of each letter (2,175 + 2,175)',
    'tagebuecher_christian_ii_anhalt': 'estimate for typed rs split between text and register is approximate; module counts equal rs@person 107,909 / rs@place 86,458 / rs@org 3,752 (+13,748 index@place entry_place rows)',
}



# Why the v2 counts differ from v1, per collection (audit of 2026-10-16, owner decisions of 2026-10-16). Written from the module changes.
# Rules that apply to EVERY collection (and are not repeated): the refined ``zone`` and ``person_class`` columns; ``excluded`` rows
# (editorial commentary, apparatus, footnote, regest, repository, characters, celestial bodies, placeholders, empty surfaces) stay in the
# table and leave the default counts; tags without a letter and without a reference are dropped; a tag whose register entry has another
# kind keeps its kind and inherits no id (key '<entry>~<kind>'); a date range wider than 20 years has no doc_year and a wide range that
# straddles 1450 or 1700 has in_period None; surfaces lose line-break hyphens and trailing punctuation; nested editorial notes are not part
# of a surface.
V2_REASONS: dict[str, str] = {
    'abacus': 're-harvest 2026-10-17 (abc-data 0edf72e): only the TEI headers changed, the mentions are identical; biblical (945) and mythological (85) persons are classed',
    'bdpn': 'celestial bodies and cosmological terms (87 of 211 place entries, 1,784 mentions) are person_class celestial and excluded from the place counts (never to be geocoded)',
    'betamasaheft_geez': 'Places/Persons authority files (ff57906 / 802eb52) give the 402 LOC and 261 PRS ids names, Ge\'ez variants and 8 coordinates; running date of the chronicle instead of one min-max range; ETH peoples are class people',
    'biblioteca_italiana': 'dated by the header century code (L\'Ermenegildo 700 = Settecento: out of period); the untyped <name> are still not read (no NER)',
    'bullinger': 're-harvest dc9fb52 (HTR re-import, 10,227 letters differ) and the project register (6,043 persons, 3,077 places, 91 org groups): ids now have names, GND, GeoNames and coordinates; @cert carried; the editors\' footnotes and the regests are excluded',
    'camena_cera': 'Burman volumes (imprint 1727) are undated instead of out of period; letter heads give sender / addressee / destination roles',
    'camena_poemata': 'myth / religion / nation classes; repeated chunk-header names excluded',
    'camena_thesaurus': 'myth / religion classes (36% of the persons are gods and Christ)',
    'casebooks': 'one mention per (case, person): 152,681 duplicated role rows removed; PLACE register still not obtainable (site answers 404)',
    'cavriana': 're-harvest 581aab1 (TGN ids re-derived, 4 letters added, 4 removed, groups and ethnic rs read); repository city excluded',
    'celt': 'divine and mythical beings (nomen sacrum, deity, angel...) are class mythological; ranges wider than 20 years have no doc_year and are not in_period True when they straddle 1450 / 1700',
    'cidtc': 'entities named from the header @ana (not the commonest epithet); epithet-only persons flagged',
    'ckcc_epistolarium': '~280 letters present in two corpora (same date, sender, recipient) are counted once; old place codes (denha004...) resolved to the register',
    'controversia_confessio': 'modern commentary (note1, annotation, footnote) and apparatus excluded (45% of the rows); each text dated by its title block, else a range without a midpoint',
    'corpus17_french': 'dramatis personae are class character (489 of 504 persons)',
    'corpus_corporum': 'dates: 400/500 read as centuries for Renaissance authors, post-1700 imprints replaced by the author\'s death year; trailing punctuation dropped; myth / religion / nation classes',
    'cosuiza': 'empty <editor><persName/></editor> rows dropped',
    'croala': 'author variants with the same @ref collapsed; Anonymus / Auctores varii are placeholders',
    'dasypodius_dictionarium_latinogermanicum': 'no extraction change (Wikipedia-title QIDs are a reconcile step)',
    'dibilit_german': 'dated by the DTA first-publication date instead of the author\'s period; header rows carry no context',
    'dracor_eng': 're-harvest da50c42 (TEI as served by DraCor since 2026-07-21): 741 plays with a cast (468 before); the cast is class character and excluded (93% of the person rows)',
    'dracor_ita': 'cast list is class character and excluded',
    'elan_crenum': 'split initials repaired (3,149 of 6,327 surfaces); register still unavailable (gricad-gitlab robots.txt)',
    'elan_fnvr': 'all register persons flagged fictional / mythological (class character / mythological)',
    'elan_polygrenaille': 'unchanged: the site now shows 294 persName that the June TEI export lacks, but the TEI cannot be fetched (gricad-gitlab robots.txt)',
    'epistemon_bvh': 'class from key prefix (persf / persm) and a lexicon of divine / mythical / fictional names; orgName peuple is class people',
    'ethica_complementoria': 'variant files of the same transcription counted once',
    'fruchtbringende_briefe': 'register-person / register-place (HAB ed000213) loaded: names, 715 GND, 276 TGN, 262 coordinates',
    'freem_bvh': 'imprint-only dating (a 1996 reference no longer widens the range)',
    'freem_other': 'Descartes salutation headings reduced to the addressee\'s name',
    'germanc': 'placeholders and deictics excluded; no extraction change',
    'grand_tour': 'each mention dated by the preceding dated diary entry instead of one midpoint per journal',
    'h15corpus': 'archive settlements (repository, 43% of the place rows) excluded; a name whose <geo> contradict each other in different letters gets no TEI point (v2.4; v2: their median)',
    'hainhofer_reisen': 'LOD register (heiDATA v26): 3,502 persons, 1,036 places, 55 org groups with GND / GeoNames / TGN; editor annotations are commentary (excluded)',
    'hainhofer_stammbuch': 'site registers (201 persons with GND, 24 places with GeoNames); 81% of the persons are in the modern commentary (excluded); the Ort div gives 103 place mentions (role written_at)',
    'hearth_tax': 'one key per taxpayer row (no homonym piles); empty rows dropped; vacant houses and the archive address excluded; Norfolk certificates dated by their taxation dates',
    'kaiserin_eleonora_magdalena': 're-harvest fc4898f and the emt-entities register (1,391 persons, 471 places, 72 institutions); the archive country and city (repository, 31% of the place rows) excluded',
    'karlstadt_kritische_gesamtausgabe': 'register (ed000216/register) loaded: 379 GND, 114 GeoNames; footnotes (27% of the rows) and apparatus excluded',
    'koenigsfelden': 'no extraction change',
    'leibniz_reihe_viii': 'keys without parentheses, life dates or dash variants (73 split groups merged); index markers inside the apparatus (819) excluded; empty terms dropped',
    'letradas': 'no extraction change',
    'lettres_calvin': 'sent_from only for the place of sending; "Envoyée à" is sent_to; places in notes are mentions',
    'luene_nonnenbriefe': 'person register (599 entries) loaded under kind keys; "unbekanntes Kloster" is a placeholder',
    'magdeburger_zenturien_methode_hab': 'register-person (HAB ed000086) loaded: 315 GND; sender / addressee also when the title has a line break after "an"',
    'map_of_early_modern_london': '906 more place files (Edition 7.0): 224 ids with GeoJSON; opaque place keys fall from 11,313 to under 700; Wikipedia list links no longer used as ids unless the title matches; 11 undated texts dated from resp/date; editorial notes (819) excluded',
    'obvil_critique': 'La Bruyère\'s persons are fictional characters (class character)',
    'old_bailey': 'untyped persName without surname interp dropped; each defendant / victim / witness has its own trial-local key; nested tags read innermost-first',
    'paesi_novamente': 'entities named from the editor\'s slug (kozhikode, cap_vert...) so that a geocoder can use the slug',
    'pceec2': 're-harvest 99b7012: records identical, new SHA; decade dates ("1570S") parsed as 1570-1579',
    'poliziano_sappho': 'no extraction change',
    'polemicagongorina_obvil': 'empty quotation markers (34) are role quote_source and excluded; address forms dropped',
    'postscriptum': 'cdd.xml register (5,069 persons and institutions); one place entity per coordinate; S.l. / Anonimo<n> / empty strings are placeholders',
    'qhod_sema': 'Zenodo 2.0 archive: register 847 persons / 624 places; kuse (1623/24) and vrancic-data (1553-1568) added; vipa (1719/20) not read',
    'reed': 'record-heading localities (45% of the place tags) are role provenance and excluded',
    'rejestr_poznan': 'the register is transcribed twice (ASK91, ASK3): the parallel witness is excluded (1,330 mentions remain)',
    'religionsfrieden_quellenedition': 'no extraction change (parallel texts are flagged by the text-overlap step)',
    'rubens_online': 'HAB registers (person, place, corporate) loaded: 482 GND, 211 TGN; the archive places (526, repository) excluded',
    'rup': 're-harvest 4b4815c: one more poem; persons.xml#P0023 dangles upstream',
    'sandrart': 'no extraction change',
    'selbstzeugnisse_hab_diarium': 'each mention dated by the preceding dated entry (1,697 dates) instead of the midpoint 1614',
    'selbstzeugnisse_hab_sz2': 'each mention dated by the preceding date, outliers ignored',
    'sozinianische_briefwechsel': 'the regest (profileDesc/abstract, 20% of the rows) is zone regest and excluded; nested commentary notes no longer in surfaces',
    'ssrq_swisslaw': 'entity key without the .NN variant suffix; document dated by its original, not the copies; register still not available (owner decision: leave as is)',
    'st_galler_missiven': 'local register from the header forms (1,721 entities named by the normalised header form)',
    'suriano': 'the archive (Archivio di Stato di Venezia, 725 rows) is repository and excluded',
    'tagebuecher_christian_ii_anhalt': 'tags whose register entry has another kind (180 "persons" that are places or peoples) keep their kind and inherit no id (class metonym)',
    'urfehdebuecher_basel_x': 'each mention dated by the entry\'s date (one year per entry) instead of 1566',
}


def deviations_vs_v1(ct: pd.DataFrame, v1_collections: Path, dest: Path) -> pd.DataFrame:
    """Per collection: v1 -> v2 canonical mentions (by kind), the default (non-excluded) counts, what is excluded, and why."""
    v1 = pd.read_csv(v1_collections, sep='\t').set_index('slug')
    rows = []
    for _, r in ct.iterrows():
        s = r['slug']
        a = v1.loc[s] if s in v1.index else None
        row = {'slug': s}
        for col in ('mentions_canonical', 'person', 'place', 'org'):
            row[f'v1_{col}'] = int(a[col]) if a is not None else None
            row[f'v2_{col}'] = int(r[col] if col != 'mentions_canonical' else r['mentions_canonical'])
            row[f'delta_{col}'] = (row[f'v2_{col}'] - row[f'v1_{col}']) if a is not None else None
        row['v2_default'] = int(r['mentions_default'])
        row['v2_default_person'] = int(r['person_default'])
        row['v2_default_place'] = int(r['place_default'])
        row['v2_default_org'] = int(r['org_default'])
        row['v2_excluded'] = int(r['mentions_excluded'])
        row['excluded_reasons'] = r.get('excluded_reasons', '')
        for zc in ZONE_CLASSES:
            row[f'zone_{zc}'] = int(r[f'zone_{zc}'])
        row['reason'] = V2_REASONS.get(s, 'no collection-specific change; the general v2 rules apply' if a is not None else 'new collection')
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(dest / 'deviations_vs_v1.tsv', sep='\t', index=False)
    return out


# collections whose place surfaces are the scribe's address strings ('of Stonystratford', 'in Southwork', 'at bishops gat'): the leading
# preposition is dropped from the key (all collections) and from the displayed name (these ones). Audit issue 147: 3,693 of 4,148 casebooks places.
PREPOSITION_SLUGS = {'casebooks'}
_PREP = re.compile(r'^(?:of|in|at|by|from|to|near)\s+(?=\S{3})', re.I)


def _norm(s: Optional[str]) -> str:
    if not isinstance(s, str) or not s:
        return ''
    s = unicodedata.normalize('NFKD', s.casefold())
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'[\W_]+', ' ', s).strip()
    if re.match(r'^(?:of|in|at|by|from|to|near) \S{3}', s) and len(s) > 5:        # casebooks addresses: 'of Ashbey' = Ashbey (58k mentions)
        s = s.split(' ', 1)[1].strip()
    return s


def _t(msg: str):
    print(f'[merge {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# scan inventory
# ----------------------------------------------------------------------------------------------------------------------
def load_scan_full() -> list[dict]:
    rows = []
    for sp in C.scan_files():                 # v2: the study scan, then the scan of the 2026-10-17 re-harvest
        with gzip.open(sp, 'rt', encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                if 'sha1' not in r:
                    continue
                for k in ('idno', 'geo', 'ns', 'root'):
                    r.pop(k, None)
                r['dup'] = bool(r.get('dup'))
                r.setdefault('types', {})
                r.setdefault('ent_text', 0)
                rows.append(r)
    return rows


def patch_iter_scan(rows: list[dict]):
    """Serve C.iter_scan from memory so that the modules' iter_files do not re-read the 380k-line scan 78 times."""
    def iter_scan(selectors=None, include_dups=False):
        sels = list(selectors) if selectors is not None else None
        for r in rows:
            if r.get('dup') and not include_dups:
                continue
            if sels is not None and not C.matches(r['path'], r['coll'], sels):
                continue
            if C.is_forbidden(r['path']):
                continue
            yield C.FileRecord.from_scan(r)
    C.iter_scan = iter_scan


def _exp_of(r: dict) -> Counter:
    """Expected tagged entities of one scan record, {(zone, kind): n}, zones txt/hdr only.

    persName/placeName/geogName/orgName counts come straight from ``c``. Typed ``rs``/``name`` (scan ``types``:
    '<tag>|<type>', all zones together) are spread over the zones in proportion to that tag's counts per zone, so
    that typed rs inside registers ('reg') do not inflate the text inventory.
    """
    out: Counter = Counter()
    c = r['c']
    for k, v in c.items():
        z, _, tag = k.partition('|')
        if z not in ('txt', 'hdr'):
            continue
        if tag == 'persName':
            out[(z, 'person')] += v
        elif tag in ('placeName', 'geogName') or (tag == 'settlement' and z == 'txt'):
            out[(z, 'place')] += v
        elif tag == 'orgName':
            out[(z, 'org')] += v
    for k, v in r['types'].items():
        tag, _, ty = k.partition('|')
        if tag not in ('rs', 'name'):
            continue
        kind = C.TYPE_MAP.get(ty.lower())
        if not kind:
            continue
        tot = sum(c.get(f'{z}|{tag}', 0) for z in ('txt', 'hdr', 'reg', 'other'))
        if tot <= 0:
            continue
        for z in (('txt', 'hdr') if tag == 'rs' else ('txt',)):     # header <name> = modern editors / funders
            out[(z, kind)] += v * c.get(f'{z}|{tag}', 0) / tot
    return out


# ----------------------------------------------------------------------------------------------------------------------
# helpers on tables
# ----------------------------------------------------------------------------------------------------------------------
def unify_character_keys(tbl: pa.Table) -> tuple:
    """A person that some mention of the edition marks as a character (cast list, a lexicon of fictional names) is a character in every mention
    of the same key (``local_key``, else the same surface): Panurge was excluded 417 times and counted 55 times, because the key prefix
    ``persh`` (historical) or the spelling spared the lexicon. v2 audit, check 6. Returns (table, number of rows changed)."""
    cols = ['kind', 'local_key', 'surface', 'person_class', 'excluded', 'excluded_reason']
    d = tbl.select(cols).to_pandas()
    person = (d['kind'] == 'person').to_numpy()
    if not person.any() or not (d['person_class'] == 'character').any():
        return tbl, 0
    surf = d['surface'].fillna('')
    nm = {x: _norm(x) for x in surf.unique()}
    sk = surf.map(nm)
    key = d['local_key'].where(d['local_key'].notna(), np.where(sk != '', 'surf:' + sk, None))
    chars = set(key[person & (d['person_class'] == 'character').to_numpy() & key.notna().to_numpy()])
    hit = person & key.isin(chars).to_numpy() & (d['person_class'] != 'character').to_numpy()
    if not hit.any():
        return tbl, 0
    d.loc[hit, 'person_class'] = 'character'
    d.loc[hit, 'excluded'] = True
    old = d.loc[hit, 'excluded_reason'].fillna('')
    d.loc[hit, 'excluded_reason'] = [';'.join(dict.fromkeys([x for x in o.split(';') if x] + ['character'])) for o in old]
    for c in ('person_class', 'excluded', 'excluded_reason'):
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, tbl.schema.field(c), pa.array(d[c].to_numpy(), tbl.schema.field(c).type, from_pandas=True))
    return tbl, int(hit.sum())


def exclude_generic(tbl: pa.Table) -> tuple:
    """v2.1: tagged pronouns and common nouns (``generic_terms``) leave the default counts: ``excluded`` True and the reason
    ``pronoun_or_common_noun`` is added (idempotent; the rows stay; a mention already excluded keeps its first reason). A common noun that carries an authority id is an identification
    and is kept; a pronoun never is. Returns (table, number of mentions newly excluded, DataFrame(kind, class, surface_key, n))
    of every generic mention (already excluded for another reason or not)."""
    names = tbl.column_names
    if 'surface' not in names or 'kind' not in names:
        return tbl, 0, pd.DataFrame(columns=['kind', 'class', 'surface_key', 'n'])
    d = tbl.select(['kind', 'surface', 'excluded', 'excluded_reason']).to_pandas()
    ok = d['kind'].isin(['person', 'place']).to_numpy()
    if not ok.any():
        return tbl, 0, pd.DataFrame(columns=['kind', 'class', 'surface_key', 'n'])
    pair = pd.Series(list(zip(d['kind'], d['surface'].fillna(''))))
    cls_of = {p: GEN.generic_kind(p[0], p[1]) for p in pair[ok].unique()}
    cls = pair.map(cls_of).where(ok, None)
    hit = cls.notna().to_numpy()
    if not hit.any():
        return tbl, 0, pd.DataFrame(columns=['kind', 'class', 'surface_key', 'n'])
    a = tbl.column('auth').combine_chunks()
    anyauth = np.zeros(len(d), dtype=bool)
    for k in AUTH_KEYS:
        anyauth |= a.field(k).is_valid().to_numpy(zero_copy_only=False)
    oth = a.field('other')
    anyauth |= (pc.fill_null(pc.list_value_length(oth), 0).to_numpy(zero_copy_only=False) > 0)
    rep = pd.DataFrame({'kind': d['kind'][hit].to_numpy(), 'class': cls[hit].to_numpy(),
                        'surface_key': d['surface'][hit].fillna('').map(GEN._key).to_numpy()})
    rep['n'] = 1
    rep = rep.groupby(['kind', 'class', 'surface_key'], as_index=False)['n'].sum()
    drop = hit & ((cls == 'pronoun').to_numpy() | ((cls == 'common_noun').to_numpy() & ~anyauth))
    have = d['excluded'].fillna(False).astype(bool).to_numpy()          # already out of the counts for another reason: the first reason stays
    new = drop & ~have
    if not new.any():
        return tbl, 0, rep
    d.loc[new, 'excluded'] = True
    d.loc[new, 'excluded_reason'] = [';'.join(dict.fromkeys([x for x in ('' if pd.isna(o) else str(o)).split(';') if x] + [GEN.REASON]))
                                     for o in d.loc[new, 'excluded_reason']]
    for c in ('excluded', 'excluded_reason'):
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, tbl.schema.field(c), pa.array(d[c].to_numpy(), tbl.schema.field(c).type, from_pandas=True))
    return tbl, int(new.sum()), rep


def flag_generic_with_id(tbl: pa.Table) -> tuple:
    """v2.2: ``generic_with_id`` marks a mention whose surface is a common noun (alhier, Stadt, lande, Reich, pays, Schloß, this towne, Meer; Vater, Bruder,
    Kaiser, Gemahlin ...) and that carries an authority id. The mention stays linked: the id says which entity the noun refers to here. But the noun is not a
    name: it is never used as the display name of the entity (``reconcile_build.aggregate_global``) and the flag lets a reader filter these links. Returns
    (table with the boolean column, number flagged). Idempotent."""
    names = tbl.column_names
    flag = np.zeros(tbl.num_rows, dtype=bool)
    if 'surface' in names and 'kind' in names and tbl.num_rows:
        d = tbl.select(['kind', 'surface']).to_pandas()
        ok = d['kind'].isin(['person', 'place']).to_numpy()
        if ok.any():
            pair = pd.Series(list(zip(d['kind'], d['surface'].fillna(''))))
            cls = pair[ok].drop_duplicates()
            cls_of = {p: GEN.generic_kind(p[0], p[1]) for p in cls}
            gen = pair.map(cls_of).where(ok, None).eq('common_noun').to_numpy()
            if gen.any():
                a = tbl.column('auth').combine_chunks()
                anyauth = np.zeros(len(d), dtype=bool)
                for k in AUTH_KEYS:
                    anyauth |= a.field(k).is_valid().to_numpy(zero_copy_only=False)
                oth = a.field('other')
                anyauth |= (pc.fill_null(pc.list_value_length(oth), 0).to_numpy(zero_copy_only=False) > 0)
                flag = gen & anyauth
    if 'generic_with_id' in names:
        tbl = tbl.drop(['generic_with_id'])
    return tbl.append_column('generic_with_id', pa.array(flag)), int(flag.sum())


def exclude_multi_referent(tbl: pa.Table) -> tuple:
    """v2.2: a mention whose ``ref_raw`` names two or more entries of the edition's register (``generic_terms.multi_referent``) is out of the default counts
    (``excluded`` True, reason ``multi_referent`` added; a mention already excluded keeps its first reason first). The extractor kept the first referent as the
    local key, so ``Churfürsten`` or ``Schwestern`` counted for one prince or one sister (Anhalt diary: 6,097 mentions, Eleonora Magdalena: 407). The
    alternative, one row per referent, would count a single mention several times. Returns (table, number newly excluded, DataFrame(coll, kind, surface_key, n_refs, n))."""
    if 'ref_raw' not in tbl.column_names or not tbl.num_rows:
        return tbl, 0, pd.DataFrame(columns=['kind', 'surface_key', 'n_refs', 'n'])
    refs = tbl.column('ref_raw').to_pandas()
    has = refs.fillna('').str.count('#').ge(2) | refs.fillna('').str.contains(r'\w:\S+\s+\S*\w:\S', regex=True)
    hit = np.zeros(len(refs), dtype=bool)
    idx = np.flatnonzero(has.to_numpy())
    for i in idx:
        hit[i] = GEN.multi_referent(refs.iat[i])
    if not hit.any():
        return tbl, 0, pd.DataFrame(columns=['kind', 'surface_key', 'n_refs', 'n'])
    d = tbl.select(['kind', 'surface', 'excluded', 'excluded_reason']).to_pandas()
    rep = pd.DataFrame({'kind': d['kind'][hit].to_numpy(), 'surface_key': d['surface'][hit].fillna('').map(GEN._key).to_numpy(),
                        'n_refs': [len(GEN.referents(r)) for r in refs[hit]]})
    rep['n'] = 1
    rep = rep.groupby(['kind', 'surface_key', 'n_refs'], as_index=False)['n'].sum()
    have = d['excluded'].fillna(False).astype(bool).to_numpy().copy()
    d.loc[hit, 'excluded_reason'] = [';'.join(dict.fromkeys([x for x in ('' if pd.isna(o) else str(o)).split(';') if x] + [GEN.REASON_MULTI]))
                                     for o in d.loc[hit, 'excluded_reason']]
    d.loc[hit, 'excluded'] = True
    for c in ('excluded', 'excluded_reason'):
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, tbl.schema.field(c), pa.array(d[c].to_numpy(), tbl.schema.field(c).type, from_pandas=True))
    return tbl, int((hit & ~have).sum()), rep


def _flat(tbl: pa.Table) -> pd.DataFrame:
    """Mentions table -> DataFrame with the auth struct flattened (a_<scheme>, a_other_n)."""
    cols = [c for c in tbl.column_names if c != 'auth']
    df = tbl.select(cols).to_pandas()
    a = tbl.column('auth').combine_chunks()
    for k in AUTH_KEYS:
        df['a_' + k] = a.field(k).to_pandas()
    oth = a.field('other')
    df['a_other_n'] = pc.fill_null(pc.list_value_length(oth), 0).to_numpy(zero_copy_only=False)
    df['_other'] = None
    idx = np.flatnonzero(df['a_other_n'].to_numpy() > 0)
    if len(idx):
        vals = oth.take(pa.array(idx)).to_pylist()
        df.loc[df.index[idx], '_other'] = pd.Series(vals, index=df.index[idx], dtype=object)
    return df


def _any_named(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    for k in AUTH_KEYS:
        m |= df['a_' + k].notna()
    return m


LOCAL_SCHEMA = pa.schema([
    ('coll_slug', pa.string()), ('renascor_id', pa.string()), ('entity_key', pa.string()), ('key_type', pa.string()),
    ('kind', pa.string()), ('name', pa.string()), ('variants', pa.list_(pa.string())), ('norm_name', pa.string()),
    ('n_mentions', pa.int64()), ('n_in_period', pa.int64()), ('n_undated', pa.int64()), ('n_files', pa.int64()),
    ('first_year', pa.int32()), ('last_year', pa.int32()),
    *[(c, pa.int64()) for c in ['pre1450'] + QCOLS + ['post1700', 'undated_year']],
    *[(f'n_{r}', pa.int64()) for r in ROLE_COLS + ('other',)],
    ('n_txt', pa.int64()), ('n_hdr', pa.int64()),
    *[(k, pa.string()) for k in AUTH_KEYS], ('other_auth', pa.list_(pa.string())),
    ('lat', pa.float64()), ('lon', pa.float64()), ('coord_source', pa.string()), ('has_coords', pa.bool_()),
    ('birth', pa.string()), ('death', pa.string()), ('note_type', pa.string()),
    ('licence', pa.string()), ('tier', pa.string()),
    ('person_class', pa.string()), ('n_default', pa.int64()), ('n_in_period_default', pa.int64()),   # v2
])


def _year_bin(y: pd.Series) -> pd.Series:
    out = pd.Series('undated_year', index=y.index, dtype=object)
    ok = y.notna()
    yy = y[ok].astype(int)
    b = pd.Series('post1700', index=yy.index, dtype=object)
    b[yy < Q0] = 'pre1450'
    mid = (yy >= Q0) & (yy <= 1700)
    idx = np.minimum((yy[mid] - Q0) // 25, QN - 1)
    b[mid] = [QCOLS[i] for i in idx]
    out[ok] = b
    return out


def local_table(df: pd.DataFrame, ents: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Edition-scoped distinct entities of ``kind`` from the canonical mention rows of ONE collection."""
    d = df[(df['kind'] == kind) & df['canonical']].copy()
    if d.empty:
        return pd.DataFrame(columns=[f.name for f in LOCAL_SCHEMA])
    uniq = d['surface'].fillna('').unique()
    nmap = {s: _norm(s) for s in uniq}
    d['_n'] = d['surface'].fillna('').map(nmap)
    has_key = d['local_key'].notna()
    d['entity_key'] = np.where(has_key, d['local_key'], 'surf:' + d['_n'])
    d['key_type'] = np.where(d['in_register'].fillna(False).astype(bool), 'register',
                             np.where(has_key, 'opaque', 'surface'))
    d = d[has_key | (d['_n'] != '')]            # nothing to identify an empty, unreferenced tag by
    if d.empty:
        return pd.DataFrame(columns=[f.name for f in LOCAL_SCHEMA])
    if 'part' not in d:                       # tolerate v1-format mention tables
        d['part'] = d['zone']
    if 'excluded' not in d:
        d['excluded'] = False
    if 'person_class' not in d:
        d['person_class'] = None
    d['_ex'] = d['excluded'].fillna(False).astype(bool)
    d['_ip'] = (d['in_period'] == True).astype('int64')   # noqa: E712
    d['_ipd'] = d['_ip'] * (~d['_ex']).astype('int64')
    d['_dflt'] = (~d['_ex']).astype('int64')
    d['_un'] = d['in_period'].isna().astype('int64')
    d['_q'] = _year_bin(d['doc_year'])
    d['_r'] = d['role'].where(d['role'].isin(ROLE_COLS), 'other')
    g = d.groupby('entity_key', sort=False)
    out = pd.DataFrame({
        'coll_slug': g['coll_slug'].first(), 'renascor_id': g['renascor_id'].first(),
        'key_type': g['key_type'].first(), 'kind': kind,
        'n_mentions': g.size(),
        'n_in_period': g['_ip'].sum(), 'n_undated': g['_un'].sum(),
        'n_default': g['_dflt'].sum(), 'n_in_period_default': g['_ipd'].sum(),
        'n_files': g['source_path'].nunique(),
        'first_year': g['doc_year'].min(), 'last_year': g['doc_year'].max(),
        'licence': g['licence'].first(), 'tier': g['tier'].first(),
    })
    for k in AUTH_KEYS:
        out[k] = g['a_' + k].first()
    for col, lab in (('_q', None), ):
        ct = pd.crosstab(d['entity_key'], d['_q'])
        for c in ['pre1450'] + QCOLS + ['post1700', 'undated_year']:
            out[c] = ct[c].reindex(out.index).fillna(0).astype('int64') if c in ct else 0
    ct = pd.crosstab(d['entity_key'], d['_r'])
    for r in ROLE_COLS + ('other',):
        out[f'n_{r}'] = ct[r].reindex(out.index).fillna(0).astype('int64') if r in ct else 0
    pcl = d[d['person_class'].notna()].groupby('entity_key')['person_class'].agg(lambda s: s.value_counts().index[0])
    out['person_class'] = pcl.reindex(out.index)
    ct = pd.crosstab(d['entity_key'], d['part'])
    for z in ('txt', 'hdr'):
        out[f'n_{z}'] = ct[z].reindex(out.index).fillna(0).astype('int64') if z in ct else 0
    # surfaces: most frequent -> name, top 5 -> variants
    sc = d[d['surface'].fillna('') != ''].groupby(['entity_key', 'surface'], sort=False).size().reset_index(name='n')
    sc = sc.sort_values(['entity_key', 'n'], ascending=[True, False]).groupby('entity_key').head(5)
    top = sc.groupby('entity_key')['surface'].apply(list)
    out['variants'] = top.reindex(out.index)
    out['variants'] = out['variants'].map(lambda v: v if isinstance(v, list) else [])
    if kind == 'place' and set(d['coll_slug'].unique()) & PREPOSITION_SLUGS:
        out['variants'] = out['variants'].map(lambda v: [_PREP.sub('', x) for x in v])
    out['name'] = out['variants'].map(lambda v: v[0] if v else None)
    # other auth ids
    o = d[d['a_other_n'] > 0][['entity_key', '_other']]
    if len(o):
        oo = o.explode('_other').dropna().drop_duplicates()
        oa = oo.groupby('entity_key')['_other'].apply(lambda s: sorted(s)[:8])
        out['other_auth'] = oa.reindex(out.index)
    else:
        out['other_auth'] = None
    out['other_auth'] = out['other_auth'].map(lambda v: v if isinstance(v, list) else [])
    # register data: name, coords, dates, note_type, auth
    for c in ('lat', 'lon'):
        out[c] = np.nan
    for c in ('coord_source', 'birth', 'death', 'note_type'):
        out[c] = None
    if ents is not None and len(ents):
        e = ents.drop_duplicates('local_key').set_index('local_key')
        inreg = out.index.intersection(e.index)
        if len(inreg):
            sub = e.loc[inreg]
            nm = sub['name']
            out.loc[inreg, 'name'] = nm.where(nm.notna() & (nm != ''), out.loc[inreg, 'name'])
            for c in ('lat', 'lon', 'coord_source', 'birth', 'death', 'note_type'):
                out.loc[inreg, c] = sub[c]
            for k in AUTH_KEYS:      # register ids fill what the tags lacked
                out.loc[inreg, k] = out.loc[inreg, k].where(out.loc[inreg, k].notna(), sub['a_' + k])
            if 'variants' in sub:
                for key, vs in sub['variants'].items():
                    cur = list(out.at[key, 'variants'])
                    for v in (vs if vs is not None else []):
                        if v not in cur and len(cur) < 8:
                            cur.append(v)
                    out.at[key, 'variants'] = cur
    out['has_coords'] = out['lat'].notna() & out['lon'].notna()
    out['norm_name'] = out['name'].map(_norm)
    out = out.reset_index()
    out['first_year'] = out['first_year'].astype('Int32')
    out['last_year'] = out['last_year'].astype('Int32')
    return out[[f.name for f in LOCAL_SCHEMA]]


def _to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)


# ----------------------------------------------------------------------------------------------------------------------
# main merge
# ----------------------------------------------------------------------------------------------------------------------
def merge(out_root: Path, dest: Path, overlaps: bool = True, v1_collections: Optional[Path] = None) -> dict:
    out_root, dest = Path(out_root), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    slugs = sorted(p.name for p in out_root.iterdir()
                   if p.is_dir() and not p.name.startswith('_') and p.name != dest.name
                   and (p / 'mentions.parquet').exists() and (p / 'stats.json').exists())
    _t(f'{len(slugs)} collections')
    stats = {s: json.load(open(out_root / s / 'stats.json')) for s in slugs}
    failures = {s: st['errors'][:5] for s, st in stats.items() if st.get('file_errors')}

    scan = load_scan_full()
    patch_iter_scan(scan)
    _t(f'scan: {len(scan)} records')
    by_path = {r['path']: r for r in scan}

    # ---- pass 1: sources per slug -> sha1 overlaps --------------------------------------------------------------
    src_counts: dict[str, Counter] = {}
    forbidden_rows = Counter()
    for s in slugs:
        t = pq.read_table(out_root / s / 'mentions.parquet', columns=['source_path'])
        vc = pc.value_counts(t['source_path']).to_pylist()
        src_counts[s] = Counter({v['values']: v['counts'] for v in vc})
        for p, n in src_counts[s].items():
            if C.is_forbidden(p):
                forbidden_rows[s] += n
    groups: dict[str, list] = defaultdict(list)          # sha1 -> [(slug, path, n, dup)]
    for s, cnt in src_counts.items():
        for p, n in cnt.items():
            r = by_path.get(p)
            if r:
                groups[r['sha1']].append((s, p, n, r['dup']))
    loser: dict[tuple, str] = {}                           # (slug, path) -> note
    sha_rows = []
    for sha, lst in groups.items():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda x: (x[3], -x[2], x[0], x[1]))   # prefer non-dup flagged, then most mentions
        win = lst[0]
        for s, p, n, dup in lst[1:]:
            loser[(s, p)] = f'same bytes as {win[0]}:{win[1][-80:]}'
            sha_rows.append({'sha1': sha, 'loser_slug': s, 'loser_path': p, 'loser_mentions': n,
                             'winner_slug': win[0], 'winner_path': win[1], 'winner_mentions': win[2]})
    for lo, wi, why in BASENAME_OVERLAPS:
        if lo in src_counts and wi in src_counts:
            wb = {p.rsplit('/', 1)[-1] for p in src_counts[wi]}
            for p, n in src_counts[lo].items():
                if (lo, p) not in loser and p.rsplit('/', 1)[-1] in wb:
                    loser[(lo, p)] = f'text overlap with {wi} (same file name): {why}'[:200]
                    sha_rows.append({'sha1': '', 'loser_slug': lo, 'loser_path': p, 'loser_mentions': n,
                                     'winner_slug': wi, 'winner_path': '(same basename)', 'winner_mentions': 0})
    n_before = len(loser)
    if overlaps:
        text_overlaps(out_root, slugs, dest, loser)
        for (sl, p), why in loser.items():
            if why.startswith('text overlap with') and 'rare-name' in why:
                sha_rows.append({'sha1': '', 'loser_slug': sl, 'loser_path': p, 'loser_mentions': src_counts[sl][p],
                                 'winner_slug': why, 'winner_path': '(rare-name overlap)', 'winner_mentions': 0})
    pd.DataFrame(sha_rows, columns=['sha1', 'loser_slug', 'loser_path', 'loser_mentions', 'winner_slug',
                                    'winner_path', 'winner_mentions']).to_csv(dest / 'duplicate_sources.tsv', sep='\t',
                                                                              index=False)
    _t(f'sha1 overlaps: {len(sha_rows)} (slug,path) duplicates')

    # ---- pass 2: write merged tables + per-collection numbers + local tables ---------------------------------------
    mw = pq.ParquetWriter(dest / 'mentions.parquet', MENTION_OUT, compression='zstd')
    ew = pq.ParquetWriter(dest / 'entities.parquet', ENTITY_OUT, compression='zstd')
    lw = {k: pq.ParquetWriter(dest / f'{n}_local.parquet', LOCAL_SCHEMA, compression='zstd')
          for k, n in (('place', 'places'), ('person', 'persons'), ('org', 'orgs'))}
    rows = []
    zcls: list = []
    generic_rows: list = []
    multi_rows: list = []
    for s in slugs:
        t0 = time.time()
        tbl = pq.read_table(out_root / s / 'mentions.parquet')
        if forbidden_rows.get(s):
            keep = pc.invert(pa.array([C.is_forbidden(p) for p in tbl['source_path'].to_pylist()]))
            tbl = tbl.filter(keep)
        tbl, n_ch = unify_character_keys(tbl)
        if n_ch:
            _t(f'{s}: {n_ch} mentions follow the character class of their key')
        tbl, n_gen, rep_gen = exclude_generic(tbl)
        if len(rep_gen):
            generic_rows.append(rep_gen.assign(coll_slug=s))
        if n_gen:
            _t(f'{s}: {n_gen} pronouns / common nouns leave the default counts')
        tbl, n_gid = flag_generic_with_id(tbl)
        if n_gid:
            _t(f'{s}: {n_gid} common nouns carry an authority id (generic_with_id: kept, linked, never a display name)')
        tbl, n_mr, rep_mr = exclude_multi_referent(tbl)
        if len(rep_mr):
            multi_rows.append(rep_mr.assign(coll_slug=s))
        if n_mr:
            _t(f'{s}: {n_mr} mentions of several entities at once leave the default counts (multi_referent)')
        sp = tbl['source_path'].to_pylist()
        canon = np.ones(len(sp), dtype=bool)
        note = [None] * len(sp)
        txt_loser = TEXT_OVERLAP_LOSERS.get(s)
        for i, p in enumerate(sp):
            lo = loser.get((s, p))
            if lo:
                canon[i], note[i] = False, lo
            elif txt_loser:
                canon[i], note[i] = False, f'text overlap with {txt_loser[0]}: {txt_loser[1]}'
        tbl = tbl.append_column('canonical', pa.array(canon)).append_column('overlap_note', pa.array(note, pa.string()))
        mw.write_table(tbl.select([f.name for f in MENTION_OUT]).cast(MENTION_OUT))
        df = _flat(tbl)
        ents = pq.read_table(out_root / s / 'entities.parquet')
        edf = ents.select([c for c in ents.column_names if c != 'auth']).to_pandas()
        ea = ents.column('auth').combine_chunks()
        for k in AUTH_KEYS:
            edf['a_' + k] = ea.field(k).to_pandas()
        # entities with canonical counts
        cc = df[df['canonical'] & df['local_key'].notna()].groupby('local_key').size()
        ncan = edf['local_key'].map(cc).fillna(0).astype('int64')
        et = ents.append_column('n_mentions_canonical', pa.array(ncan.to_numpy(), pa.int64()))
        ew.write_table(et.select([f.name for f in ENTITY_OUT]).cast(ENTITY_OUT))
        # local tables
        loc_counts = {}
        for kind, w in lw.items():
            lt = local_table(df, edf[edf['kind'] == kind] if len(edf) else edf, kind)
            if len(lt):
                w.write_table(_to_table(lt, LOCAL_SCHEMA))
            loc_counts[kind] = lt
        # collection row
        can = df[df['canonical']]
        inp = can[can['in_period'] == True]    # noqa: E712
        n = max(len(can), 1)
        named = _any_named(can)
        anyauth = named | (can['a_other_n'] > 0)
        letters = can[can['role'].isin(LETTER_ROLES)]
        lt_all = letters[['source_path', 'doc_id']].drop_duplicates()
        lt_in = letters[letters['in_period'] == True][['source_path', 'doc_id']].drop_duplicates()   # noqa: E712
        both = (letters.groupby(['source_path', 'doc_id'])['role'].agg(lambda r: {'sender', 'addressee'} <= set(r)))
        row = {
            'slug': s, 'renascor_ids': ';'.join(sorted(set(can['renascor_id'].dropna()))) if len(can) else '',
            'licence': stats[s]['licence'], 'tier': stats[s]['tier'],
            'files_read': stats[s]['files_read'], 'file_errors': stats[s]['file_errors'],
            'mentions': len(df), 'mentions_canonical': len(can), 'mentions_noncanonical': len(df) - len(can),
            'txt': int((can['part'] == 'txt').sum()), 'hdr': int((can['part'] == 'hdr').sum()),
            'undated': int(can['in_period'].isna().sum()), 'out_of_period': int((can['in_period'] == False).sum()),  # noqa: E712
            'resolved_share': round(float(can['local_key'].notna().mean()), 4) if len(can) else 0,
            'in_register_share': round(float((can['in_register'] == True).mean()), 4) if len(can) else 0,  # noqa: E712
            'auth_named_share': round(float(named.mean()), 4) if len(can) else 0,
            'auth_any_share': round(float(anyauth.mean()), 4) if len(can) else 0,
            'nonstandard_role_rows': stats[s]['nonstandard_role_rows'],
            'entities': len(edf), 'entities_with_coords': int(edf['lat'].notna().sum()) if len(edf) else 0,
            'letters_with_roles': len(lt_all), 'letters_in_period': len(lt_in),
            'letters_sender_and_addressee': int(both.sum()) if len(both) else 0,
            'seconds_extract': stats[s].get('seconds'),
        }
        for z in ('txt', 'hdr'):
            for k in ('person', 'place', 'org'):
                row[f'ext_{z}_{k}'] = int(((df['part'] == z) & (df['kind'] == k)).sum())    # all rows, incl. non-canonical
        # ---- v2: refined zone, class, default counts (canonical rows only) ----
        exc = can['excluded'].fillna(False).astype(bool)
        dflt = can[~exc]
        row['mentions_default'] = len(dflt)
        row['mentions_excluded'] = int(exc.sum())
        for zc in ZONE_CLASSES:
            row[f'zone_{zc}'] = int((can['zone'] == zc).sum())
        for pc_ in PERSON_CLASSES:
            row[f'class_{pc_}'] = int((can['person_class'] == pc_).sum())
        reasons = Counter()
        for r_ in can.loc[exc, 'excluded_reason'].fillna(''):
            for x in str(r_).split(';'):
                if x:
                    reasons[x] += 1
        row['excluded_reasons'] = json.dumps(dict(reasons.most_common()), ensure_ascii=False) if reasons else ''
        for k in ('person', 'place', 'org'):
            row[f'{k}_default'] = int((dflt['kind'] == k).sum())
            row[f'{k}_default_in_period'] = int(((dflt['kind'] == k) & (dflt['in_period'] == True)).sum())   # noqa: E712
        zcls.append(can.assign(_slug=s).groupby(['_slug', 'kind', 'zone', 'person_class', 'excluded'], dropna=False).size()
                    .reset_index(name='n'))
        for k in ('person', 'place', 'org'):
            row[k] = int((can['kind'] == k).sum())
            row[f'{k}_in_period'] = int((inp['kind'] == k).sum())
            lt = loc_counts[k]
            row[f'local_{k}s'] = len(lt)
            row[f'local_{k}s_in_period'] = int((lt['n_in_period'] > 0).sum()) if len(lt) else 0
            row[f'local_{k}s_with_coords'] = int(lt['has_coords'].sum()) if len(lt) else 0
        rows.append(row)
        _t(f'{s}: {len(df)} rows ({len(can)} canonical), {time.time() - t0:.1f}s')
        del tbl, df
    mw.close(); ew.close()
    for w in lw.values():
        w.close()
    ct = pd.DataFrame(rows)
    if zcls:
        pd.concat(zcls).rename(columns={'_slug': 'slug'}).to_csv(dest / 'counts_by_zone_class.tsv', sep='\t', index=False)
    if multi_rows:         # v2.2: mentions that name several entities at once, by edition, kind and surface
        pd.concat(multi_rows)[['coll_slug', 'kind', 'surface_key', 'n_refs', 'n']].sort_values(
            ['n', 'coll_slug'], ascending=[False, True]).to_csv(dest / 'multi_referent_terms.tsv', sep='\t', index=False)
    if generic_rows:       # v2.1: every tagged pronoun / common noun by edition, kind, class and surface (the owner can flip the default)
        pd.concat(generic_rows)[['coll_slug', 'kind', 'class', 'surface_key', 'n']].sort_values(
            ['n', 'coll_slug'], ascending=[False, True]).to_csv(dest / 'generic_terms.tsv', sep='\t', index=False)

    # ---- inventory comparison ----------------------------------------------------------------------------------
    from .collections import discover
    reg = discover()
    scan_by_coll: dict[str, list] = defaultdict(list)
    for r in scan:
        scan_by_coll[r['coll']].append(r)
    dev_rows, inv_rows = [], []
    sel_cache = {}
    for s in slugs:
        col = reg.get(s)
        if col is None:
            continue
        sels = list(col.module.COLLS)
        sel_cache[s] = sels
    # one pass over the scan for all modules
    colls_set = {s: set(v) for s, v in sel_cache.items()}
    prefixes = {s: tuple(v) for s, v in sel_cache.items()}
    exp_all: dict[str, Counter] = {s: Counter() for s in sel_cache}
    exp_sel: dict[str, Counter] = {s: Counter() for s in sel_cache}
    nfiles_all = Counter()
    sel_paths: dict[str, set] = {}
    for s in sel_cache:
        try:
            ctx = reg[s].make_context()
            ctx.all_files = []
            files = reg[s].iter_files(ctx, False)
            sel_paths[s] = {f.path for f in files}
        except Exception:      # noqa: BLE001  prepare-dependent iter_files: fall back to the whole selection
            sel_paths[s] = None
    for r in scan:
        if r['dup'] or C.is_forbidden(r['path']):
            continue
        e = None
        for s in sel_cache:
            if r['coll'] in colls_set[s] or r['path'].startswith(prefixes[s]):
                if e is None:
                    e = _exp_of(r)
                exp_all[s].update(e)
                nfiles_all[s] += 1
                if sel_paths[s] is None or r['path'] in sel_paths[s]:
                    exp_sel[s].update(e)
    cti = ct.set_index('slug')
    for s in sel_cache:
        ex = exp_sel[s]
        rec = {'slug': s, 'files_collection': nfiles_all[s], 'files_read': int(cti.at[s, 'files_read'])}
        for z in ('txt', 'hdr'):
            for k in ('person', 'place', 'org'):
                rec[f'inv_{z}_{k}'] = ex.get((z, k), 0)
                rec[f'inv_all_{z}_{k}'] = exp_all[s].get((z, k), 0)
        inv_rows.append(rec)
    inv = pd.DataFrame(inv_rows).set_index('slug')
    inv = inv.reset_index()
    for z in ('txt', 'hdr'):
        for k in ('person', 'place', 'org'):
            inv[f'ext_{z}_{k}'] = inv['slug'].map(cti[f'ext_{z}_{k}'])
    inv.to_csv(dest / 'inventory_comparison.tsv', sep='\t', index=False)
    for _, r in inv.iterrows():
        for k in ('person', 'place', 'org'):
            ex_ = float(r[f'inv_txt_{k}'] + r[f'inv_hdr_{k}'])
            ext = float(r[f'ext_txt_{k}'] + r[f'ext_hdr_{k}'])
            if max(ex_, ext) < 30:
                continue
            dev = (ext - ex_) / max(ex_, 1)
            if abs(dev) > 0.10 and abs(ext - ex_) >= 20:
                dev_rows.append({'slug': r['slug'], 'kind': k, 'inventory': round(ex_), 'extracted': int(ext),
                                 'diff': round(ext - ex_), 'deviation': round(dev, 3),
                                 'inv_txt': round(r[f'inv_txt_{k}']), 'ext_txt': int(r[f'ext_txt_{k}']),
                                 'inv_hdr': round(r[f'inv_hdr_{k}']), 'ext_hdr': int(r[f'ext_hdr_{k}']),
                                 'files_read/collection': f"{int(r['files_read'])}/{int(r['files_collection'])}",
                                 'reason': DEVIATION_REASONS.get(r['slug'], '')})
    pd.DataFrame(dev_rows, columns=['slug', 'kind', 'inventory', 'extracted', 'diff', 'deviation', 'inv_txt', 'ext_txt',
                                    'inv_hdr', 'ext_hdr', 'files_read/collection', 'reason']).to_csv(
        dest / 'deviations.tsv', sep='\t', index=False)
    ct = ct.merge(inv[['slug', 'files_collection']], on='slug', how='left')
    ct.to_csv(dest / 'collections.tsv', sep='\t', index=False)
    if v1_collections and Path(v1_collections).exists():
        deviations_vs_v1(ct, Path(v1_collections), dest)

    # ---- harvest-level duplicate pairs and uncovered collections ----------------------------------------------
    first = {}
    for r in scan:
        if not r['dup']:
            first[r['sha1']] = r['coll']
    pairs, pents = Counter(), Counter()
    for r in scan:
        if r['dup']:
            k = (r['coll'], first.get(r['sha1']))
            pairs[k] += 1
            pents[k] += r['ent_text']
    pd.DataFrame([{'dup_coll': a, 'kept_coll': b, 'files': n, 'tagged_entities_in_text': pents[(a, b)]}
                  for (a, b), n in pairs.most_common()]).to_csv(dest / 'harvest_dup_pairs.tsv', sep='\t', index=False)
    allsel = set().union(*colls_set.values())
    allpre = tuple(p for v in prefixes.values() for p in v)
    unc = defaultdict(lambda: [0, 0, 0])
    for r in scan:
        if r['dup'] or C.is_forbidden(r['path']):
            continue
        if r['coll'] in allsel or r['path'].startswith(allpre):
            continue
        e = _exp_of(r)
        u = unc[r['coll']]
        u[0] += 1
        u[1] += sum(v for (z, k), v in e.items() if z == 'txt')
        u[2] += r['ent_text']
    pd.DataFrame([{'coll': c, 'files': v[0], 'txt_persName_placeName_orgName_typed': v[1], 'ent_text': v[2]}
                  for c, v in sorted(unc.items(), key=lambda kv: -kv[1][1])]).to_csv(
        dest / 'uncovered_colls.tsv', sep='\t', index=False)
    del scan, by_path

    # ---- totals ------------------------------------------------------------------------------------------------
    totals = totals_from(dest, ct)
    totals['failures'] = failures
    totals['forbidden_rows_dropped'] = dict(forbidden_rows)
    totals['noncanonical_source_paths'] = len(sha_rows)
    totals['noncanonical_source_mentions'] = int(sum(r['loser_mentions'] for r in sha_rows))
    json.dump(totals, open(dest / 'totals.json', 'w'), indent=1, default=str)
    return totals


def totals_from(dest: Path, ct: pd.DataFrame) -> dict:
    t: dict = {}
    for k in ('person', 'place', 'org'):
        t[f'mentions_{k}'] = int(ct[k].sum())
        t[f'mentions_{k}_in_period'] = int(ct[f'{k}_in_period'].sum())
    t['mentions_total'] = int(ct['mentions'].sum())
    t['mentions_canonical'] = int(ct['mentions_canonical'].sum())
    t['letters_with_roles'] = int(ct['letters_with_roles'].sum())
    t['letters_with_roles_in_period'] = int(ct['letters_in_period'].sum())
    t['letters_sender_and_addressee'] = int(ct['letters_sender_and_addressee'].sum())
    for nm, kind in (('places', 'place'), ('persons', 'person'), ('orgs', 'org')):
        lt = pq.read_table(dest / f'{nm}_local.parquet').to_pandas()
        inp = lt[lt['n_in_period'] > 0]
        d = {'distinct_local': len(lt), 'distinct_local_in_period': len(inp),
             'mentions_in_period_covered': int(inp['n_in_period'].sum())}
        for tag, sub in (('all', lt), ('in_period', inp)):
            n = max(len(sub), 1)
            d[f'share_coords_{tag}'] = round(float(sub['has_coords'].mean()), 4) if len(sub) else 0
            for k in AUTH_KEYS:
                d[f'share_{k}_{tag}'] = round(float(sub[k].notna().mean()), 4) if len(sub) else 0
            anyn = np.zeros(len(sub), dtype=bool)
            for k in AUTH_KEYS:
                anyn |= sub[k].notna().to_numpy()
            d[f'share_any_named_auth_{tag}'] = round(float(anyn.mean()), 4) if len(sub) else 0
            oth = sub['other_auth'].map(lambda v: len(v) > 0 if v is not None else False).to_numpy()
            d[f'share_other_auth_{tag}'] = round(float(oth.mean()), 4) if len(sub) else 0
            geo = sub['geonames'].notna() | sub['tgn'].notna() | sub['pleiades'].notna()
            d[f'share_coordinate_authority_{tag}'] = round(float(geo.mean()), 4) if len(sub) else 0
            d[f'share_coords_or_coord_authority_{tag}'] = round(float((sub['has_coords'] | geo).mean()), 4) if len(sub) else 0
            d[f'share_key_register_{tag}'] = round(float((sub['key_type'] == 'register').mean()), 4) if len(sub) else 0
            d[f'share_key_opaque_{tag}'] = round(float((sub['key_type'] == 'opaque').mean()), 4) if len(sub) else 0
            d[f'share_key_surface_{tag}'] = round(float((sub['key_type'] == 'surface').mean()), 4) if len(sub) else 0
        t[nm] = d
    return t


# ----------------------------------------------------------------------------------------------------------------------
# text-level overlap detection (different bytes, same text)
# ----------------------------------------------------------------------------------------------------------------------
AGGREGATORS = {'corpus_corporum', 'freem_other', 'dta_1600'}   # corpora that re-host texts of dedicated editions


def text_overlaps(out_root: Path, slugs: list, dest: Path, loser: dict, min_shared: int = 30,
                  min_score: float = 0.4, drop_score: float = 0.8) -> pd.DataFrame:
    """Document pairs from different collections sharing many RARE normalised surfaces (document frequency 2-4).

    Pairs with score >= drop_score are real text duplicates: the aggregator's document (AGGREGATORS), else the one
    with fewer mentions, is added to ``loser`` (non-canonical). Everything >= min_score is listed in
    text_overlap_pairs.tsv.
    """
    _t('text overlaps ...')
    parts = []
    for sl in slugs:
        t = pq.read_table(out_root / sl / 'mentions.parquet', columns=['source_path', 'kind', 'surface']).to_pandas()
        t = t[t['kind'].isin(['person', 'place']) & t['surface'].notna()]
        t['coll_slug'] = sl
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df[[(sl, p) not in loser for sl, p in zip(df['coll_slug'], df['source_path'])]]
    nm = {x: _norm(x) for x in df['surface'].unique()}
    df['n'] = df['surface'].map(nm)
    df = df[df['n'].str.len() >= 5]
    nmen = df.groupby(['coll_slug', 'source_path']).size()
    df['doc'] = df['coll_slug'] + '|' + df['source_path']
    df = df[['doc', 'n', 'coll_slug']].drop_duplicates(['doc', 'n'])
    df['doc'] = df['doc'].astype('category')
    df['d'] = df['doc'].cat.codes
    slug_of = df.groupby('d')['coll_slug'].first()
    freq = df.groupby('n')['d'].transform('nunique')
    rare = df[(freq >= 2) & (freq <= 4)][['d', 'n']]
    nrare = rare.groupby('d').size()
    m = rare.merge(rare, on='n', suffixes=('_a', '_b'))
    m = m[m['d_a'] < m['d_b']]
    m = m[m['d_a'].map(slug_of).to_numpy() != m['d_b'].map(slug_of).to_numpy()]
    pr = m.groupby(['d_a', 'd_b']).size().reset_index(name='shared')
    pr = pr[pr['shared'] >= min_shared]
    pr['score'] = pr['shared'] / np.minimum(pr['d_a'].map(nrare), pr['d_b'].map(nrare))
    pr = pr[pr['score'] >= min_score]
    cats = df['doc'].cat.categories
    pr['doc_a'] = cats[pr['d_a'].to_numpy()]
    pr['doc_b'] = cats[pr['d_b'].to_numpy()]
    pr['slug_a'] = pr['doc_a'].str.split('|').str[0]
    pr['slug_b'] = pr['doc_b'].str.split('|').str[0]
    actions = []
    for _, r in pr.iterrows():
        act = ''
        if r['score'] >= drop_score:
            sa, pa = r['slug_a'], r['doc_a'].split('|', 1)[1]
            sb, pb = r['slug_b'], r['doc_b'].split('|', 1)[1]
            na, nb = nmen.get((sa, pa), 0), nmen.get((sb, pb), 0)
            if min(na, nb) >= 0.5 * max(na, nb) and (sa in AGGREGATORS) != (sb in AGGREGATORS):
                lo = (sa, pa, sb) if sa in AGGREGATORS else (sb, pb, sa)      # same size: the aggregator copy goes
            else:                                                              # the smaller doc is contained in the bigger
                lo = (sa, pa, sb) if (na, sa) < (nb, sb) else (sb, pb, sa)
            if (lo[0], lo[1]) not in loser:
                loser[(lo[0], lo[1])] = f'text overlap with {lo[2]} (rare-name score {r["score"]:.2f})'
            act = f'drop {lo[0]}'
        actions.append(act)
    pr['action'] = actions
    pr['mentions_a'] = [nmen.get((a, d.split('|', 1)[1]), 0) for a, d in zip(pr['slug_a'], pr['doc_a'])]
    pr['mentions_b'] = [nmen.get((a, d.split('|', 1)[1]), 0) for a, d in zip(pr['slug_b'], pr['doc_b'])]
    res = pr[['slug_a', 'slug_b', 'doc_a', 'doc_b', 'shared', 'score', 'mentions_a', 'mentions_b', 'action']].sort_values('shared', ascending=False)
    res.to_csv(dest / 'text_overlap_pairs.tsv', sep='\t', index=False)
    agg = (res.groupby(['slug_a', 'slug_b']).agg(doc_pairs=('shared', 'size'), shared=('shared', 'sum'),
                                                  mean_score=('score', 'mean')).reset_index()
           .sort_values('doc_pairs', ascending=False))
    agg.to_csv(dest / 'text_overlap_slug_pairs.tsv', sep='\t', index=False)
    _t(f'text overlaps: {len(res)} doc pairs, {len(agg)} slug pairs, {len(res[res.action != ""])} dropped')
    return agg


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='ratlas merge')
    ap.add_argument('--out-root', type=Path, default=None)
    ap.add_argument('--dest', type=Path, default=None)
    ap.add_argument('--no-overlaps', action='store_true')
    ap.add_argument('--v1-collections', type=Path, default=None, help='v1 collections.tsv: writes deviations_vs_v1.tsv')
    a = ap.parse_args(argv)
    root = a.out_root or C.out_root()
    dest = a.dest or (root / 'all')
    t = merge(root, dest, overlaps=not a.no_overlaps, v1_collections=a.v1_collections)
    print(json.dumps(t, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
