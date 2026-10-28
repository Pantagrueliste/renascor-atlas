import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

SYN = b'''<TEI.2><text><body><div0 type="sessionsPaper" id="16740429"><p>
<persName id="a" type="defendantName">Thomas Mullinex<interp inst="a" type="surname" value="Mullinex"/></persName> and
<persName id="b" type="victimName">Walter Carey</persName>, <persName id="c" type="jurorName">J Doe</persName>,
<persName id="d">Mary</persName> <persName id="g">the prisoner<interp inst="g" type="given" value="x"/></persName> <persName id="h">Michael Hair <persName id="i">Mary Hair<interp inst="i" type="surname" value="Hair"/></persName><interp inst="h" type="surname" value="Hair"/></persName> in <placeName id="e">Enfield</placeName>
<rs id="f" type="offenceDescription"><interp inst="f" type="offenceCategory" value="theft"/>theft</rs></p></div0></body></text></TEI.2>'''


def test_synthetic_roles_dates_and_rs_ignored():
    col = get('old_bailey')
    ctx = col.make_context()
    rec = C.FileRecord(path='eng_dh/oldbailey/sessionsPapers/16740429.xml', coll='eng_dh')
    ms, _ = col.extractor.extract_root(C.parse_xml(SYN), rec, ctx)
    roles = collections.Counter((m.kind, m.role) for m in ms)
    assert roles == {('person', 'defendant'): 1, ('person', 'victim'): 1, ('person', 'juror'): 1,
                     ('person', 'mention'): 1, ('place', 'mention'): 1}
    assert {m.doc_year for m in ms} == {1674} and {m.date_source for m in ms} == {'filename'}
    assert all(m.in_period for m in ms)
    # v2: untyped persName without a surname interp ('Mary', 'the prisoner') is not a name; nested persName: the inner one is read
    assert sorted(m.surface for m in ms if m.kind == 'person' and m.role == 'mention') == ['Mary Hair']
    assert {m.local_key for m in ms if m.kind == 'person'} == {'old_bailey:a', 'old_bailey:b', 'old_bailey:c', 'old_bailey:i'}
    assert all(m.local_key is None for m in ms if m.kind == 'place')


def test_file_selection_only_the_period():
    col = get('old_bailey')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    assert len(files) == 307
    assert max(int(r.stem.lstrip('OA')[:4]) for r in files) == 1700


def test_real_files():
    col = get('old_bailey')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    rec = next(r for r in files if r.path.endswith('sessionsPapers/16740429.xml'))
    ms, _ = col.extract(rec, ctx)
    c = collections.Counter((m.kind, m.role) for m in ms)
    assert c[('person', 'defendant')] == 12 and c[('person', 'victim')] == 9 and c[('place', 'mention')] == 3   # typed names are kept
    rec = next(r for r in files if r.path.endswith('OA16760517.xml'))
    ms, _ = col.extract(rec, ctx)
    assert len(ms) == 27 and {m.doc_year for m in ms} == {1676}
    assert all(m.licence == 'CC-BY-NC-4.0' and m.tier == 'T2' for m in ms)
