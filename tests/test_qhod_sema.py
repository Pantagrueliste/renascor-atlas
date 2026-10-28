import collections

import pytest

from ratlas import common as C
from ratlas.collections import get

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')


def _setup():
    col = get('qhod_sema')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    ents = col.prepare(ctx)
    return col, ctx, files, ents


def test_register_cited_only_with_coordinates():
    col, ctx, files, ents = _setup()
    # v2: Zenodo 2.0 archive: 16 sema letters + 76 kuse + the vrancic documents + the register
    assert len(files) > 90 and sum('/sema/' in f.path for f in files) >= 16
    kinds = collections.Counter(e.kind for e in ents)
    assert kinds['person'] > 100 and kinds['place'] > 100 and len(ents) < 900
    assert sum(e.lat is not None for e in ents if e.kind == 'place') > 60
    assert any(e.auth['gnd'] for e in ents) and any(e.auth['geonames'] for e in ents)
    assert all(not any(x.startswith('url:https://mpr') for x in e.auth['other']) for e in ents)
    assert any(x.startswith('mpr:') for e in ents for x in e.auth['other'])


def test_letter_roles_and_dates():
    col, ctx, files, _ = _setup()
    rec = next(r for r in files if r.path.endswith('Selim_II_Urkunde_1.xml'))
    ms, es = col.extract(rec, ctx)
    assert es == []
    c = collections.Counter((m.kind, m.role) for m in ms)
    assert c[('person', 'sender')] == 1 and c[('person', 'addressee')] == 1 and c[('place', 'sent_from')] == 1
    assert c[('place', 'sent_to')] == 1 and c[('person', 'author')] == 1
    sender = next(m for m in ms if m.role == 'sender')
    assert sender.auth['gnd'] == '138054231' and sender.surface == 'Selim II.'
    assert {m.doc_year_from for m in ms} == {1567} and {m.date_source for m in ms} == {'letter'}
    res = [m for m in ms if m.local_key]
    assert len(res) >= 8 and all(m.local_key.startswith('qhod_sema:mpr') for m in res)
    assert all(m.tier == 'T1' and m.licence == 'CC-BY-4.0' for m in ms)
