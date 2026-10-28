from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

LETTER = 'suriano/datasource/tei/03/016.xml'


def test_letter_header_names():
    ms, ents, ctx = H.extract_paths('suriano', [LETTER])
    by = {(m.role, m.kind): m for m in ms}
    assert by[('sender', 'person')].surface == 'Christofforo Suriano'
    assert by[('sender', 'person')].local_key == 'suriano:cs' and by[('sender', 'person')].in_register is False
    assert by[('addressee', 'org')].surface == 'Doge and Senate of Venice'
    assert by[('sent_from', 'place')].surface == 'The Hague'
    assert by[('sent_to', 'place')].surface == 'Venice' and by[('repository', 'place')].surface == 'Venice'
    assert {m.doc_year for m in ms} == {1616} and {m.date_source for m in ms} == {'letter'}
    assert all(m.context is None for m in ms)       # header fields have no running text


def test_teisimple_volumes_are_not_read():
    ms, _, ctx = H.extract_paths('suriano', [LETTER])
    assert not any('teiSimple' in f.path for f in ctx.all_files)
