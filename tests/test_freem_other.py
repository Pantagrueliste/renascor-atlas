import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import freem_other as fo

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

B = 'originals_2026-10-15/github-neolatin-french-tei/FreEMmax_OA/0_source/'


def test_licence_and_header_paths():
    assert fo._spdx('http://creativecommons.org/licenses/by-nc-nd/3.0/fr/') == 'CC-BY-NC-ND-3.0-FR'
    assert fo._spdx('https://creativecommons.org/licenses/by-nc-sa/4.0/') == 'CC-BY-NC-SA-4.0'
    assert fo._spdx('http://creativecommons.org/licenses/CC BY-NC-SA 4.0/') is None
    assert fo._header_path(B + 'Descartes/clerselier_1.xml').endswith('1_header/Descartes/clerselier_1_dAlemBERT.xml')


def test_real_files():
    ms, _, ctx = H.extract_paths('freem_other', [B + 'Descartes/clerselier_2.xml', B + 'IG/francois_a_londres.xml'])
    desc = [m for m in ms if 'Descartes' in m.source_path]
    ig = [m for m in ms if '/IG/' in m.source_path]
    assert len(desc) >= 70 and {m.role for m in desc} == {'addressee'} and {m.doc_year for m in desc} == {1667}
    assert {m.licence for m in desc} == {'CC-BY-NC-ND-3.0-FR'} and all(m.context is None for m in desc)
    assert not any('REPONSE' in m.surface.upper() for m in desc)
    assert {m.doc_year for m in ig} == {1758} and {m.in_period for m in ig} == {False}
    assert any(m.kind == 'org' and m.tag == 'rs@institution' for m in ig)
    assert not any('/BVH/' in f.path for f in ctx.all_files)
