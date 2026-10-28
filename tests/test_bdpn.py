from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import bdpn

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

CARTA = 'bdpn/archivosXMLDocumentos/carta'
ALMANAQUE = 'bdpn/archivosXMLDocumentos/Discurso_y_relacion_cometographa.xml'


def test_ref_normalisation():
    assert bdpn._norm_id('catalogo_nombres_lugares.xml#S007') == 'S007'
    assert bdpn._norm_id('catalogo_nombres_lugares.xml$L004') == 'L004'
    assert bdpn._norm_id('catalogo_nombres_personas.xmlV010') == 'V010'
    assert bdpn._norm_id('catalogo_nombres_personas.xml#A35') == 'A035'
    assert bdpn._norm_id('catalogo_nombres_lugares.xml#P0017') == 'P017'
    assert bdpn._kind_of_ref('catalogo- nombres_lgares.xml#O04') == 'place'
    assert bdpn._norm_id('catalogo_nombres_lugares.xml#') is None


def test_extensionless_files_are_listed():
    ctx = C.Context(slug='bdpn', licence='CC0-1.0', tier='T1')
    ctx.options['colls'] = bdpn.COLLS
    paths = {r.path for r in bdpn.EXTRACTOR.iter_files(ctx)}
    assert CARTA in paths and 'bdpn/archivosXMLDocumentos/DiscursoEthereologico' in paths
    assert not any('catalogo' in p for p in paths)


def test_real_text_and_catalogues():
    ms, ents, ctx = H.extract_paths('bdpn', [ALMANAQUE, CARTA])
    kinds = Counter(m.kind for m in ms)
    assert kinds['place'] > 100 and kinds['person'] > 30 and kinds['place'] > kinds['org']
    assert {m.doc_year for m in ms if m.source_path == ALMANAQUE} == {1652}
    luna = next(m for m in ms if m.ref_raw and m.ref_raw.endswith('#L004'))
    assert luna.local_key == 'bdpn:place:L004'
    cat = {e.local_key: e for e in ents}
    assert cat['bdpn:place:L004'].note_type == 'celestial'      # Luna
    assert cat['bdpn:place:C015'].name == 'Ciudad de México' and cat['bdpn:place:C015'].note_type == 'place'
    assert cat['bdpn:person:D010'].name.lower().startswith('dios')
    assert sum(e.kind == 'person' for e in cat.values()) > 600


def test_compass_and_cosmological_terms_are_not_places():
    """v2.1 (manual review, bdpn_residual): thirteen terms the catalogue typed as places are cosmological or compass points: celestial = out of the counts."""
    from ratlas.collections.bdpn import _CELESTIAL, _COSMOLOGICAL
    for t in ('Eclíptica', 'Círculos horarios', 'Azimutal', 'Epiciclos', 'Cometa', 'Andromeda', 'Septentrional', 'Media Región', 'Oriente', 'Occidente', 'Norte', 'Sur', 'Poniente'):
        assert _COSMOLOGICAL.match(t), t
    for t in ('Ciudad de México', 'Norte de España', 'Sevilla', 'Oriente Medio'):
        assert not _COSMOLOGICAL.match(t) and not _CELESTIAL.search(t), t
