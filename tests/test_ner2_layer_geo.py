"""Small checks of the layer geocoding helpers (scripts/ner2_layer_geo_*.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_clean_surface():
    m = load('ner2_layer_geo_prep')
    assert m.clean_surface('  Lon- don, ') == 'London'
    assert m.clean_surface("Rome's") == 'Rome'
    assert m.clean_surface('Espan~a') == 'España'
    assert m.clean_surface('ſtrasbourg') == 'strasbourg'
    assert m.reason('A1') == 'short'
    assert m.reason('Rue 12') == 'digit'


def test_kind_and_links():
    m = load('ner2_layer_geo_points')
    r = dict(gc_source='name_exonym', gc_evidence="exonym table: 'x' = Y (area, radius 350 km)", gc_radius_km=350)
    assert m.kind_of(r) == 'a'
    r = dict(gc_source='name_geonames', gc_evidence='GeoNames 1 X (PPLA2, GB, pop 90524); key', gc_radius_km=8)
    assert m.kind_of(r) == 'c'
    assert m.doc_link('EEBOphase2/A73036', 'EEBOphase2', 'en').endswith('A73036.0001.001')
    assert m.doc_link('BibIT/bibit000033', 'BibIT', 'it') is None
    assert m.fold('Zürich') == 'zurich'


def test_head_title():
    m = load('ner2_layer_geo_points')
    raw = b'{"id": "x", "t": "\\n  \\nThe Survey of London .\\nSecond line"}'
    assert m.head_title(raw).startswith('The Survey of London')
