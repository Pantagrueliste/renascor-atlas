"""Release packaging must preserve the scientific payload and original build."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'build_release_ui.py'
spec = importlib.util.spec_from_file_location('release_ui', SCRIPT)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

PAYLOAD = '<script>const D = {"mentions":334021,"name":"A & B"};</script>'
SOURCE = '''<!doctype html><html><head></head><body><div class="wrap">
<header>Old header</header>
  <section class="filters">Old filters</section>
  <div class="main"><section class="stage"><div class="mapbox"><div id="map"></div></div><svg id="tl"></svg></section></div>
<section class="notes">Method</section>
  <p class="foot">Atlas v2.2 · 19 October 2026</p></div>''' + PAYLOAD + '</body></html>'


def test_build_preserves_data_and_reports_missing_dependencies(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'index.html').write_text(SOURCE)
    (source / 'strict').mkdir()
    (source / 'licences.tsv').write_text('internal audit table')
    shard = b'{"0":[123,456]}'
    (source / 'strict' / '00.json').write_bytes(shard)
    output = tmp_path / 'preview'
    ui.build(source, output)
    assert (source / 'index.html').read_text() == SOURCE
    assert PAYLOAD in (output / 'index.html').read_text()
    assert (output / 'strict' / '00.json').read_bytes() == shard
    assert not (output / 'licences.tsv').exists()          # the internal table never ships
    assert 'cloud.umami.is/script.js' in (output / 'index.html').read_text()   # analytics tag present
    assert 'cloud.umami.is' not in (source / 'index.html').read_text()         # source never modified
    report = json.loads((output / 'release-report.json').read_text())
    assert len(report['missing_document_shards']) == 64
    assert report['publication_ready'] is False
    assert report['image_directory_present'] is False
    assert report['letters_docs_present'] is False
    assert report['licences_table_excluded'] is True
    assert any('letters/docs.json' in n for n in report['publication_notes'])
    assert 'licences.tsv' not in (output / 'release-report.json').read_text()


def test_refuses_overwriting_or_nesting_source(tmp_path):
    for output in [tmp_path, tmp_path / 'child', tmp_path.parent]:
        with pytest.raises(ValueError, match='separate'):
            ui.build(tmp_path, output)
    source = tmp_path / 'source'
    output = tmp_path / 'existing'
    source.mkdir()
    output.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        ui.build(source, output)


def test_refuses_double_redesign():
    with pytest.raises(ValueError, match='already has'):
        ui.redesign(ui.redesign(SOURCE))


def test_fullscreen_toggle_is_an_unobtrusive_map_button():
    html = ui.redesign(SOURCE)
    header = html[html.index('<header>'):html.index('</header>')]
    assert 'fsbtn' not in header and 'sidebtn' not in header
    assert html.index('id="fsbtn"') > html.index('<div class="mapbox">')
    assert '<button type="button" id="fsbtn" class="fs-btn" aria-pressed="false"' in html
    tools_start = html.index('class="map-tools"')
    tools_html = html[tools_start:html.index('</div>', tools_start)]
    assert 'sidebtn' in tools_html and 'fsbtn' not in tools_html  # the fullscreen button is pinned to a viewport corner, not on the map


def test_inherited_assets_fill_gaps_without_overwriting_current_shards(tmp_path):
    source, inherited, output = (tmp_path / n for n in ('source', 'inherited', 'preview'))
    (source / 'docs').mkdir(parents=True)
    (source / 'letters').mkdir()
    (source / 'index.html').write_text(SOURCE)
    (source / 'docs' / '64.json').write_text('{"current":true}')
    (source / 'letters' / 'docs.json').write_text('{"mine":true}')
    (inherited / 'docs').mkdir(parents=True)
    (inherited / 'img').mkdir()
    for i in range(65):
        (inherited / 'docs' / f'{i:02}.json').write_text('{}')
    (inherited / 'img' / '00.json').write_text('{}')
    letters_docs = tmp_path / 'letters_docs.json'
    letters_docs.write_text('{"other":true}')
    ui.build(source, output, inherited, letters_docs)
    assert (output / 'docs' / '64.json').read_text() == '{"current":true}'
    assert (output / 'img' / '00.json').exists()
    assert (output / 'letters' / 'docs.json').read_text() == '{"mine":true}'   # the build's own letters win
    report = json.loads((output / 'release-report.json').read_text())
    assert report['missing_document_shards'] == []
    assert report['image_directory_present'] is True
    assert report['letters_docs_present'] is True
    assert report['publication_ready'] is True
    assert any('owner decision of 2026-11-04' in n for n in report['publication_notes'])
