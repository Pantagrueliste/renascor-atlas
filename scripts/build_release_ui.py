#!/usr/bin/env python3
"""Build a separate UI preview from a complete v29 atlas build (no data processing).

Usage: python scripts/build_release_ui.py SOURCE_BUILD OUTPUT_DIRECTORY
Missing inherited document shards are reported, never silently fetched or published.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

ASSETS = Path(__file__).resolve().parents[1] / 'web'


def redesign(html):
    if 'id="release-ui"' in html:
        raise ValueError('Source already has the release interface; use the original v29 build.')
    header = '''<a class="skip-link" href="#explore">Skip to atlas</a>
  <header>
    <div><div class="brandline">Early modern texts · 1450–1700</div>
    <h1>Renascor Atlas</h1>
    <p class="lede">Explore the places mentioned in texts from 1450–1700. Search for a place, follow a letter, or select a point to discover its sources.</p></div>
    <a class="header-link" href="#about-atlas">About the atlas ↗</a>
    <div class="facts" id="facts" aria-label="Coverage of the selected source layer"></div>
  </header>'''
    html, n = re.subn(r'<header>.*?</header>', lambda _: header, html, count=1, flags=re.S)
    if n != 1:
        raise ValueError('Expected v29 header')
    start = html.index('  <section class="filters"')
    end = html.index('  <div class="main">', start)
    html = html[:start] + '''  <section class="filters" aria-label="Explore the atlas" id="explore" tabindex="-1">
    <div class="fgroup search-group">
      <label class="flabel" for="q">Find a place or an edition</label>
      <div class="row"><span class="find"><input type="search" id="q" placeholder="Try Venice, Zurich, or Bullinger…" autocomplete="off" role="combobox" aria-expanded="false" aria-controls="sugg" aria-autocomplete="list"><ul id="sugg" role="listbox" hidden></ul></span><span class="chip" id="edchip" hidden></span></div>
    </div>
    <div class="fgroup"><span class="flabel">Explore</span><div class="seg" role="group" aria-label="Explore" id="modes">
      <button type="button" data-m="places" aria-pressed="true">Places</button><button type="button" data-m="letters" aria-pressed="false">Letters</button>
    </div></div>
    <button type="button" class="reset" id="reset-explore">Reset filters</button>
    <details class="advanced"><summary>Filters &amp; display</summary><div class="advanced-content">
      <div class="fgroup" id="readgroup"><span class="flabel">Sources</span><div class="seg" role="group" aria-label="Source layer" id="reading">
        <button type="button" data-r="strict" aria-pressed="true">Editor sources</button><button type="button" data-r="aug" aria-pressed="false">Expanded sources</button>
      </div><span class="fhint" id="readnote" role="status"></span></div>
      <div class="fgroup"><span class="flabel">Kind of place</span><div class="row" id="kinds"></div></div>
      <div class="fgroup" id="viewgroup"><span class="flabel">Map display</span><div class="seg" role="group" aria-label="Map display" id="views">
        <button type="button" data-v="circles" aria-pressed="true">Points</button><button type="button" data-v="heat" aria-pressed="false">Density</button><button type="button" data-v="columns" aria-pressed="false">Columns</button>
      </div></div>
    </div></details>
  </section>

''' + html[end:]
    html = html.replace('<section class="stage">', '<section class="stage" aria-label="Map and timeline"><div class="map-intro"><strong>Explore the atlas</strong><span>Select a point for its mentions and sources · Scroll to zoom</span></div>', 1)
    # Map-corner control: the panel toggle (#sidebtn) appears on the map in fullscreen mode.
    tools = ('<div class="map-tools">'
             '<button type="button" id="sidebtn" class="map-btn" aria-expanded="true">Hide panel</button>'
             '</div>')
    if '<div class="mapbox">' not in html:
        raise ValueError('Expected v29 mapbox')
    html = html.replace('<div class="mapbox">', '<div class="mapbox">' + tools, 1)
    # Fullscreen toggle (#fsbtn): a real button pinned to a corner of the viewport, in normal and fullscreen mode alike.
    fsbtn = '<button type="button" id="fsbtn" class="fs-btn" aria-pressed="false" title="Fullscreen" aria-label="Fullscreen">Fullscreen</button>'
    html = html.replace('<svg id="tl"', '<div class="period-controls"><label for="start-year">From</label><select id="start-year"></select><label for="end-year">to</label><select id="end-year"></select><span>or drag across the timeline</span></div>\n        <svg id="tl"', 1)
    html = re.sub(r'<section class="notes">.*?</section>',
                  lambda _: (ASSETS / 'about.html').read_text().strip(), html, count=1, flags=re.S)
    html = html.replace('<section class="notes">', '<details class="about" id="about-atlas"><summary>About the atlas &amp; how to read it</summary><section class="notes">', 1)
    html = html.replace('  <p class="foot">', '  </details>\n  <p class="foot">', 1)
    # The owner requested that internal tier labels stay out of the public interface.
    html = html.replace('${p.t===3?`<div class="note">Includes T3 data: internal use only.</div>`:""}', '')
    html = html.replace(' · places that also come from T3 collections are for internal use only.', '.')
    html = html.replace('Atlas v2.2 · 19 October 2026', 'Renascor Atlas · First-release preview · Data build v29')
    html = html.replace('${w ? "Description and names from Wikidata. " : ""}Document passages are shown only for editions whose licence allows it.', '')
    # Fix holes displaced from their containing land polygon at the date line.
    if 'function unwrapLand(fc)' in html:
        start = html.index('function unwrapLand(fc)')
        end = html.index('const land = unwrapLand(', start)
        html = html[:start] + (ASSETS / 'map-geometry.js').read_text() + '\n' + html[end:]
    # Theme and UI styles precede initialization so the map reads the same palette.
    html = html.replace('<html>', '<html lang="en">', 1)
    html = html.replace('</head>', '<style id="release-ui">' + (ASSETS / 'release.css').read_text() + '</style></head>', 1)
    # Privacy-friendly analytics (Umami cloud); release builds only, never the source build.
    html = html.replace('</head>', '<script defer src="https://cloud.umami.is/script.js" data-website-id="71fc027b-719d-483e-86b5-b835d3502618"></script></head>', 1)
    # Original app styles live inside the body; our override must follow them.
    css = '<style id="release-ui">' + (ASSETS / 'release.css').read_text() + '</style>'
    html = html.replace(css, '', 1).replace('<div class="wrap">', css + '\n<div class="wrap">', 1)
    html = html.replace('</body>', fsbtn + '\n<script>\n' + (ASSETS / 'release.js').read_text() + '\n</script>\n</body>', 1)
    return html


def build(source, output, inherited_assets=None, letters_docs=None):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from the source build.')
    if output.exists():
        raise ValueError('Output already exists; choose a fresh preview directory.')
    html = redesign((source / 'index.html').read_text())
    shutil.copytree(source, output)
    if inherited_assets is not None:
        # Historical v29 builds inherit these indexes. Never replace their newer shards.
        for folder in ('docs', 'img'):
            for asset in (inherited_assets / folder).glob('*.json'):
                target = output / folder / asset.name
                if not target.exists():
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(asset, target)
    if letters_docs is not None:
        # The page fetches letters/docs.json the first time a letter card opens; the v29 build does not ship it.
        target = output / 'letters' / 'docs.json'
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(letters_docs, target)
    (output / 'index.html').write_text(html)
    # The internal licence/audit table never ships with the public build.
    excluded = [p for p in output.rglob('licences.tsv')]
    for p in excluded:
        p.unlink()
    missing = [f'docs/{i:02}.json' for i in range(64) if not (output / f'docs/{i:02}.json').is_file()]
    img_present = (output / 'img').is_dir()
    letters_present = (output / 'letters' / 'docs.json').is_file()
    ready = not missing and img_present and letters_present
    notes = []
    if missing or not img_present:
        notes.append('Resolve inherited document/image assets.')
    if not letters_present:
        notes.append('Provide letters/docs.json (--letters-docs, the letters_docs.json of the matching atlas build) so Letters mode can load its documents.')
    notes.append('T3 collections are published as catalogued facts with full source citation per owner decision of 2026-11-04; '
                 'no context is stored for T3. Open audit points stay internal.')
    report = {'source': str(source), 'output': str(output), 'missing_document_shards': missing,
              'image_directory_present': img_present, 'letters_docs_present': letters_present,
              'publication_ready': ready,
              'licences_table_excluded': bool(excluded),
              'inherited_assets': str(inherited_assets) if inherited_assets else None,
              'publication_notes': notes}
    (output / 'release-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--inherited-assets', type=Path, help='Compatible cards_v24/out directory supplying missing docs/ and img/ shards')
    parser.add_argument('--letters-docs', type=Path, help='letters_docs.json of the matching atlas build, deployed as letters/docs.json')
    args = parser.parse_args()
    build(args.source, args.output, args.inherited_assets, args.letters_docs)
