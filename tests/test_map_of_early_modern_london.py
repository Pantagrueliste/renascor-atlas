import pytest
from ratlas import common as C
from ratlas.collections import get
from ratlas.model import Mention

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason="raw data / scan not available")


def _rec(path, xml, coll="x"):
    return C.FileRecord(path=path, coll=coll, data=xml.encode("utf8"))


def _setup(slug, suffixes=()):
    col = get(slug)
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = [f for f in files if f.path.endswith(tuple(suffixes))] if suffixes else files
    col.prepare(ctx)
    return col, ctx, files


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="X"><teiHeader><fileDesc><titleStmt><title>t</title>
<respStmt><name ref="mol:JENS1">Janelle Jenstad</name></respStmt></titleStmt>
<sourceDesc><p>Source: <date when-custom="1615" datingMethod="mol:julianSic">1615</date></p></sourceDesc></fileDesc>
<profileDesc><textClass><catRef scheme="mdt:molDocumentTypes" target="mdt:mdtPrimarySourceLibraryMayoral"/></textClass></profileDesc></teiHeader>
<text><body><p><name ref="mol:HENR4">Henry</name> at <ref target="mol:CAND1">Candlewick</ref>, <ref type="bibl" target="mol:SUGD1">Sugden</ref>
<name type="org" ref="mol:BUTC3">Butchers</name> <ref target="mol:UNKN1">unknown</ref></p></body></text></TEI>"""

PLACE = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="CAND1"><teiHeader/><text><body><div type="placeInfo"><listPlace><place>
<placeName>Candlewick Street</placeName><location type="GeoJSON"><geo>"geometry": {"type": "LineString", "coordinates": [[-0.09, 51.51], [-0.10, 51.51]]}</geo></location></place></listPlace></div></body></text></TEI>"""


def test_synthetic():
    col = get("map_of_early_modern_london")
    ctx = col.make_context()
    recs = [_rec("m/files/original/CAND1.xml", PLACE), _rec("m/files/original/PRIM1.xml", SYN)]
    ctx.all_files = recs
    es = col.prepare(ctx)
    e = ctx.shared.entities["map_of_early_modern_london:CAND1"]
    assert e.kind == "place" and abs(e.lat - 51.51) < 1e-9 and abs(e.lon + 0.095) < 1e-9
    assert e.coord_source == "tei:geo:geojson-vertex-mean"
    assert col.extract(recs[0], ctx) == ([], [])                              # modern file: not read for mentions
    ms, _ = col.extract(recs[1], ctx)
    assert [(m.kind, m.tag, m.in_register) for m in ms] == [
        ("person", "name@person", False), ("place", "ref@place", True), ("org", "name@org", False),
        ("place", "ref@place", False)]                                       # bibl ref and the respStmt name are skipped
    assert ms[0].local_key.endswith(":HENR4") and ms[0].doc_year == 1615 and ms[0].date_source == "header"


PLACEHOLDER = '<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="LLLL1"><teiHeader/><text><body><div type="placeInfo"><listPlace><place><placeName>PLACEHOLDER LOCATION</placeName></place></listPlace></div></body></text></TEI>'


def test_the_placeholder_location_is_not_a_place():
    """v2 audit, per-edition sample: mol:LLLL1 (PLACEHOLDER LOCATION) gathered 302 mentions of 300 different places."""
    col = get("map_of_early_modern_london")
    ctx = col.make_context()
    text = SYN.replace('<ref target="mol:UNKN1">unknown</ref>', '<ref target="mol:LLLL1">Pimlico path</ref> <ref target="mol:LLLL1">Almory</ref>')
    recs = [_rec("m/files/original/LLLL1.xml", PLACEHOLDER), _rec("m/files/original/PRIM1.xml", text)]
    ctx.all_files = recs
    col.prepare(ctx)
    ms, _ = col.extract(recs[1], ctx)
    ph = [m for m in ms if m.surface in ('Pimlico path', 'Almory')]
    assert len(ph) == 2 and all(m.local_key is None and m.in_register is None for m in ph)
    assert any(m.local_key and m.local_key.endswith(':HENR4') for m in ms)          # the real ids are untouched


def test_place_outside_of_london_is_a_placeholder_too():
    """v2.2 sample check: mol:POOL1 (PLACE OUTSIDE OF LONDON) gathered 84 mentions of Stratford, Hackney, Chelsea College ... at one point; each mention is keyed
    by its own surface, so the entity POOL1 has no mention and the surfaces are geocoded one by one (inside Greater London)."""
    col = get("map_of_early_modern_london")
    ctx = col.make_context()
    text = SYN.replace('<ref target="mol:UNKN1">unknown</ref>', '<ref target="mol:POOL1">Hackney</ref> <ref target="mol:POOL1">Stratford at Bow</ref>')
    pool = PLACEHOLDER.replace('LLLL1', 'POOL1').replace('PLACEHOLDER LOCATION', 'PLACE OUTSIDE OF LONDON')
    recs = [_rec("m/files/original/POOL1.xml", pool), _rec("m/files/original/PRIM1.xml", text)]
    ctx.all_files = recs
    col.prepare(ctx)
    ms, _ = col.extract(recs[1], ctx)
    ph = [m for m in ms if m.surface in ('Hackney', 'Stratford at Bow')]
    assert len(ph) == 2 and all(m.local_key is None and m.in_register is None for m in ph)


def test_real_files():
    col, ctx, files = _setup("map_of_early_modern_london")
    assert len(files) == 813
    assert any(k.endswith(":STOW6") for k in ctx.shared.entities)
    assert not any(e.note_type == "person" and e.name.startswith("Amogha") for e in ctx.shared.entities.values())
    f = next(f for f in files if f.path.endswith("stow_1633_CAND2.xml"))
    ms, _ = col.extract(f, ctx)
    assert len(ms) > 50 and all(m.doc_year == 1633 and m.date_source == "filename" for m in ms)
    assert {m.kind for m in ms} >= {"person", "place"}
    f2 = next(f for f in files if f.path.endswith("AMWE1.xml"))
    ms2 = col.extract(f2, ctx)[0]
    assert len(ms2) == 7 and [m.zone for m in ms2[:2]] == ["hdr", "hdr"]     # source author and printer, then the text


def test_moeml_is_not_covered_by_the_medici_archive_project_rule():
    """Owner decision 2026-10-17: the 'MAP' rule is the Medici Archive Project only. The Map of Early Modern London is legitimate: its harvested files
    (the 878 register files of originals_2026-10-17/map_of_early_modern_london-registers) are read."""
    import inspect
    from ratlas import common as C
    from ratlas import helpers_v2 as V
    from ratlas.collections import map_of_early_modern_london as M
    assert C.is_forbidden('renascor_2026-10/medici-archive-project/letters/1.xml') and C.is_forbidden('x/mediciarchive/y.xml')
    for p in ('renascor_2026-10/map-of-early-modern-london/files/original/ABBE1.xml',
              V.NEW + '/map_of_early_modern_london-registers/xml/original/ABBE2.xml', 'originals_2026-10-17/map_of_early_modern_london-registers/xml/original/LOND5.xml'):
        assert not C.is_forbidden(p)
    assert 'map_of_early_modern_london-registers/xml/original/' in inspect.getsource(M.Extractor._new_places)
    import re
    from ratlas import cards_check
    assert not cards_check.EXCLUDED.search(M.SLUG) and cards_check.EXCLUDED.search('medici_archive_project')
