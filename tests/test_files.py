import gzip
import io
import json
import tarfile
import zipfile

from ratlas import common as C

X = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><persName>A</persName></body></text></TEI>'


def test_archives_and_scan(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    (raw / 'c1').mkdir(parents=True)
    (raw / 'c1' / 'plain.xml').write_bytes(X)
    with zipfile.ZipFile(raw / 'c1' / 'a.zip', 'w') as z:
        z.writestr('d/one.xml', X)
        z.writestr('d/two.tei', X)
    with tarfile.open(raw / 'c1' / 'b.tar.gz', 'w:gz') as t:
        ti = tarfile.TarInfo('x/three.xml'); ti.size = len(X); t.addfile(ti, io.BytesIO(X))
    (raw / 'c1' / 'g.xml.gz').write_bytes(gzip.compress(X))
    (raw / 'london-lives-1690-1800').mkdir()
    scan = tmp_path / 'scan.jsonl.gz'
    recs = [dict(path='c1/plain.xml', coll='c1', sha1='1'), dict(path='c1/a.zip!d/one.xml', coll='c1', sha1='2'),
            dict(path='c1/a.zip!d/two.tei', coll='c1', sha1='3', dup=True),
            dict(path='c1/b.tar.gz!x/three.xml', coll='c1', sha1='4'), dict(path='c1/g.xml.gz', coll='c1', sha1='5'),
            dict(path='c1/big.xml', coll='c1', skipped='too big'),
            dict(path='london-lives-1690-1800/x.xml', coll='london-lives-1690-1800', sha1='6'),
            dict(path='c2/other.xml', coll='c2', sha1='7')]
    with gzip.open(scan, 'wt') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setenv('RATLAS_RAW', str(raw))
    monkeypatch.setenv('RATLAS_SCAN', str(scan))
    got = C.iter_collection_files(['c1'])
    assert [r.path for r in got] == ['c1/a.zip!d/one.xml', 'c1/b.tar.gz!x/three.xml', 'c1/g.xml.gz', 'c1/plain.xml']
    assert len(C.iter_collection_files(['c1'], include_dups=True)) == 5
    assert C.iter_collection_files(['london-lives-1690-1800']) == []
    assert [r.path for r in C.iter_collection_files(['c1/a.zip'])] == ['c1/a.zip!d/one.xml']
    for r in C.iter_with_data(got):
        assert r.data == X or r.path.endswith('g.xml.gz')   # .xml.gz read as gz by read_raw
    assert C.read_raw('c1/g.xml.gz') == X and C.read_raw('c1/b.tar.gz!x/three.xml') == X
    assert C.list_members('c1/a.zip') == ['d/one.xml', 'd/two.tei']
    from ratlas.pushlist import push_files
    assert push_files(['c1']) == ['c1/a.zip', 'c1/b.tar.gz', 'c1/g.xml.gz', 'c1/plain.xml']
