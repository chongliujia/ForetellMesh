from copy import deepcopy
import io
import importlib.util
import json
import shutil
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file
from foretellmesh.pma_archive import download, download_parallel, extract, member_path, validate_response
from foretellmesh.schema import ValidationError


def config_for(size):
    return {'schema_version': '1', 'source': 'prediction_market_analysis',
            'url': 'https://s3.jbecker.dev/data.tar.zst', 'upstream_schema_revision': 'a'*40,
            'expected_bytes': size, 'expected_etag': '"fixed"',
            'extract_tables': ['polymarket/markets', 'polymarket/blocks', 'polymarket/trades'],
            'max_extracted_bytes': 250*1024**3, 'max_member_bytes': 2*1024**3}


class Response(io.BytesIO):
    def __init__(self, raw, config, offset=0):
        super().__init__(raw); self.url = config['url']; self.status = 206 if offset else 200
        self.headers = {'ETag': config['expected_etag'], 'Content-Length': str(len(raw))}
        if offset:self.headers['Content-Range'] = f'bytes {offset}-{config["expected_bytes"]-1}/{config["expected_bytes"]}'


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.cp = self.root/'config.json'; self.output = self.root/'archive'

    def config(self, size):
        c = config_for(size); self.cp.write_text(json.dumps(c)); return c

    def test_wrong_object_length_or_ignored_resume_rejected(self):
        c = config_for(12)
        for change in [{'ETag': '"changed"'}, {'Content-Length': '11'}]:
            r = Response(b'123456789012', c); r.headers.update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):validate_response(r, c, 0)
        r = Response(b'123456789012', c)
        with self.assertRaisesRegex(ValidationError, 'resume'):validate_response(r, c, 4)

    def test_download_content_hash_and_repeat_verification(self):
        raw = b'content from pinned source'; c = self.config(len(raw))
        with patch('foretellmesh.pma_archive.urlopen', return_value=Response(raw,c)):
            report = download(self.cp, self.output)
        self.assertEqual(report['sha256'], sha256_file(self.output/'data.tar.zst'))
        with patch('foretellmesh.pma_archive.urlopen', side_effect=AssertionError('unexpected network')):
            self.assertEqual(report, download(self.cp, self.output))
        (self.output/'data.tar.zst').write_bytes(b'tampered')
        with self.assertRaises(ValidationError):download(self.cp, self.output)

    def test_incomplete_download_preserves_partial_and_never_promotes(self):
        c = self.config(20); r = Response(b'short',c); r.headers['Content-Length'] = '20'
        with patch('foretellmesh.pma_archive.urlopen', return_value=r), self.assertRaisesRegex(ValidationError,'incomplete'):
            download(self.cp,self.output)
        self.assertTrue((self.output/'data.tar.zst.partial').exists()); self.assertFalse((self.output/'download_manifest.json').exists())

    def test_parallel_resume_joins_verified_prefix_and_range(self):
        raw=b'prefix and remaining bytes'; c=self.config(len(raw)); self.output.mkdir()
        (self.output/'download_config.json').write_text(json.dumps(c));(self.output/'data.tar.zst.partial').write_bytes(raw[:6])
        with patch('foretellmesh.pma_archive.urlopen', return_value=Response(raw[6:],c,6)):
            report=download_parallel(self.cp,self.output)
        self.assertEqual((self.output/'data.tar.zst').read_bytes(),raw);self.assertEqual(report['bytes'],len(raw))

    def test_truncated_range_retries_without_accepting_partial_body(self):
        raw=b'prefix and remaining bytes';c=self.config(len(raw));self.output.mkdir()
        (self.output/'download_config.json').write_text(json.dumps(c));(self.output/'data.tar.zst.partial').write_bytes(raw[:6])
        truncated=Response(raw[6:10],c,6);truncated.headers['Content-Length']=str(len(raw)-6)
        with (patch('foretellmesh.pma_archive.urlopen',side_effect=[truncated,Response(raw[6:],c,6)]) as fetch,
              patch('foretellmesh.pma_archive.time.sleep')):
            download_parallel(self.cp,self.output)
        self.assertEqual(fetch.call_count,2);self.assertEqual((self.output/'data.tar.zst').read_bytes(),raw)
        self.assertEqual(len(list((self.output/'parts').glob('*.failures.jsonl'))),1)

    def test_unsafe_links_paths_and_duplicates_rejected(self):
        for name in ['/tmp/out','data/../../out','other/file','data/\\out','data/./file']:
            with self.subTest(name=name), self.assertRaises(ValidationError):member_path(tarfile.TarInfo(name),set())
        m=tarfile.TarInfo('data/polymarket/markets/link');m.type=tarfile.SYMTYPE;m.linkname='/tmp/out'
        with self.assertRaises(ValidationError):member_path(m,set())
        m=tarfile.TarInfo('data/file'); seen=set();member_path(m,seen)
        with self.assertRaises(ValidationError):member_path(m,seen)

    @unittest.skipUnless(importlib.util.find_spec('zstandard') and shutil.which('zstd'), 'optional zstandard and zstd CLI dependencies')
    def test_extract_only_registered_tables_and_bind_every_member(self):
        import zstandard
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as tar:
            for name in ['data/polymarket/markets/m.parquet','data/polymarket/trades/t.parquet',
                         'data/polymarket/blocks/b.parquet','data/polymarket/markets/._m.parquet','data/kalshi/trades/t.parquet']:
                member=tarfile.TarInfo(name);member.size=3;tar.addfile(member,io.BytesIO(b'raw'))
        raw=zstandard.ZstdCompressor().compress(stream.getvalue());c=self.config(len(raw))
        with patch('foretellmesh.pma_archive.urlopen',return_value=Response(raw,c)):download(self.cp,self.output)
        target=self.root/'extracted';report=extract(self.cp,self.output,target)
        self.assertEqual(report['selected_shards'],3)
        self.assertFalse((target/'kalshi').exists());self.assertFalse((target/'polymarket/markets/._m.parquet').exists())
        manifest=json.loads((target/'manifest.json').read_text())
        for entry in manifest['selected']:self.assertEqual(sha256_file(target/entry['file']),entry['sha256'])
        with self.assertRaises(ValidationError):extract(self.cp,self.output,target)
        # A decodable archive must still match the pinned compressed input hash.
        binding_path=self.output/'download_manifest.json';binding=json.loads(binding_path.read_text())
        binding['sha256']='0'*64;binding_path.write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValidationError,'archive binding'):
            extract(self.cp,self.output,self.root/'wrong-hash')
        self.assertFalse((self.root/'wrong-hash').exists())
        self.assertEqual(list(self.root.glob('.pma-extract-*')),[])

    @unittest.skipUnless(importlib.util.find_spec('zstandard') and shutil.which('zstd'), 'optional zstandard and zstd CLI dependencies')
    def test_truncated_zstd_checksum_fails_even_when_tar_end_is_present(self):
        import zstandard
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as tar:
            member=tarfile.TarInfo('data/file');member.size=3;tar.addfile(member,io.BytesIO(b'raw'))
        raw=zstandard.ZstdCompressor(write_checksum=True).compress(stream.getvalue())[:-1];c=self.config(len(raw))
        with patch('foretellmesh.pma_archive.urlopen',return_value=Response(raw,c)):download(self.cp,self.output)
        with self.assertRaisesRegex(ValidationError,'zstd archive'):extract(self.cp,self.output,self.root/'truncated')
        self.assertFalse((self.root/'truncated').exists())


if __name__ == '__main__':unittest.main()
