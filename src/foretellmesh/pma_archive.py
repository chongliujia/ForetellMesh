"""Download and safely extract the registered PMA archive without upstream code.

An upstream Git revision identifies the schema, not the independently hosted
archive. Pin HTTP object identity for transfer and hash complete local bytes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_dataset import now
from .schema import ValidationError


def load_config(path: Path) -> dict:
    config = strict_json(path.read_text())
    if (config.get('schema_version') != '1' or config.get('source') != 'prediction_market_analysis'
            or config.get('url') != 'https://s3.jbecker.dev/data.tar.zst'
            or not re.fullmatch(r'[0-9a-f]{40}', config.get('upstream_schema_revision', ''))
            or type(config.get('expected_bytes')) is not int or not 0 < config['expected_bytes'] <= 50*1024**3
            or not isinstance(config.get('expected_etag'), str) or config['expected_etag'].startswith('W/')
            or config.get('extract_tables') != ['polymarket/markets', 'polymarket/blocks', 'polymarket/trades']
            or config.get('max_extracted_bytes') != 250*1024**3 or config.get('max_member_bytes') != 2*1024**3):
        raise ValidationError('invalid bounded PMA archive policy')
    return config


def validate_response(response, config, offset):
    if response.url != config['url'] or response.headers.get('ETag') != config['expected_etag']:
        raise ValidationError('archive URL or HTTP object identity changed')
    total = config['expected_bytes']
    if offset:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{total-1}/{total}':
            raise ValidationError('server did not honor resume range; partial bytes preserved')
    elif response.status != 200:
        raise ValidationError('unexpected archive download status')
    if int(response.headers.get('Content-Length', '-1')) != total-offset:
        raise ValidationError('archive response length changed')


def download(config_path: Path, output: Path) -> dict:
    config = load_config(config_path); output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path); binding = output/'download_config.json'
    if binding.exists() and strict_json(binding.read_text()) != config:
        raise ValidationError('download directory is bound to a different object')
    binding.write_text(json_text(config))
    archive, partial, manifest = output/'data.tar.zst', output/'data.tar.zst.partial', output/'download_manifest.json'
    if archive.exists():
        if not manifest.exists():raise ValidationError('unmanifested completed archive')
        report = strict_json(manifest.read_text())
        if (report['config_sha256'] != config_hash or archive.stat().st_size != report['bytes']
                or sha256_file(archive) != report['sha256']):raise ValidationError('completed archive changed')
        return report
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= config['expected_bytes']:raise ValidationError('partial archive requires completion review')
    if shutil.disk_usage(output).free < config['expected_bytes']-offset+2*1024**3:
        raise ValidationError('insufficient download space')
    headers = {'User-Agent': 'ForetellMesh/0.1 public-data-research', 'If-Match': config['expected_etag'], 'Accept-Encoding': 'identity'}
    if offset:headers['Range'] = f'bytes={offset}-'
    start = now(); clock = time.monotonic(); last = clock; digest = hashlib.sha256()
    if offset:
        with partial.open('rb') as existing:
            for chunk in iter(lambda: existing.read(8*1024**2), b''):digest.update(chunk)
    with urlopen(Request(config['url'], headers=headers), timeout=60) as response:
        validate_response(response, config, offset)
        response_headers = dict(response.headers)
        received = offset
        with partial.open('ab' if offset else 'xb') as stream:
            while chunk := response.read(8*1024**2):
                received += len(chunk)
                if received > config['expected_bytes']:raise ValidationError('archive exceeded pinned size')
                stream.write(chunk); digest.update(chunk)
                if time.monotonic()-last >= 10:
                    progress = {'bytes': received, 'total_bytes': config['expected_bytes'],
                                'elapsed_seconds': round(time.monotonic()-clock, 2), 'completed': False}
                    (output/'progress.json').write_text(json_text(progress))
                    print(json_text(progress).strip(), flush=True); last = time.monotonic()
        if received != config['expected_bytes']:raise ValidationError('incomplete archive; partial bytes preserved')
    report = {'schema_version': '1', 'source': config['source'], 'url': config['url'],
              'upstream_schema_revision': config['upstream_schema_revision'], 'config_sha256': config_hash,
              'http_etag': config['expected_etag'], 'http_headers': response_headers,
              'started_at': start, 'completed_at': now(), 'resumed_from_bytes': offset,
              'bytes': received, 'sha256': digest.hexdigest(),
              'hash_trust': 'Computed over local complete download; no publisher SHA-256 signature supplied.'}
    manifest.write_text(json_text(report)); partial.rename(archive)
    (output/'progress.json').write_text(json_text({'bytes': received, 'total_bytes': received, 'completed': True}))
    return report


def member_path(member, seen):
    path = PurePosixPath(member.name)
    if (path.is_absolute() or '..' in path.parts or '\\' in member.name or not path.parts
            or path.parts[0] != 'data' or path.as_posix() in seen
            or member.name.rstrip('/') != path.as_posix() or not (member.isdir() or member.isfile())):
        raise ValidationError('unsafe, duplicate or unsupported archive member')
    seen.add(path.as_posix())
    return path


def download_parallel(config_path: Path, output: Path) -> dict:
    """Finish a stopped sequential prefix with bounded range readers.

Parts are separate files; no sparse files or unverified holes are accepted.
The prefix and each full range are hashed before assembly. If-Range is needed
by the public endpoint to reliably honor Range requests.
"""
    config = load_config(config_path); binding = output/'download_config.json'
    if not binding.exists() or strict_json(binding.read_text()) != config:
        raise ValidationError('parallel transfer requires the pinned sequential directory')
    archive, prefix = output/'data.tar.zst', output/'data.tar.zst.partial'
    if archive.exists():return download(config_path, output)
    parts = output/'parts'; parts.mkdir(exist_ok=True)
    plan_path = output/'parts_plan.json'
    if plan_path.exists():
        plan = strict_json(plan_path.read_text())
        if plan['config_sha256'] != sha256_file(config_path):raise ValidationError('range plan config changed')
    else:
        offset = prefix.stat().st_size
        plan = {'config_sha256': sha256_file(config_path), 'prefix_bytes': offset, 'prefix_sha256': sha256_file(prefix),
                'ranges': [[start, min(start+128*1024**2, config['expected_bytes'])-1]
                           for start in range(offset, config['expected_bytes'], 128*1024**2)]}
        plan_path.write_text(json_text(plan))
    offset = plan['prefix_bytes']
    if (type(offset) is not int or not 0 <= offset < config['expected_bytes']
            or plan['ranges'] != [[start, min(start+128*1024**2, config['expected_bytes'])-1]
                                 for start in range(offset, config['expected_bytes'], 128*1024**2)]):
        raise ValidationError('range plan has missing, overlapping or reordered extents')
    if prefix.stat().st_size != plan['prefix_bytes'] or sha256_file(prefix) != plan['prefix_sha256']:
        raise ValidationError('partial prefix changed while planning range transfer')
    if shutil.disk_usage(output).free < 2*config['expected_bytes']+2*1024**3:
        raise ValidationError('insufficient space for verified range assembly')
    def part(interval):
        start, end = interval; path = parts/f'{start}-{end}.bin'; record_path = path.with_suffix('.json')
        if record_path.exists():
            record = strict_json(record_path.read_text())
            if path.stat().st_size != end-start+1 or record['interval'] != interval:
                raise ValidationError('completed range changed')
            # Assembly rehashes every cached part before publishing the full
            # archive; avoid reading tens of GiB twice during resume.
            return record
        headers = {'User-Agent': 'ForetellMesh/0.1 public-data-research', 'Accept-Encoding': 'identity',
                   'Range': f'bytes={start}-{end}', 'If-Range': config['expected_etag']}
        began = now()
        with urlopen(Request(config['url'], headers=headers), timeout=60) as response:
            if (response.url != config['url'] or response.status != 206
                    or response.headers.get('ETag') != config['expected_etag']
                    or response.headers.get('Content-Range') != f'bytes {start}-{end}/{config["expected_bytes"]}'
                    or int(response.headers.get('Content-Length', '-1')) != end-start+1):
                raise ValidationError('range identity, extent or object changed')
            digest = hashlib.sha256(); count = 0
            with path.open('wb') as stream:
                while chunk := response.read(4*1024**2):
                    count += len(chunk)
                    if count > end-start+1:raise ValidationError('range exceeds pinned extent')
                    stream.write(chunk); digest.update(chunk)
        if count != end-start+1:raise ValidationError('truncated range response')
        record = {'interval': interval, 'sha256': digest.hexdigest(), 'bytes': count, 'started_at': began, 'completed_at': now()}
        record_path.write_text(json_text(record)); return record
    def retry_part(interval):
        for attempt in range(3):
            try:return part(interval)
            except (ValidationError, URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
                if isinstance(exc, ValidationError) and str(exc) != 'truncated range response':raise
                failure = {'interval': interval, 'attempt': attempt+1, 'time': now(),
                           'error_type': type(exc).__name__, 'error': str(exc)}
                with (parts/f'{interval[0]}-{interval[1]}.failures.jsonl').open('a') as log:
                    log.write(json_text(failure).replace('\n','')+'\n')
                if attempt == 2:raise
                time.sleep(attempt+1)
    began = now(); clock = time.monotonic(); received = plan['prefix_bytes']; records = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(retry_part, interval) for interval in plan['ranges']]
        try:
            for future in as_completed(futures):
                record = future.result(); records.append(record); received += record['bytes']
                progress = {'bytes': received, 'total_bytes': config['expected_bytes'],
                            'elapsed_seconds': round(time.monotonic()-clock, 2), 'completed': False, 'method': 'sixteen_range_readers'}
                (output/'progress.json').write_text(json_text(progress))
                if len(records) % 16 == 0:print(json_text(progress).strip(), flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise
    records.sort(key=lambda record: record['interval'][0])
    merged = output/'data.tar.zst.assembled'; digest = hashlib.sha256(); count = 0
    with merged.open('wb') as target:
        for path, expected in [(prefix, plan['prefix_sha256'])]+[(parts/f'{r["interval"][0]}-{r["interval"][1]}.bin', r['sha256']) for r in records]:
            piece = hashlib.sha256()
            with path.open('rb') as source:
                while chunk := source.read(8*1024**2):target.write(chunk); digest.update(chunk); piece.update(chunk); count += len(chunk)
            if piece.hexdigest() != expected:raise ValidationError('range changed during assembly')
    if count != config['expected_bytes']:raise ValidationError('assembled size mismatch')
    report = {'schema_version': '1', 'source': config['source'], 'url': config['url'],
              'upstream_schema_revision': config['upstream_schema_revision'], 'config_sha256': sha256_file(config_path),
              'http_etag': config['expected_etag'], 'started_at': began, 'completed_at': now(),
              'bytes': count, 'sha256': digest.hexdigest(), 'prefix_sha256': plan['prefix_sha256'],
              'range_workers': 16,
              'range_plan_sha256': sha256_file(plan_path), 'range_records': records,
              'hash_trust': 'Local complete SHA-256 and HTTP object binding; no publisher signature supplied.'}
    (output/'download_manifest.json').write_text(json_text(report)); merged.rename(archive)
    (output/'progress.json').write_text(json_text({'bytes': count, 'total_bytes': count, 'completed': True}))
    return {k: v for k, v in report.items() if k != 'range_records'}


def extract(config_path: Path, archive_root: Path, output: Path) -> dict:
    config = load_config(config_path)
    if output.exists():raise ValidationError('PMA extraction output exists')
    manifest = strict_json((archive_root/'download_manifest.json').read_text()); archive = archive_root/'data.tar.zst'
    if (manifest['config_sha256'] != sha256_file(config_path) or archive.stat().st_size != config['expected_bytes']):
        raise ValidationError('archive binding changed before extraction')
    # The CLI checks complete frames, including trailers after tar's end.
    # Hash its exact compressed input in the same pass to avoid three disk reads.
    executable = shutil.which('zstd')
    if executable is None:raise ValidationError('PMA extraction requires the zstd CLI for complete-frame verification')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pma-extract-', dir=output.parent) as temporary:
        stage = Path(temporary)/'extracted'; stage.mkdir()
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([executable, '--decompress', '--stdout', '--quiet'],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            compressed_digest = hashlib.sha256(); feed_errors = []; received = [0]
            def feed():
                try:
                    with archive.open('rb') as source, process.stdin:
                        while chunk := source.read(8*1024**2):
                            compressed_digest.update(chunk); received[0] += len(chunk); process.stdin.write(chunk)
                except Exception as exc:feed_errors.append(exc)
            feeder = threading.Thread(target=feed, name='pma-archive-hash'); feeder.start()
            try:
                report = extract_members(config_path, config, manifest, archive_root, process.stdout, stage)
                feeder.join(); status = process.wait()
                errors.seek(0); error_text = errors.read(500).decode('utf-8', errors='replace')
                if status:raise ValidationError('incomplete or corrupt zstd archive: '+error_text)
                if feed_errors:raise ValidationError('archive read failed: '+str(feed_errors[0]))
                if received[0] != config['expected_bytes'] or compressed_digest.hexdigest() != manifest['sha256']:
                    raise ValidationError('archive binding changed during extraction')
                report['completed_at'] = now()
                (stage/'manifest.json').write_text(json_text(report))
                if output.exists():raise ValidationError('PMA extraction output appeared during processing')
                stage.rename(output)
            except (tarfile.TarError, OSError) as exc:
                raise ValidationError('incomplete or corrupt zstd archive/tar stream: '+str(exc)) from exc
            finally:
                if process.poll() is None:process.kill()
                process.stdout.close(); process.wait(); feeder.join(); process.stdin.close()
    return {k: v for k, v in report.items() if k != 'selected'} | {'selected_shards': len(report['selected'])}


def extract_members(config_path, config, manifest, archive_root, decoded, output):
    """Write only to unpublished staging; caller validates frame and input hash."""
    selected, seen, counts, total = [], set(), Counter(), 0
    start = now()
    with tarfile.open(fileobj=decoded, mode='r|') as tar:
        with (output/'members.jsonl').open('w') as ledger:
            for member in tar:
                path = member_path(member, seen)
                if not member.isfile():continue
                parts = path.parts; table = '/'.join(parts[1:3])
                keep = table in config['extract_tables'] and len(parts) == 4 and parts[-1].endswith('.parquet') and not parts[-1].startswith('._')
                counts[table] += 1
                entry = {'name': member.name, 'bytes': member.size, 'selected': keep}
                if keep:
                    if not 0 <= member.size <= config['max_member_bytes']:raise ValidationError('archive member size exceeds bound')
                    total += member.size
                    if total > config['max_extracted_bytes'] or shutil.disk_usage(output).free < member.size+2*1024**3:
                        raise ValidationError('PMA extraction storage bound exceeded')
                    destination = output.joinpath(*parts[1:]); destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(); read = 0
                    with tar.extractfile(member) as source, destination.open('xb') as target:
                        while chunk := source.read(1024**2):target.write(chunk); digest.update(chunk); read += len(chunk)
                    if read != member.size:raise ValidationError('truncated archive member')
                    entry.update(file=destination.relative_to(output).as_posix(), sha256=digest.hexdigest())
                    selected.append(entry)
                    if len(selected) % 500 == 0:print(f'Extracted {len(selected)} Polymarket shards, {total} bytes', flush=True)
                ledger.write(json_text(entry).replace('\n', '')+'\n')
        # Drain to force the decoder through every frame and final checksum.
        while decoded.read(1024**2):pass
    report = {'schema_version': '1', 'kind': 'pma_selective_extraction', 'config_sha256': sha256_file(config_path),
              'archive_sha256': manifest['sha256'], 'download_manifest_sha256': sha256_file(archive_root/'download_manifest.json'),
              'started_at': start, 'completed_at': now(), 'selected': selected,
              'zstd_complete_frame_check': 'passed',
              'selected_bytes': total, 'all_file_counts_by_table': dict(sorted(counts.items())),
              'member_ledger_sha256': sha256_file(output/'members.jsonl')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'download-parallel', 'extract']); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.command == 'extract' and args.archive is None:parser.error('extract requires --archive')
    if args.command == 'download':result = download(args.config, args.output)
    elif args.command == 'download-parallel':result = download_parallel(args.config, args.output)
    else:result = extract(args.config, args.archive, args.output)
    print(json_text(result))


if __name__ == '__main__':main()
