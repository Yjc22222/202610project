"""Download and extract ASVspoof 2019 LA (Windows/Linux/macOS).
AASIST, Copyright (c) 2021-present NAVER Corp. MIT license.
"""
import argparse
from http.client import IncompleteRead
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATASET_URL = ('https://datashare.ed.ac.uk/server/api/core/bitstreams/'
               'a9f87c35-f055-4015-80e2-2fdff0d46269/content')
CHUNK_SIZE = 1024 * 1024
REQUIRED_DIRS = tuple('ASVspoof2019_LA_' + split + '/flac'
                      for split in ('train', 'dev', 'eval'))
REQUIRED_FILES = (
    'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
    'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
    'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
    'ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt',
)


def download(url, archive, retries=4):
    """Keep incomplete downloads in LA.zip.part; resume using HTTP Range."""
    partial = archive.with_suffix(archive.suffix + '.part')
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'AASIST-dataset-downloader/1.0',
                   'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = 'bytes={}-'.format(offset)
        try:
            try:
                response = urlopen(Request(url, headers=headers), timeout=60)
            except HTTPError as error:
                if (error.code == 416 and offset and
                        error.headers.get('Content-Range') == 'bytes */{}'.format(offset)):
                    partial.replace(archive)
                    return
                raise
            with response:
                if 'text/html' in response.headers.get('Content-Type', '').lower():
                    raise ValueError('Server returned a web page instead of the ZIP archive.')
                if response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith('bytes {}-'.format(offset)):
                        raise ValueError('Server returned an unexpected download range.')
                    total_text = content_range.rsplit('/', 1)[-1]
                    total = int(total_text) if total_text != '*' else None
                elif response.status == 200:
                    if offset:
                        print('Server does not support resume; restarting download.', flush=True)
                    offset = 0
                    size = response.headers.get('Content-Length')
                    total = int(size) if size else None
                else:
                    raise ValueError('Unexpected HTTP status: {}'.format(response.status))
                print('Downloading {} (starting at {:.1f} MB)'.format(
                    archive.name, offset / 1024**2), flush=True)
                if total and shutil.disk_usage(archive.parent).free < total - offset:
                    raise OSError('Not enough free disk space for the download.')
                received = offset
                last_report = time.monotonic()
                with partial.open('ab' if offset else 'wb') as output:
                    while True:
                        block = response.read(CHUNK_SIZE)
                        if not block:
                            break
                        output.write(block)
                        received += len(block)
                        if time.monotonic() - last_report >= 2:
                            progress = '{:.1f} MB'.format(received / 1024**2)
                            if total:
                                progress += ' / {:.1f} MB ({:.1f}%)'.format(
                                    total / 1024**2, received / total * 100)
                            print(progress, flush=True)
                            last_report = time.monotonic()
                if total is not None and received != total:
                    raise OSError('Incomplete download: {} of {} bytes.'.format(received, total))
            partial.replace(archive)
            print('Download complete: {}'.format(archive), flush=True)
            return
        except (OSError, URLError, ValueError, IncompleteRead) as error:
            if isinstance(error, HTTPError) and error.code in (400, 401, 403, 404):
                raise
            if isinstance(error, ValueError) or attempt == retries:
                raise
            print('Download interrupted: {}. Retry {}/{}; partial file kept.'.format(
                error, attempt, retries - 1), flush=True)
            time.sleep(2)



def download_parallel(url, archive, connections=4):
    """Resume independent byte ranges, preserving an older sequential download."""
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor, wait

    headers = {'User-Agent': 'AASIST-dataset-downloader/1.0', 'Accept-Encoding': 'identity',
               'Range': 'bytes=0-0'}
    with urlopen(Request(url, headers=headers), timeout=30) as probe:
        content_range = probe.headers.get('Content-Range', '')
        if probe.status != 206 or not content_range.startswith('bytes 0-0/'):
            print('Server does not support parallel ranges; using one connection.', flush=True)
            return download(url, archive)
        total = int(content_range.rsplit('/', 1)[1])
        probe.read(1)
    folder = archive.with_suffix('.zip.parts')
    folder.mkdir(exist_ok=True)
    manifest = folder / 'download.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['url'] != url or saved['total'] != total:
            raise ValueError('Existing download parts belong to a different file: {}'.format(folder))
        connections = saved['connections']
    else:
        manifest.write_text(json.dumps({'url': url, 'total': total,
                                        'connections': connections}), encoding='utf-8')
    if not 1 <= connections <= 16:
        raise ValueError('Invalid connection count in download manifest.')
    ranges = [(total * i // connections, total * (i + 1) // connections)
              for i in range(connections)]
    parts = [folder / ('part{:02d}'.format(i)) for i in range(connections)]
    old_partial = archive.with_suffix('.zip.part')
    # Copy the existing sequential prefix into the corresponding ranges.
    if old_partial.exists():
        old_size = min(old_partial.stat().st_size, total)
        with old_partial.open('rb') as old:
            for part, (start, end) in zip(parts, ranges):
                have = part.stat().st_size if part.exists() else 0
                seed_end = min(end, old_size)
                if seed_end > start + have:
                    old.seek(start + have)
                    remaining = seed_end - start - have
                    with part.open('ab') as output:
                        while remaining:
                            block = old.read(min(CHUNK_SIZE, remaining))
                            if not block:
                                raise OSError('Old partial file changed during migration. Stop the old downloader first.')
                            output.write(block)
                            remaining -= len(block)
    sizes = [part.stat().st_size if part.exists() else 0 for part in parts]
    for size, (start, end) in zip(sizes, ranges):
        if size > end - start:
            raise ValueError('A download part is larger than expected: {}'.format(folder))
    if shutil.disk_usage(archive.parent).free < 2 * total - sum(sizes):
        raise OSError('Not enough disk space for parallel parts and the assembled ZIP.')
    stop = threading.Event()

    def worker(index):
        part = parts[index]
        start, end = ranges[index]
        for attempt in range(4):
            if stop.is_set():
                return
            have = part.stat().st_size if part.exists() else 0
            if have == end - start:
                return
            request_headers = dict(headers)
            request_headers['Range'] = 'bytes={}-{}'.format(start + have, end - 1)
            try:
                with urlopen(Request(url, headers=request_headers), timeout=30) as response:
                    expected = 'bytes {}-{}/{}'.format(start + have, end - 1, total)
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Unexpected range response for part {}'.format(index))
                    with part.open('ab') as output:
                        while not stop.is_set():
                            block = response.read(64 * 1024)
                            if not block:
                                break
                            if sizes[index] + len(block) > end - start:
                                raise ValueError('Server sent too many bytes for part {}'.format(index))
                            output.write(block)
                            sizes[index] += len(block)
                    if stop.is_set():
                        return
                    if sizes[index] != end - start:
                        raise OSError('Incomplete response for part {}'.format(index))
                return
            except (OSError, URLError, IncompleteRead) as error:
                if isinstance(error, HTTPError) and error.code in (400, 401, 403, 404):
                    raise
                if attempt == 3:
                    raise
                print('Part {} interrupted; retrying: {}'.format(index + 1, error), flush=True)
                stop.wait(2)

    print('Downloading with {} connections; {:.1f} MB already saved.'.format(
        connections, sum(sizes) / 1024**2), flush=True)
    previous_size = sum(sizes)
    previous_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(worker, i) for i in range(connections)]
        pending = set(futures)
        try:
            while pending:
                done, pending = wait(pending, timeout=1)
                for future in done:
                    future.result()
                now = time.monotonic()
                if now - previous_time >= 2:
                    current = sum(sizes)
                    speed = (current - previous_size) / (now - previous_time)
                    eta = '{:.0f} min'.format((total - current) / speed / 60) if speed else 'waiting'
                    print('{:.1f}% | {:.1f}/{:.1f} MB | {:.2f} MB/s | ETA {}'.format(
                        current / total * 100, current / 1024**2, total / 1024**2,
                        speed / 1024**2, eta), flush=True)
                    previous_time, previous_size = now, current
        finally:
            stop.set()
    if any(size != end - start for size, (start, end) in zip(sizes, ranges)):
        raise OSError('Download incomplete. Run the same command to resume.')
    assembled = folder / 'assembled.zip'
    print('All parts downloaded. Assembling LA.zip...', flush=True)
    with assembled.open('wb') as output:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, output, CHUNK_SIZE)
    if assembled.stat().st_size != total or not zipfile.is_zipfile(assembled):
        raise ValueError('Assembled archive is invalid; download parts have been kept.')
    assembled.replace(archive)
    # Only delete this downloader's known files, after successful assembly.
    for part in parts:
        part.unlink()
    manifest.unlink()
    if not any(folder.iterdir()):
        folder.rmdir()
    print('Download complete: {}'.format(archive), flush=True)


def validate_layout(dataset_dir):
    missing = [name for name in REQUIRED_DIRS if not (dataset_dir / name).is_dir()]
    missing += [name for name in REQUIRED_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise ValueError('Dataset layout is incomplete. Missing under {}: {}'.format(
            dataset_dir, ', '.join(missing)))
    print('Dataset ready: {}'.format(dataset_dir), flush=True)


def extract(archive, destination):
    """CRC-check each extracted file; reject paths outside the destination."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as dataset:
        members = dataset.infolist()
        for member in members:
            target = (root / member.filename.replace('\\', '/')).resolve()
            if target != root and root not in target.parents:
                raise ValueError('Unsafe ZIP member: {}'.format(member.filename))
        required_space = sum(member.file_size for member in members)
        if shutil.disk_usage(root).free < required_space:
            raise OSError('Extraction needs {:.1f} GB of free disk space.'.format(
                required_space / 1024**3))
        print('Extracting {} entries; file contents will be CRC-checked.'.format(
            len(members)), flush=True)
        last_report = time.monotonic()
        for index, member in enumerate(members, 1):
            target = root / member.filename.replace('\\', '/')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            try:
                with dataset.open(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, CHUNK_SIZE)
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if time.monotonic() - last_report >= 2:
                print('Extracted {}/{} entries'.format(index, len(members)), flush=True)
                last_report = time.monotonic()
    validate_layout(destination / 'LA')


def main():
    parser = argparse.ArgumentParser(description='Download and extract ASVspoof 2019 LA.')
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parent,
                        help='Parent folder for LA/ and LA.zip (default: project folder).')
    parser.add_argument('--url', default=DATASET_URL, help='Alternative LA.zip download URL.')
    parser.add_argument('--connections', type=int, choices=range(1, 17), default=4,
                        help='Parallel connections (default: 4; 1 uses the old sequential download).')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download-only', action='store_true', help='Download without extracting.')
    mode.add_argument('--extract-only', action='store_true', help='Extract an existing LA.zip offline.')
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / 'LA.zip'
    if not archive.exists():
        if args.extract_only:
            raise FileNotFoundError('Archive not found: {}'.format(archive))
        if args.connections == 1:
            download(args.url, archive)
        else:
            download_parallel(args.url, archive, args.connections)
    if not zipfile.is_zipfile(archive):
        raise ValueError('Invalid ZIP archive: {}. Rename/remove it, then retry.'.format(archive))
    if not args.download_only:
        extract(archive, destination)
        print('For AASIST.conf, use database_path: "{}"'.format(
            (destination / 'LA').as_posix()), flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled. Run the same command again to resume.', file=sys.stderr)
        sys.exit(130)
    except (OSError, URLError, ValueError, IncompleteRead, zipfile.BadZipFile, RuntimeError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        print('Fix the issue and run again. Downloads keep LA.zip.part; extraction can be retried.',
              file=sys.stderr)
        sys.exit(1)



