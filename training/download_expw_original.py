"""Resume the public ExpW archive using bounded, verified ranges, then check ZIP CRCs."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import zipfile

SOURCE = 'https://www.kaggle.com/api/v1/datasets/download/shahzadabbas/expression-in-the-wild-expw-dataset'
SIZE = 8117492353
CHUNK = 16 * 1024 * 1024
DEST = Path('D:/ExpW-source')

def main():
    DEST.mkdir(exist_ok=True)
    session = json.loads((DEST / 'download-session.json').read_text())
    url = session['url']
    path = DEST / 'expw-original.zip'
    journal = DEST / 'download-ranges.jsonl'
    if not path.exists():
        with path.open('wb') as f:
            f.truncate(SIZE)
    if path.stat().st_size != SIZE:
        raise ValueError('Unexpected output size; do not overwrite unknown data')
    completed = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            completed[row['start']] = row
    # Verify previously downloaded ranges before trusting resume records.
    with path.open('rb') as f:
        for start, row in list(completed.items()):
            f.seek(start)
            if hashlib.sha256(f.read(row['bytes'])).hexdigest() != row['sha256']:
                del completed[start]

    def fetch(start):
        end = min(SIZE - 1, start + CHUNK - 1)
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Range mismatch')
                    raw = response.read(end - start + 2)
                if len(raw) != end - start + 1:
                    raise ValueError('Truncated range')
                return start, raw
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f'Range {start} failed: {type(exc).__name__}') from None
                time.sleep(2 ** attempt)
        raise AssertionError

    starts = list(range(0, SIZE, CHUNK))
    pending = [start for start in starts if start not in completed]
    print(f'ExpW ranges complete={len(completed)}/{len(starts)}', flush=True)
    with journal.open('a') as log, ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, start) for start in pending}
        for future in as_completed(futures):
            start, raw = future.result()
            futures.remove(future)
            with path.open('r+b', buffering=0) as f:
                f.seek(start)
                f.write(raw)
            row = {'start': start, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            log.write(json.dumps(row) + '\n')
            log.flush()
            completed[start] = row
            del raw
            if len(completed) % 10 == 0 or len(completed) == len(starts):
                print(f'ExpW ranges complete={len(completed)}/{len(starts)}', flush=True)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if '/' in info.filename or '\\' in info.filename or info.filename.startswith('.'):
                raise ValueError('Unexpected archive path')
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f'CRC failure: {bad}')
        for info in z.infolist():
            if info.filename.startswith('origin.7z.'):
                # Published mirror accidentally adds .001 to the first volume name.
                name = 'origin.7z.001' if info.filename == 'origin.7z.001.001' else info.filename
                destination = DEST / name
                with z.open(info) as source, destination.open('wb') as target:
                    while raw := source.read(1024 * 1024):
                        target.write(raw)
                print('extracted', name, info.file_size, flush=True)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    report = {'status': 'complete', 'source': SOURCE,
              'original_source': 'https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html',
              'archive_bytes': SIZE, 'sha256': digest, 'zip_crc_verified': True,
              'labels': 'Original label.lst with per-face boxes and seven class ids; mapping recorded in original readme.txt.',
              'usage': 'Local academic research; do not distribute source images in deployment.'}
    (DEST / 'download-report.json').write_text(json.dumps(report, indent=2))
    (DEST / 'download-session.json').unlink(missing_ok=True)
    print('EXPW_DOWNLOAD_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
