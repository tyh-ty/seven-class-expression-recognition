"""Download pinned eNTERFACE research mirror; split by subject before any modeling.

Original paper DOI10.1109/ICDEW.2006.145 states publicly available for scientific
research. The mirror's Apache claim is not substituted for source terms.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

REPO = 'NoahMartinezXiang/2006_eNTERFACE05'
REVISION = '61cc20c56bee600e299f19041f28df511f73fa47'
URL = f'https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/2006_eNTERFACE05.zip'
SIZE = 844060422
SHA = 'd8533c04e974728f1e86e0e800000ec719d0220ba246c7e3786d943080c71e6d'
OUT = Path('D:/eNTERFACE-source')
ROOT = Path(__file__).resolve().parents[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / 'enterface05-pinned.zip'
    verified = archive.is_file() and archive.stat().st_size == SIZE
    if verified:
        with archive.open('rb') as handle:
            verified = hashlib.file_digest(handle, 'sha256').hexdigest() == SHA
    if not verified:
        partial = archive.with_suffix('.partial')
        with partial.open('wb') as handle:
            handle.truncate(SIZE)
        starts = list(range(0, SIZE, 8*1024*1024))
        def fetch(start):
            end = min(SIZE-1, start+8*1024*1024-1)
            for attempt in range(4):
                try:
                    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=35) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                            raise ValueError('Byte range mismatch')
                        raw = response.read(end-start+2)
                    if len(raw) != end-start+1:
                        raise ValueError('Truncated chunk')
                    # Each worker writes its nonoverlapping range then releases bytes.
                    with partial.open('r+b', buffering=0) as handle:
                        handle.seek(start)
                        handle.write(raw)
                    return len(raw)
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
        with ThreadPoolExecutor(4) as pool:
            futures = {pool.submit(fetch, start) for start in starts}
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                futures.remove(future)
                if i % 10 == 0 or i == len(starts):
                    print('download', i, '/', len(starts), flush=True)
        with partial.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != SHA:
                raise ValueError('Archive SHA256 mismatch')
        partial.replace(archive)
    with zipfile.ZipFile(archive) as z:
        entries = [e for e in z.infolist() if e.filename.lower().endswith('.avi')]
        parsed = []
        for e in entries:
            match = re.fullmatch(r'2006_eNTERFACE05/subject (\d+)/(anger|disgust|fear|happiness|sadness|surprise)/(?:sentence (\d+)/)?[^/]+\.avi', e.filename)
            if not match or not 0 < e.file_size < 50*1024*1024:
                raise ValueError(f'Unexpected entry: {e.filename}')
            subject, label, sentence = match.groups()
            parsed.append((e, int(subject), label, int(sentence) if sentence is not None else None))
        subjects = sorted({r[1] for r in parsed}, key=lambda s: hashlib.sha256(f'enterface-subject-split-v1:{s}'.encode()).hexdigest())
        if len(subjects) < 30:
            raise ValueError('Unexpectedly small subject cohort')
        split_map = {s: ('holdout' if i < 6 else 'val' if i < 12 else 'train') for i, s in enumerate(subjects)}
        rows = []
        for e, subject, label, sentence in parsed:
            split = split_map[subject]
            row = {'source_entry': e.filename, 'subject': subject, 'label': label, 'sentence': sentence,
                   'split': split, 'bytes': e.file_size, 'crc32': e.CRC}
            if split != 'holdout':
                destination = OUT / split / f'subject-{subject:02d}' / label / Path(e.filename).name
                if not destination.resolve().is_relative_to(OUT.resolve()):
                    raise ValueError('Archive target escaped intended directory')
                destination.parent.mkdir(parents=True, exist_ok=True)
                raw = z.read(e)  # Checks per-file archive CRC.
                destination.write_bytes(raw)
                row.update(file=str(destination), sha256=hashlib.sha256(raw).hexdigest())
            rows.append(row)
    report = {'source_paper': 'https://doi.org/10.1109/ICDEW.2006.145',
              'source_permission_evidence': 'training/overnight-20260927/source-review/enterface-paper-research.json; paper abstract: This database is made publicly available for scientific research purposes.',
              'mirror': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION,
              'archive_sha256': SHA, 'archive_bytes': SIZE,
              'limitations': 'Third-party copy; equivalence to an original author checksum cannot be established. Controlled elicited emotional speech, not spontaneous natural webcam or micro-expression ground truth. No neutral class.',
              'split_method': 'SHA256 ordering of subject IDs with seed enterface-subject-split-v1; first6holdout/next6validation/resttraining. Declared before reading video pixels or model outputs. Cross-dataset identity overlap not exhaustively established.',
              'holdout_videos_extracted': False, 'subject_splits': split_map,
              'counts': dict(Counter(f"{r['split']}/{r['label']}" for r in rows)), 'videos': rows}
    (OUT/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps({'subjects': len(subjects), 'counts': report['counts']}), flush=True)


if __name__ == '__main__':
    main()
