"""Verify a release against current source and pinned assets before removing old releases."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

def digest(handle):
    return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);args=p.parse_args()
    root=args.source.resolve();archive_path=args.zip.resolve()
    lock=json.loads((root/'deployment/runtime-lock.json').read_text(encoding='utf-8-sig'))
    assets=[lock['python'],lock['vc_redist'],*lock['wheels']]
    expected={'start-demo.cmd','启动服务.cmd','server.py','scripts/start-demo.ps1','scripts/launcher.py',
              'training/emotion_preprocessing.py','deployment/runtime-lock.json','deployment/requirements-runtime.txt'}
    expected.update(model['file'].replace('\\','/') for model in lock['models'])
    expected.update('deployment/cache/'+asset['file'] for asset in assets)
    expected.update('web/'+file.name for file in (root/'web').iterdir() if file.is_file())
    with ZipFile(archive_path) as archive:
        failed=archive.testzip()
        if failed:raise ValueError('ZIP CRC failed: '+failed)
        entries={}
        for item in archive.infolist():
            if item.is_dir():continue
            name=item.filename.replace('\\','/')
            parts=PurePosixPath(name).parts
            if len(parts)<2 or parts[0]!=archive_path.stem or any(part in ('.','..') for part in parts):
                raise ValueError('Unexpected archive path: '+name)
            relative='/'.join(parts[1:])
            if relative in entries:raise ValueError('Duplicate archive entry: '+relative)
            entries[relative]=item
        if set(entries)!=expected:
            raise ValueError(json.dumps({'missing':sorted(expected-set(entries)),'extra':sorted(set(entries)-expected)}))
        for relative,item in entries.items():
            source=(root/relative).resolve()
            if not source.is_relative_to(root):raise ValueError('Source path escapes workspace')
            with archive.open(item) as handle:archived_hash=digest(handle)
            with source.open('rb') as handle:source_hash=digest(handle)
            if archived_hash!=source_hash:raise ValueError('Archive/source mismatch: '+relative)
        for asset in assets:
            with archive.open(entries['deployment/cache/'+asset['file']]) as handle:actual=digest(handle)
            if actual!=asset['sha256'].lower():raise ValueError('Pinned runtime asset mismatch: '+asset['file'])
        for model in lock['models']:
            with archive.open(entries[model['file'].replace('\\','/')]) as handle:actual=digest(handle)
            if actual!=model['sha256'].lower():raise ValueError('Pinned model mismatch: '+model['file'])
    with archive_path.open('rb') as handle:archive_hash=digest(handle)
    print(json.dumps({'status':'verified','zip':str(archive_path),'files':len(expected),'sha256':archive_hash,
                      'checks':['ZIP CRC','exact deployable file allowlist','every file matches source','pinned runtime/model hashes']}),flush=True)

if __name__=='__main__':main()
