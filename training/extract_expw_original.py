"""Wait for the verified ExpW archive and extract its original volumes into a bounded directory."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

BASE=Path('D:/ExpW-source').resolve()
DEADLINE=datetime.fromisoformat('2026-09-28T08:00:00+08:00')

def main():
    while not (BASE/'download-report.json').is_file():
        if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Authorized work deadline reached before download completed')
        time.sleep(30)
    report=json.loads((BASE/'download-report.json').read_text())
    if not report.get('zip_crc_verified'):raise ValueError('Download not verified')
    archive=BASE/'origin.7z.001';destination=(BASE/'original').resolve()
    if destination.parent!=BASE:raise ValueError('Destination outside ExpW source directory')
    executable=Path('D:/7-Zip/7z.exe')
    command=[str(executable),'l','-slt',str(archive)]
    listing=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',errors='replace',check=True,creationflags=subprocess.CREATE_NO_WINDOW).stdout
    body=listing.split('----------',1)[-1]
    entries=[]
    for line in body.splitlines():
        if line.startswith('Path = '):
            relative=Path(line[7:])
            target=(destination/relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(destination):raise ValueError('Unsafe archive path')
            entries.append(str(relative))
    if not entries:raise ValueError('Archive listing empty')
    (BASE/'original-entries.json').write_text(json.dumps(entries,indent=2))
    destination.mkdir(exist_ok=True)
    subprocess.run([str(executable),'x',str(archive),'-o'+str(destination),'-aos','-y','-bsp0'],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    images=list(destination.rglob('*.jpg'))
    result={'status':'complete','image_count':len(images),'destination':str(destination),'source_archive_sha256':report['sha256'],'extractor':'7-Zip CRC checked during extraction','time':datetime.now(timezone.utc).isoformat()}
    if len(images)<60000:raise ValueError(f'Unexpectedly few original images: {len(images)}')
    (BASE/'extraction-report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
