"""Fetch pinned third-party mirror; do not equate it with verified author bytes."""
import hashlib
import json
from pathlib import Path
import requests

root=Path('models/research/poster-v2-affectnet7')
root.mkdir(parents=True,exist_ok=True)
revision='396b49075db21805ff831193b492c606dc1178d1'
name='affectnet-7-model_best.pth'
url=f'https://hf-mirror.com/minhnguyent546/poster_v2_model_artifacts/resolve/{revision}/{name}'
expected='7ca3533b076594ccd573d30c9b666167c8241e811a70f9cfe2b6a4586dcaf5b5'
provenance={'url':url,'revision':revision,'expected_sha256':expected,
            'mirror_type':'third-party; author-byte equivalence and label order not yet verified',
            'official_source':'https://github.com/Talented-Q/POSTER_V2', 'status':'downloading'}
(root/'mirror-source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
try:
    h=hashlib.sha256(); size=0
    with requests.get(url,stream=True,timeout=(25,60)) as r:
        r.raise_for_status()
        with (root/(name+'.part')).open('wb') as f:
            for block in r.iter_content(1024*1024):
                f.write(block);h.update(block);size+=len(block)
                if size % (20*1024*1024)==0: print('bytes',size,flush=True)
    if h.hexdigest()!=expected: raise ValueError('Mirror LFS SHA256 mismatch')
    (root/(name+'.part')).replace(root/name)
    provenance.update(status='downloaded-mirror-sha-verified',bytes=size,sha256=h.hexdigest())
except Exception as e:
    provenance.update(status='failed',error=str(e))
    raise
finally:
    (root/'mirror-source.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance),flush=True)
