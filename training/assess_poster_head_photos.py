"""Compare each prospectively fixed new-head checkpoint to the released ViT."""
import json
from pathlib import Path
import subprocess
import sys

work=Path('training/noon-20260928')
run=Path('training/runs/poster-frozen-seven-v1')
paired=json.loads((work/'paired-photos.json').read_text(encoding='utf-8'))
plan=json.loads((work/'sam-paired-plan.json').read_text(encoding='utf-8'))
base_key=next(k for k in paired['reports'] if Path(k).resolve()==Path(plan['baseline']).resolve())
reports={base_key:paired['reports'][base_key]}
for name in ('best','epoch-010','epoch-020','epoch-030'):
    reports[str(run/(name+'.pt'))]=json.loads((run/(name+'-photos.json')).read_text(encoding='utf-8'))
combined={'device':'CUDA FP32 frozen visual features + CPU FP32 trained head',
          'inference_precision':'float32','reports':reports,
          'scope':'New known-label head, original anonymous head discarded. Repeated selection validation, not new blind test.'}
dest=work/'poster-head-photos.json';dest.write_text(json.dumps(combined,indent=2),encoding='utf-8')
subprocess.run([sys.executable,'-X','utf8','training/assess_noon_photos.py','--report',str(dest),
                '--output',str(work/'poster-head-assessment.json')],check=True)
