"""Queue all fixed consistency candidates for video diagnosis, without promotion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    work = Path('training/noon-20260928')
    queue_file = work/'queue.json'
    queue = json.loads(queue_file.read_text(encoding='utf-8'))
    if any(j['id'].startswith('consistency-video') for j in queue):
        raise ValueError('Video diagnosis already queued; do not duplicate')
    selection = json.loads((work/'consistency-paired-plan.json').read_text(encoding='utf-8'))
    photos = next(j for j in selection['jobs'] if j['id']=='consistency-photos')['args']
    checkpoints = photos[photos.index('--candidates')+1:photos.index('--dataset')]
    assessment = json.loads((work/'consistency-photo-assessment.json').read_text(encoding='utf-8'))
    if any(r['photo_pass'] for r in assessment['results']):
        raise ValueError('A passing candidate needs full release acceptance, not this failed-method diagnostic')
    cache = Path('data/video-validation-combined-10fps-v1')
    manifest_path = cache/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert len(manifest['videos']) == 538
    assert not any('holdout' in v['source'].lower() or 'test' in v['source'].lower() for v in manifest['videos'])
    output = work/'consistency-video'
    plan = {
        'declared_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Diagnose failed method on all538 fixed video clips and exact frontend smoothing. No deployment permission.',
        'selection': 'All six previously fixed epoch1/2/4 candidates plus released baseline; no success-only candidate or clip selection.',
        'precision': 'PyTorch GPU float32 with TF32 disabled; not measured CPU or browser latency.',
        'cache': str(cache), 'cache_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'candidates': [{'file':p, 'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in checkpoints],
        'photo_assessment_sha256': hashlib.sha256((work/'consistency-photo-assessment.json').read_bytes()).hexdigest(),
        'photo_pass_count': 0,
        'limits': 'Repeated selection validation; old final2610/180 excluded. Clip emotion labels are not per-frame natural-expression truth. CAER is scene diagnosis only.',
        'deployment': 'Even a video improvement does not override failed photo/perclass gates. Current FP32 remains deployed.'}
    (work/'consistency-video-plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    jobs = [
        {'id':'consistency-video-all-fixed','args':['training/evaluate_video_models.py','--checkpoint',*checkpoints,
            '--cache',str(cache),'--output',str(output),'--batch-size','40','--inference-precision','float32'],
            'depends_on':['consistency-assessment'],'require_after':[str(output/'summary.json')]},
        {'id':'consistency-video-intensity','args':['training/stratify_video_validation.py','--reports',str(output),
            '--output',str(work/'consistency-video-intensity.json')],
            'depends_on':['consistency-video-all-fixed'],'require_after':[str(work/'consistency-video-intensity.json')]},
        {'id':'consistency-video-eyewear','args':['training/stratify_enterface_video.py','--reports',str(output),
            '--output',str(work/'consistency-video-eyewear.json')],
            'depends_on':['consistency-video-all-fixed'],'require_after':[str(work/'consistency-video-eyewear.json')]}]
    temp = queue_file.with_suffix('.tmp')
    temp.write_text(json.dumps(queue+jobs,indent=2),encoding='utf-8');temp.replace(queue_file)
    print(json.dumps({'queued':[j['id'] for j in jobs],'clips':538,'candidates':len(checkpoints),'deployed':False}),flush=True)


if __name__=='__main__':main()
