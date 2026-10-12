"""Evaluate only the frozen selected system and baseline on a final reserved split."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort

from emotion_preprocessing import emotion_tensor_path
from evaluate_onnx_ensemble import CLASSES, metrics
from temporal_runtime import TemporalRuntime

ROOT = Path(__file__).resolve().parents[1]


def read_frozen(path, allow_contract_smoke=False):
    decision = json.loads(path.read_text(encoding='utf-8'))
    if decision.get('contract_smoke_only') and not allow_contract_smoke:
        raise ValueError('A contract test fixture cannot open reserved data')
    if decision.get('frozen_before_holdout') is not True:
        raise ValueError('Freeze the final system before opening reserved examples')
    if tuple(decision.get('classes', ())) != CLASSES:
        raise ValueError('Frozen system class order was not verified')
    for name in ('baseline', 'selected'):
        system = decision[name]
        if abs(sum(g['weight'] for g in system['graphs'])-1.) > 1e-8:
            raise ValueError('Frozen visual weights must sum to one')
        for graph in system['graphs']:
            if graph['weight'] < 0 or hashlib.sha256(Path(graph['path']).read_bytes()).hexdigest() != graph['sha256']:
                raise ValueError('Frozen visual graph changed')
        if system.get('temporal'):
            if len(system['graphs']) != 1:
                raise ValueError('Temporal model was not trained for a visual ensemble')
            temporal = system['temporal']
            if hashlib.sha256(Path(temporal['path']).read_bytes()).hexdigest() != temporal['sha256']:
                raise ValueError('Frozen temporal graph changed')
    return decision


class FrozenPair:
    def __init__(self, decision):
        self.decision = decision
        self.graphs = {}
        self.temporal = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        for name in ('baseline', 'selected'):
            for graph in decision[name]['graphs']:
                key = (graph['sha256'], graph['grayscale'])
                if key not in self.graphs:
                    session = ort.InferenceSession(graph['path'], sess_options=options, providers=['CPUExecutionProvider'])
                    if session.get_inputs()[0].shape[1:] != [3,224,224] or session.get_outputs()[0].shape[-1] != 7:
                        raise ValueError('Unexpected visual graph dimensions')
                    self.graphs[key] = session
            temporal = decision[name].get('temporal')
            self.temporal[name] = TemporalRuntime(Path(temporal['path']), Path(decision[name]['graphs'][0]['path'])) if temporal else None

    def scores(self, path):
        tensor_cache, score_cache = {}, {}
        for (digest, grayscale), session in self.graphs.items():
            if grayscale not in tensor_cache:
                tensor_cache[grayscale] = emotion_tensor_path(path, grayscale=grayscale)[None]
            logits = session.run(None, {session.get_inputs()[0].name:tensor_cache[grayscale]})[0].reshape(7).astype(np.float64)
            values = np.exp(logits-logits.max())
            score_cache[digest,grayscale] = values/values.sum()
        return {name:sum(g['weight']*score_cache[g['sha256'],g['grayscale']] for g in self.decision[name]['graphs'])
                for name in ('baseline','selected')}


def confusion_report(labels, predictions):
    matrix = np.zeros((7,8),dtype=np.int64)
    for label,predicted in zip(labels,predictions):
        matrix[CLASSES.index(label),CLASSES.index(predicted) if predicted is not None else 7] += 1
    result = metrics(matrix[:,:7])
    correct = matrix[:,:7].diagonal()
    support = matrix.sum(1)
    precision = correct/np.maximum(1,matrix[:,:7].sum(0))
    recall = correct/np.maximum(1,support)
    f1 = 2*precision*recall/np.maximum(1e-12,precision+recall)
    result.update(examples=len(labels),accuracy=float(correct.sum()/max(1,len(labels))),
                  macro_f1=float(f1.mean()),macro_f1_present=float(f1[support>0].mean()),
                  confusion_with_no_face_column=matrix.tolist(),no_face=int(matrix[:,7].sum()))
    for i,label in enumerate(CLASSES):
        result['per_class'][label].update(correct=int(correct[i]),support=int(support[i]),recall=float(recall[i]),
                                          precision=float(precision[i]),f1=float(f1[i]))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--mode',choices=('photos','videos'),required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    decision = read_frozen(args.decision)
    fingerprint = hashlib.sha256(args.decision.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    start_file = args.output/(args.mode+'-started.json')
    if start_file.exists():
        if json.loads(start_file.read_text())['decision_sha256'] != fingerprint:
            raise ValueError('Reserved split was already opened for a different frozen selection')
    else:
        start_file.write_text(json.dumps({'decision_sha256':fingerprint,'opened_at':datetime.now(timezone.utc).isoformat(),
            'input':str(args.input),'rule':'Resume only samefrozen system aftertechnicalfailure. Do not tune using finalresults.'},indent=2),encoding='utf-8')
    final_file = args.output/(args.mode+'-final.json')
    if final_file.exists():
        report = json.loads(final_file.read_text())
        if report['decision_sha256'] != fingerprint:
            raise ValueError('Final result fingerprint mismatch')
        print('Already completed same frozen evaluation',str(final_file),flush=True)
        return
    cv2.setNumThreads(1)
    pair = FrozenPair(decision)
    started = time.perf_counter()
    if args.mode == 'photos':
        if args.input.name != 'expw-holdout':
            raise ValueError('Final photo input must be the reserved ExpW split')
        rows = {name:[] for name in ('baseline','selected')}
        files = [(label,path) for label in CLASSES for path in sorted((args.input/label).glob('*')) if path.suffix.lower() in ('.jpg','.png','.jpeg')]
        if len(files) != 2610:
            raise ValueError('Reserved ExpW size changed')
        for i,(label,path) in enumerate(files,1):
            scores = pair.scores(path)
            for name,values in scores.items():
                rows[name].append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'label':label,'predicted':CLASSES[int(values.argmax())],'probabilities':values.tolist()})
            if i%500==0:
                print(json.dumps({'photos':i,'total':len(files)}),flush=True)
        evaluated = {name:dict(confusion_report([r['label'] for r in records],[r['predicted'] for r in records]),records=records)
                     for name,records in rows.items()}
    else:
        manifest = json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('split') != 'holdout' or manifest.get('partial_smoke_only') or len(manifest['videos']) != 180:
            raise ValueError('Expected complete reserved eNTERFACE cohort')
        videos = {name:copy.deepcopy(manifest['videos']) for name in ('baseline','selected')}
        for vi,video in enumerate(manifest['videos']):
            for runtime in pair.temporal.values():
                if runtime:
                    runtime.reset(video['id'])
            for fi,frame in enumerate(video['frames']):
                if frame['status'] != 'detected':
                    for runtime in pair.temporal.values():
                        if runtime:
                            runtime.reset(video['id'])
                    continue
                scores = pair.scores(args.input/frame['crop'])
                for name,values in scores.items():
                    if pair.temporal[name]:
                        values = pair.temporal[name].transform(video['id'],values,frame['bbox'],frame['seconds'])
                    videos[name][vi]['frames'][fi]['probabilities'] = values.tolist()
            if (vi+1)%20==0:
                print(json.dumps({'videos':vi+1,'total':len(manifest['videos'])}),flush=True)
        evaluated = {}
        for name,clips in videos.items():
            for video in clips:
                eligible = [f for f in video['frames'] if video['known_label_window'][0]<=f['fraction']<=video['known_label_window'][1]]
                detected = [f for f in eligible if 'probabilities' in f]
                video['aggregate_prediction'] = CLASSES[int(np.mean([f['probabilities'] for f in detected],axis=0).argmax())] if detected else None
            measured = confusion_report([v['label'] for v in clips],[v['aggregate_prediction'] for v in clips])
            raw = {'model':name,'decision_sha256':fingerprint,'target_sampling_fps':10,'crop_format':'png',
                   'videos':clips,'metrics':{'eNTERFACE-holdout':measured},
                   'scope':'Final fixed system, six heldback actors; controlledspeech cliplabels are not per-frame facialtruth.'}
            path = args.output/(name+'-videos.json')
            path.write_text(json.dumps(raw,indent=2),encoding='utf-8')
            display = path.with_stem(path.stem+'-display')
            subprocess.run([shutil.which('node'),str(ROOT/'scripts/evaluate_video_smoothing.cjs'),str(path),str(display)],
                           check=True,stdout=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            shown = json.loads(display.read_text())
            measured['stable_correct_half_second_clip_rate'] = sum(v['stableCorrectAtLeastHalfSecond'] for v in shown['clips'])/len(clips)
            measured['display_trace'] = str(display)
            for label in CLASSES:
                subset = [v for v in shown['clips'] if v['label']==label]
                measured['per_class'][label]['stable_correct_half_second_clip_rate'] = sum(v['stableCorrectAtLeastHalfSecond'] for v in subset)/len(subset) if subset else None
            evaluated[name] = measured
    report = {'decision':str(args.decision),'decision_sha256':fingerprint,'mode':args.mode,
              'completed_at':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-started,
              'systems':decision,'results':evaluated,
              'scope':'Previously reserved final split. Evaluate only the frozen selected system and original baseline. No tuning after results. Photo recall and clip/continuous-display measures are separate; not a guarantee for a new webcam or person.'}
    final_file.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in result.items() if k not in ('records','confusion','confusion_with_no_face_column','per_class')}
                      for name,result in evaluated.items()}),flush=True)


if __name__=='__main__':
    main()
