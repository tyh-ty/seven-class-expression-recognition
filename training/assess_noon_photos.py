"""Apply the predeclared noon photo gates relative to the actually released model."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,default=Path('training/noon-20260928/paired-photos.json'))
    p.add_argument('--plan',type=Path,default=Path('training/noon-20260928/sam-paired-plan.json'))
    p.add_argument('--output',type=Path,default=Path('training/noon-20260928/paired-photo-assessment.json'))
    args=p.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'));r=json.loads(args.report.read_text(encoding='utf-8'))
    if r['inference_precision']!='float32':raise ValueError('BF16 training proxy cannot qualify a released FP32 model')
    reports={str(Path(k).resolve()):v for k,v in r['reports'].items()}
    baseline=reports[str(Path(plan['baseline']).resolve())];gates=plan['deployment_gate']
    results=[]
    for key,c in reports.items():
        if key==str(Path(plan['baseline']).resolve()):continue
        if set(c)!=set(baseline):raise ValueError('Candidate dataset mismatch')
        regressions=[]
        for domain,first in baseline.items():
            second=c[domain]
            if first['examples']!=second['examples']:raise ValueError('Different dataset sizes')
            # Match ordered files/labels to prevent a smaller/easier cohort appearing to improve.
            a=[(v['file'],v['label']) for v in first['records']]
            b=[(v['file'],v['label']) for v in second['records']]
            if a!=b:raise ValueError('Different samples or labels')
            for label,old in first['per_class'].items():
                new=second['per_class'][label]
                if old['support']!=new['support']:raise ValueError('Support mismatch')
                if old['support'] and new['recall']-old['recall'] < -gates['per_class_recall_max_drop']-1e-6:
                    regressions.append({'domain':domain,'class':label,'support':old['support'],
                                        'old_recall':old['recall'],'new_recall':new['recall']})
        checks={'clean_fer_floor':c['ferplus_clean']['macro_f1']+1e-6>=gates['clean_fer_macro_f1_min'],
                'expw_gain':c['expw_original']['macro_f1']-baseline['expw_original']['macro_f1']+1e-6>=gates['expw_macro_f1_improvement_min'],
                'expanded_real_eyewear_no_drop':c['real_eyeglasses_expanded']['macro_f1']+1e-6>=baseline['real_eyeglasses_expanded']['macro_f1'],
                'strict_real_eyewear_no_drop':c['real_eyeglasses_strict']['macro_f1']+1e-6>=baseline['real_eyeglasses_strict']['macro_f1'],
                'no_perclass_recall_regression_gt5pp':not regressions}
        results.append({'checkpoint':key,'checks':checks,'photo_pass':all(checks.values()),
                        'regressions':regressions,'deployed':False,
                        'summary':{name:{'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for name,v in c.items()},
                        'pending_if_pass':['538 clip replay and real-eyewear/anger/fear stability','actual CPU ONNX parity and speed','browser integration']})
    result={'scope':'Repeated selection validation; old final2610/180 not read or reused.',
            'baseline':plan['baseline'],'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            'report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),'results':results}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{'checkpoint':r['checkpoint'],'checks':r['checks'],'photo_pass':r['photo_pass']} for r in results]),flush=True)


if __name__=='__main__':main()
