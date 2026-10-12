"""Apply prospective video checks without treating them as deployment permission."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('training/runs/temporal-residual-v1'))
    p.add_argument('--videos',type=Path,default=Path('training/overnight-20260927/video-temporal-residual-v1'))
    p.add_argument('--plan',type=Path,default=Path('training/overnight-20260927/temporal-residual-plan.json'))
    p.add_argument('--baseline-name', default='baseline-int8')
    p.add_argument('--opened-historical-holdout',action='store_true',
                   help='Historical final tests are already seen; do not describe them as untouched.')
    args=p.parse_args()
    plan=json.loads(args.plan.read_text())['prospective_acceptance']
    summary=json.loads((args.videos/'summary.json').read_text())
    rav={r['model']:r for r in json.loads((args.run/'ravdess-intensity.json').read_text())['reports'].values()}
    ent={r['model']:r for r in json.loads((args.run/'enterface-eyewear.json').read_text())['reports'].values()}
    base=args.baseline_name
    evaluated=[]
    for name,row in summary.items():
        if name==base:continue
        checks={}
        regressions=[]
        for source in ('RAVDESS-val','eNTERFACE-val'):
            first=summary[base]['metrics'][source];second=row['metrics'][source]
            checks[source+'_macro_f1_gain']=second['macro_f1_present']-first['macro_f1_present']+1e-9>=plan['both_RAV_and_enterface_clip_macro_f1_gain_min']
            for emotion,values in second['per_class'].items():
                difference=values['recall']-first['per_class'][emotion]['recall']
                if values['support']>=20 and difference < -plan['per_class_drop_max_if_support20']-1e-9:
                    regressions.append({'source':source,'class':emotion,'recall_change':difference})
        for label,first,second in [('RAVDESS',rav[base]['all'],rav[name]['all']),
                                   ('eNTERFACE',ent[base]['groups']['all'],ent[name]['groups']['all'])]:
            checks[label+'_halfsecond_gain']=second['stable_correct_half_second_clip_rate']-first['stable_correct_half_second_clip_rate']+1e-9>=plan['both_RAV_and_enterface_halfsecond_stability_gain_min']
        checks['no_major_class_regression']=not regressions
        for emotion in ('anger','fear'):
            old=rav[base]['normal_intensity']['per_class'][emotion]
            new=rav[name]['normal_intensity']['per_class'][emotion]
            delta=(new['stable_correct_half_second_clip_rate']-old['stable_correct_half_second_clip_rate'])*new['support_clips']
            required=plan['RAV_normal_anger_halfsecond_gain_clips_min'] if emotion=='anger' else 0
            checks['normal_'+emotion+'_stable_clips']=delta+1e-8>=required
        for actor in ent[base]['eyewear_subjects']:
            old=ent[base]['per_subject'][str(actor)]['per_class']['fear']
            new=ent[name]['per_subject'][str(actor)]['per_class']['fear']
            aggregate=(new['clip_aggregate_recall']-old['clip_aggregate_recall'])*new['support_clips']
            stable=(new['stable_correct_half_second_clip_rate']-old['stable_correct_half_second_clip_rate'])*new['support_clips']
            checks[f'eyewear_subject_{actor}_fear_aggregate']=aggregate+1e-8>=plan['each_heldback_eyewear_subject_fear_correct_clips_gain_min']
            checks[f'eyewear_subject_{actor}_fear_stable']=stable+1e-8>=plan['each_heldback_eyewear_subject_fear_stable_halfsecond_gain_clips_min']
            if plan.get('each_heldback_eyewear_subject_anger_no_drop', False):
                old_anger=ent[base]['per_subject'][str(actor)]['per_class']['anger']
                new_anger=ent[name]['per_subject'][str(actor)]['per_class']['anger']
                checks[f'eyewear_subject_{actor}_anger_aggregate']=new_anger['clip_aggregate_recall']+1e-8>=old_anger['clip_aggregate_recall']
                checks[f'eyewear_subject_{actor}_anger_stable']=new_anger['stable_correct_half_second_clip_rate']+1e-8>=old_anger['stable_correct_half_second_clip_rate']
        evaluated.append({'model':name,'epoch':row['metadata']['epoch'],'checks':checks,'preliminary_video_pass':all(checks.values()),
                          'regressions':regressions,'deployment_accepted':False,
                          'pending_if_preliminary_pass':['Exact frozen CAER visible-face agreement','Static photo probability parity','Runtime latency','Browser integration']})
    holdout_note='Historicalfinal2610/180alreadyseen; excludedfromthisselectionexperiment.' if args.opened_historical_holdout else 'Finalholdouts untouched.'
    result={'scope':'Fixed prospective video checks. Controlledspeech/sourcecliplabels and twoeyewearactors do not prove natural7class or microexpression accuracy. '+holdout_note,
            'evaluated':evaluated}
    (args.run/'video-acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(evaluated),flush=True)


if __name__=='__main__':
    main()
