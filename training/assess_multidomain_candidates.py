"""Apply the recorded validation rubric; this never deploys a model or reads holdout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('expw_original', 'real_eyeglasses_expanded', 'ravdess', 'cremad', 'kdef')
NUMERIC_EPSILON = 1e-9  # Inclusive policy boundaries, not a statistical tolerance.


def assess(reports, baseline_key, rule):
    baseline = reports[baseline_key]
    results = []
    for key, candidate in reports.items():
        mean_gain = sum(candidate[d]['macro_f1'] - baseline[d]['macro_f1'] for d in DOMAINS) / len(DOMAINS)
        af_gain = sum(candidate[d]['per_class'][c]['recall'] - baseline[d]['per_class'][c]['recall']
                      for d in DOMAINS for c in ('anger', 'fear')) / (len(DOMAINS) * 2)
        regressions = []
        for domain in DOMAINS:
            for emotion, values in candidate[domain]['per_class'].items():
                change = values['recall'] - baseline[domain]['per_class'][emotion]['recall']
                if values['support'] >= 20 and change < -rule['max_single_class_recall_drop_on_domain_with_at_least20_examples'] - NUMERIC_EPSILON:
                    regressions.append({'domain': domain, 'class': emotion,
                                        'support': values['support'], 'recall_change': change})
        checks = {
            'clean_fer_f1': candidate['ferplus_clean']['macro_f1'] - baseline['ferplus_clean']['macro_f1'] + NUMERIC_EPSILON >= -rule['ferplus_macro_f1_drop_max'],
            'mean_domain_gain': mean_gain + NUMERIC_EPSILON >= rule['mean_macro_f1_gain_on_expw_glasses_ravdess_cremad_kdef_min'],
            'expw_gain': candidate['expw_original']['macro_f1'] - baseline['expw_original']['macro_f1'] + NUMERIC_EPSILON >= rule['expw_macro_f1_gain_min'],
            'expanded_glasses_no_f1_regression': candidate['real_eyeglasses_expanded']['macro_f1'] - baseline['real_eyeglasses_expanded']['macro_f1'] + NUMERIC_EPSILON >= -rule['real_glasses_macro_f1_drop_max'],
            'strict_glasses_no_f1_regression': candidate['real_eyeglasses_strict']['macro_f1'] - baseline['real_eyeglasses_strict']['macro_f1'] + NUMERIC_EPSILON >= -rule['real_glasses_macro_f1_drop_max'],
            'anger_fear_mean': af_gain + NUMERIC_EPSILON >= -rule['mean_anger_fear_recall_drop_max'],
            'single_class_drop': not regressions,
        }
        results.append({'checkpoint': key, 'checks': checks, 'photo_rule_pass': all(checks.values()),
                        'mean_domain_f1_gain': mean_gain, 'mean_anger_fear_recall_change': af_gain,
                        'major_per_class_regressions': regressions,
                        'domains': {d: {k: candidate[d][k] for k in ('examples', 'accuracy', 'macro_f1', 'per_class')}
                                    for d in ('ferplus_clean', *DOMAINS, 'real_eyeglasses_strict')}})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-key')
    args = parser.parse_args()
    policy = json.loads((ROOT / 'training/overnight-20260927/selection-policy.json').read_text(encoding='utf-8'))
    data = json.loads(args.report.read_text(encoding='utf-8'))
    reports = data['reports']
    baseline_key = args.baseline_key or data.get('baseline_key') or next(k for k in reports if Path(k).as_posix().replace('\\', '/').endswith('vit-small-seven-glasses-v1/best_target.pt'))
    evaluated = assess(reports, baseline_key, policy['same_architecture_candidate_rule'])
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'report': str(args.report),
              'baseline': baseline_key, 'heldout_read': False,
              'scope': 'Apply existing photo rubric only. Photo qualification is not product acceptance: video/per-class stability, quantization and interactive CPU latency still required. Validation is used repeatedly, not an unbiased final test.',
              'evaluated': evaluated}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps([{'checkpoint': r['checkpoint'], 'pass': r['photo_rule_pass'],
                       'failed': [k for k, v in r['checks'].items() if not v],
                       'fer_f1': r['domains']['ferplus_clean']['macro_f1'],
                       'expw_f1': r['domains']['expw_original']['macro_f1'],
                       'eyewear_f1': r['domains']['real_eyeglasses_expanded']['macro_f1'],
                       'eyewear_accuracy': r['domains']['real_eyeglasses_expanded']['accuracy']}
                      for r in evaluated]), flush=True)


if __name__ == '__main__':
    main()
