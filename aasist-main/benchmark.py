"""Evaluate a frozen threshold on independent labelled audio."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
from app_runtime import AASISTDetector
from app_metrics import classification_metrics, select_threshold
from app_utils import read_manifest, write_json, write_predictions_csv, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Independent test CSV: path,label,group,type.')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config'/'inference.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() != '.json':
        parser.error('--output must be a .json path.')
    rows = read_manifest(args.manifest)
    detector = AASISTDetector(args.config, require_profile=True, device=args.device)
    groups = {row['group'] for row in rows if row['group']}
    if groups & set(detector.profile.get('validation_groups', [])):
        raise ValueError('Test groups overlap threshold-validation groups; use independent recordings/speakers.')
    if file_sha256(args.manifest) == detector.profile.get('validation_manifest_sha256'):
        raise ValueError('Cannot benchmark on the calibration manifest.')
    validation_hashes = set(detector.profile.get('validation_audio_sha256', []))
    for row in rows:
        path = Path(row['path'])
        if path.is_file() and file_sha256(path) in validation_hashes:
            raise ValueError('Test audio duplicates threshold-validation audio: ' + str(path))
    records = detector.score_manifest(rows)
    metrics = classification_metrics(records)
    by_type = {kind: classification_metrics([r for r in records if r['type'] == kind])
               for kind in sorted({r['type'] for r in records if r['type']})}
    valid = [r for r in records if r['status'] == 'ok']
    # EER is supplementary; its oracle threshold is never used for test decisions.
    eer = None
    if {r['label'] for r in valid} == {'bonafide', 'spoof'}:
        eer = select_threshold([r['raw_score'] for r in valid], [r['label'] for r in valid], 'eer')['validation_eer']
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'pipeline_id': detector.pipeline_id, 'threshold': detector.threshold,
        'threshold_selection': detector.profile['selection']['method'],
        'manifest_sha256': file_sha256(args.manifest), 'positive_class': 'spoof',
        'class_counts': dict(Counter(row['label'] for row in rows)),
        'metrics': metrics, 'by_type': by_type, 'supplementary_eer_on_scored': eer,
        'reference_targets': {'accuracy_on_all_at_least_0.90': metrics['accuracy_on_all'] >= .9,
                              'spoof_recall_on_all_at_least_0.90': metrics['spoof_recall_on_all'] >= .9},
        'metric_notes': ['Fractions, not percentages. Failures count as incorrect in accuracy_on_all.',
                        'Confusion matrix contains classified files; failures are listed separately.',
                        'Reference targets do not establish general performance beyond this test set.',
                        'EER uses test scores only as a descriptive metric, not to select the deployment threshold.'],
        'records': records}
    write_json(args.output, report)
    write_predictions_csv(args.output.with_suffix('.csv'), records)
    errors = [r for r in records if r['status'] != 'ok' or r['prediction'] != r['label']]
    write_predictions_csv(args.output.with_name(args.output.stem + '.errors.csv'), errors)
    print('Accuracy (all inputs): {:.2%}'.format(metrics['accuracy_on_all']))
    print('Spoof recall (all spoof inputs): {:.2%}'.format(metrics['spoof_recall_on_all']))
    print('Coverage: {:.2%}; failed files: {}'.format(metrics['coverage'], metrics['failed']))
    print('Report:', args.output.resolve())
    return 1 if metrics['failed'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
