"""Choose a threshold using a labelled validation manifest; never train weights."""
import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
from app_runtime import AASISTDetector
from app_metrics import select_threshold, classification_metrics
from app_utils import read_manifest, write_json, file_sha256, write_predictions_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Validation CSV: path,label,group,type.')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config'/'inference.json')
    parser.add_argument('--output', type=Path, help='Default: threshold_profile from config. Must match that path for inference.')
    parser.add_argument('--method', choices=('balanced_accuracy', 'eer'), default='balanced_accuracy')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() != '.json':
        parser.error('--output must be a .json path.')
    rows = read_manifest(args.manifest)
    detector = AASISTDetector(args.config, require_profile=False, device=args.device)
    output = args.output or detector.profile_path
    records = detector.score_manifest(rows)
    write_predictions_csv(output.with_suffix('.validation.csv'), records)
    failures = [r for r in records if r['status'] == 'error']
    if failures:
        raise ValueError('{} validation files failed. Fix them before calibrating; no new threshold was saved. See validation CSV.'.format(len(failures)))
    selection = select_threshold([r['raw_score'] for r in records], [r['label'] for r in records], args.method)
    threshold = selection['threshold']
    for record in records:
        record.update(status='ok', threshold=threshold,
                      prediction='bonafide' if record['raw_score'] > threshold else 'spoof')
    profile = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'pipeline_id': detector.pipeline_id, 'pipeline': detector.identity,
        'threshold': threshold, 'decision': 'bonafide iff raw_score > threshold; otherwise spoof',
        'selection': selection, 'validation_metrics': classification_metrics(records),
        'validation_manifest_sha256': file_sha256(args.manifest),
        'validation_sample_count': len(rows),
        'validation_groups': sorted({row['group'] for row in rows if row['group']}),
        'validation_audio_sha256': sorted({r['sha256'] for r in records}),
        'warning': 'Validation metrics are for threshold selection, not independent acceptance results.'}
    write_json(output, profile)
    write_predictions_csv(output.with_suffix('.validation.csv'), records)
    print('Threshold saved:', output.resolve())
    print('Threshold:', threshold, '| method:', args.method)
    print('Independent benchmark data must not overlap validation data.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
