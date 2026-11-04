"""Detect AI-generated speech in a file or folder; scores are not probabilities."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
from audio_io import SUPPORTED_EXTENSIONS
from app_runtime import AASISTDetector
from app_utils import write_json, write_predictions_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='An audio file or folder.')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config'/'inference.json')
    parser.add_argument('--output', required=True, type=Path, help='JSON report path; a CSV is written beside it.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--scores-only', action='store_true', help='No threshold or classification; output raw scores only.')
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() != '.json':
        parser.error('--output must be a .json path.')
    source = args.input.resolve()
    ignored = []
    if source.is_dir():
        discovered = sorted(p for p in (source.rglob('*') if args.recursive else source.iterdir()) if p.is_file())
        files = [p for p in discovered if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        ignored = [str(p) for p in discovered if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
        if not files:
            raise ValueError('No supported audio files found in folder.')
    else:
        files = [source]
    detector = AASISTDetector(args.config, require_profile=not args.scores_only, device=args.device)
    records = []
    for index, file in enumerate(files, 1):
        record = detector.score_file(file)
        records.append(record)
        if record['status'] == 'error':
            print('[{}/{}] ERROR {}: {}'.format(index, len(files), file.name, record['error']['message']), flush=True)
        else:
            verdict = record['prediction'] or 'unclassified'
            print('[{}/{}] {} score={:.6f} {}'.format(index, len(files), verdict, record['raw_score'], file.name), flush=True)
    failed = sum(r['status'] == 'error' for r in records)
    write_json(args.output, {'schema_version': 1, 'task': 'AI speech detection only',
        'score_direction': 'higher supports bonafide; score is not a probability',
        'scope_note': 'bonafide means AI forgery was not detected; it does not establish recording integrity.',
        'pipeline_id': detector.pipeline_id, 'device': detector.device,
        'threshold': detector.threshold, 'total': len(records), 'failed': failed,
        'ignored_non_audio_files': ignored, 'records': records})
    write_predictions_csv(args.output.with_suffix('.csv'), records)
    print('Report:', args.output.resolve(), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
