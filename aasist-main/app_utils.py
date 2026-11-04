"""Small shared helpers for manifests, reports, and reproducibility."""
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_manifest(path):
    path = Path(path).resolve()
    rows, seen = [], set()
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {'path', 'label'} <= set(reader.fieldnames):
            raise ValueError('Manifest requires path,label columns; optional: group,type.')
        for number, row in enumerate(reader, 2):
            label = (row.get('label') or '').strip().lower()
            name = (row.get('path') or '').strip()
            if label not in ('bonafide', 'spoof') or not name:
                raise ValueError('Invalid manifest row {}: labels must be bonafide or spoof.'.format(number))
            audio = Path(name)
            if not audio.is_absolute():
                audio = path.parent / audio
            audio = audio.resolve()
            identity = str(audio).casefold()
            if identity in seen:
                raise ValueError('Duplicate audio path in manifest: ' + str(audio))
            seen.add(identity)
            rows.append({'path': str(audio), 'label': label,
                         'group': (row.get('group') or '').strip(),
                         'type': (row.get('type') or '').strip()})
    if not rows or {row['label'] for row in rows} != {'bonafide', 'spoof'}:
        raise ValueError('Manifest must contain both bonafide and spoof audio.')
    return rows


def write_predictions_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['file', 'label', 'group', 'type', 'status', 'prediction', 'raw_score',
              'threshold', 'duration_seconds', 'segment_count', 'elapsed_seconds', 'error_code', 'error_message']
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            row = dict(record)
            row['duration_seconds'] = record.get('audio', {}).get('duration_seconds')
            row['error_code'] = record.get('error', {}).get('code')
            row['error_message'] = record.get('error', {}).get('message')
            writer.writerow(row)
