"""Build labelled development/evaluation CSVs from the local ASVspoof 2019 LA data."""
import argparse
import csv
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True, help='The LA folder.')
    parser.add_argument('--split', choices=('dev', 'eval'), required=True)
    parser.add_argument('--per-class', type=int, default=500, help='Sample size per class; 0 uses all files.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.per_class < 0:
        parser.error('--per-class must be nonnegative.')
    root = args.dataset_root.resolve()
    protocol = root/'ASVspoof2019_LA_cm_protocols'/('ASVspoof2019.LA.cm.' + args.split + '.trl.txt')
    pools = {'bonafide': [], 'spoof': []}
    for line in protocol.read_text(encoding='utf-8').splitlines():
        speaker, key, _, attack, label = line.split()
        audio = root/('ASVspoof2019_LA_' + args.split)/'flac'/(key + '.flac')
        pools[label].append({'path': str(audio), 'label': label,
                             'group': 'ASVspoof2019:' + speaker,
                             'type': 'bonafide' if label == 'bonafide' else attack})
    rng = random.Random(args.seed)
    selected = []
    for label, pool in pools.items():
        if args.per_class > len(pool):
            parser.error('Not enough {} files for requested sample.'.format(label))
        selected.extend(rng.sample(pool, args.per_class) if args.per_class else pool)
    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=('path', 'label', 'group', 'type'))
        writer.writeheader()
        writer.writerows(selected)
    print('Wrote {} {} files to {}'.format(len(selected), args.split, args.output.resolve()))


if __name__ == '__main__':
    main()
