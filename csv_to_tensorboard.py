"""Convert a recorded run's CSV files without changing its original records."""
import argparse
import csv
import itertools
import json
from pathlib import Path

from utils.tensorboard_logging import TrainingTensorBoard


def ordered_rows(path):
    previous = -1
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            step = int(row['global_step'])
            if step < previous:
                raise ValueError(f'Non-monotone global_step in {path}')
            previous = step
            yield row


def convert(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    paths = {name: run / 'logs' / (name + '.csv')
             for name in ('train', 'train_substeps', 'validation')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # Refuse append/overwrite: repeated imports must not duplicate scalar steps.
    destination.mkdir(parents=True, exist_ok=False)
    writer = TrainingTensorBoard(destination)
    counts = {}
    try:
        counts['train'] = 0
        for row in ordered_rows(paths['train']):
            writer.training(row)
            counts['train'] += 1
        for name, callback in [('train_substeps', writer.optimizer), ('validation', writer.validation)]:
            counts[name] = 0
            for _, group in itertools.groupby(ordered_rows(paths[name]), lambda r: int(r['global_step'])):
                rows = list(group)
                callback(rows)
                counts[name] += len(rows)
    finally:
        writer.close()
    # The source commit belongs to the original experiment, not the converter.
    metadata = json.loads((run / 'metadata.json').read_text())
    import hashlib
    manifest = {'source_run': str(run), 'source_commit': metadata['git_commit'],
                'counts': counts,
                'csv_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in paths.items()},
                'note': 'Scalar steps preserved. Event wall times are conversion time, not training time.'}
    (destination / 'csv_import.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.run, args.output or args.run / 'tensorboard'), indent=2))
