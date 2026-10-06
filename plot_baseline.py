"""Plot recorded physical losses and comparable fixed-sequence validation curves."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    out = args.run_dir / 'plots'
    out.mkdir(exist_ok=True)
    for filename, xkey, suffix in [('train.csv', 'global_step', '_mean'), ('validation.csv', 'global_step', '_regular_mean')]:
        source = args.run_dir / 'logs' / filename
        if not source.exists() or not source.stat().st_size:
            continue
        with source.open() as stream:
            rows = list(csv.DictReader(stream))
        keys = [k for k in rows[0] if k.endswith(suffix) and ('loss' in k or 'penetration' in k or 'strain' in k)]
        if not keys:
            continue
        fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.6 * len(keys)), squeeze=False)
        labels = sorted({r.get('sequence', 'training') for r in rows})
        for axis, key in zip(axes[:, 0], keys):
            for label in labels:
                selected = [r for r in rows if r.get('sequence', 'training') == label]
                axis.plot([int(r[xkey]) for r in selected], [float(r[key]) for r in selected], label=label)
            axis.set_ylabel(key)
            axis.set_yscale('symlog', linthresh=1e-5)  # Gravity can legitimately be negative.
            axis.grid(alpha=0.25)
            if len(labels) > 1:
                axis.legend(fontsize=7)
        axes[-1, 0].set_xlabel('global training step')
        fig.tight_layout()
        fig.savefig(out / (Path(filename).stem + '.png'), dpi=140)
        plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
