"""Paired-seed HOOD comparisons with common evaluator and objective warnings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


LOWER_IS_BETTER = ('mesh_', 'principal_strain_rms', 'principal_extreme_strain_fraction',
                   'pinned_error', 'frame_inference_time', 'model_forward_time',
                   'inference_frames_total', 'evaluation_cuda_peak', 'rollout_cuda_peak')


def metadata(run):
    return json.loads((Path(run) / 'metadata.json').read_text())


def load_csv(path):
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def history(run, name, visited=None):
    run = Path(run).resolve()
    visited = set() if visited is None else visited
    if run in visited:
        raise ValueError('Cycle in resume lineage')
    visited.add(run)
    info = metadata(run)
    current = load_csv(run / 'logs' / (name + '.csv'))
    parent, cutoff = None, None
    if info.get('resume'):
        parent, cutoff = Path(info['resume']['parent_run']), info['resume']['from_step']
    elif name != 'validation' and current.empty and info.get('evaluated_checkpoint'):
        parent = Path(info['evaluated_checkpoint']['path']).parent.parent
        cutoff = info['global_step']
    if parent is not None:
        earlier = history(parent, name, visited)
        if not earlier.empty:
            earlier = earlier[earlier.global_step <= cutoff]
            if name == 'validation':
                a = json.loads((parent / 'validation_manifest.json').read_text()).get('evaluation_signature')
                b = json.loads((run / 'validation_manifest.json').read_text()).get('evaluation_signature')
                if a != b:
                    raise ValueError('Resume lineage has incompatible evaluation definitions; re-evaluate checkpoints')
            current = pd.concat([earlier, current], ignore_index=True)
    return current


def training(run):
    frame, substeps = history(run, 'train'), history(run, 'train_substeps')
    if frame.empty:
        return frame
    if not substeps.empty:
        if 'optimizer_step_count' not in substeps:
            substeps['optimizer_step_count'] = substeps.optimized.fillna(0).astype(int).cumsum()
        groups = substeps.groupby('global_step', sort=True)
        extra = pd.DataFrame({'optimizer_step_count': groups.optimizer_step_count.max(),
                              'optimizer_updates_in_batch': groups.optimized.sum()})
        for key in ('lr_used', 'lr_next', 'grad_norm_before_clip', 'collision_weight', 'collision_raw_loss'):
            if key in substeps:
                extra[key + '_mean'] = groups[key].mean()
                extra[key + '_last'] = groups[key].last()
        for key in extra:
            if key not in frame:
                frame[key] = frame.global_step.map(extra[key])
    return frame.sort_values('global_step').drop_duplicates('global_step', keep='last')


def controls(run):
    config = OmegaConf.to_container(OmegaConf.load(Path(run) / 'config.yaml'), resolve=True)
    result = {key: config[key] for key in ('model', 'dataloader', 'runner')}
    result['runner']['postcvpr'].pop('device', None)
    result['budget'] = {key: config['experiment'].get(key) for key in ('max_iter', 'n_epochs')}
    source = Path(run)
    info = metadata(source)
    if info.get('evaluated_checkpoint'):
        source = Path(info['evaluated_checkpoint']['path']).parent.parent
    visited = set()
    while metadata(source).get('resume'):
        if source in visited:
            raise ValueError('Cycle in resume provenance')
        visited.add(source)
        source = Path(metadata(source)['resume']['parent_run'])
    if (source / 'assets.json').is_file():
        assets = json.loads((source / 'assets.json').read_text())
        result['asset_hashes'] = {key: value['sha256'] for key, value in assets.items()}
    if (source / 'training_data_manifest.json').is_file():
        record = json.loads((source / 'training_data_manifest.json').read_text())
        result['training_sequence_hashes'] = [(item['id'], item['sha256']) for item in record['sequences']]
    info = metadata(source)
    result['initialization'] = info.get('initialization_checkpoint', {}).get('sha256', 'random_seed_initialization')
    return result


def pair_runs(baseline, modified):
    def by_seed(paths):
        result = {}
        for path in paths:
            seed = metadata(path)['seed']
            if seed in result:
                raise ValueError(f'Duplicate seed {seed}; pass only the final resume segment for each seed')
            result[seed] = Path(path).resolve()
        return result
    left, right = by_seed(baseline), by_seed(modified)
    if left.keys() != right.keys():
        raise ValueError('Baseline and modified must have the same seed set')
    return [(seed, left[seed], right[seed]) for seed in sorted(left)]


def common_curves(series, column, axis):
    arrays = []
    for data in series:
        if column not in data or axis not in data:
            return None
        subset = data[[axis, column]].dropna()
        if axis == 'optimizer_step_count':
            subset = subset[subset[axis] > 0]
        arrays.append(subset.drop_duplicates(axis, keep='last').set_index(axis)[column])
    joined = pd.concat(arrays, axis=1, join='inner').dropna()
    return joined if not joined.empty else None


def plot_curves(base, modified, columns, axis, destination, objective=False):
    destination.mkdir(parents=True, exist_ok=True)
    for column in columns:
        b, m = common_curves(base, column, axis), common_curves(modified, column, axis)
        if b is None or m is None:
            continue
        indices = b.index.intersection(m.index)
        if len(indices) == 0:
            continue
        b, m = b.loc[indices], m.loc[indices]
        report = pd.DataFrame({'axis': indices, 'baseline_mean': b.mean(axis=1).values,
                               'baseline_std': b.std(axis=1, ddof=1).fillna(0).values,
                               'modified_mean': m.mean(axis=1).values,
                               'modified_std': m.std(axis=1, ddof=1).fillna(0).values})
        report.to_csv(destination / (column + '.csv'), index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        for prefix, color in [('baseline', '#2864ad'), ('modified', '#d46624')]:
            mean, std = report[prefix + '_mean'].to_numpy(), report[prefix + '_std'].to_numpy()
            ax.plot(indices, mean, color=color, label=prefix)
            ax.fill_between(indices, mean - std, mean + std, color=color, alpha=.15)
        ax.set_xlabel(axis); ax.set_ylabel(column); ax.grid(alpha=.2); ax.legend()
        ax.set_title(column + ('\nTraining objectives can differ; curves are diagnostics' if objective else ''))
        fig.tight_layout(); fig.savefig(destination / (column + '.png'), dpi=140); plt.close(fig)


def compare(baseline, modified, output, axis='global_step', exploratory=False):
    pairs = pair_runs(baseline, modified)
    output = Path(output).resolve()
    checks, signatures, tables, base_train, mod_train = [], set(), [], [], []
    for seed, b, m in pairs:
        same_controls = controls(b) == controls(m)
        if not same_controls and not exploratory:
            raise ValueError(f'Non-objective hyperparameters/budgets differ for seed {seed}; use --exploratory for diagnostics')
        for run in (b, m):
            manifest = json.loads((run / 'validation_manifest.json').read_text())
            signature = manifest.get('evaluation_signature')
            if not signature:
                raise ValueError('Historical validation uses incompatible metrics. Run validate_experiment.py on both checkpoints first.')
            signatures.add(signature)
        bv, mv = history(b, 'validation'), history(m, 'validation')
        if bv.empty or mv.empty:
            raise ValueError('Both experiments need validation records')
        key = ['global_step', 'sequence', 'garment']
        joined = bv.merge(mv, on=key, suffixes=('_baseline', '_modified'), validate='one_to_one')
        if joined.empty:
            if not exploratory:
                raise ValueError('No equal-step validation checkpoints; compare matched budgets')
            # Explicitly label unequal budgets; never call this a formal comparison.
            bv, mv = bv[bv.global_step == bv.global_step.max()], mv[mv.global_step == mv.global_step.max()]
            joined = bv.merge(mv, on=['sequence', 'garment'], suffixes=('_baseline', '_modified'), validate='one_to_one')
            joined['global_step'] = -1
        for _, row in joined.iterrows():
            if row.frames_baseline != row.frames_modified:
                raise ValueError('Validation lengths differ')
            for column in bv.select_dtypes(include=np.number).columns:
                if column in ('global_step', 'optimizer_step_count', 'frames') or column not in mv:
                    continue
                a, z = float(row[column + '_baseline']), float(row[column + '_modified'])
                change = z - a
                # Signed clearance and gravity/acceleration are not unconditionally
                # lower-is-better quality measures. Never invent improvement there.
                lower = column.startswith(LOWER_IS_BETTER) and 'clearance' not in column
                relative = 100 * change / abs(a) if abs(a) > 1e-12 else np.nan
                tables.append({'seed': seed, 'global_step': int(row.global_step),
                               'sequence': row.sequence, 'garment': row.garment, 'metric': column,
                               'baseline': a, 'modified': z, 'absolute_change': change,
                               'relative_change_percent': relative,
                               'relative_improvement_percent': -relative if lower else np.nan,
                               'direction': 'lower' if lower else 'descriptive',
                               'matched_budget': int(row.global_step >= 0)})
        bt, mt = training(b), training(m)
        base_train.append(bt); mod_train.append(mt)
        checks.append({'seed': seed, 'baseline': str(b), 'modified': str(m), 'same_controls': same_controls,
                       'baseline_status': metadata(b)['status'], 'modified_status': metadata(m)['status']})
    if len(signatures) != 1:
        raise ValueError('Evaluator/data/material/initial-condition signatures differ; use one common protocol')
    # All checks precede creation. Never overwrite a previous comparison.
    output.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(tables)
    metrics.to_csv(output / 'metrics_comparison.csv', index=False)
    # Aggregate each seed across sequences first; do not pretend frames are independent seeds.
    paired = metrics.groupby(['global_step', 'metric', 'seed'])[['baseline', 'modified', 'absolute_change']].mean()
    aggregate = paired.groupby(['global_step', 'metric']).agg(['mean', 'std', 'count'])
    aggregate.to_csv(output / 'paired_seed_statistics.csv')
    train_cols = set.intersection(*[set(frame.columns) for frame in base_train + mod_train])
    losses = sorted(c for c in train_cols if c.endswith(('_loss_mean', '_loss_last')) and not c.startswith('total_loss_'))
    monitor = sorted(c for c in train_cols if c.startswith(('lr_', 'grad_norm_', 'collision_weight', 'step_time_ms', 'collision_raw_loss')))
    plot_curves(base_train, mod_train, losses, axis, output / 'training_curves', objective=True)
    plot_curves(base_train, mod_train, monitor, axis, output / 'training_state')
    # Common evaluation trends across matched checkpoints, averaged per seed over sequences.
    evaluation_base, evaluation_modified = [], []
    for seed, _, _ in pairs:
        own = metrics[(metrics.seed == seed) & (metrics.global_step >= 0)]
        for label, collection in [('baseline', evaluation_base), ('modified', evaluation_modified)]:
            frame = own.pivot_table(index='global_step', columns='metric', values=label, aggfunc='mean').reset_index()
            collection.append(frame)
    if evaluation_base and not evaluation_base[0].empty:
        columns = [key for key in evaluation_base[0] if key != 'global_step']
        plot_curves(evaluation_base, evaluation_modified, columns, 'global_step', output / 'validation_comparison')
    # Frame trajectories at the latest common checkpoint for every seed/sequence.
    for seed, b, m in pairs:
        bsteps = {p.name for p in (b / 'validation').glob('step_*')}
        msteps = {p.name for p in (m / 'validation').glob('step_*')}
        common = bsteps & msteps
        if not common:
            continue
        step = sorted(common)[-1]
        for path in sorted((b / 'validation' / step).glob('*_frames.csv')):
            other = m / 'validation' / step / path.name
            if not other.is_file():
                raise ValueError('Missing matching per-frame sequence')
            x, y = pd.read_csv(path), pd.read_csv(other)
            columns = [key for key in x if key.startswith(('mesh_vertex_penetration', 'mesh_penetrating_depth', 'mesh_penetration_area'))]
            plot_curves([x], [y], columns, 'frame', output / 'per_frame' / f'seed{seed}' / path.stem)
    write = {'pairs': checks, 'evaluation_signature': next(iter(signatures)), 'axis': axis,
             'exploratory': exploratory, 'paired_seeds': len(pairs),
             'notes': ['No performance claim from unequal objectives or short smoke runs.',
                       'Shading is sample SD across seeds, not a confidence interval.',
                       'Relative percentages are undefined when the baseline is zero.',
                       'Inference timing excludes metrics; evaluation_wall_time includes metrics and CPU transfer.',
                       'Resumed training history is clipped at the parent checkpoint before concatenation.']}
    (output / 'comparison.json').write_text(json.dumps(write, indent=2) + '\n')
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs='+', type=Path, required=True)
    parser.add_argument('--modified', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axis', choices=['global_step', 'optimizer_step_count'], default='global_step')
    parser.add_argument('--exploratory', action='store_true')
    args = parser.parse_args()
    compare(args.baseline, args.modified, args.output, args.axis, args.exploratory)
