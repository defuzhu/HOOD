"""One bounded C run, then matched A evaluation and comparison; never auto-restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data = Path(os.environ['HOOD_DATA'])
    reference = data / 'experiments/modified_contact_relaxed_barrier_seed57_20261011T084658Z_fc628647/checkpoints/step_0000060000.pth'
    config = 'friction/lagged_coulomb_extended_seed57'
    cfg = OmegaConf.load(project / ('configs/' + config + '.yaml'))
    contact = cfg.criterions['postcvpr.collision_penalty']
    friction = cfg.criterions['postcvpr.friction_energy']
    if (contact.contact_geometry, contact.contact_potential, friction.mode) != (
            'signed_mesh', 'area_relaxed_barrier', 'lagged_coulomb'):
        raise RuntimeError('Wrong physical models for C-group launch')
    if not reference.is_file() or not cfg.recording.tensorboard or not cfg.recording.require_clean:
        raise RuntimeError('Missing A checkpoint or disabled provenance/TensorBoard')
    train = [sys.executable, '-u', str(project/'train_baseline.py'), 'config='+config,
             'seed=57', 'experiment.max_iter=60000']
    evaluate = [sys.executable, '-u', str(project/'validate_experiment.py'), 'checkpoint='+str(reference),
                'protocol='+str(project/'configs/evaluation/friction_extended.yaml'),
                'experiment.name=friction_reference_A_extended']
    if args.dry_run:
        print(json.dumps({'train':train, 'reference_evaluation':evaluate,
                          'comparison':'A reference vs C, matching 60000 steps'}, indent=2))
        return
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    workflow = data/'experiments'/'workflows'/('friction_c_'+identity)
    workflow.mkdir(parents=True, exist_ok=False)
    record = {'status':'running', 'stage':'training', 'budget_batches':60000,
              'project':str(project), 'train_command':train, 'reference_command':evaluate}
    def save():
        temporary = workflow/'status.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(workflow/'status.json')
    def execute(command, key):
        found = None
        with subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                if line.startswith('RUN_DIR='):
                    found = Path(line.strip().split('=',1)[1])
                    record[key] = str(found); save()
            if process.wait():
                raise RuntimeError(f'{key} process failed with exit code {process.returncode}')
        if found is None:
            raise RuntimeError(f'Missing {key} RUN_DIR')
        meta = json.loads((found/'metadata.json').read_text())
        if meta['status'] != 'completed' or meta['global_step'] != 60000:
            raise RuntimeError(f'{key} did not complete its expected 60000-step endpoint')
        return found
    save()
    print('WORKFLOW_DIR='+str(workflow), flush=True)
    try:
        current = execute(train, 'training_run')
        record['stage']='reference_evaluation'; save()
        old = execute(evaluate, 'reference_run')
        record['stage']='comparison'; save()
        output = data/'experiments/comparisons'/('friction_A_vs_C_'+identity)
        subprocess.run([sys.executable, str(project/'compare_experiments.py'),
                        '--baseline',str(old),'--modified',str(current),'--output',str(output)],
                       cwd=project, check=True)
        record.update(status='completed', stage='completed', comparison=str(output)); save()
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}'); save()
        raise


if __name__ == '__main__':
    main()
