#!/usr/bin/env python3
"""Freeze the next leg-assisted stop-response candidate; never launch PPO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan
    from wheeled_tasks.chassis.transient_response_plan import make_transient_plan
    parent = ROOT / 'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'
    if digest(parent) != 'e55569a9b6bdc5f60cabfda232c12ac25cb3ec589de9b15fc9fddc398dc6d576':
        raise ValueError('The user-reviewed 200 Hz candidate changed')
    info = torch.load(parent, map_location='cpu', weights_only=True)['infos']
    base = read_plan(ROOT, info['plan_file'])
    if digest(ROOT / info['plan_file']) != info['plan_sha256'] or info['critic_warmup_remaining']:
        raise ValueError('Parent plan or remaining warmup differs')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = make_transient_plan(base, str(parent.relative_to(ROOT)), digest(parent),
                               info['plan_file'], info['plan_sha256'])
    plan_file = out / 'plan.json'
    plan_file.write_bytes(encode(plan))
    read_plan(ROOT, plan_file)
    subprocess.run([sys.executable, str(ROOT / 'scripts/train_flat_base.py'),
        '--plan', str(plan_file), '--output', str(out / 'preflight'),
        '--updates', '8', '--preflight-only'], cwd=ROOT, check=True)
    static = json.loads((out / 'preflight/progress.json').read_text())
    revision = dict(schema='flat-training-plan-revision-v1',
        parent_checkpoint_sha256=digest(parent), old_plan_file=info['plan_file'],
        old_plan_sha256=info['plan_sha256'], new_plan_sha256=digest(plan_file),
        old_source_sha256=info['source_sha256'], new_source_sha256=static['source_sha256'],
        changed_plan_fields=sorted(k for k in base.keys() | plan.keys() if base.get(k) != plan.get(k)))
    (out / 'revision.json').write_bytes(encode(revision))
    receipt = dict(status='prepared_no_optimizer_updates', new_optimizer_updates=0,
        training_started=False, native_scene_started=False, qualified=False,
        hardware_deployment_ready=False, parent_checkpoint=str(parent),
        parent_checkpoint_sha256=digest(parent), plan_sha256=digest(plan_file),
        source_sha256=static['source_sha256'], response_targets=plan['response_targets'],
        candidate_block=dict(optimizer_updates=8, critic_only_updates=2, actor_updates=6,
            automatic_launch=False),
        limitations=['CPU preparation is not behavior or mechanical validation',
            'gate and bound change the initial function; no identity claim',
            'encoder rolling is a slip-sensitive proxy; no friction or velocity oracle enters the actor',
            'pitch allowance does not raise traction or relax mechanical limits',
            'response/stop distance/slip require matched native traces before selecting a checkpoint'])
    (out / 'preparation.json').write_bytes(encode(receipt))
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=receipt['status'], new_optimizer_updates=0,
        reverse_target_m_s=2.5, transient_sample_fraction=.65)))


if __name__ == '__main__':
    main()
