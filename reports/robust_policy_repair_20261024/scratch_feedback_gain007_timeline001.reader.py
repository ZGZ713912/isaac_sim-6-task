"""Read saved fresh policies on CPU; record local derivatives in RSL events."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from rsl_rl.utils.logger import Logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_scratch_feedback import inspect

REPORTS = Path(__file__).parent
OUTPUT = REPORTS / 'scratch_feedback_gain007_timeline001.json'
EVENTS = REPORTS / 'scratch_feedback_gain007_timeline001.events'
assert not OUTPUT.exists() and not EVENTS.exists()
torch.set_num_threads(2)
runs = [
    ('pilot007_contact_cost_zero', ('initial', '96', '192', '256', '384', '496', 'final')),
    ('pilot008_leg_target_rate020', ('96',)),
]
rows, events = [], {}
for name, models in runs:
    run = ROOT / 'runs/scratch_standing_20261026' / name
    cfg = json.loads((run / 'config/contract.json').read_text())
    agent = json.loads((run / 'agent_config.json').read_text())
    prior_path = ROOT / cfg['control_math_source']
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == cfg['control_math_sha256']
    wheel_kd = json.loads(prior_path.read_text())['actuators']['wheel']['kd']
    writer_path = EVENTS / name
    writer_path.mkdir(parents=True)
    logger = Logger(log_dir=str(writer_path), cfg=agent, env_cfg=cfg, num_envs=1,
                    is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device='cpu')
    logger.init_logging_writer()
    try:
        for model in models:
            row = inspect(run / f'model_{model}.pt')
            jac = torch.tensor(row['actor_input_jacobian'])
            kp, kd = cfg['v5_control']['leg_kp'], cfg['v5_control']['leg_kd']
            leg = kp * .25 * .1 * jac[:4, 16:20] - kd * torch.eye(4)
            wheel = wheel_kd * (jac[4:, 20:22] - torch.eye(2))
            leg_gain = float(torch.linalg.eigvalsh((leg + leg.T) * .5).max())
            wheel_gain = float(torch.linalg.eigvalsh((wheel + wheel.T) * .5).max())
            row.update(run_name=name, leg_command_gain_max_nm_per_rad_s=leg_gain,
                       wheel_command_gain_max_nm_per_rad_s=wheel_gain)
            rows.append(row)
            metrics = dict(
                leg_command_gain_max_nm_per_rad_s=leg_gain,
                wheel_command_gain_max_nm_per_rad_s=wheel_gain,
                previous_action_spectral_radius=row['previous_action_spectral_radius'],
                leg_target_error_max_rad=max(abs(value) for value in row['initial_leg_target_error_rad']),
                leg_gaussian_std_raw=sum(row['gaussian_std_raw'][:4]) / 4.,
                wheel_gaussian_std_raw=sum(row['gaussian_std_raw'][4:]) / 2.,
            )
            for key, value in metrics.items():
                logger.writer.add_scalar(f'DiagnosticScratchNominalFeedback200Hz/{key}', value, row['completed_updates'])
        logger.writer.flush()
    finally:
        logger.writer.close()
    events[name] = dict(directory=str(writer_path), files_sha256={
        event.name: hashlib.sha256(event.read_bytes()).hexdigest()
        for event in writer_path.glob('events.out.tfevents.*')})
report = dict(
    schema='scratch-nominal-feedback-derivative-timeline-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    rows=rows, tensorboard=events,
    step='Actual completed fresh PPO updates of each sealed checkpoint; not simulation seconds.',
    inspected_source_sha256=hashlib.sha256((ROOT / 'scripts/inspect_scratch_feedback.py').read_bytes()).hexdigest(),
    reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_PPO=0, native_steps=0, optimizer_constructed=False, behavior_qualification=False, hardware_ready=False,
    scope='Local command derivatives at the same ideal middle-height geometry with other inputs fixed. '
          'This is a CPU model inspection, not a replay of the intermediate policies in physics or proof of closed-loop stability.',
)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
OUTPUT.with_suffix('.inspect_source.py').write_bytes((ROOT / 'scripts/inspect_scratch_feedback.py').read_bytes())
print(json.dumps([{key: row[key] for key in ('run_name', 'completed_updates', 'leg_command_gain_max_nm_per_rad_s',
      'wheel_command_gain_max_nm_per_rad_s', 'previous_action_spectral_radius')} for row in rows]))
