"""Read the training contract and checked RMCS YAML; never alter either repository."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RMCS = Path('/home/noir/Documents/workspace/RMCS')
RUN = BASE/'runs/fresh_unified_20261027/pilot016_scoped_failure_requested_mixed1536'
YAML = RMCS/'rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-rl.yaml'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


task = json.loads((RUN/'task_contract.json').read_text())
cfg = json.loads((RUN/'config/contract.json').read_text())
parameters = yaml.safe_load(YAML.read_text())
bridge, chassis, commands = [parameters[name]['ros__parameters']
    for name in ('rl_bridge', 'wheel_leg_controller', 'wheel_leg_chassis_controller')]
terms = [dict(token.split('=', 1) for token in shlex.split(term)) for term in bridge['action_terms']]
raw = task['settings']
required_leg_target_bound = raw['leg_action_clip']*raw['leg_position_scale']
spec = ROOT/cfg['asset_directory']/'model_spec.json'
sources = [YAML, RMCS/'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/action.cpp',
    RMCS/'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/configuration.cpp',
    RMCS/'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/observation.cpp',
    RMCS/'rmcs_ws/src/rmcs_rl/src/policy/policy_model.cpp',
    RMCS/'rmcs_ws/src/rmcs_rl/src/policy/onnx_runtime.cpp']
report = dict(schema='fresh-deployment-interface-audit-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(), checked_yaml=str(YAML), training_run=str(RUN),
    rmcs_commit=subprocess.check_output(['git', '-C', str(RMCS), 'rev-parse', 'HEAD'], text=True).strip(),
    rmcs_rl_submodule_commit=subprocess.check_output(
        ['git', '-C', str(RMCS/'rmcs_ws/src/rmcs_rl'), 'rev-parse', 'HEAD'], text=True).strip(),
    checked_source_sha256={str(path): sha(path) for path in sources},
    training_task_sha256=sha(RUN/'task_contract.json'), model_spec_sha256=sha(spec),
    model_spec_matches=sha(spec) == bridge['expected_model_spec_sha256'],
    current_file_configuration=dict(policy_hz=bridge['policy_rate'], pd_hz=chassis['pd_frequency'],
        input_dim=bridge['rl_obs_size'], history=bridge['history_length'], action_dim=bridge['rl_action_size'],
        max_action_age_s=bridge['max_action_age'], bridge_timeout_s=bridge['inference_timeout'],
        chassis_timeout_s=chassis['inference_timeout'], leg_target_clip_rad=[term['clip'] for term in terms[:4]],
        wheel_target_clip_rad_s=[term['clip'] for term in terms[4:]],
        chassis_yaw_cap_rad_s=chassis['yaw_rate_max'], input_yaw_cap_rad_s=commands['yaw_rate_max'],
        input_spin_rad_s=commands['spin_yaw_rate'], input_spin_rpm=commands['spin_yaw_rate']*60./(2.*math.pi)),
    required_future_export=dict(policy_hz=1./cfg['policy_dt'], physical_hz=1./cfg['physics_dt'],
        input=dict(name='obs', shape=[1, 35], dtype='float32'),
        output=dict(name='actions', shape=[1, 6], dtype='float32'), deterministic_mean=True,
        three_policy_periods_action_age_s=3*cfg['policy_dt'],
        actual_getup_leg_action_clip=raw['leg_action_clip'], leg_target_clip_rad=[-required_leg_target_bound, required_leg_target_bound],
        wheel_action_clip=raw['wheel_action_clip'], wheel_target_clip_rad_s=[-90., 90.],
        spin_target_rpm=100., spin_target_rad_s=100.*2.*math.pi/60., minimum_body_rpm=80.,
        height_coefficients_and_normalization='Embed the exact archived actor mean, normalization, height coefficients and wheel velocity increment once',
        metadata_normalization='Do not add rmcs_obs_mean/std to an already normalized ONNX graph',
        leg_clipping_source='Use checkpoint infos.settings leg_action_clip16; cfg.v5_control.action_clip3 is overridden by RobustGetupEnv at construction',
        safety='Retain remote reset/loss, CAN/IMU timeout, clear_error, nonfinite/stale inference stops, relative-angle torque projection and40Nm saturation'),
    export_or_deployment_performed=False, policy_quality_or_hardware_qualification=False,
    limitation='Read-only file configuration audit. The running vehicle process and its selected YAML were not inspected. '
        'No trained candidate has completed the full capability tests or been selected for deployment.',
    new_PPO=0, native_steps=0, producer_sha256=sha(Path(__file__)))
for directory in (ROOT/REL, BASE/REL):
    (directory/'fresh_deployment_interface_audit001.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
(BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(model_spec_matches=report['model_spec_matches'],
    current_policy_hz=bridge['policy_rate'], required_policy_hz=1/cfg['policy_dt'],
    current_spin_rpm=commands['spin_yaw_rate']*60/(2*math.pi),
    required_leg_target_bound_rad=required_leg_target_bound, performed_deployment=False)))
