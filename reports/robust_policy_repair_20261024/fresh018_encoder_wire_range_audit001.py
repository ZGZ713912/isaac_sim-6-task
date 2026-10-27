"""Read-only comparison of simulated leg speeds with configured DM wire levels.

This audits source and actual RSL maxima, not a live CAN register or the motor
firmware's behavior outside its configured velocity range. No limits or noise
are inserted into the active simulator, actor, PD or hardware configuration.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
RMCS = Path('/home/noir/Documents/workspace/RMCS')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/fresh_unified_20261027/pilot018_velocity_quartic_physical_speed_mixed1536'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def parameter(document, key):
    matches = [(name, block['ros__parameters'][key]) for name, block in document.items()
               if isinstance(block, dict) and key in block.get('ros__parameters', {})]
    if len(matches) != 1:
        raise ValueError('Require one explicit YAML parameter: ' + key)
    return matches[0]


def main():
    name = Path(__file__).stem
    output = ROOT/REL/(name+'.json')
    archive = BASE/REL/name
    if output.exists() or archive.exists():
        raise ValueError('Preserve the original audit')
    before_status = subprocess.check_output(['git','-C',str(RMCS),'status','--porcelain'], text=True)
    files = {key: RMCS/path for key, path in {
        'driver': 'rmcs_ws/src/rmcs_core/src/hardware/device/dm_motor.hpp',
        'feedback_mapping': 'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/observation.cpp',
        'mapping_configuration': 'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/configuration.cpp',
        'yaml': 'rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-rl.yaml',
    }.items()}
    native_names = ['src/wheeled_tasks/chassis/scut_observation.py',
                    'src/wheeled_tasks/chassis/getup_feedback.py',
                    'src/wheeled_tasks/chassis/robust_getup.py']
    files.update({path: ROOT/path for path in native_names})
    sources = {key: dict(path=str(path), sha256=sha(path)) for key, path in files.items()}
    texts = {key: path.read_text() for key, path in files.items()}
    frozen = read(RUN/'source_hashes.json')
    for path in native_names:
        assert sources[path]['sha256'] == frozen[path] == sha(RUN/'source_snapshot'/path)
    assert 'uint_to_float(vel_u, -velocity_max_, velocity_max_, 12)' in texts['driver']
    assert 'uint_to_float(pos_u, -position_max_, position_max_, 16)' in texts['driver']
    assert 'dq_.head<4>() = leg_jacobian_ * motor_dq;' in texts['feedback_mapping']
    assert 'self.dq_history[read, self.rows], self.dq)' in texts[native_names[1]]
    assert 'motor_dq[:, POLICY_FROM_CONTROL] * .1' in texts[native_names[0]]
    document = yaml.safe_load(texts['yaml'])
    _, velocity_max = parameter(document, 'dm_mit_velocity_max')
    _, position_max = parameter(document, 'dm_mit_position_max')
    _, matrix = parameter(document, 'leg_motor_to_model')
    _, gain_p = parameter(document, 'joint_position_kp')
    _, gain_d = parameter(document, 'joint_position_kd')
    _, control_mode = parameter(document, 'joint_control_mode')
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4,4)
    velocity_max = np.asarray(velocity_max, dtype=np.float64)
    position_max = np.asarray(position_max, dtype=np.float64)
    assert np.array_equal(matrix, -np.eye(4)) and np.all(velocity_max == 45.)
    assert np.all(position_max == 12.5) and control_mode == 'torque'
    model_speed_bound = np.abs(matrix) @ velocity_max
    velocity_step = 2.*velocity_max/(2**12-1)
    position_step = 2.*position_max/(2**16-1)
    levels = -velocity_max[0] + np.arange(4096)*velocity_step[0]
    assert len(levels) == 4096 and levels[0] == -45. and levels[-1] == 45.
    assert np.allclose(np.diff(levels), velocity_step[0], atol=1e-14, rtol=1e-12)
    assert not np.any(levels == 0.)
    record_path = ROOT/REL/'fresh_unified018_full352_012.json'
    original = record_path.read_bytes()
    record = json.loads(original)
    assert record['all_checks_pass'] and all(record['checks'].values())
    prefix = record['event_prefix']
    with open(prefix['path'], 'rb') as stream:
        raw = stream.read(prefix['bytes'])
    assert len(raw) == prefix['bytes'] and hashlib.sha256(raw).hexdigest() == prefix['sha256']
    with tempfile.TemporaryDirectory() as temporary:
        event = Path(temporary)/Path(prefix['path']).name
        event.write_bytes(raw)
        events = EventAccumulator(str(event), size_guidance={'scalars':0}).Reload()
        def scalar(tag):
            return {point.step:point.value for point in events.Scalars(tag)}
        peaks = scalar('ScratchStandingExtrema/peak_leg_speed_rad_s')
        phases = scalar('PhysicalPeakWitness/phase_code')
        admission = scalar('DomainAdmission/admitted_randomized_lanes')
    count = record['extrema']['complete_RSL_updates']
    witness = record['extrema']['global_peak_witness']
    peak_index = record['extrema']['peak_update_one_based']-1
    assert len(peaks) >= count and all(i in peaks and i in phases for i in range(count))
    assert admission[peak_index] == 188. and peaks[peak_index] == witness['speed_rad_s']
    samples_path = RUN/'randomization_study.initial_samples.json'
    samples = read(samples_path)['samples']
    enabled = np.asarray(samples['enabled'], dtype=bool).reshape(-1)
    sigma = np.asarray(samples['policy_noise_sigma'])
    assert np.allclose(sigma[enabled,12:16]/.1, .0175)
    assert enabled[int(witness['lane'])]
    exceed = [i for i in range(count) if peaks[i] > float(model_speed_bound.max())]
    archive.mkdir()
    for key,path in files.items():
        target = archive/'source_snapshot'/path.relative_to(RMCS if path.is_relative_to(RMCS) else ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    assert subprocess.check_output(['git','-C',str(RMCS),'status','--porcelain'], text=True) == before_status
    assert all(sha(Path(item['path'])) == item['sha256'] for item in sources.values())
    assert record_path.read_bytes() == original
    result = dict(schema='fresh018-encoder-wire-range-source-audit-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        RMCS_HEAD=subprocess.check_output(['git','-C',str(RMCS),'rev-parse','HEAD'],text=True).strip(),
        configured_host_control_mode=control_mode, API_to_model_matrix=matrix.tolist(),
        configured_wire=dict(velocity_bits=12, position_bits=16,
            velocity_api_bounds_rad_s=[[-float(v),float(v)] for v in velocity_max],
            velocity_model_bounds_rad_s=[[-float(v),float(v)] for v in model_speed_bound],
            velocity_LSB_rad_s=velocity_step.tolist(), position_LSB_rad=position_step.tolist(),
            smallest_abs_velocity_level_rad_s=float(np.abs(levels).min()), zero_exactly_representable=False),
        conditional_nearest_level_bounds=dict(assumption='Nearest-level in-range quantization; firmware rounding and error distribution are unverified',
            half_LSB_position_PD_error_Nm=(gain_p*position_step/2.).tolist(),
            half_LSB_velocity_PD_error_Nm=(gain_d*velocity_step/2.).tolist()),
        simulation=dict(leg_policy_velocity_noise_sigma_rad_s=.0175,
            low_level_PD_uses_float_simulator_velocity=True, low_level_PD_quantization_modeled=False,
            scut_builder_velocity_clip_before_domain_rad_s=100./.1,
            nominal_noise_enabled=False, startup_samples_sha256=sha(samples_path)),
        actual_training_evidence=dict(source_report=str(record_path), report_sha256=hashlib.sha256(original).hexdigest(),
            event_prefix=prefix, complete_updates=count, global_peak_update=peak_index+1,
            global_peak_witness=witness, peak_lane_randomized=True, peak_lane_PPO_admitted=True,
            rollout_maxima_above_configured_model_range=len(exceed),
            phases_of_those_rollout_maxima={str(code):sum(phases[i]==code for i in exceed) for code in range(4)},
            scope='One actual maximum per rollout; neither total out-of-range samples nor their duration can be reconstructed.'),
        limitations=['Repository YAML/source only; no live vehicle configuration or motor register readback.',
            'Wire representability is not a mechanical speed limit. Above-range firmware saturation/wrapping/error behavior is unknown.',
            'No continuous motor energy or full preceding peak trajectory was recorded by the single-maximum witness.',
            'This sensor-model gap does not establish the cause of nominal standing drift or the benefit of any intervention.'],
        current_training_changed=False, RMCS_files_modified=False, new_native_steps=0, new_PPO=0,
        optimizer_constructed=False, hardware_motion=False, behavior_qualified=False, hardware_ready=False,
        producer_sha256=sha(Path(__file__)))
    encoded = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    output.write_text(encoded)
    (BASE/REL/output.name).write_text(encoded)
    (archive/'result.json').write_text(encoded)
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'wire':result['configured_wire'], 'out_of_range_rollout_maxima':len(exceed),
        'peak':witness, 'new_native_steps':0, 'new_PPO':0}))


if __name__ == '__main__':
    main()
