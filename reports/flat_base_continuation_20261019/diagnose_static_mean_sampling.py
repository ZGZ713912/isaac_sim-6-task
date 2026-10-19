"""Matched standing-only simulation; action noise cannot qualify a release.

Compare896/960 using the same four seeded initial perturbations per height and
common random noise. No optimizer or training checkpoint is created. The native
run requires the existing960 queue to be completed, exited and sealed first.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from wheeled_tasks.chassis.flat_base_continuation import write_bundle, encode, digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--preflight-only', action='store_true')
parser.add_argument('--deployment-pd', action='store_true',
    help='Separate matched1000Hz physics diagnostic; the actor stays50Hz')
args = parser.parse_args()
queue_path = RUN / 'feedback960_queue001.json'
queue = json.loads(queue_path.read_text())
completion = HERE / 'feedback960_queue_completion001.json'
if not args.preflight_only:
    if (queue['status'] != 'complete' or not completion.exists()
            or any(Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid'))):
        raise SystemExit('Wait for the existing960 queue to finish, exit and be sealed')
    if not json.loads(completion.read_text())['passed']:
        raise SystemExit('The preceding native completion receipt failed')
frozen = {**queue['source_sha256'], str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
if args.deployment_pd:
    control_source = ROOT / 'contracts/v6_wheel_response_control_v1.json'
    control_sha = 'cc30b2b0572999224c8feb79ffa7afb3df1b8a28800bca4a153663b9ea8d4bf8'
    if digest(control_source) != control_sha:
        raise SystemExit('The frozen1000Hz control source changed')
    frozen[str(control_source.relative_to(ROOT))] = control_sha
if not all(digest(ROOT / n) == value for n, value in frozen.items()):
    raise SystemExit('The frozen diagnostic source differs from the completed native source')
checkpoints = {896: RUN / 'pilot017/model_000896.pt', 960: RUN / 'pilot018/model_000960.pt'}
checkpoint_hashes = {str(k): digest(p) for k, p in checkpoints.items()}
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
config, seed = write_bundle(ROOT, args.output / 'config', 32, evaluation=True,
    plan_file='contracts/flat_base_finetune_v13.json')
settings = deepcopy(config['evaluation'])
originals = [c for c in settings['cases'] if c['name'] in ('stand_230mm', 'stand_305mm', 'stand_430mm')]
if len(originals) != 3:
    raise ValueError('The three original standing cases are required')
cases, specification = [], {}
for original in originals:
    for reset_condition in ('fixed_jitter', 'nominal_training_start'):
        for command_condition in ('immediate_height', 'rmcs_height_reference'):
            for step in checkpoints:
                for label, sigma_scale in (('mean', 0.), ('half_sigma', .5), ('training_sigma', 1.)):
                    case = deepcopy(original)
                    name = f"{original['name']}__{reset_condition}_{command_condition}_p{step}_{label}"
                    case.update(name=name, reset_seed_key=original['name'], episode_seconds=10.)
                    cases.append(case)
                    specification[name] = dict(source_case=original['name'], checkpoint=step,
                        action_sigma_scale=sigma_scale, condition=label, reset_condition=reset_condition,
                        command_condition=command_condition)
settings.update(cases=cases, episode_seconds=10., diagnostic_only=True,
    protocol_id='flat-static-matched-mean-sampling-diagnostic-v1')
count = 4 * len(cases)
design_path = args.output / 'config/design.json'
design = json.loads(design_path.read_text())
row = design['stages']['speed']['roles']['quick']
for mapping in (config, row['contract']):
    mapping.update(evaluation=settings, episode_seconds=10., target_num_envs=count,
        ppo_minibatch_samples=count * config['num_steps_per_env'] // config['num_mini_batches'],
        scene_groups=[dict(name=c['name'], fraction=1 / len(cases), terrain=['flat']) for c in cases])
row['invocation']['num_envs'] = count
if args.deployment_pd:
    for mapping in (config, row['contract']):
        mapping.update(physics_dt=.001, control_math_source=str(control_source.relative_to(ROOT)),
            control_math_sha256=control_sha)
    reference_control = json.loads(control_source.read_text())
    row['control']['timing'] = reference_control['timing']
    row['runtime'].update(reference_control['timing'])
design['scope'] = 'matched_static_sampling_diagnostic_no_optimization_no_release'
design_path.write_bytes(encode(design))
config['design_preflight']['sha256'] = digest(design_path)
(args.output / 'config/contract.json').write_bytes(encode(config))
from train_chassis import preflight
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
_, manifest = preflight(args.output / 'config/contract.json')
control = load_chassis_control(ROOT / config['control_math_source'])
static = design_preflight(config, control, ROOT, stage='speed', role='quick', num_envs=count, seed=seed)
from flat_rmcs_command_reference import read_reference_contract
reference_settings = json.loads((ROOT / 'contracts/flat_base_finetune_v13.json').read_text())['training_command_reference']
reference_contract = read_reference_contract(reference_settings['rmcs_root'])
if reference_contract != reference_settings['contract']:
    raise ValueError('Diagnostic command math differs from the actual training contract')
report = dict(schema='flat-static-matched-mean-sampling-diagnostic-v1', status='static_preflight_passed',
    pid=os.getpid(), recorded_at=datetime.now(timezone.utc).isoformat(), seed=seed, action_noise_seed=191031,
    policy_hz=50, physics_hz=round(1 / config['physics_dt']), num_envs=count,
    case_count=len(cases), episodes_per_condition=4,
    source_sha256=frozen, checkpoint_sha256=checkpoint_hashes, static_preflight=static,
    preceding_completion_sha256=digest(completion) if completion.exists() else None,
    conditions=specification, qualification_eligible=False, behavior_qualified=False,
    command_reference_contract=reference_contract,
    command_reference_scope=f"Same50Hz height-reference math; this diagnostic uses{round(1 / config['physics_dt'])}Hz physics. "
        'Immediate-height arms reproduce the training-clock fixed evaluation instead.',
    reset_conditions=dict(fixed_jitter=dict(base_height_m=.324, height_jitter_m=[0., .002],
        roll_range_rad=[-.01, .01], pitch_range_rad=[-.02, .02]),
        nominal_training_start=dict(base_height_m=.32, height_jitter_m=[0., 0.],
            roll_range_rad=[0., 0.], pitch_range_rad=[0., 0.])),
    native_environment_steps=0, optimizer_updates_performed=0, checkpoint_files_written=0,
    hardware_motion_started=False,
    limitations='Seeded standing replay only. Action sampling changes the controller distribution and '
        'cannot qualify deterministic deployment or establish a unique cause of any regression.')


def save():
    path = args.output / 'audit.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(encode(report))
    temporary.replace(path)


save()
if args.preflight_only:
    print(json.dumps(dict(status=report['status'], num_envs=count, case_count=len(cases), native_steps=0)))
    raise SystemExit(0)
launcher = env = None
try:
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(dict(headless=True, device='cuda:0', enable_cameras=False,
        kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_tasks.chassis.eval_env import FixedCaseEnv
    from wheeled_tasks.chassis.root_kinematics import update_robot_data
    from wheeled_tasks.chassis.flat_command_reference import RmcsFlatCommandReference

    class DiagnosticEnv(FixedCaseEnv):
        def reset(self, ids):
            super().reset(ids)
            selected = ids[torch.tensor([specification[self.scene_groups[i]]['reset_condition'] ==
                'nominal_training_start' for i in ids.tolist()], device=self.device, dtype=torch.bool)]
            if not len(selected):
                return
            # These cached positions and the identity quaternion are exactly
            # ChassisEnv.reset's ordinary flat training birth, not a new pose.
            root = torch.zeros(len(selected), 7, device=self.device)
            root[:, :3], root[:, 6] = self.reset_positions[selected], 1.
            self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=selected)
            self.robot.write_root_com_velocity_to_sim_index(
                root_velocity=torch.zeros(len(selected), 6, device=self.device), env_ids=selected)
            update_robot_data(self, self.dt)
            self.update_targets()
            self.update_command_reference(selected)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    actors, standard_deviations = {}, {}
    for step, path in checkpoints.items():
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['infos']['successful_updates_total'] != step or digest(path) != checkpoint_hashes[str(step)]:
            raise ValueError('Diagnostic parent checkpoint changed')
        actors[step] = mean_from_checkpoint(state, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx').to('cuda:0').eval()
        std = state['actor_state_dict']['distribution.std_param']
        if std.shape != (6,) or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('The saved effective training distribution is invalid')
        if standard_deviations and not torch.equal(std, next(iter(standard_deviations.values())).cpu()):
            raise ValueError('The paired checkpoints have different effective training distributions')
        standard_deviations[step] = std.to('cuda:0')
    report['effective_std_policy_order'] = {str(step): value.cpu().tolist()
        for step, value in standard_deviations.items()}
    report['noise_sigma_semantics'] = 'Exact checkpoint std, frozen during these continuations; initial plan std is not reinitialized'
    env = DiagnosticEnv(config, manifest, control, ROOT, stage_name='speed', num_envs=count,
        device='cuda:0', seed=seed, level=1., run_role='quick')
    reference = RmcsFlatCommandReference(count, env.device, reference_contract)
    original_reset, original_observations = env.reset, env.get_observations
    desired = torch.tensor([env.cases[name]['command'] for name in env.scene_groups], device=env.device)
    reference_mask = torch.tensor([specification[name]['command_condition'] == 'rmcs_height_reference'
        for name in env.scene_groups], device=env.device)

    def reset(ids):
        original_reset(ids)
        reference.reset(ids)

    def get_observations():
        command = reference.update(desired, env.skills.yaw(), env.episode_length_buf)
        env.commands.copy_(torch.where(reference_mask[:, None], command, desired))
        env.command_acceleration.zero_()
        if env.references is not None:
            env.raw_height.copy_(env.commands[:, 2])
            env.update_command_reference()
        return original_observations()

    env.reset, env.get_observations = reset, get_observations
    observations = env.reset_suite()
    members = {name: torch.tensor([i for i, group in enumerate(env.scene_groups) if group == name],
        device=env.device, dtype=torch.long) for name in specification}
    if any(len(ids) != 4 for ids in members.values()):
        raise ValueError('Every diagnostic condition requires four replicas')
    initial = observations['policy'].clone()
    reset_pairs = {}
    for original in originals:
        for reset_condition in ('fixed_jitter', 'nominal_training_start'):
            for command_condition in ('immediate_height', 'rmcs_height_reference'):
                names = [n for n, spec in specification.items() if spec['source_case'] == original['name']
                    and spec['reset_condition'] == reset_condition and spec['command_condition'] == command_condition]
                anchor = members[names[0]]
                reset_pairs[original['name'] + '/' + reset_condition + '/' + command_condition] = dict(
                    jitter_samples_bit_exact=all(torch.equal(env.fixed_reset_samples[anchor], env.fixed_reset_samples[members[n]]) for n in names),
                    initial_obs_max_abs_difference=max(float((initial[anchor] - initial[members[n]]).abs().max()) for n in names))
    if not all(v['jitter_samples_bit_exact'] for v in reset_pairs.values()):
        raise ValueError('The diagnostic initial perturbations are not paired')
    nominal_ids = torch.tensor([i for i, name in enumerate(env.scene_groups)
        if specification[name]['reset_condition'] == 'nominal_training_start'], device=env.device)
    nominal_pose = env.robot.data.root_link_pose_w.torch[nominal_ids]
    nominal_readback = dict(position_max_abs_error_m=float((nominal_pose[:, :3] - env.reset_positions[nominal_ids]).abs().max()),
        quaternion_max_abs_error=float((nominal_pose[:, 3:] - nominal_pose.new_tensor([0., 0., 0., 1.])).abs().max()))
    if max(nominal_readback.values()) > 1e-6:
        raise ValueError('The declared ordinary training birth was not read back')
    noise = torch.zeros(500, count, 6, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(report['action_noise_seed'])
    for original in originals:
        common = torch.randn(500, 4, 6, generator=generator, device=env.device)
        for name, spec in specification.items():
            if spec['source_case'] == original['name']:
                noise[:, members[name]] = common * standard_deviations[spec['checkpoint']] * spec['action_sigma_scale']
    actor_members = {step: torch.tensor([i for i, name in enumerate(env.scene_groups)
        if specification[name]['checkpoint'] == step], device=env.device) for step in actors}
    alive = torch.ones(count, device=env.device, dtype=torch.bool)
    traces = {key: [] for key in ('active', 'observations', 'actions', 'position', 'velocity', 'commands', 'height', 'gap', 'done')}
    report.update(status='running', reset_pairing=reset_pairs, nominal_birth_readback=nominal_readback,
        startup=env.startup_report)
    save()
    started = time.monotonic()
    with torch.inference_mode():
        for tick in range(500):
            before = observations['policy'].clone()
            action = torch.empty(count, 6, device=env.device)
            for step, ids in actor_members.items():
                action[ids] = actors[step](before[ids])
            action += noise[tick]
            observations, _, done, extras = env.step(action)
            report['native_environment_steps'] = tick + 1
            data = extras['diagnostics']
            values = dict(active=alive, done=done, observations=before, actions=action,
                **{k: data[k] for k in ('position', 'velocity', 'commands', 'height', 'gap')})
            for key, value in values.items():
                traces[key].append(value.cpu().numpy().copy())
            alive &= ~done.bool()
            if (tick + 1) % 100 == 0:
                print('STATIC_SAMPLING', json.dumps(dict(tick=tick + 1, active=int(alive.sum()),
                    elapsed_s=round(time.monotonic() - started, 2))), flush=True)
    arrays = {key: np.asarray(values) for key, values in traces.items()}
    np.savez_compressed(args.output / 'traces.npz', **arrays, groups=np.asarray(env.scene_groups),
        action_noise=noise.cpu().numpy(), initial_observations=initial.cpu().numpy())
    displacement = np.linalg.norm(arrays['position'][:, :, :2] - arrays['position'][0:1, :, :2], axis=-1)
    results = {}
    for name, ids in members.items():
        ids = ids.cpu().numpy()
        valid = arrays['active'][:, ids]
        late = valid.copy()
        late[:50] = False
        vx = arrays['velocity'][:, ids, 0]
        results[name] = dict(**specification[name], active_frames_per_lane=valid.sum(0).tolist(),
            drift_max_m=float(np.where(valid, displacement[:, ids], 0.).max()),
            drift_max_m_per_lane=np.where(valid, displacement[:, ids], 0.).max(0).tolist(),
            post1s_vx_mae_m_s=float(np.abs(vx)[late].mean()) if late.any() else None,
            post1s_vx_signed_mean_m_s=float(vx[late].mean()) if late.any() else None)
    if not all(digest(ROOT / n) == value for n, value in frozen.items()):
        raise ValueError('Frozen diagnostic source changed during simulation')
    with SummaryWriter(str(args.output)) as writer:
        writer.add_text('FlatDiagnostic/static_sampling/scope',
            'Paired counterfactual only; action noise cannot qualify deterministic deployment.', 960)
        for result in results.values():
            prefix = '/'.join(('FlatDiagnostic/static_sampling', result['source_case'],
                result['reset_condition'], result['command_condition'], result['condition']))
            for metric in ('drift_max_m', 'post1s_vx_mae_m_s', 'post1s_vx_signed_mean_m_s'):
                if result[metric] is not None:
                    writer.add_scalar(prefix + '/' + metric, result[metric], result['checkpoint'])
            writer.add_scalar(prefix + '/full_horizon_fraction',
                sum(n == 500 for n in result['active_frames_per_lane']) / 4., result['checkpoint'])
        writer.flush()
    report.update(status='audited', results=results, native_environment_steps=500,
        traces_sha256=digest(args.output / 'traces.npz'), elapsed_s=time.monotonic() - started,
        tensorboard_events={p.name: digest(p) for p in args.output.glob('events.out.tfevents.*')},
        tensorboard_step_semantics='Checkpoint local successful update count896/960; same native trace metrics, no optimizer updates')
except Exception:
    report.update(status='failed', error=traceback.format_exc())
    traceback.print_exc()
finally:
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    if env is not None:
        env.close()
    if launcher is not None:
        launcher.app.close()
raise SystemExit(0 if report['status'] == 'audited' else 1)
