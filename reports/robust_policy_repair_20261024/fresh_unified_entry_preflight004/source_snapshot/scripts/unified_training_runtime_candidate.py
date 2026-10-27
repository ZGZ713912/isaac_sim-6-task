"""Prepare native training sequences, real case clocks and upright grounding.

The live learner source is untouched. The factory can be used only after the
declared native patch is applied in a new frozen, fully fresh experiment.
All motor actions and every physical step remain owned by RobustGetupEnv.
"""
import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path

import torch

from unified_training_cases_candidate import TrainingCases, WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
NATIVE = 'src/wheeled_tasks/chassis/robust_getup_env.py'
GROUND = 'src/wheeled_tasks/chassis/grounded_standing_reset.py'
VERSION = 'case-sequences-and-timeouts-v1'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Need the exact original native anchor: '+old[:80])
    return source.replace(old, new, 1)


def native_bytes(original):
    value = original.decode()
    value = replace_once(value, 'class RobustGetupEnv(ChassisEnv):\n',
        'class RobustGetupEnv(ChassisEnv):\n'
        '    TRAINING_CASE_RUNTIME_VERSION = "'+VERSION+'"\n')
    value = replace_once(value,
        '        if self.fixed_cases is not None:\n            for stage in range(self.sequence_time.shape[1]):\n',
        '        if self.fixed_cases is not None or getattr(self, "training_command_sequences", False):\n'
        '            for stage in range(self.sequence_time.shape[1]):\n')
    value = replace_once(value, '        timeout = self.episode_length_buf >= self.max_episode_length\n',
        '        timeout = self.episode_length_buf >= self.episode_limits\n')
    compile(value, NATIVE, 'exec')
    return value.encode()


def ground_bytes(original):
    value = original.decode()
    value = replace_once(value, 'def install(env):\n', 'def install(env, *, include_motion=False):\n')
    value = replace_once(value, '    state = dict(version=VERSION, placement_calls=0, environments=0,\n',
        '    state = dict(version=VERSION+("-all-upright-v1" if include_motion else ""), placement_calls=0, environments=0,\n')
    value = replace_once(value,
        '        selected = ids[(~env.recovery_lane[ids]) & (env.pending_command[ids, :2].abs() < .01).all(-1)]\n',
        '        eligible = ~env.recovery_lane[ids]\n'
        '        if not include_motion:\n'
        '            eligible &= (env.pending_command[ids, :2].abs() < .01).all(-1)\n'
        '        selected = ids[eligible]\n')
    compile(value, GROUND, 'exec')
    return value.encode()


class ActualEpisodeClock:
    """Reject random episode offsets and require actual native step/reset time."""
    def __init__(self, count, device):
        self.expected = torch.zeros(count, device=device, dtype=torch.long)

    def reset(self, ids):
        self.expected[ids] = 0

    def before_step(self, actual):
        if not torch.equal(actual, self.expected):
            raise ValueError('Training sequences require real episode time; disable init_at_random_ep_len')
        return self.expected.clone()

    def after_step(self, previous, actual, done):
        expected = torch.where(done, 0, previous+1)
        if not torch.equal(actual, expected):
            raise ValueError('Native training must advance one real policy interval or reset at actual done')
        self.expected.copy_(actual)


def make_environment_class():
    """Lazy imports let the future entry construct Isaac before its assets."""
    from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
    from wheeled_tasks.chassis.root_kinematics import update_robot_data
    from wheeled_tasks.chassis.grounded_standing_reset import install as ground

    if getattr(RobustGetupEnv, 'TRAINING_CASE_RUNTIME_VERSION', None) != VERSION:
        raise ValueError('Apply and verify the declared native case-runtime patch before use')

    class FreshUnifiedTrainingEnv(RobustGetupEnv):
        def __init__(self, *args, training_case_seed=271701, foundation_updates=192,
                     full_tasks_at=768, **kwargs):
            self.fresh_cases_ready = False
            if kwargs.get('fixed_cases') is not None:
                raise ValueError('This is a training environment; fixed evaluation is separate')
            super().__init__(*args, **kwargs)
            if (not self.cfg['record_diagnostics'] or self.getup_cfg['quiet_training']
                    or not math.isclose(self.max_episode_length*self.policy_dt, 30.)):
                raise ValueError('Predeclare30s, diagnostics and motion-enabled settings before construction')
            self.training_catalogue = TrainingCases(self.bank, self.num_envs, training_case_seed,
                foundation_updates=foundation_updates, full_tasks_at=full_tasks_at)
            self.height_pose = self.training_catalogue.height_pose
            self.training_case_names = tuple(WEIGHTS)
            self.training_group = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.training_reset_numbers = [0]*self.num_envs
            self.training_reset_cases = [None]*self.num_envs
            self.actual_episode_clock = ActualEpisodeClock(self.num_envs, self.device)
            self.training_command_sequences = True
            self.fresh_cases_ready = True
            self.reset(torch.arange(self.num_envs, device=self.device))
            # The optional native helper also initializes moving/spinning
            # upright births, retaining its selected-lane sensor-cache refresh.
            ground(self, include_motion=True)
            self.startup_report['fresh_training_cases'] = self.training_catalogue.summary(0)
            self.startup_report['fresh_training_cases'].update(native_runtime_version=VERSION,
                init_at_random_ep_len=False, state_writes='episode_reset_only',
                motor_actions='all native six-motor RL outputs; no trajectory controller')

        @torch.inference_mode()
        def reset(self, ids):
            if not self.fresh_cases_ready:
                return super().reset(ids)
            if self.fixed_cases is not None:
                raise ValueError('Never disable training autoresets with fixed evaluation cases')
            lanes = ids.tolist()
            group_ids = []
            for lane in lanes:
                case = self.training_catalogue.case(lane, self.training_reset_numbers[lane], self.actor_updates)
                self.training_reset_cases[lane] = case
                self.training_reset_numbers[lane] += 1
                group_ids.append(self.training_case_names.index(case['training_group']))
            self.training_group[ids] = self.training_group.new_tensor(group_ids)
            self.fixed_cases = self.training_reset_cases
            try:
                super().reset(ids)
            finally:
                self.fixed_cases = None
            cases = [self.training_reset_cases[lane] for lane in lanes]
            self.episode_limits[ids] = self.episode_limits.new_tensor([
                round(case['duration_seconds']/self.policy_dt) for case in cases])
            root = self.robot.data.root_link_pose_w.torch[ids].clone()
            # Pitch perturbs only validated upright births. World heading
            # rotates every bank pose without changing its floor clearance.
            pitch = root.new_tensor([math.radians(case['pitch_deg']) for case in cases])
            x, y, z, w = root[:, 3:].unbind(-1)
            s, c = (pitch/2).sin(), (pitch/2).cos()
            root[:, 3:] = torch.stack((x*c-z*s, y*c+w*s, z*c+x*s, w*c-y*s), -1)
            heading = root.new_tensor([case['heading_rad'] for case in cases])
            x, y, z, w = root[:, 3:].unbind(-1)
            s, c = (heading/2).sin(), (heading/2).cos()
            root[:, 3:] = torch.stack((c*x-s*y, s*x+c*y, c*z+s*w, c*w-s*z), -1)
            self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
            update_robot_data(self, self.dt)
            self.previous_potential[ids] = self.potential()[ids]
            self.actual_episode_clock.reset(ids)

        def sample_commands(self, ids):
            if not self.fresh_cases_ready:
                return super().sample_commands(ids)
            # The native training resampler may call this every4s. Preserve
            # the declared case sequence rather than draw an unrelated task.

        @torch.inference_mode()
        def step(self, actions):
            previous = self.actual_episode_clock.before_step(self.episode_length_buf)
            groups = self.training_group.clone()
            obs, reward, done, extras = super().step(actions)
            self.actual_episode_clock.after_step(previous, self.episode_length_buf, done)
            state = extras['diagnostics']  # True elapsed state, before any reset.
            active = torch.nn.functional.one_hot(groups, len(self.training_case_names)).to(state['height'].dtype)
            active *= extras['policy_learning_mask'][:, None]
            count = active.sum(0)
            divisor = count.clamp_min(1.)
            values = dict(active_lanes=count, episode_end_events=(active*done[:, None]).sum(0),
                planar_speed_rms_m_s=((active*state['velocity'][:, :2].square().sum(-1, keepdim=True)).sum(0)/divisor).sqrt(),
                tilt_mean_deg=(active*torch.acos((-state['gravity'][:, 2:3]).clamp(-1., 1.))*180./math.pi).sum(0)/divisor,
                height_mae_m=(active*(state['height']-state['commands'][:, 2]).abs()[:, None]).sum(0)/divisor)
            extras['log']['FreshTasks/exposure_fraction'] = state['height'].new_tensor(
                self.training_catalogue.exposure(self.actor_updates))
            for index, name in enumerate(self.training_case_names):
                for metric, value in values.items():
                    extras['log']['FreshTasks/'+name+'/'+metric] = value[index]
            return obs, reward, done, extras

    return FreshUnifiedTrainingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources, patch = {}, []
    for name, transform in ((NATIVE, native_bytes), (GROUND, ground_bytes)):
        original = (ROOT/name).read_bytes()
        candidate = transform(original)
        destination = args.output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(candidate)
        patch.extend(difflib.unified_diff(original.decode().splitlines(True), candidate.decode().splitlines(True),
            fromfile='a/'+name, tofile='b/'+name))
        sources[name] = dict(original_sha256=hashlib.sha256(original).hexdigest(),
            candidate_sha256=hashlib.sha256(candidate).hexdigest())
    (args.output/'integration.patch').write_text(''.join(patch).replace('\n \n', '\n\n'))
    result = dict(schema='fresh-unified-training-runtime-candidate-v1', source=sources,
        version=VERSION, candidate_applied=False, new_native_steps=0, new_PPO=0,
        actual_native_class_constructed=False, live_source_changed=False, qualified=False, hardware_ready=False)
    (args.output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
