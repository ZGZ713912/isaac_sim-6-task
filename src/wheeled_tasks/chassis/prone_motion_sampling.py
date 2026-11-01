"""Execution-only rollout packing; control, rewards and RNG are unchanged."""
from __future__ import annotations

import torch

from .tensor_cuda_graph import TensorCudaGraph


class PhysicalSamples:
    """Copy fixed-shape samples in dtype batches into fresh policy storage.

    Each returned field is contiguous, just like stack(clones). Prior policy
    outputs retain their storage when the next interval starts. Mutable input
    buffers may be overwritten immediately after append on the same stream.
    """
    def __init__(self, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError('Physical sample interval must have a positive integer length')
        self.steps = steps
        self.layout = None
        self.groups = {}
        self.cursor = 0

    def begin(self):
        self.cursor = 0
        self.output = {key: torch.empty(group['size']*self.steps, dtype=key[0], device=key[1])
                       for key, group in self.groups.items()}
        return self

    def _initialize(self, record):
        self.layout = {name: (value.shape, value.dtype, value.device) for name, value in record.items()}
        for name, value in record.items():
            key = (value.dtype, value.device)
            group = self.groups.setdefault(key, dict(fields=[], size=0))
            group['fields'].append((name, group['size'], value.numel()))
            group['size'] += value.numel()
        for key, group in self.groups.items():
            group['staging'] = torch.empty(group['size'], dtype=key[0], device=key[1])
            indices = []
            for step in range(self.steps):
                indices.append(torch.cat([torch.arange(offset*self.steps + step*size,
                    offset*self.steps + (step+1)*size, dtype=torch.long)
                    for _, offset, size in group['fields']]))
            group['indices'] = torch.stack(indices).to(key[1])
        self.output = {key: torch.empty(group['size']*self.steps, dtype=key[0], device=key[1])
                       for key, group in self.groups.items()}

    def append(self, record):
        if self.layout is None:
            self._initialize(record)
        if self.cursor >= self.steps:
            raise RuntimeError('Too many physical samples in a policy interval')
        if tuple(record) != tuple(self.layout) or any(
                (value.shape, value.dtype, value.device) != self.layout[name]
                for name, value in record.items()):
            raise ValueError('Physical sample schema changed inside an environment')
        for key, group in self.groups.items():
            torch.cat([record[name].reshape(-1) for name, _, _ in group['fields']],
                      out=group['staging'])
            self.output[key].index_copy_(0, group['indices'][self.cursor], group['staging'])
        self.cursor += 1

    def finish(self):
        if self.cursor != self.steps:
            raise RuntimeError('Incomplete physical sample interval')
        result = {}
        for key, group in self.groups.items():
            for name, offset, size in group['fields']:
                result[name] = self.output[key][offset*self.steps:(offset+size)*self.steps].view(
                    self.steps, *self.layout[name][0])
        return {name: result[name] for name in self.layout}


def role_reductions(fields, role_index, active, outside, role_ids):
    """Fixed-size reductions; excluded NaN values cannot pollute a group."""
    masks = (role_ids[:, None] == role_index[None]) & active[None]
    count = masks.sum(-1)
    means = torch.where(masks[:, :, None], fields[None], 0.).sum(1) / count.clamp_min(1)[:, None]
    spring = torch.where(masks[:, None, :, None], outside[None].float(), 0.).sum((1, 2, 3))
    spring = spring / (count.clamp_min(1)*outside.shape[0]*outside.shape[2])
    return means, spring, count > 0


def recovery_reductions(lane, family, leg, first, enabled, terminated, seconds, groups):
    """All family/leg event statistics with fixed-shape tensor reductions."""
    masks = lane[None] & (groups[:, :1] == family[None]) & (groups[:, 1:] == leg[None])
    completed = masks & first[None]
    return torch.stack(((masks & enabled[None]).float().sum(-1), completed.float().sum(-1),
        (masks & terminated[None]).float().sum(-1),
        torch.where(completed, seconds[None], 0.).sum(-1)), -1)


class ProneMotionSampling:
    def __init__(self, env):
        self.samples = PhysicalSamples(env.decimation)
        self.indices = {name: torch.tensor(value, dtype=torch.long, device=env.device)
            for name, value in dict(motors=env.ids, legs=list(env.v5.LEGS), wheels=[2, 5],
                native_legs=[env.ids[i] for i in env.v5.LEGS],
                native_wheels=[env.ids[2], env.ids[5]], springs=env.spring_ids,
                gimbal=getattr(env, 'gimbal_ids', []), wheel_bodies=env.wheel_ids).items()}
        from .prone_motion import ROLES
        self.role_ids = torch.arange(len(ROLES), device=env.device)
        self.replay = None
        from .multi_pose_recovery import FAMILIES, LEG_CLASSES
        self.recovery_groups = torch.tensor([(f,l) for f in range(len(FAMILIES))
            for l in range(len(LEG_CLASSES))], device=env.device)
        self.recovery_replay = None

    def role_metrics(self, fields, role_index, active, outside):
        args = (fields, role_index, active, outside, self.role_ids)
        if fields.device.type == 'cuda' and self.replay is None:
            self.replay = TensorCudaGraph(role_reductions, *args)
        return self.replay(*args) if self.replay is not None else role_reductions(*args)

    def recovery_metrics(self, lane, family, leg, first, enabled, terminated, seconds):
        args = (lane, family, leg, first, enabled, terminated, seconds, self.recovery_groups)
        if lane.device.type == 'cuda' and self.recovery_replay is None:
            self.recovery_replay = TensorCudaGraph(recovery_reductions, *args)
        return self.recovery_replay(*args) if self.recovery_replay is not None else recovery_reductions(*args)


def apply_fixed_cases(env, ids, rows):
    """Set exactly the selected lanes using three bulk host/device transfers."""
    chosen = torch.tensor([row['pose_index'] for row in rows], dtype=torch.long, device=env.device)
    flags = torch.tensor([[row['recovery'], row.get('push', False)] for row in rows],
                         dtype=torch.bool, device=env.device).reshape(len(rows), 2)
    values = env.pending_command.new_tensor([[*row['command'], row.get('push_sign', 1.)]
                                             for row in rows]).reshape(len(rows), 4)
    env.recovery_lane[ids], env.push_selected[ids] = flags[:, 0], flags[:, 1]
    env.pending_command[ids], env.push_sign[ids] = values[:, :3], values[:, 3]
    entries = [(j, stage, start, command) for j, row in enumerate(rows)
               for stage, (start, command) in enumerate(row.get('sequence', []))]
    if entries:
        index = torch.tensor([[j, stage] for j, stage, _, _ in entries], dtype=torch.long, device=env.device)
        sequence = env.pending_command.new_tensor([[start, *command] for _, _, start, command in entries])
        lanes, stages = ids[index[:, 0]], index[:, 1]
        env.sequence_time[lanes, stages] = sequence[:, 0]
        env.sequence_command[lanes, stages] = sequence[:, 1:]
    return chosen


def apply_course_cases(env, ids, rows):
    """Batch metadata writes without changing row order or sampling RNG."""
    env.course_targets[ids] = env.course_targets.new_tensor([row['target'] for row in rows]).reshape(len(rows), 3)
    flags = torch.tensor([[row.get('moving_height', False), row.get('abrupt_brake', False)] for row in rows],
                         dtype=torch.bool, device=env.device).reshape(len(rows), 2)
    env.course_moving_height[ids], env.course_abrupt_brake[ids] = flags[:, 0], flags[:, 1]
    duration = [(row.get('evaluation_seconds', env.plan['evaluation']['seconds']) if env.evaluation_cases is not None
        else env.plan['training_prone_seconds'] if row['recovery']
        else env.plan['standing_seconds'] if row['role'] == 'stand' else env.plan['motion_seconds']) for row in rows]
    env.episode_limits[ids] = torch.tensor([round(value/env.policy_dt) for value in duration],
                                          dtype=env.episode_limits.dtype, device=env.device)
