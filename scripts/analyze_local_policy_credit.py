#!/usr/bin/env python3
"""Read-only old-policy score-gradient decomposition, never an optimizer update."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch


def command_buckets(commands, eligible):
    """Disjoint labels at recorded current commands, retaining every owned row."""
    if commands.shape[:-1] != eligible.shape or commands.shape[-1] != 3 or eligible.dtype != torch.bool:
        raise ValueError('Expected aligned commands and boolean ownership')
    if not bool(torch.isfinite(commands[eligible]).all()):
        raise ValueError('Nonfinite owned command')
    vx, yaw, height = commands.unbind(-1)
    stand = (vx.abs() < .001) & (yaw.abs() < .001)
    rows = dict(stand_low=eligible & stand & (height < .27),
        stand_mid=eligible & stand & (height >= .27) & (height <= .38),
        stand_high=eligible & stand & (height > .38),
        forward=eligible & (vx > .1) & (yaw.abs() < .001),
        backward=eligible & (vx < -.1) & (yaw.abs() < .001),
        yaw_positive=eligible & (vx.abs() < .001) & (yaw > .1),
        yaw_negative=eligible & (vx.abs() < .001) & (yaw < -.1))
    assigned = torch.stack(list(rows.values())).any(0)
    rows['other'] = eligible & ~assigned
    assert torch.equal(torch.stack(list(rows.values())).sum(0), eligible.int())
    return rows


def score_gradient(actor, inputs, actions, coefficients, denominator, chunk=16384):
    """Gradient of -sum(c*log pi)/denominator at this actor, without sampling.

    At the old policy this equals the unclipped native PPO surrogate gradient.
    It excludes entropy, critic, mini-batch permutations, clipping and Adam.
    """
    from tensordict import TensorDict
    if denominator <= 0 or len(inputs) != len(actions) or len(actions) != len(coefficients):
        raise ValueError('Expected aligned score rows and positive denominator')
    if not all(bool(torch.isfinite(v).all()) for v in (inputs, actions, coefficients)):
        raise ValueError('Nonfinite score sample')
    params = tuple(actor.parameters())
    gradient = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    for start in range(0, len(inputs), chunk):
        sl = slice(start, start + chunk)
        mean = actor(TensorDict({'policy': inputs[sl]}, batch_size=[len(inputs[sl])]))
        actor.distribution.update(mean)
        loss = -(actor.get_output_log_prob(actions[sl]) * coefficients[sl]).sum() / denominator
        derivatives = torch.autograd.grad(loss, params, allow_unused=False)
        gradient += torch.cat([v.detach().reshape(-1).double() for v in derivatives])
    return gradient


def cosine(a, b):
    norm = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / norm) if norm else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--event-dir', required=True, type=Path)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert not args.output.exists()
    d = read(args.run_dir / 'diagnosis.json')
    assert d['status'] == 'completed' and d['ppo_updates'] == 0 and d['policy_credit_recording']
    assert all(d['unchanged'].values()) and not d['native_advantage_normalize_per_minibatch']
    checkpoint = Path(d['checkpoint'])
    assert sha(checkpoint) == d['checkpoint_sha256']
    assert sha(args.run_dir / 'rollouts.npz') == d['arrays_sha256']
    assert checkpoint.parent == args.event_dir
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = read(checkpoint.parent / 'agent_config.json')['actor'].copy()
    assert cfg.pop('class_name') == 'MLPModel' and not cfg['obs_normalization']
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    torch.set_num_threads(4)
    # Constructor initialization must not consume the analysis process RNG.
    with torch.random.fork_rng(devices=[]):
        actor = MLPModel(TensorDict({'policy': torch.zeros(1, 35)}, batch_size=[1]),
            {'actor': ['policy']}, 'actor', 6, **cfg)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    before = deepcopy(actor.state_dict())
    rng_before = torch.get_rng_state().clone()
    a = np.load(args.run_dir / 'rollouts.npz', allow_pickle=False)
    size = d['rollout_ticks']
    results = []
    aggregate_native = aggregate_within = None
    for block, start in enumerate(range(0, len(a['advantages']), size)):
        sl = slice(start, start + size)
        owned = torch.from_numpy(a['learning'][sl]).bool()
        commands = torch.from_numpy(a['commands'][sl])
        masks = command_buckets(commands, owned)
        inputs = torch.from_numpy(a['policy_input'][sl])
        actions = torch.from_numpy(a['native_action'][sl])
        adv = torch.from_numpy(a['advantages'][sl])
        n = int(owned.sum())
        assert n > 0 and inputs.shape[-1] == 35
        with torch.no_grad():
            m = actor(TensorDict({'policy': inputs[owned]}, batch_size=[n]))
            actor.distribution.update(m)
            mean_error = float((m - torch.from_numpy(a['native_mean'][sl])[owned]).abs().max())
            std_error = float((actor.output_std - torch.from_numpy(a['native_std'][sl])[owned]).abs().max())
            log_error = float((actor.get_output_log_prob(actions[owned])
                - torch.from_numpy(a['native_log_prob'][sl])[owned]).abs().max())
        assert mean_error < 1e-5 and std_error < 1e-7 and log_error < .002
        native = score_gradient(actor, inputs[owned], actions[owned], adv[owned], n)
        within = torch.zeros_like(native)
        between = torch.zeros_like(native)
        groups, gradients = {}, {}
        adv_mean = float(adv[owned].mean())
        between_variance = 0.
        for name, mask in masks.items():
            count = int(mask.sum())
            if count == 0:
                groups[name] = dict(samples=0)
                continue
            mean = adv[mask].mean()
            g = score_gradient(actor, inputs[mask], actions[mask], adv[mask], n)
            centered = score_gradient(actor, inputs[mask], actions[mask], adv[mask] - mean, n)
            base = score_gradient(actor, inputs[mask], actions[mask], torch.ones_like(adv[mask]) * mean, n)
            gradients[name] = g
            within += centered
            between += base
            between_variance += count / n * (float(mean) - adv_mean) ** 2
            groups[name] = dict(samples=count, fraction=count/n, advantage_mean=float(mean),
                advantage_positive_fraction=float((adv[mask] > 0).float().mean()),
                residual_mean=float(a['residuals'][sl][mask.numpy()].mean()),
                score_gradient_l2=float(g.norm()), within_bucket_score_gradient_l2=float(centered.norm()),
                bucket_mean_score_gradient_l2=float(base.norm()),
                contribution_dot_total=float(torch.dot(g, native)),
                contribution_cosine_total=cosine(g, native))
        reconstruction = float((within + between - native).abs().max())
        bound = 1e-5 * max(1., float(native.abs().max()))
        assert reconstruction < bound, (reconstruction, bound)
        results.append(dict(rollout=block, samples=n, cpu_gpu_mean_max_error=mean_error,
            cpu_gpu_std_max_error=std_error, cpu_gpu_log_prob_max_error=log_error,
            native_score_gradient_l2=float(native.norm()),
            within_bucket_score_gradient_l2=float(within.norm()),
            bucket_mean_score_gradient_l2=float(between.norm()),
            native_vs_within_cosine=cosine(native, within),
            bucket_mean_advantage_variance_fraction=between_variance/float(adv[owned].var(correction=0)),
            maximum_gradient_decomposition_error=reconstruction, groups=groups,
            group_cosines={left:{right:cosine(g, h) for right,h in gradients.items()}
                for left,g in gradients.items()}))
        aggregate_native = native if aggregate_native is None else aggregate_native + native
        aggregate_within = within if aggregate_within is None else aggregate_within + within
        print('rollout', block, 'native/within norms', float(native.norm()), float(within.norm()),
            'cosine', cosine(native, within), flush=True)
    assert all(torch.equal(v, actor.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng_before, torch.get_rng_state())
    report = dict(schema='v6-local-policy-credit-gradient-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        ppo_updates=0, optimizer_steps=0, actor_unchanged=True, analysis_cpu_rng_unchanged=True,
        diagnosis=str(args.run_dir/'diagnosis.json'), diagnosis_sha256=sha(args.run_dir/'diagnosis.json'),
        script_sha256=sha(Path(__file__)), checkpoint_sha256=d['checkpoint_sha256'],
        original_rsl_event_directory=str(args.event_dir), learning_clock=d['curriculum_actor_update'],
        rollouts=results, aggregate_native_vs_within_cosine=cosine(aggregate_native, aggregate_within),
        scope='Exact old-policy first score-gradient before native epochs, ratio clipping, gradient clipping, entropy, critic and Adam; these vectors are NOT an actual optimizer step. Command-bucket centering is an offline algebraic baseline decomposition only, no modified advantage is passed to PPO. Three sequential rollouts from one saved checkpoint/RNG are not independent training seeds. Task anti-alignment or a large bucket-mean contribution is not a unique cause, convergence prediction or evidence for removing tasks/rewards.')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.event_dir)) as writer:
        for r in results:
            prefix = f"LocalDiagnostics/PolicyCredit/rollout{r['rollout']}"
            for k,v in r.items():
                if isinstance(v,(int,float)) and v is not None:
                    writer.add_scalar(prefix+'/'+k, v, report['learning_clock'])
            for group, row in r['groups'].items():
                for k,v in row.items():
                    if isinstance(v,(int,float)) and v is not None:
                        writer.add_scalar(prefix+'/'+group+'/'+k, v, report['learning_clock'])
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
