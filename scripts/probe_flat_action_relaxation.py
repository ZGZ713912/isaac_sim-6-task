#!/usr/bin/env python3
"""Isolate all/leg/wheel relaxation against identical sealed 1418 weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-variants', type=Path, nargs='+', required=True)
    parser.add_argument('--comparison-indices', type=int, nargs='+', required=True)
    parser.add_argument('--inference-backend', default='torch_cuda')
    args, _ = parser.parse_known_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if (args.comparison_indices != [0, 2, 3, 4, 5]
            or len(args.candidate_variants) != 3
            or any(sha(p) != sha(args.candidate) for p in args.candidate_variants)
            or args.inference_backend != 'torch_cuda'):
        raise ValueError('Require baseline0 plus four same-checkpoint CUDA arms 2..5')
    from flat_action_relaxation_probe import ActionRelaxationProbe, VARIANTS
    from probe_flat_stop_heading import seal_before_close
    import wheeled_algo.command_residual as actor_module
    import flat_rmcs_command_reference as reference_module
    import probe_flat_speed_prior as evaluator
    original_mean, original_install = actor_module.mean_from_checkpoint, reference_module.install_reference
    audit = dict(schema='flat-action-relaxation-diagnostic-v1',
        new_optimizer_updates=0, training_or_export_eligible=False,
        candidate_sha256=sha(args.candidate), assignments={},
        variants=VARIANTS, observation_previous_action_indices=list(range(22, 28)),
        policy_hz=200, physics_hz=1000,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT/'scripts/flat_action_relaxation_probe.py')})
    variants = iter(('all', 'control', 'legs', 'wheels'))
    candidate_count = 0
    import torch
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=True)
    expected_state = candidate['actor_state_dict']

    def load(state, source):
        nonlocal candidate_count
        result = original_mean(state, source)
        if (state['infos']['successful_updates_total'] == candidate['infos']['successful_updates_total']
                and state['actor_state_dict'].keys() == expected_state.keys()
                and all(torch.equal(v.cpu(), expected_state[k]) for k, v in state['actor_state_dict'].items())):
            variant = next(variants)
            index = 2 + candidate_count
            candidate_count += 1
            audit['assignments'][str(index)] = variant
            return ActionRelaxationProbe(result, variant)
        return result

    output = args.output.resolve()

    def seal():
        if not output.is_dir():
            return
        report_file = output/'report.json'
        if report_file.exists():
            report = json.loads(report_file.read_text())
            audit['base_report_sha256_before_override_marker'] = sha(report_file)
            report['diagnostic_inference_override'] = dict(
                schema=audit['schema'], assignments=audit['assignments'],
                receipt='action_relaxation_override.json', training_or_export_eligible=False)
            # Preserve the native report but deny direct campaign admission.
            report['inference_backend'] = 'diagnostic_action_relaxation_cuda'
            report_file.write_text(json.dumps(report, indent=2)+'\n')
            audit['native_report_sha256'] = sha(report_file)
        (output/'action_relaxation_override.json').write_text(json.dumps(audit, indent=2)+'\n')

    def install(env, contract, **kwargs):
        result = original_install(env, contract, **kwargs)
        if candidate_count != 4:
            raise ValueError('Not all four sealed candidate arms were loaded')
        seal_before_close(env, seal)
        return result

    actor_module.mean_from_checkpoint = load
    reference_module.install_reference = install
    try:
        return evaluator.main()
    finally:
        actor_module.mean_from_checkpoint = original_mean
        reference_module.install_reference = original_install


if __name__ == '__main__':
    raise SystemExit(main())
