"""Explicit fresh-network initialization for the absolute-encoder prone task."""
from copy import deepcopy

import torch

from .scratch_standing import initialize_fresh, tensor_hash


CONTRACT = dict(schema="wheel-leg-prone-learning-interface-v1",
    observation_normalization="fixed_policy35_empirical_critic_suffix",
    fresh_actor_initialization="trainable_pose_hold_v1",
    action_semantics="absolute_output_shaft_encoder_position",
    leg_position_scale_rad=.25, all_parameters_trainable=True,
    runtime_encoder_skip=False)


def validate(config):
    if config != CONTRACT:
        raise ValueError("Use the complete versioned prone learning interface")
    return deepcopy(config)


def initialize_prone_fresh(algorithm, nominal_action, plan):
    config = plan.get("learning_interface")
    if config is None:
        return initialize_fresh(algorithm, nominal_action, initial_std=plan["initial_std"])
    validate(config)
    from .prone_observation import PhysicalScaleProneMotionModel
    if not isinstance(algorithm.actor, PhysicalScaleProneMotionModel):
        raise ValueError("Encoder initialization requires the fixed physical policy35 interface")
    layers = [x for x in algorithm.actor.mlp.modules() if isinstance(x, torch.nn.Linear)]
    # RSL reuses one activation object at several Sequential positions;
    # modules() deduplicates it, so inspect the actual forward sequence.
    activations = [x for x in algorithm.actor.mlp if isinstance(x, torch.nn.ELU)]
    if (len(layers) < 3 or len(activations) != len(layers)-1
            or layers[0].out_features < 16 or any(x.out_features < 4 for x in layers[1:-1])
            or layers[-1].out_features != 6):
        raise ValueError("Pose-hold initialization needs two ELU hidden layers, first16 and later4 units")
    receipt = initialize_fresh(algorithm, nominal_action, initial_std=plan["initial_std"])
    with torch.no_grad():
        # A flat region around the standing bias retains position stiffness.
        # Far from it, initially hold the observed pose instead of commanding
        # a ~2rad step to standing. ELU combinations implement this *only in
        # initial trainable weights*, with no runtime encoder skip or phase.
        # H(z)=[2*ELU(k(z-b))-ELU(k(z-c))+1]/k, c=2b-1/k.
        # H(z)-H(-z) is near0 at standing and approaches z for |z|>c.
        k, b = 20., .25
        c = 2.*b-1./k
        center = nominal_action[:4]*.25
        first = layers[0]
        first.weight[:16].zero_()
        for joint in range(4):
            rows = slice(4*joint, 4*joint+4)
            first.weight[rows, 10+joint] = first.weight.new_tensor([k, k, -k, -k])
            first.bias[rows] = torch.stack((-k*(center[joint]+b), -k*(center[joint]+c),
                k*(center[joint]-b), k*(center[joint]-c)))
        second = layers[1]
        second.weight[:4].zero_()
        second.bias[:4].fill_(4.)
        for joint in range(4):
            second.weight[joint, 4*joint:4*joint+4] = second.weight.new_tensor([2., -1., -2., 1.])/k
        for layer in layers[2:-1]:
            layer.weight[:4].zero_()
            layer.bias[:4].zero_()
            layer.weight[:4, :4].copy_(torch.eye(4, device=layer.weight.device))
        output = layers[-1]
        output.weight[:4].zero_()
        output.bias[:4].copy_(nominal_action[:4]-16.)
        output.weight[:4, :4].copy_(4.*torch.eye(4, device=output.weight.device))
        output.weight[4:, :4].zero_()
    receipt.update(schema="prone-absolute-encoder-initialization-v2",
        actor_tensor_sha256=tensor_hash(algorithm.actor.state_dict()),
        output_bias_source="fresh trainable pose hold with standing stiffness; no policy, teacher or scripted targets",
        output_bias_policy_order=output.bias.cpu().tolist(),
        learning_interface=deepcopy(config), initial_leg_mean="standing-centered flat region, distant encoder pose hold",
        initial_standing_region_rad=b, initial_pose_transition_end_rad=c,
        runtime_action_decoder_changed=False, normalizers_loaded=False)
    return receipt
