"""Fresh Gaussian initialization; native sampling and PPO remain unchanged."""
import math

POLICY_AXES = ("left_leg1", "left_leg2", "right_leg1", "right_leg2", "left_wheel", "right_wheel")


def validate_policy_std(values):
    if (not isinstance(values, list) or len(values) != 6
            or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError("Initial policy std requires six finite positive values in policy order")
    return [float(v) for v in values]


def initialize_policy_std(algorithm, values):
    """Copy into the existing parameter before any Adam step or checkpoint load.

    RSL-RL's public init_std argument is scalar. Preserve the parameter object
    registered with Adam, all network weights, and RNG. Never apply this to an
    already used optimizer or after restoring a checkpoint.
    """
    import torch
    from rsl_rl.modules import GaussianDistribution
    values = validate_policy_std(values)
    distribution = algorithm.actor.distribution
    optimizer = algorithm.optimizer
    if type(distribution) is not GaussianDistribution or distribution.std_type != "scalar":
        raise ValueError("Axis initialization requires the native scalar Gaussian parameterization")
    parameter = distribution.std_param
    if (optimizer.state or parameter.shape != (6,) or not parameter.requires_grad
            or not any(p is parameter for g in optimizer.param_groups for p in g["params"])):
        raise ValueError("Axis initialization requires an unused optimizer owning the learnable std")
    if any(not distribution.std_range[0] <= v <= distribution.std_range[1] for v in values):
        raise ValueError("Initial policy std would be clamped by the native distribution")
    with torch.no_grad():
        parameter.copy_(parameter.new_tensor(values))
    return parameter.detach().cpu().tolist()


def install_axis_std_logging(logger):
    """Use the original RSL-RL writer, native update step and action-std values."""
    original = logger.log

    def log(*args, **kwargs):
        result = original(*args, **kwargs)
        if logger.writer is not None:
            std = kwargs["action_std"].detach().reshape(-1, 6).mean(0).cpu().tolist()
            for name, value in zip(POLICY_AXES, std):
                logger.writer.add_scalar("Policy/std/" + name, value, kwargs["it"])
        return result

    logger.log = log
