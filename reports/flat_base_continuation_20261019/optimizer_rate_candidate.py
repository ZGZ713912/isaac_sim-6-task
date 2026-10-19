"""Unadopted rate-contract helper for a declared future flat PPO migration.

The caller authenticates the plan revision and restores the full parent first.
Only the actor group's learning rate may then change; moments are retained.
This helper is not connected to the running frozen trainer.
"""
from math import isfinite


def apply_flat_optimizer_rates(optimizer, actor_lr, critic_lr, *, parent_rates=None):
    def rate(value):
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError("Flat optimizer learning rates must be finite positive numbers")
        return float(value)

    target = (rate(actor_lr), rate(critic_lr))
    groups = optimizer.param_groups
    if len(groups) != 2 or [g.get("name") for g in groups] != ["actor", "critic"]:
        raise ValueError("Flat Adam requires ordered actor and critic parameter groups")
    before = tuple(rate(g["lr"]) for g in groups)
    if parent_rates is None:
        if before != target:
            raise ValueError("Loaded Adam learning rates differ from the unchanged plan")
    else:
        if len(parent_rates) != 2 or before != tuple(rate(v) for v in parent_rates):
            raise ValueError("Loaded Adam does not match the declared parent learning rates")
        if before[1] != target[1]:
            raise ValueError("This migration only permits a declared actor learning-rate change")
    # Complete validation before touching the optimizer. No state, moment,
    # step count, parameter ownership or other hyperparameter is replaced.
    groups[0]["lr"] = target[0]
    return dict(schema="flat-optimizer-rate-contract-v1",
        before=dict(zip(("actor", "critic"), before, strict=True)),
        after=dict(zip(("actor", "critic"), target, strict=True)),
        declared_actor_rate_change=before[0] != target[0],
        moments_steps_and_other_hyperparameters="retained")
