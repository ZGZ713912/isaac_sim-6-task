"""Explicit diagnostic interventions applied after the ordinary seeded reset.

No reset draw, random generator, noise amplitude, delay, body property or
policy is changed. These counterfactuals cannot qualify a deployment export.
"""


def neutralize_evaluation_bias(env, component):
    if component not in (None, "policy_encoder", "policy_gyro", "pd_encoder"):
        raise ValueError("Unknown evaluation bias component")
    if component is None:
        return None
    if component == "pd_encoder":
        values = env.getup_feedback.bias
    elif component == "policy_encoder":
        values = env.domain.bias[:, 6:10]
    else:
        # Gyro observations use the existing half-rad/s scale.
        values = env.domain.bias[:, :3]
    before = float(values.abs().max())
    values.zero_()
    return dict(component=component, before_max_abs=before, after_max_abs=0.,
                application="after_seeded_reset_before_first_policy_observation",
                scope="evaluation_only_counterfactual_not_deployment_qualification")
