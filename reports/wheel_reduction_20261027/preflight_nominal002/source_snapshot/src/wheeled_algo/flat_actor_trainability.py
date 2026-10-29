"""Explicit parameter freezing for continuation of the recovered 35D actor."""


OUTPUT_LAYER = frozenset(("mlp.6.weight", "mlp.6.bias"))


def configure_flat_actor_training(actor, *, enabled, output_layer_only=False):
    """Keep the full actor and Adam state; select which gradients may update it."""
    parameters = dict(actor.named_parameters())
    if output_layer_only:
        if (not OUTPUT_LAYER <= parameters.keys() or parameters["mlp.6.weight"].shape != (6, 64)
                or parameters["mlp.6.bias"].shape != (6,)):
            raise ValueError("Output-layer continuation requires the recovered 35-256-128-64-6 actor")
        selected = OUTPUT_LAYER if enabled else frozenset()
    else:
        selected = parameters.keys() if enabled else frozenset()
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected)
        if name not in selected:
            parameter.grad = None
    return dict(trainable_names=sorted(selected), trainable_parameters=sum(parameters[n].numel() for n in selected),
        total_parameters=sum(p.numel() for p in parameters.values()),
        exploration_std_frozen=output_layer_only or not enabled)
