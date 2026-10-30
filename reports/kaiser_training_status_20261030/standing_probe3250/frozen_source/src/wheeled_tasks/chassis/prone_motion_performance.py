"""Optional execution-only acceleration for the unchanged prone-motion task."""
from __future__ import annotations

import torch

from .tensor_cuda_graph import TensorCudaGraph


class RouteReadCache:
    """Scope to a native read epoch and explicit reset generation.

    Masses and the mechanical model are startup-static for this task. Domain
    changes construct a new environment/cache. No Tensor._version is consulted:
    environments can also be constructed inside RSL's inference_mode.
    """
    def __init__(self, env, function):
        self.env, self.function = env, function
        self.key, self.value = None, None

    def __call__(self):
        key = (self.env.root_kinematics.read_epoch,self.env.prone_reset_generation)
        if self.key != key:
            self.value = self.function()
            self.key = key
        return self.value


def install(env, backend):
    if backend != "cuda_graph":
        raise ValueError("Unknown prone-motion performance backend")
    if hasattr(env, "performance_backend"):
        raise RuntimeError("Performance backend already installed")
    with torch.inference_mode():
        encoder = env.robot.data.joint_pos.torch[:, env.ids][:, list(env.v5.LEGS)].clone()
        ff = env.spring_feedforward
        ff.tensor_replay = TensorCudaGraph(ff._evaluate, encoder)
        geometry = env.prone_encoder_geometry
        replay = TensorCudaGraph(geometry, encoder, torch.zeros_like(encoder))
        # Keep the original module alive, including all authenticated buffers.
        env.prone_encoder_geometry = replay
        env.performance_geometry_module = geometry
        spring = env.identified_spring
        compression = env.v5.s0-env.robot.data.joint_pos.torch[:, env.spring_ids]
        if spring is ff:
            spring.force_tensor_replay = TensorCudaGraph(spring._evaluate_force, compression)
        else:
            spring.tensor_replay = TensorCudaGraph(spring._evaluate,compression)
    env.route_state = RouteReadCache(env,env.route_state)
    env.performance_backend = backend
    env.startup_report["performance_backend"] = dict(name=backend,
        captured=["encoder_spring_feedforward", "encoder_leg_geometry","physical_gas_spring"],
        route_cache="native read epoch and explicit reset generation; startup-static model",
        physics_or_optimizer_captured=False, finite_checks_retained=True,
        original_torch_kernels=True, fresh_output_storage=True,
        physics_hz=round(1./env.dt),policy_hz=round(1./env.policy_dt))
