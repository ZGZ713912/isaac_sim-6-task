"""Observe one native RSL-RL 5.5.1 PPO update without replacing its losses.

No forward pass, sampling, backward pass, clipping, or optimizer call is added.
The class-level generator hook also sees RecoveryRolloutMask's compact storage;
binding a generator to the original storage would silently bypass that mask.
Only one observer may be installed in this single-learner diagnostic process.
"""

from importlib.metadata import version
import math

import torch


def validate_local_audit_request(enabled, *, local, resume, updates, scene, revision):
    if enabled and (not local or not resume or updates != 1 or scene or revision):
        raise ValueError("PPO audit requires one same-contract local full-resume update, without a scene probe or revision")


class NativePPOAudit:
    """Read native minibatches, gradients before/after clipping, and Adam steps."""

    _active = None

    def __init__(self, algorithm):
        from rsl_rl.algorithms import PPO
        from rsl_rl.storage import RolloutStorage

        if NativePPOAudit._active is not None:
            raise ValueError("Only one native PPO audit may be installed")
        if (version("rsl-rl-lib") != "5.5.1" or type(algorithm) is not PPO
                or type(algorithm.storage) is not RolloutStorage
                or algorithm.actor.is_recurrent or algorithm.critic.is_recurrent
                or algorithm.symmetry or algorithm.rnd or algorithm.is_multi_gpu
                or algorithm.use_mixed_precision or type(algorithm.optimizer) is not torch.optim.Adam):
            raise ValueError("PPO audit supports only the pinned feedforward native Adam path")
        self.algorithm = algorithm
        self.rows = []
        self.closed = False
        self._batch = None
        self._values = None
        self._grads = {"actor": {}, "critic": {}}
        self._storage_class = RolloutStorage
        self._original_generator = RolloutStorage.mini_batch_generator
        self._original_step = algorithm.optimizer.step
        self._previous_step_attribute = algorithm.optimizer.__dict__.get("step")
        self._handles = []
        NativePPOAudit._active = self

        def batches(storage, *args, **kwargs):
            for batch in self._original_generator(storage, *args, **kwargs):
                if storage is not algorithm.storage:
                    yield batch
                    continue
                self._batch, self._values = batch, None
                self._grads = {"actor": {}, "critic": {}}
                try:
                    yield batch
                finally:
                    self._batch, self._values = None, None

        self._generator_hook = batches
        RolloutStorage.mini_batch_generator = batches
        self._handles.append(algorithm.critic.register_forward_hook(self._capture_values))
        for group in ("actor", "critic"):
            for name, parameter in getattr(algorithm, group).named_parameters():
                if parameter.requires_grad:
                    self._handles.append(parameter.register_hook(self._gradient_hook(group, name)))
        algorithm.optimizer.step = self._step

    def _capture_values(self, module, inputs, output):
        if self._batch is not None:
            self._values = output.detach()

    def _gradient_hook(self, group, name):
        def observe(gradient):
            if self._batch is not None:
                self._grads[group][name] = gradient.detach().square().sum()
            # Returning None preserves the exact gradient supplied by autograd.
        return observe

    @staticmethod
    def _norm(tensors):
        values = list(tensors)
        return math.sqrt(sum(float(value) for value in values))

    def _adam_steps(self):
        values = [float(state["step"]) for state in self.algorithm.optimizer.state.values() if "step" in state]
        return (min(values), max(values)) if values else (0., 0.)

    def _step(self, *args, **kwargs):
        if self._batch is None or self._values is None:
            raise RuntimeError("Native optimizer step lacks its observed minibatch/critic output")
        alg, batch, values = self.algorithm, self._batch, self._values
        with torch.no_grad():
            # These native distribution queries only read the distribution from
            # the existing PPO forward pass. They never draw another action.
            params = alg.actor.output_distribution_params
            if len(params) != 2:
                raise RuntimeError("PPO audit requires native mean/std Gaussian parameters")
            log_prob = alg.actor.get_output_log_prob(batch.actions)
            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantage = batch.advantages.squeeze(-1)
            clipped_ratio = ratio.clamp(1 - alg.clip_param, 1 + alg.clip_param)
            clipped_value = batch.values + (values - batch.values).clamp(-alg.clip_param, alg.clip_param)
            mse = (values - batch.returns).square()
            clipped_mse = (clipped_value - batch.returns).square()
            clipped_value_loss = (torch.maximum(mse, clipped_mse) if alg.use_clipped_value_loss else mse)
            flat_value = ((values - batch.values).abs() > alg.clip_param) & (clipped_mse > mse)
            row = {
                "optimizer_index": len(self.rows), "samples": int(batch.actions.shape[0]),
                "optimizer_committed": False,
                "ratio_clip_fraction": float(((ratio < 1 - alg.clip_param) | (ratio > 1 + alg.clip_param)).float().mean()),
                "surrogate_clipped_branch_fraction": float((-advantage * clipped_ratio > -advantage * ratio).float().mean()),
                "ratio_mean": float(ratio.mean()),
                "old_to_new_kl_nats": float(alg.actor.get_kl_divergence(batch.old_distribution_params, params).mean()),
                "value_mse": float(mse.mean()), "native_value_loss": float(clipped_value_loss.mean()),
                "old_value_mse": float((batch.values - batch.returns).square().mean()),
                "value_clipped_flat_fraction": float(flat_value.float().mean()) if alg.use_clipped_value_loss else 0.,
                "return_mean": float(batch.returns.mean()), "return_std": float(batch.returns.std()),
                "advantage_std": float(advantage.std()), "entropy_nats": float(alg.actor.output_entropy.mean()),
                "learning_rate": float(alg.optimizer.param_groups[0]["lr"]),
                "adam_step_before_min": self._adam_steps()[0], "adam_step_before_max": self._adam_steps()[1],
            }
            for axis, std in enumerate(params[1].mean(dim=0).cpu().tolist()):
                row[f"std_policy_axis{axis}"] = std
            before = {}
            for group in ("actor", "critic"):
                parameters = list(getattr(alg, group).parameters())
                row[f"{group}_gradient_l2_before_clip"] = self._norm(self._grads[group].values())
                row[f"{group}_gradient_l2_after_clip"] = self._norm(
                    parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None)
                before[group] = [parameter.detach().clone() for parameter in parameters]
        self.rows.append(row)
        result = self._original_step(*args, **kwargs)  # Exactly the original call, once.
        with torch.no_grad():
            row["optimizer_committed"] = True
            row["adam_step_after_min"], row["adam_step_after_max"] = self._adam_steps()
            for group in ("actor", "critic"):
                row[f"{group}_parameter_delta_l2"] = self._norm(
                    (parameter.detach() - previous).square().sum()
                    for parameter, previous in zip(getattr(alg, group).parameters(), before[group]))
        return result

    def report(self):
        numeric = [key for key in self.rows[0] if key != "optimizer_committed"] if self.rows else []
        summary = {f"{key}/{aggregation}": operation([row[key] for row in self.rows if key in row])
                   for key in numeric for aggregation, operation in
                   (("mean", lambda values: sum(values) / len(values)), ("min", min), ("max", max))}
        finite = all(math.isfinite(value) for row in self.rows for value in row.values())
        # Preserve an invalid diagnostic in JSON without masking the trainer's
        # original failure/consumption receipt during its finally block.
        safe_rows = [{key: value if math.isfinite(value) else None for key, value in row.items()}
                     for row in self.rows]
        safe_summary = {key: value if math.isfinite(value) else None for key, value in summary.items()}
        return {"schema": "native-ppo-audit-v1", "optimizer_calls": len(self.rows),
                "committed_optimizer_calls": sum(row["optimizer_committed"] for row in self.rows),
                "all_recorded_values_finite": finite, "closed": self.closed,
                "rows": safe_rows, "summary": safe_summary,
                "units": "fractions in [0,1]; KL/entropy nats; gradients/parameter deltas Euclidean norm; native return/loss units",
                "scope": "Observed native minibatches at optimizer.step, before the actual Adam call. No added forward/sample/backward/step. One rollout is not behavior acceptance or a causal training comparison."}

    def log(self, writer, step):
        report = self.report()
        if writer is not None:
            for key, value in report["summary"].items():
                if value is not None:
                    writer.add_scalar(f"LocalDiagnostics/NativePPOAudit/{key}", value, step)
            writer.add_scalar("LocalDiagnostics/NativePPOAudit/optimizer_calls", report["optimizer_calls"], step)

    def close(self):
        if self.closed:
            return
        self._storage_class.mini_batch_generator = self._original_generator
        optimizer = self.algorithm.optimizer
        if self._previous_step_attribute is None:
            del optimizer.step
        else:
            optimizer.step = self._previous_step_attribute
        for handle in self._handles:
            handle.remove()
        self._batch, self._values = None, None
        self.closed = True
        NativePPOAudit._active = None
