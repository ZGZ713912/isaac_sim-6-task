"""Diagnostic removal of later leg feedback at the already-trained height.

This never becomes a training policy, controller fallback, or export. Each
lane's four learned targets are held at a declared decision boundary; both
wheel outputs still come from one native actor call on each current frame.
"""
import torch


class HoldInitialLegTargetProbe(torch.nn.Module):
    def __init__(self, actor, *, hold_after_seconds=0.):
        super().__init__()
        if isinstance(hold_after_seconds, bool) or hold_after_seconds not in (0., 3.):
            raise ValueError("Declare a0s or3s leg-hold boundary")
        self.actor = actor
        self.hold_after_seconds = float(hold_after_seconds)
        self.capture_call = round(self.hold_after_seconds / .005)
        self.initial_legs = None
        self.selected = None
        self.calls = 0
        self.native_actions = []
        self.wheel_error = 0.

    def forward(self, observation):
        frame = observation["policy"]
        if frame.ndim != 2 or frame.shape[-1] != 35 or len(frame) < 21:
            raise ValueError("Leg-hold diagnosis requires the original21-case35D evaluation")
        native = self.actor(observation)
        if native.shape != (len(frame), 6) or not torch.isfinite(native).all():
            raise ValueError("Need one finite six-action native actor result")
        selected = torch.isclose(frame[:, 3], frame.new_tensor(.305 * 5.), rtol=0., atol=1e-6)
        if self.calls == 0:
            if selected[:21].nonzero().flatten().tolist() != list(range(7, 14)):
                raise ValueError("Expected the seven trained-height lanes at indices7..13")
            self.selected = selected.detach().clone()
        elif selected.shape != self.selected.shape or not torch.equal(selected, self.selected):
            raise ValueError("Fixed-case height population changed during the diagnostic")
        if self.calls == self.capture_call:
            self.initial_legs = native[:, :4].detach().clone()
        output = native.clone()
        if self.initial_legs is not None:
            output[self.selected, :4] = self.initial_legs[self.selected]
        if not torch.equal(output[:, 4:], native[:, 4:]):
            raise RuntimeError("Leg-hold diagnostic altered a native wheel output")
        self.native_actions.append(native[:21].detach().cpu().clone())
        self.calls += 1
        return output

    def finish(self, output, writer, completed, digest):
        import numpy as np
        from pathlib import Path

        if self.calls != 1600 or len(self.native_actions) != self.calls or self.initial_legs is None:
            raise ValueError("Require every actor call in the original8s/200Hz evaluation")
        path = Path(output) / "unmodified_actor_actions.npz"
        if path.exists():
            raise ValueError("Preserve the earlier diagnostic trace")
        np.savez_compressed(path, action_policy=torch.stack(self.native_actions).numpy(),
            held_leg_targets_policy=self.initial_legs[:21].cpu().numpy(),
            selected=self.selected[:21].cpu().numpy())
        writer.add_scalar("LegHoldProbe/actor_calls", self.calls, completed)
        writer.add_scalar("LegHoldProbe/selected_original_cases", int(self.selected[:21].sum()), completed)
        writer.add_scalar("LegHoldProbe/native_wheel_output_max_error", self.wheel_error, completed)
        writer.add_scalar("LegHoldProbe/hold_after_s", self.hold_after_seconds, completed)
        writer.add_scalar("LegHoldProbe/native_prefix_policy_frames", self.capture_call + 1, completed)
        writer.flush()
        return dict(schema="scratch-hold-leg-target-counterfactual-v2", actor_calls=self.calls,
            selected_height_m=.305, selected_case_indices=list(range(7, 14)),
            held_after_s=self.hold_after_seconds, native_prefix_policy_frames=self.capture_call + 1,
            first_modified_policy_time_s=(self.capture_call + 1) * .005,
            held_after="native actor action at the declared boundary; all earlier actions stay native",
            leg_policy_columns=[0, 1, 2, 3], wheel_policy_columns=[4, 5],
            wheel_output="one unmodified native actor result on the current35D observation each step",
            original_actions=dict(path=str(path), sha256=digest(path)),
            first_action_unchanged=True, other_heights_policy_unchanged=True,
            new_PPO=0, optimizer_constructed=False, learned_policy=False,
            adoption_by_training=False, export_forbidden=True, behavior_qualification=False,
            scope="Read-only causal diagnostic of subsequent leg feedback. Mechanical protection, physics, "
                  "reset and strict100% contact acceptance stay unchanged. No six-motor policy qualification.")
