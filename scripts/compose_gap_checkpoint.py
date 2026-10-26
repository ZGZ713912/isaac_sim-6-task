#!/usr/bin/env python3
"""Compose named learned heads and their Adam state for a measured ablation.

No update is performed. The source clock and all consumed work are retained;
the donor ancestry is recorded explicitly instead of claiming a linear run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch

from export_robust_getup import load_mean
from train_robust_getup import digest, write_json


HEADS = {
    "quiet": ("quiet.", "quiet_low.", "quiet_high.", "quiet_velocity."),
    "motion": ("arc.",),
    "recovery": ("recovery.",),
}


def actor_parameter_names(mean, state):
    names = ["mlp.log_std"] + [
        "mlp.mean." + name for name, parameter in mean.named_parameters()
        if parameter.requires_grad
    ]
    groups = state["optimizer_state_dict"]["param_groups"]
    if len(groups) != 2 or [g["name"] for g in groups] != ["actor", "critic"]:
        raise ValueError("Require the authenticated two-group optimizer")
    if len(names) != len(groups[0]["params"]):
        raise ValueError("Actor parameter registration differs from the optimizer")
    result = dict(zip(names, groups[0]["params"], strict=True))
    for name, index in result.items():
        moment = state["optimizer_state_dict"]["state"].get(index)
        if moment is not None and any(
            moment[key].shape != state["actor_state_dict"][name].shape
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("Adam shape does not match " + name)
    return result


def compose(source, donor, output, heads, experiment_consumed):
    if isinstance(experiment_consumed, bool) or not isinstance(experiment_consumed, int) or experiment_consumed < 0:
        raise ValueError("Composition requires the actual nonnegative experiment consumption")
    source, donor, output = [Path(p).resolve() for p in (source, donor, output)]
    if output.exists():
        raise ValueError("Composition never overwrites an existing checkpoint")
    a, ai = load_mean(source)
    b, bi = load_mean(donor)
    # Module flags carry behavior beyond tensors. All actor-side settings must
    # match, while training severity, commands and reward settings may differ.
    keys = ("leg_action_clip", "wheel_action_clip", "quiet_actor_alpha",
            "quiet_wheel_bound", "gap_entry_handoff_version",
            "gap_upright_target_limit_version", "standing_position_observation",
            "quiet_high_leg_prediction_s", "quiet_velocity_feature_gain")
    if any(ai["settings"].get(k) != bi["settings"].get(k) for k in keys):
        raise ValueError("Donor and source actor contracts differ")
    left, right = [torch.load(p, map_location="cpu", weights_only=False) for p in (source, donor)]
    lm, rm = actor_parameter_names(a, left), actor_parameter_names(b, right)
    if lm.keys() != rm.keys() or left["actor_state_dict"].keys() != right["actor_state_dict"].keys():
        raise ValueError("Donor and source parameter names differ")
    result = deepcopy(left)
    prefixes = tuple("mlp.mean." + prefix for head in heads for prefix in HEADS[head])
    restored = [name for name in left["actor_state_dict"] if name.startswith(prefixes)]
    if not restored:
        raise ValueError("No learned head selected")
    for name in restored:
        result["actor_state_dict"][name] = right["actor_state_dict"][name].clone()
        if name in lm:
            target, origin = lm[name], rm[name]
            result["optimizer_state_dict"]["state"].pop(target, None)
            if origin in right["optimizer_state_dict"]["state"]:
                result["optimizer_state_dict"]["state"][target] = deepcopy(
                    right["optimizer_state_dict"]["state"][origin])
    ancestry = dict(source=str(source), source_sha256=digest(source),
                    source_clock=ai["completed_updates"], donor=str(donor),
                    donor_sha256=digest(donor), donor_clock=bi["completed_updates"],
                    restored_heads=list(heads), restored_tensors=restored,
                    optimizer="Donor Adam moments and steps only for restored parameters",
                    new_optimizer_updates=0, linear_ancestry=False)
    result["infos"]["head_composition"] = ancestry
    result["infos"]["hardware_ready"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    verified, _ = load_mean(output)
    for name, tensor in verified.state_dict().items():
        full = "mlp.mean." + name
        expected = b.state_dict()[name] if full in restored else a.state_dict()[name]
        if not torch.equal(tensor, expected):
            raise ValueError("Composed tensor differs: " + name)
    write_json(output.with_suffix(".json"), dict(
        schema="wheel-leg-head-composition-v1", **ancestry,
        checkpoint=str(output), checkpoint_sha256=digest(output),
        preserved="Other actor tensors, critic, RNG and source clock unchanged",
        actual_experiment_consumed=experiment_consumed, behavior_qualified=False, hardware_ready=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=tuple(HEADS), required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    args = parser.parse_args()
    print(compose(args.source, args.donor, args.output, args.heads, args.experiment_consumed))


if __name__ == "__main__":
    main()
