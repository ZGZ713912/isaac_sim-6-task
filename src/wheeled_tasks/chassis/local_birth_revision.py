"""One exact local v4-1024 -> v5 birth revision, preserving all learning state."""
from copy import deepcopy
import json
from pathlib import Path

from . import local_scratch as local


SOURCE_PLAN_SHA = "dca8aba16f8b54c89c4e85441ea95f1e76e9746d14317f7373e323ea8f71dd2c"
REVISION = dict(
    schema="v6-local-birth-revision-v1", source_plan_file=local.PLAN_FILE,
    source_plan_sha256=SOURCE_PLAN_SHA,
    checkpoint_file="/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot008/model_final.pt",
    checkpoint_sha256="f57d05f7f32c9a89ed400b7b2350b71879968175faa12a190858eb59cd8b9c0e",
    source_contract_sha256="bf692fd6c861820a13dac2c94494aea1a0a216dddd4bbfd575f0388171a8e699",
    completion_sha256="355e0e49a50efacae425c79414559aca6770470c934a21a2c3eef09392b44743",
    seal_sha256="e285e1f7d4393de1c0d26e2c807e3caf675eefc25165a99de6d1bf51b094c7c5",
    learning_updates=1024, prior_consumption=1088, lineage_update_ceiling=18912,
    first_block_update_ceiling=128, training_transitions=50331648,
    evidence=[
        dict(file="reports/local_training_20261017/pilot008.json",
             sha256="9a161097a23c5f4b5f54e4938efc2f2273c4ea74fe1ce5a1fba98a0e67a2298d"),
        dict(file="reports/local_training_20261017/birth_diagnosis002.json",
             sha256="1040e0d370114029cabe2dfb21eae18c931df8af0aa1c0957057a63179b18e78"),
        dict(file="docs/V6_LOCAL_BIRTH_RAMP_PLAN_20261017.md",
             sha256="c054d52005b986d0649081036e48e5248d5472003722eac218b01be25fc5283b")])


def validate_plan(plan, root):
    root = Path(root)
    if local.digest(root / local.PLAN_FILE) != SOURCE_PLAN_SHA:
        raise ValueError("Local birth revision source plan changed")
    expected = json.loads((root / local.PLAN_FILE).read_text())
    expected.update(initialization="full_state_continuation_from_v4_1024", local_birth_revision=deepcopy(REVISION))
    if plan != expected:
        raise ValueError("Local birth revision plan/budget differs from its exact v4 derivation")
    for row in REVISION["evidence"]:
        if local.digest(root / row["file"]) != row["sha256"]:
            raise ValueError("Local birth revision evidence changed: " + row["file"])


def validate_contracts(source, target, root):
    """Authenticate each independent design before allowing the precise diff."""
    if (source.get("local_scratch", {}).get("plan_file") != local.PLAN_FILE
            or target.get("local_scratch", {}).get("plan_file") != local.BIRTH_RAMP_PLAN_FILE):
        raise ValueError("Local birth revision requires exact local v4 -> v5 contracts")
    local.validate_config(target, root)
    _, original = local.build_bundle(root, Path(source["design_preflight"]["file"]).parent,
        source["target_num_envs"], plan_file=local.PLAN_FILE)
    if original["worker"] != source:
        raise ValueError("Local birth revision source is not its independent v4 derivation")
    allowed = {"nonstandard_start", "local_scratch", "design_preflight", "budget_spec_file", "budget_spec_sha256"}
    if {k: v for k, v in source.items() if k not in allowed} != {
            k: v for k, v in target.items() if k not in allowed}:
        raise ValueError("Local birth revision changes an undeclared contract field")
    from .nonstandard_start import nonstandard_start_settings, ramped_start_settings
    if (source["nonstandard_start"] != nonstandard_start_settings(0)
            or target["nonstandard_start"] != ramped_start_settings(0)
            or source["target_num_envs"] != 512):
        raise ValueError("Local birth revision changes pose bank, schedule, capture or population")
    return dict(schema=REVISION["schema"], changed_contract_fields=sorted(allowed),
        consumption="all_prior_updates_preserved_no_refund", additional_critic_warmup=0)


def validate_parent(checkpoint, target, root):
    from .full_curriculum import checkpoint_contract_path
    checkpoint = Path(checkpoint).resolve(strict=True)
    source_path = checkpoint_contract_path(checkpoint)
    source = json.loads(source_path.read_text())
    result = validate_contracts(source, target, root)
    count = local.validate_parent(checkpoint, source)
    if (checkpoint != Path(REVISION["checkpoint_file"])
            or local.digest(checkpoint) != REVISION["checkpoint_sha256"]
            or local.digest(source_path) != REVISION["source_contract_sha256"]
            or local.digest(checkpoint.parent / "completion.json") != REVISION["completion_sha256"]
            or local.digest(checkpoint.with_suffix(".checkpoint.json")) != REVISION["seal_sha256"]
            or count != REVISION["learning_updates"]):
        raise ValueError("Local birth revision parent differs from its named finalized v4-1024 checkpoint")
    completion = json.loads((checkpoint.parent / "completion.json").read_text())
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    if (seal.get("global_actor_update") != count
            or completion["parent_training_transitions"] + completion["successful_updates"]
                * source["target_num_envs"] * source["num_steps_per_env"] != REVISION["training_transitions"]):
        raise ValueError("Local birth revision parent clocks changed")
    result.update(parent_updates=count, inherited_consumption=REVISION["prior_consumption"],
        total_prior_consumption=count + REVISION["prior_consumption"],
        lineage_update_ceiling=REVISION["lineage_update_ceiling"])
    return result
