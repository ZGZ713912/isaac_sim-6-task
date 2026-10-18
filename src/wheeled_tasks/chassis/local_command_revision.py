"""One exact local v5-2176 -> v6 planar rehearsal with no learning-state reset."""
from copy import deepcopy
import json
from pathlib import Path

from . import local_scratch as local


SOURCE_PLAN_SHA = "877d2d6d6c3a5a3b945a3d38b28f53d323069045006ebaed7f8fba2e7bfc5bc9"
REVISION = dict(
    schema="v6-local-command-revision-v1", source_plan_file=local.BIRTH_RAMP_PLAN_FILE,
    source_plan_sha256=SOURCE_PLAN_SHA,
    checkpoint_file="/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot011/model_final.pt",
    checkpoint_sha256="872bc86d2007e5a664507278f4ec6c35c4acbe4a0e936cf306b14768c5710034",
    source_contract_sha256="65abce0937c0e7222935674ae89642f8489f19622f873b7b65617116b33d5a15",
    completion_sha256="78bd12bb7f41073a6ee5e9ceec6d13545b1ac5a096871551253c8d61daaa79e0",
    seal_sha256="2ca7546f137707f89e4e4949ba9c2e3872a993aaba72b84b0c048faeb02df080",
    learning_updates=2176, prior_consumption=1088, lineage_update_ceiling=18912,
    first_block_update_ceiling=128, training_transitions=106954752,
    evidence=[
        dict(file="reports/local_training_20261017/pilot011.json",
             sha256="38b51f6d3385e81eb44694f5970ab993ab840eb7599cafb867448c3ee96f7687"),
        dict(file="reports/local_training_20261017/credit_diagnosis001.json",
             sha256="68830e57cacea96b98719590aee95ddef18a4211d78507c645632aa96aa8382a"),
        dict(file="docs/V6_LOCAL_COMMAND_REHEARSAL_PLAN_20261018.md",
             sha256="41ffcc2eb1ca149b1779cbec6e28cc3692798b5f5dc7317e0f6b98084d2f749a")])


def validate_plan(plan, root):
    root = Path(root)
    if local.digest(root / local.BIRTH_RAMP_PLAN_FILE) != SOURCE_PLAN_SHA:
        raise ValueError("Local command revision source plan changed")
    expected = deepcopy(local.read_plan(root, local.BIRTH_RAMP_PLAN_FILE))
    expected.update(initialization="full_state_continuation_from_v5_2176",
                    local_command_revision=deepcopy(REVISION))
    if plan != expected:
        raise ValueError("Local command revision differs from its exact v5 derivation")
    for row in REVISION["evidence"]:
        if local.digest(root / row["file"]) != row["sha256"]:
            raise ValueError("Local command revision evidence changed: " + row["file"])


def validate_contracts(source, target, root):
    if (source.get("local_scratch", {}).get("plan_file") != local.BIRTH_RAMP_PLAN_FILE
            or target.get("local_scratch", {}).get("plan_file") != local.REHEARSAL_PLAN_FILE):
        raise ValueError("Local command revision requires exact local v5 -> v6 contracts")
    local.validate_config(target, root)
    _, original = local.build_bundle(root, Path(source["design_preflight"]["file"]).parent,
        source["target_num_envs"], plan_file=local.BIRTH_RAMP_PLAN_FILE)
    if original["worker"] != source:
        raise ValueError("Local command revision source is not its independent v5 derivation")
    allowed = {"locomotion_difficulty", "local_scratch", "design_preflight", "budget_spec_file", "budget_spec_sha256"}
    if {k: v for k, v in source.items() if k not in allowed} != {
            k: v for k, v in target.items() if k not in allowed}:
        raise ValueError("Local command revision changes an undeclared contract field")
    from .locomotion_difficulty import difficulty_settings, local_rehearsal_settings
    if (source["locomotion_difficulty"] != difficulty_settings(0)
            or target["locomotion_difficulty"] != local_rehearsal_settings(0)
            or source["target_num_envs"] != 512):
        raise ValueError("Local command revision changes clocks, heights or population")
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
        raise ValueError("Local command revision parent differs from named finalized v5-2176")
    completion = json.loads((checkpoint.parent / "completion.json").read_text())
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    if (seal.get("global_actor_update") != count
            or completion["parent_training_transitions"] + completion["successful_updates"]
                * source["target_num_envs"] * source["num_steps_per_env"] != REVISION["training_transitions"]):
        raise ValueError("Local command revision parent clocks changed")
    result.update(parent_updates=count, inherited_consumption=REVISION["prior_consumption"],
        total_prior_consumption=count + REVISION["prior_consumption"],
        lineage_update_ceiling=REVISION["lineage_update_ceiling"])
    return result
