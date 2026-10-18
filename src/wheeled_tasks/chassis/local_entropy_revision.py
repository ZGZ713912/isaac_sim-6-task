"""One same-state local entropy arm, charging all completed control work."""
from copy import deepcopy
import json
from pathlib import Path

from . import local_scratch as local


REVISION = {'schema': 'v6-local-entropy-revision-v1',
 'source_plan_file': 'contracts/v6_local_scratch_200hz_v6.json',
 'source_plan_sha256': 'd39f52c61b6eedacdb3589f5b5d48676ecf9eef9534c63eca7c4e01a387bf23e',
 'checkpoint_file': '/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot013/model_final.pt',
 'checkpoint_sha256': '2dfd8288c7df3e2da99b3ca9064b45fe9ca7b36f334b96a42e339f3174be3552',
 'source_contract_sha256': '693aad6a37ff305f459589e8932990a674676bbdc909d426b2234a5c7777c8ab',
 'completion_sha256': 'a88412786bf44272f7ef861ad6c7832f90648336a9a004ce1588a98b27497640',
 'seal_sha256': '454baf6077297f8feebdc13de22b32262efc4e2425ae655b52104ec7de507ade',
 'learning_updates': 2560,
 'prior_consumption': 1216,
 'lineage_update_ceiling': 18784,
 'first_block_update_ceiling': 128,
 'training_transitions': 125829120,
 'source_entropy_coef': 0.0,
 'target_entropy_coef': 0.0005,
 'control': {'checkpoint_file': '/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot014/model_final.pt',
             'checkpoint_sha256': 'bd5a35112e89596ad1ab98bde1de1933a4f36b0b6253c6ffcaeff67ddcc5177f',
             'completion_file': '/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot014/completion.json',
             'completion_sha256': '101f50df0df88adc1ab0f7b09b24a0d6aed1e3a393d4774efb6515875c14ae53',
             'evaluation_file': '/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/eval014/evaluation.json',
             'evaluation_sha256': '3f71be7976869618db3b13ebd548c1c84c73174521097cf9d75a2368abc90fe6',
             'learning_updates': 2688,
             'additional_consumption': 128},
 'evidence': [{'file': 'reports/local_training_20261017/pilot013.json',
               'sha256': '78fef25a88d89ab8690a2717d245d8bfd3bfbddd8847b4b37df9bc0cf99783e0'},
              {'file': 'reports/local_training_20261017/action_credit_diagnosis001.json',
               'sha256': '2d7240a19e66f1257e1cdf63550fdd401baa04162881ced0f6e4d41fec992544'},
              {'file': 'docs/V6_LOCAL_ENTROPY_AB_PLAN_20261018.md',
               'sha256': '3c8ac17db08e8fdf354f5e9185bd3aef293e99304f96aa5b16635eccb00cc9f5'},
              {'file': 'reports/local_training_20261017/entropy_ab001_plan.json',
               'sha256': '5664d36e15f2e352fa4bca0b6d17e0660eb94608b013aea759bd4fedd8ceca79'},
              {'file': 'reports/local_training_20261017/pilot014.json',
               'sha256': 'ad2f5857d46b1221ed8a8da7f9c4ff6c61b277badf46eaff7474dd920a44e21b'},
              {'file': 'reports/local_training_20261017/entropy_ab001_consumption.json',
               'sha256': 'b932110cd564ea4a9f61451b353f6b4a92e693c11ff5ac486648d397b3655811'}]}



def validate_plan(plan, root):
    root = Path(root)
    if local.digest(root / local.REHEARSAL_PLAN_FILE) != REVISION["source_plan_sha256"]:
        raise ValueError("Local entropy revision source v6 plan changed")
    expected = deepcopy(local.read_plan(root, local.REHEARSAL_PLAN_FILE))
    expected.update(initialization="full_state_same2560_entropy_ab_treatment",
        entropy_coef=REVISION["target_entropy_coef"],
        prior_experiment_consumption=REVISION["prior_consumption"],
        prior_experiment_receipt_file=REVISION["evidence"][-1]["file"],
        prior_experiment_receipt_sha256=REVISION["evidence"][-1]["sha256"],
        update_ceiling=REVISION["lineage_update_ceiling"],
        local_entropy_revision=deepcopy(REVISION))
    if plan != expected:
        raise ValueError("Local entropy revision differs from its exact same-state derivation")
    for row in REVISION["evidence"]:
        if local.digest(root / row["file"]) != row["sha256"]:
            raise ValueError("Local entropy revision evidence changed: " + row["file"])


def validate_contracts(source, target, root):
    if (source.get("local_scratch", {}).get("plan_file") != local.REHEARSAL_PLAN_FILE
            or target.get("local_scratch", {}).get("plan_file") != local.ENTROPY_AB_PLAN_FILE):
        raise ValueError("Local entropy revision requires exact local v6 -> v7 contracts")
    budget = local.validate_config(target, root)
    _, original = local.build_bundle(root, Path(source["design_preflight"]["file"]).parent,
        source["target_num_envs"], plan_file=local.REHEARSAL_PLAN_FILE)
    if original["worker"] != source:
        raise ValueError("Local entropy revision source is not its independent v6 derivation")
    allowed = {"entropy_coef", "local_scratch", "design_preflight", "budget_spec_file", "budget_spec_sha256"}
    if {k: v for k, v in source.items() if k not in allowed} != {
            k: v for k, v in target.items() if k not in allowed}:
        raise ValueError("Local entropy revision changes an undeclared contract field")
    if (source["entropy_coef"] != REVISION["source_entropy_coef"]
            or target["entropy_coef"] != REVISION["target_entropy_coef"]
            or source["target_num_envs"] != 512
            or budget["inherited_consumption"] != 1216
            or budget["update_ceiling"] != 18784):
        raise ValueError("Local entropy revision changed its population, entropy or consumed budget")
    return dict(schema=REVISION["schema"], changed_contract_fields=sorted(allowed),
        source_entropy_coef=source["entropy_coef"], target_entropy_coef=target["entropy_coef"],
        consumption="all_old_lineages_and_control_updates_preserved_no_refund",
        additional_critic_warmup=0)


def validate_parent(checkpoint, target, root):
    from .full_curriculum import checkpoint_contract_path
    root = Path(root)
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
        raise ValueError("Local entropy arm must fork the exact finalized pilot013-2560, never the control endpoint")
    completion = json.loads((checkpoint.parent / "completion.json").read_text())
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    if (completion["status"] != "completed" or seal.get("global_actor_update") != count
            or completion["parent_training_transitions"] + completion["successful_updates"]
                * source["target_num_envs"] * source["num_steps_per_env"] != REVISION["training_transitions"]):
        raise ValueError("Local entropy revision parent clocks changed")
    control = REVISION["control"]
    for field in ("checkpoint", "completion", "evaluation"):
        if local.digest(control[field + "_file"]) != control[field + "_sha256"]:
            raise ValueError("Local entropy arm control evidence changed: " + field)
    control_completion = json.loads(Path(control["completion_file"]).read_text())
    control_evaluation = json.loads(Path(control["evaluation_file"]).read_text())
    if (control_completion["status"] != "completed"
            or control_completion["parent_updates"] != count
            or control_completion["successful_updates"] != control["additional_consumption"]
            or control_completion["final_learning_lineage_updates"] != control["learning_updates"]
            or control_evaluation["status"] != "evaluated"):
        raise ValueError("Local entropy arm requires the completed and evaluated control")
    inherited = json.loads((root / REVISION["evidence"][-1]["file"]).read_text())
    if (inherited["status"] != "behavior_rejected" or inherited["consumption"] != 1088 + 128
            or inherited["total_all_experiments_before_treatment"] != count + 1216
            or inherited["lineage_update_ceiling"] + inherited["consumption"] != 20000
            or inherited["components"][1]["phase_consumption"] != control_completion["successful_updates"]):
        raise ValueError("Local entropy arm would refund control consumption")
    result.update(parent_updates=count, inherited_consumption=REVISION["prior_consumption"],
        total_prior_consumption=count + REVISION["prior_consumption"],
        lineage_update_ceiling=REVISION["lineage_update_ceiling"])
    return result
