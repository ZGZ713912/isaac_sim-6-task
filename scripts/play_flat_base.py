#!/usr/bin/env python3
"""Interactive ONNX playback of the validated flat-base continuation candidate."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
MODEL = ROOT / "models/flat_static_grip1338_20261020/policy.onnx"
CHECKPOINT = ROOT / "runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt"
CONTRACT = CHECKPOINT.parent / "config/contract.json"


def prepare_flat_gui_experience(output):
    """Keep the installed Lab GUI settings, omitting its unused missing drawer."""
    import importlib.util
    from chassis_suite_support import digest, write_json

    package = Path(importlib.util.find_spec("isaaclab").origin)
    source = next(parent / "apps/isaaclab.python.kit" for parent in package.parents
                  if (parent / "apps/isaaclab.python.kit").is_file())
    text = source.read_text()
    dependency = '"isaacsim.util.debug_draw" = {}'
    if text.count(dependency) != 1:
        raise ValueError("Review the installed IsaacLab GUI experience before adapting it")
    text = text.replace(dependency, "# Force overlays use USD BasisCurves; debug_draw is not required.")
    # This installed renderer and PhysX expose different Fabric interface versions.
    # Use Kit's USD scene delegate for this single-robot GUI. The suite also
    # constructs its physics context with use_fabric=False to keep links live.
    text = text.replace("app.useFabricSceneDelegate = true", "app.useFabricSceneDelegate = false")
    text = text.replace("rtx.hydra.readTransformsFromFabricInRenderDelegate = true",
                        "rtx.hydra.readTransformsFromFabricInRenderDelegate = false")
    # The generated experience lives in the report, so restore the source paths.
    text = text.replace('"${app}", # needed to find other app files',
                        json.dumps(str(source.parent)) + ", # original IsaacLab apps")
    text = text.replace('"${app}/../source", # needed to find extensions in Isaac Lab',
                        json.dumps(str(source.parent.parent / "source")) + ", # IsaacLab extensions")
    target = Path(output) / "flat_gui.kit"
    target.write_text(text)
    receipt = dict(source_file=str(source), source_sha256=digest(source),
        file=str(target.resolve()), sha256=digest(target),
        omitted_dependency="isaacsim.util.debug_draw", installed_experience_modified=False,
        cpu_affinity=sorted(os.sched_getaffinity(0)), render_transport="usd")
    write_json(Path(output) / "gui_experience.json", receipt)
    return receipt


def prepare_flat_playback(root, contract_path, output, scenario):
    from chassis_suite_support import digest, project_scene, write_json
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight

    if scenario != "flat":
        raise ValueError("This candidate's interactive entry supports flat ground")
    root, contract_path, output = Path(root), Path(contract_path), Path(output)
    source, manifest = preflight(contract_path)
    keys = ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim", "history_length")
    if (source['policy_dt'] not in (.02,.005) or
            tuple(source[key] for key in keys) != (.001, source['policy_dt'], 35, 83, 6, 1)):
        raise ValueError("Flat continuation requires its declared 1000 Hz physics, 50/200 Hz policy and 35/83/6 contract")
    binding = source["design_preflight"]
    design_path = root / binding["file"]
    if digest(design_path) != binding["sha256"]:
        raise ValueError("Source design changed")
    design = json.loads(design_path.read_text())
    row = deepcopy(design["stages"][binding["stage"]]["roles"]["train"])
    seed = row["invocation"]["seed"]["value"]
    control = load_chassis_control(root / source["control_math_source"])
    original = design_preflight(source, control, root, stage=binding["stage"], role="train",
        seed=seed, num_envs=row["invocation"]["num_envs"])
    # Project the runtime and the independently frozen expectation separately.
    config = project_scene(source, "flat", seed)
    row["contract"] = project_scene(row["contract"], "flat", seed)
    row["contract"]["evaluation.seed"] = seed
    # Training's short group horizons cannot apply to a manual playback case.
    config["flat_group_episode_seconds"] = {}
    row["contract"]["flat_group_episode_seconds"] = {}
    row["invocation"]["num_envs"] = 1
    projected = {key: deepcopy(value) for key, value in design.items() if key != "stages"}
    projected.update(run_scope="interactive_unscored", qualification=None,
        source_contract_sha256=digest(contract_path), source_design_sha256=digest(design_path),
        source_role="train", scenario="flat", stages={binding["stage"]: {"roles": {"playback": row}}})
    output.mkdir(parents=True, exist_ok=False)
    profile = output / "playback_expectations.json"
    write_json(profile, projected)
    config["design_preflight"] = dict(file=str(profile.resolve()), sha256=digest(profile), stage=binding["stage"])
    receipt = design_preflight(config, control, root, stage=binding["stage"], role="playback", seed=seed, num_envs=1)
    write_json(output / "playback_contract.json", config)
    write_json(output / "CPU_preflight.json", dict(status="CPU_design_preflight_passed_scene_not_run",
        source_train_role=original, playback=receipt, qualification=None, scene_run=False,
        ground_material_range=config["contact_material"]["ground_static_friction_range"],
        wheel_material=config["contact_material"]["wheel_static_friction"]))
    return source, config, manifest, control, seed, receipt


def validated_flat_session(policy, contract_path, contract):
    import numpy as np
    import onnxruntime as ort
    import torch
    from chassis_suite_support import digest

    policy, contract_path = Path(policy), Path(contract_path)
    proof = json.loads((policy.parent / "verification.json").read_text())
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    info = checkpoint["infos"]
    if (digest(policy) != proof["model_sha256"]
            or digest(CHECKPOINT) != proof["checkpoint_sha256"]
            or digest(contract_path) != info["contract_sha256"]
            or digest(ROOT / proof["training_plan_file"]) != proof["training_plan_sha256"]
            or info["plan_sha256"] != proof["training_plan_sha256"]
            or not proof["deployment_interface_verified"]
            or not proof["behavior_passed"]):
        raise ValueError("Flat ONNX, checkpoint, source contract or export verification changed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape[-1] != contract["actor_dim"]
            or inputs[0].type != "tensor(float)" or len(outputs) != 1 or outputs[0].shape[-1] != 6):
        raise ValueError("Flat ONNX observation/action interface mismatch")
    probe = session.run(None, {"obs": np.zeros((1, 35), dtype=np.float32)})[0]
    if probe.shape != (1, 6) or not np.isfinite(probe).all():
        raise ValueError("Flat ONNX inference failed")
    return session, dict(onnx_sha256=digest(policy), contract_sha256=digest(contract_path),
        asset_manifest_sha256=contract["asset_manifest_sha256"],
        successful_updates_total=info["successful_updates_total"], actor_updates_total=info["actor_updates_total"],
        export_verification_sha256=digest(policy.parent / "verification.json"),
        run_scope="interactive_unscored", behavior_qualified=False)


def main():
    if "--headless" not in sys.argv and "--preflight-only" not in sys.argv:
        cores = int(os.environ.get("FLAT_PLAY_CPU_CORES", "4"))
        if not 1 <= cores <= 8:
            raise ValueError("FLAT_PLAY_CPU_CORES must be between 1 and 8")
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:cores])
        print(f"GUI CPU affinity: {sorted(os.sched_getaffinity(0))}", flush=True)
    import play_chassis_suite as suite
    from play_v5_grounded import PolicyView

    class FlatView(PolicyView):
        def __init__(self, env, args):
            super().__init__(env, args)
            self.auto.model.set_value(False)
            # Start at the flat policy's full keyboard envelope. The loaded
            # contract still limits acceleration before each actor observation.
            self.speed_limit.model.set_value(3.)
            self.yaw_limit.model.set_value(4.)

    suite.prepare_playback = prepare_flat_playback
    suite.validated_session = validated_flat_session
    suite.SuiteView = FlatView
    # The suite owns the existing GUI lifecycle, command timing and telemetry.
    sys.argv[1:1] = ["--onnx", str(MODEL), "--contract", str(CONTRACT), "--scenario", "flat"]
    return suite.main(gui_experience_factory=prepare_flat_gui_experience)


if __name__ == "__main__":
    raise SystemExit(main())
