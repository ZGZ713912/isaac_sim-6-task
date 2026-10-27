"""Declared read-only domain replacement around an authenticated frozen actor.

The resolver is loaded by absolute path so an old run's import path cannot
silently provide a different resolver. Frozen physics/task/actor code is kept.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(cfg, design, task, study_file, profile, strength):
    source = ROOT / "src/wheeled_tasks/chassis/randomization_study.py"
    module_spec = importlib.util.spec_from_file_location("declared_domain_resolver", source)
    resolver = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(resolver)
    data = Path(study_file).read_bytes()
    resolved = resolver.resolve(json.loads(data), profile, strength)
    resolved.update(input_file=str(Path(study_file).resolve()), input_sha256=hashlib.sha256(data).hexdigest())
    previous = deepcopy(task.pop("randomization_study", None))
    resolver.apply(cfg, task, resolved)
    row = design["stages"]["speed"]["roles"]["train"]
    for key in ("dynamics_randomization", "contact_material"):
        if key in row["contract"]:
            row["contract"][key] = deepcopy(cfg[key])
    row["runtime"]["contact_material"] = deepcopy(cfg["contact_material"])
    row["runtime"]["modules"]["dynamics_randomization"] = cfg["dynamics_randomization"]["enabled"]
    intervention = dict(resolved=resolved, replaced_study=previous,
        resolver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="read-only environmental sensitivity; no claim of learning from these domains")
    design["review_basis"]["only_runtime_override"]["randomization_study"] = intervention
    return intervention, resolver


def samples(env, count):
    """Actual draws after evaluation reset, indexed exactly like native traces."""
    domain = env.domain
    fields = dict(enabled=domain.enabled, motor_strength=domain.strength,
        leg_kp_scale=domain.kp, leg_kd_scale=domain.kd, spring_strength=domain.spring,
        policy_sensor_delay_ticks=domain.sensor_delay, policy_bias=domain.bias,
        policy_noise_sigma=domain.sigma, pd_encoder_bias=env.getup_feedback.bias)
    if domain.imu_mount is not None:
        fields.update(domain.imu_mount.parameters())
    return {name: value[:count].detach().cpu().tolist() for name, value in fields.items()}
