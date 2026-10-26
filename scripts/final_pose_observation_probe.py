"""Declared, read-only reset-cache counterfactual for frozen v1 evidence.

This transforms only the final placement's HistoryStack reset. It never
changes geometry, dynamics, policy weights or standing acceptance.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys


RESET_SOURCE = "src/wheeled_tasks/chassis/grounded_standing_reset.py"
INSERT = ("        # The parent reset already queried this policy tick, before final\n"
          "        # pitch/grounding. Refresh only placed lanes; preserve other histories\n"
          "        # and the domain RNG, gains, delays and sensor cursor.\n"
          "        env.history.reset(selected)\n")


def candidate_bytes(original):
    text = original.decode("utf-8")
    version = "VERSION = 'dual-wheel-grounded-reset-v1'"
    anchor = "        frame = ChassisEnv.get_observations(env)['policy']"
    if text.count(version) != 1 or text.count(anchor) != 1 or "env.history.reset(selected)" in text:
        raise ValueError("Reset counterfactual requires the unmodified frozen v1 placement")
    return text.replace(version, "VERSION = 'dual-wheel-grounded-reset-v2-final-observation'").replace(
        anchor, INSERT + anchor).encode("utf-8")


def prepare(archive, output):
    original_path = Path(archive) / RESET_SOURCE
    original = original_path.read_bytes()
    patched = candidate_bytes(original)
    path = Path(output) / "grounded_standing_reset_final_pose.py"
    if path.exists():
        raise ValueError("Preserve the diagnostic reset source")
    path.write_bytes(patched)
    return dict(source=str(original_path), original_sha256=hashlib.sha256(original).hexdigest(),
        modified_source=str(path), modified_sha256=hashlib.sha256(patched).hexdigest(),
        change="history.reset(selected) immediately before final placed sensor-history initialization",
        new_PPO=0, actor_weights_changed=False, geometry_changed=False,
        candidate_adopted=False, behavior_qualification=False, hardware_ready=False)


def activate(path):
    name = "wheeled_tasks.chassis.grounded_standing_reset"
    if name in sys.modules:
        raise ValueError("The diagnostic must be selected before any placement module is imported")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
