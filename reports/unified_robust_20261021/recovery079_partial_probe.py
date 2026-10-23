"""Read-only frozen-state check of a fully sealed intermediate079 checkpoint."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_native_state_audit import named_slots, finite
from iteration045_runner import write
from wheeled_algo.robust_getup import tilt_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = RUNS / "quiet_severe_reference065/model_migrated.pt"
    output = RUNS / "pilot033_dynamic_recovery079"
    sealed = output / "model_536.pt"
    before, bi = load_mean(parent)
    after, ai = load_mean(sealed)
    old, new = [torch.load(path, map_location="cpu", weights_only=False) for path in (parent, sealed)]
    clock = ai["completed_updates"]
    increment = clock - bi["completed_updates"]
    assert clock == 537 and bi["completed_updates"] == 505 and increment == 32
    assert all(finite(new[key]) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
    changed = [key for key, value in new["actor_state_dict"].items() if not equal(value, old["actor_state_dict"][key])]
    assert changed and all(key.startswith("mlp.mean.recovery.") for key in changed)
    oslots, nslots = named_slots(old, before), named_slots(new, after)
    assert all(equal(value, nslots["actor"][key]) for key, value in oslots["actor"].items()
               if not key.startswith("mlp.mean.recovery."))
    source = RUNS / "high_recenter_full066/robust/trajectory.npz"
    with np.load(source) as data:
        obs = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    zero = tilt_gate(obs, True, True).ravel() == 0.
    with torch.no_grad():
        baseline = before(obs)
        candidate = after(obs)
        assert torch.equal(baseline[zero], candidate[zero])
    progress = json.loads((output / "progress.json").read_text())
    value = dict(schema="recovery079-readonly-partial-v1", sealed_checkpoint_sha256=sha(sealed),
        checkpoint_filename="model_536.pt", partial_lineage_clock=clock, sealed_new_updates=increment,
        charged_completed_at_least=1192+increment,
        actual_frozen_actor_std_and_named_adam_exact=True, changed_actor=changed, finite=True,
        recorded_inputs=len(obs), zero_authority_frames=int(zero.sum()),
        zero_authority_outputs_bitwise_exact=True, trajectory_sha256=sha(source),
        live_at_read={key: progress.get(key) for key in ("status", "new_optimizer_updates", "completed_updates", "elapsed_s")},
        evaluation_still_required=True, full_training_admitted=False,
        interpretation="Function replay and tensor scope, not new closed-loop standing/recovery evidence; live updates can exceed this sealed intermediate")
    write(REPORTS / "iteration079_partial_sealed.json", value)
    with SummaryWriter(str(output), filename_suffix=".partial079_audit") as writer:
        writer.add_scalar("Diagnosis/Recovery079/partial_sealed_new_updates", increment, clock)
        writer.add_scalar("Diagnosis/Recovery079/frozen_state_exact", 1., clock)
        writer.add_scalar("Diagnosis/Recovery079/zero_authority_replay_frames", int(zero.sum()), clock)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
