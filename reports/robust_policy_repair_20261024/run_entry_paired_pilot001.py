"""Native training entry point with read-only first-rollout provenance."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
if "--entry-recovery-only" not in sys.argv or "--recovery-only" not in sys.argv:
    raise ValueError("This receipt wrapper is limited to the explicit entry-only pilot")
output = Path(sys.argv[sys.argv.index("--output") + 1])

from wheeled_algo.recovery_rollout import RecoveryRolloutMask
native_update = RecoveryRolloutMask.update


def tensor_receipt(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def measured_update(self):
    if not getattr(self, "_entry_first_rollout_saved", False):
        storage = self.storage
        data = {"observations/" + str(key): tensor_receipt(value)
                for key, value in storage.observations.items()}
        for name in ("actions", "dones", "rewards", "values", "actions_log_prob"):
            data[name] = tensor_receipt(getattr(storage, name))
        data["learning_mask"] = tensor_receipt(self.learning)
        data["bootstrap_mask"] = tensor_receipt(self.bootstrap)
        for index, value in enumerate(storage.distribution_params):
            data["distribution_param_" + str(index)] = tensor_receipt(value)
        state = dict(schema="entry-pilot-first-native-rollout-hashes-v1", tensors=data,
            phase="After native GAE, before the first optimizer update; rewards and returns may differ across the declared cost intervention",
            sampling_likelihood_GAE_optimizer_changed=False,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        target = output / "first_rollout_hashes.json"
        if target.exists():
            raise ValueError("Never overwrite a first-rollout receipt")
        target.write_text(json.dumps(state, indent=2) + "\n")
        archive = output / "source" / Path(__file__).resolve().relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(Path(__file__).read_bytes())
        self._entry_first_rollout_saved = True
    return native_update(self)


RecoveryRolloutMask.update = measured_update
from train_robust_getup import main
main()
