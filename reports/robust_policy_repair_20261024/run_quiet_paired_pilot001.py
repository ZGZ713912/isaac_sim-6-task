"""Read-only receipt for the first native quiet rollout, before optimization."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if '--quiet-only' not in sys.argv or '--quiet-velocity-only' not in sys.argv:
    raise ValueError('This wrapper permits only the explicit24-parameter quiet profile')
output=Path(sys.argv[sys.argv.index('--output')+1])
from wheeled_algo.recovery_rollout import RecoveryRolloutMask
native_update=RecoveryRolloutMask.update

def receipt(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())

def measured_update(self):
    if not getattr(self,'_quiet_first_rollout_saved',False):
        storage=self.storage
        data={'observations/'+str(k):receipt(v) for k,v in storage.observations.items()}
        for name in ('actions','dones','rewards','values','actions_log_prob'):
            data[name]=receipt(getattr(storage,name))
        data['learning_mask']=receipt(self.learning)
        data['bootstrap_mask']=receipt(self.bootstrap)
        for i,v in enumerate(storage.distribution_params):
            data['distribution_param_'+str(i)]=receipt(v)
        target=output/'first_rollout_hashes.json'
        if target.exists():
            raise ValueError('Never overwrite a native first-rollout receipt')
        target.write_text(json.dumps(dict(schema='quiet-first-native-rollout-v1',tensors=data,
            phase='After native GAE, before optimizer; only rewards may differ in the first paired rollout',
            sampling_likelihood_GAE_optimizer_changed=False,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
        archive=output/'source'/Path(__file__).resolve().relative_to(ROOT)
        archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(Path(__file__).read_bytes())
        self._quiet_first_rollout_saved=True
    return native_update(self)

RecoveryRolloutMask.update=measured_update
from train_robust_getup import main
main()
