"""Recorded-state diagnosis of the trainer's missing actor type selections."""
from pathlib import Path
import hashlib
import json
import sys
import time

sys.path[:0] = ['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_preservation import EntryPreservingGapActor
from wheeled_algo.paired_recovery import wrap

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
checkpoint = B/'motion_damping080002/model_probe.pt'
mean,info = load_mean(checkpoint)
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation',
             'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation',
             'diagnosis024_entry_phase_fit_fixed/evaluation'):
    path = B/name/'trajectory.npz'
    z = np.load(path)
    obs = torch.tensor(z['observations'][::20].reshape(-1,35))
    leg_peak = wheel_peak = 0.
    different = 0
    with torch.inference_mode():
        for batch in obs.split(512):
            expected = mean(batch)
            # The inspected training factory currently selects this parent
            # class for all split-entry settings and ignores both newer flags.
            actual = EntryPreservingGapActor.forward(mean,batch)
            leg_peak = max(leg_peak,float(wrap((expected[:,:4]-actual[:,:4])*.25).abs().max()))
            wheel_peak = max(wheel_peak,float((expected[:,4:]-actual[:,4:]).abs().max()))
            different += int((expected!=actual).any(-1).sum())
    rows.append(dict(trace=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frames=len(obs),different_frames=different,leg_target_difference_peak_rad=leg_peak,
        wheel_target_difference_peak_nm=wheel_peak))
training = Path('scripts/train_robust_getup.py')
out = dict(schema='training-decoder-parity-diagnosis-v1',checkpoint=str(checkpoint),
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    training_source_sha256=hashlib.sha256(training.read_bytes()).hexdigest(),
    observed_training_factory='EntryPreservingGapActor selected for split-entry, no selection of periodic encoder or motion damping classes',
    required_flags={k:info['settings'].get(k) for k in ('gap_entry_encoder_version','motion_leg_damping_version','motion_leg_damping_alpha')},
    rows=rows,scope='Unbound forward of the inspected selected class, with exactly the checkpoint parameters and fields. CPU function diagnosis only, no native training rollout.',
    decision='PPO closed until factory parity and learned-checkpoint serialization are implemented and checked',
    new_PPO=0,actual_experiment_consumed=2603,qualification=None,hardware_started=False,created_unix=time.time())
(R/'training_actor_parity_diagnosis001.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
