"""Finite private sampled-policy exposure with explicit distribution overrides."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
if '--sampled-policy' not in sys.argv:
    raise ValueError('This entry point is sampled-policy diagnosis only')
index=sys.argv.index('--entry-leg-std-ratio');ratio=float(sys.argv[index+1]);del sys.argv[index:index+2]
from entry_exploration_probe001 import alter_entry_std
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean,SampledGapPolicyEvaluation
import evaluate_gap_robustness as evaluate
native_forward=GapRecoveryNoiseMean.forward
native_receipt=SampledGapPolicyEvaluation.sampling_receipt
native_write=evaluate.write_json

def forward(self,obs):return alter_entry_std(native_forward(self,obs),obs,ratio)

def receipt(self):
    value=native_receipt(self)
    value.update(entry_leg_std_ratio=ratio,entry_leg_full_authority_scale=.2*ratio,
        deterministic_mean_changed=False,wheel_distribution_changed=False,
        distribution='checkpoint_orbit_v2_conditioned_normal_with_explicit_entry_leg_std_probe',
        scope='Four leg std only, continuous upright-entry gate; explicit private zero-PPO diagnostic',
        source_wrapper=str(Path(__file__)))
    return value

def write(path,value):
    if Path(path).name=='task_contract.json':
        value['evaluation_overrides']['entry_leg_std_ratio']=ratio
        value['evaluation_overrides']['sampled_policy_exposure']['distribution']='checkpoint_orbit_v2_conditioned_normal_with_explicit_entry_leg_std_probe'
    return native_write(path,value)

GapRecoveryNoiseMean.forward=forward
SampledGapPolicyEvaluation.sampling_receipt=receipt
evaluate.write_json=write
evaluate.main()
