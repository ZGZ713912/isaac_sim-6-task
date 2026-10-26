"""Execute the actual evaluation wrapper with a CPU-only evaluator substitute."""
from pathlib import Path
import sys,types,runpy,time,json
sys.path[:0]=['src','scripts']
import torch
from train_robust_getup import write_json,digest
from wheeled_algo import robust_getup,quiet_output_actor
R=Path('reports/robust_policy_repair_20261024');out=R/'quiet_authority_wrapper_smoke002';out.mkdir(exist_ok=False)
original=(robust_getup.RobustGetupActor.forward,quiet_output_actor.QuietOutputGetupActor.forward)
gate=robust_getup.quiet_gate;calls=[]
evaluate=types.ModuleType('evaluate_gap_robustness');evaluate.digest=digest
evaluate.write_json=lambda p,v:calls.append((str(p),v))
def main():
 assert robust_getup.quiet_gate is gate
 assert robust_getup.RobustGetupActor.forward is not original[0]
 evaluate.write_json(out/'task_contract.json',dict(evaluation_overrides={}))
 evaluate.write_json(out/'progress.json',dict(source_sha256={},evaluation_overrides={}))
evaluate.main=main;sys.modules['evaluate_gap_robustness']=evaluate
sys.argv=[str(R/'evaluate_quiet_authority_probe002.py'),'--retained-quiet-authority','.5']
runpy.run_path(sys.argv[0],run_name='__main__')
metadata=calls[0][1]['evaluation_overrides']['quiet_gyro_authority_probe']
checks=dict(actual_wrapper_executed=True,policy_patch_present_during_main=True,
 shared_gate_stays_native=robust_getup.quiet_gate is gate,
 methods_restored=original==(robust_getup.RobustGetupActor.forward,quiet_output_actor.QuietOutputGetupActor.forward),
 mean_intervention_explicit=metadata['gaussian_mean_changed'] is True,
 std_unchanged_explicit=metadata['sampling_std_changed'] is False,
 diagnostic_qualification_excluded=metadata['eligible_for_checkpoint_mean_qualification'] is False,
 two_sources_archived=len(calls[1][1]['source_sha256'])==2)
write_json(R/'quiet_authority_wrapper_smoke002.json',dict(schema='policy-only-probe-wrapper-smoke-v1',checks=checks,metadata=metadata,
 wrapper_sha256=digest(R/'evaluate_quiet_authority_probe002.py'),probe_sha256=digest(R/'quiet_gyro_authority_probe001.py'),
 simulated_steps=0,new_PPO=0,created_unix=time.time()))
print(json.dumps(checks));assert all(checks.values())
