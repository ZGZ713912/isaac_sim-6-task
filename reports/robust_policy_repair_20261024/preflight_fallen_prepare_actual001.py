"""Run the reviewable prepare draft on CPU; active training sources stay frozen."""
from pathlib import Path
import sys,types,argparse,json,time,importlib.util
sys.path[:0]=['src','scripts']
import torch
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
source=Path('scripts/train_robust_getup.py').resolve();draft=Path('scripts/train_robust_getup.py')
module=types.ModuleType('fallen_prepare_draft');module.__file__=str(source);exec(compile(draft.read_text(),str(source),'exec'),module.__dict__)
class Captured(Exception):pass
native=argparse.ArgumentParser.parse_args;args=None
command=['--output',str(B/'fallen_prepare_actual001'),'--bank',str(R/'pose_bank_rmcs003.json'),'--resume',str(B/'fallen_output_root001/model_initial.pt'),
 '--updates','8','--num-envs','256','--rollout-steps','512','--seed','241510','--isolated-origins','--gap-robustness','--gap-severity','.8',
 '--gap-actuator-timing','native','--actor-lr','.0001','--friction-min','.3','--domain-scale','.35','--eval-seconds','2','--curriculum-updates','512','--curriculum-origin','569','--recovery-only','--motion-recovery-focus']
def capture(self,*a,**kw):
 global args
 args=native(self,*a,**kw);raise Captured()
try:
 argparse.ArgumentParser.parse_args=capture;sys.argv=['train']+command
 try:module.main()
 except Captured:pass
finally:argparse.ArgumentParser.parse_args=native
assert args is not None
cfg,task=module.prepare(args)
path=Path('src/wheeled_algo/fallen_output_checkpoint.py');spec=importlib.util.spec_from_file_location('wheeled_algo.fallen_output_checkpoint_draft',path);contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
root=torch.load(args.resume,map_location='cpu',weights_only=False)['infos']['settings'];contract.validate_training_settings(task,root)
delta={k:dict(before=root.get(k,'MISSING'),after=task.get(k,'MISSING')) for k in root.keys()|task.keys() if root.get(k)!=task.get(k)}
checks=dict(only_exposure_changed=set(delta)=={'version','motion_fraction','motion_push_fraction'},source_function_floor_zero=task['quiet_stance_retained_authority']==0.,
 motion80pct=task['motion_fraction']==.8,all_motion_pushes=task['motion_push_fraction']==1.,
 source_matches_reviewed_draft=digest(source)==json.loads((R/'fallen_output_code_draft001.json').read_text())['trainer_draft_sha256'])
out=dict(schema='fallen-output-trainer-prepare-draft-v1',checks=checks,draft_sha256=digest(draft),command=command,settings_delta=delta,
 source_checkpoint=str(args.resume),source_sha256=digest(args.resume),new_PPO=0,training_admitted=False,qualification=None,created_unix=time.time())
write_json(R/'fallen_prepare_actual_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks
