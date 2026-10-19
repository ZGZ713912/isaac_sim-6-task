"""Bind exact parent restoration, isolated no-write backward passes and RSL events."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;RUN=ROOT/'runs/flat_base_20261019'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
qpath=RUN/'population_capacity_queue002.json';q=read(qpath)
output=HERE/'population_capacity_completion002.json'
if output.exists():raise SystemExit('Preserve completed capacity receipt')
if q['status']!='complete' or any(Path('/proc/'+str(q[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for the capacity queue and both children to exit')
cp=RUN/'pilot023/model_001248.pt';parent=torch.load(cp,map_location='cpu',weights_only=True)
def exact(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
checks=dict(two_successful_native_children=len(q['history'])==2 and all(row['exit_code']==0 for row in q['history']),
    no_optimizer_writes=q['new_optimizer_updates']==0 and not q['subsequent_training_queued'],
    exact_checkpoint=sha(cp)==q['checkpoint_sha256'],
    exact_entry=sha(ROOT/'scripts/train_flat_base.py')==q['new_entry_sha256'],
    all_other_parent_execution_files_exact=all(sha(ROOT/n)==v for n,v in parent['infos']['source_sha256'].items() if n!='scripts/train_flat_base.py'))
rows={}
for row in q['history']:
    name=row['phase'];count=row['num_envs'];folder=RUN/name;path=folder/'progress.json';d=read(path)
    initial=torch.load(folder/'initial_state.pt',map_location='cpu',weights_only=True)
    proof={n+'_exact_parent':exact(initial[n],parent[n]) for n in
        ('actor_state_dict','critic_state_dict','optimizer_state_dict','torch_rng_state','cuda_rng_state')}
    # The environment generator is checked before the declared fresh physical reset.
    proof.update(native_six_restore_checks=len(d['full_restore_verification'])==6 and all(d['full_restore_verification'].values()),
        preserved_consumption=d['consumed_optimizer_updates_total']==1249 and d['successful_updates_total']==1248,
        preserved_historical_frames=d['training_transitions_total']==61341696,
        no_new_critic_warmup=d['critic_warmup_remaining']==0,
        exact_report=sha(path)==row['progress_sha256'],
        capacity_mode_complete=d['status']=='capacity_complete' and d['new_optimizer_updates']==0,
        full_rollout_and_backward=d['capacity_probe']['native_policy_frames']==96*count
            and d['capacity_probe']['blocked_optimizer_writes']==12 and d['capacity_probe']['optimizer_writes']==0,
        all_capacity_state_checks=d['capacity_probe']['passed'] and all(d['capacity_probe']['checks'].values()),
        GPU_margin_measured=row['peak_gpu_used_mib']<15*1024)
    events=EventAccumulator(str(folder),size_guidance={'scalars':0});events.Reload()
    tags=('collection_s','backward_s','policy_frames_per_collection_s','torch_peak_allocated_bytes',
          'torch_peak_reserved_bytes','optimizer_writes','blocked_optimizer_writes','passed')
    good=[]
    for tag in tags:
        points=events.Scalars(f'FlatCapacity/population_{count}/'+tag)
        value=d['capacity_probe'][tag]
        good.append(len(points)==1 and points[0].step==1248
            and math.isclose(points[0].value,value,rel_tol=2e-7,abs_tol=2e-8))
    proof['eight_RSL_TensorBoard_capacity_metrics']=all(good)
    checks.update({name+'/'+k:v for k,v in proof.items()})
    rows[name]=dict(checks=proof,progress_sha256=sha(path),initial_state_sha256=sha(folder/'initial_state.pt'),
        capacity=d['capacity_probe'],peak_gpu_used_mib=row['peak_gpu_used_mib'],
        event_file_sha256={p.name:sha(p) for p in folder.glob('events.out.tfevents.*')})
if not all(checks.values()):print(json.dumps(checks,indent=2));raise SystemExit('Capacity identity or logger check failed')
output.write_text(json.dumps(dict(schema='flat-native-capacity-completion-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,queue_sha256=sha(qpath),
    entry_sha256=q['new_entry_sha256'],checkpoint_sha256=sha(cp),probes=rows,
    scope='Engineering capacity and exact state restoration only; no convergence or paired learning improvement claim',
    new_optimizer_updates=0,behavior_qualified=False,hardware_motion_started=False),indent=2,allow_nan=False)+'\n')
print(json.dumps({name:dict(collection_s=row['capacity']['collection_s'],
    backward_s=row['capacity']['backward_s'],frames_per_s=row['capacity']['policy_frames_per_collection_s'],
    peak_gpu_used_mib=row['peak_gpu_used_mib']) for name,row in rows.items()},indent=2))
