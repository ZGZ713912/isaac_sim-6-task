"""Compare the two actual first rollouts from the same1280 static composition."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
out = HERE / 'wheel_from1280_first_rollout_pair001.json'
if out.exists():
    raise SystemExit('Preserve completed first-rollout comparison')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = [RUN/name/'initial_state.pt' for name in ('pilot026','pilot027_from1280')]
states = [torch.load(p,map_location='cpu',weights_only=True) for p in paths]

def exact(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

values=[]
for path in paths:
    events=EventAccumulator(str(path.parent),size_guidance={'scalars':0})
    events.Reload()
    row={}
    for tag in events.Tags()['scalars']:
        if tag.startswith(('FlatBehavior/','FlatBirth/','FlatTermination/')):
            points=[p.value for p in events.Scalars(tag) if p.step==1281]
            if len(points)!=1:
                raise ValueError('Missing/duplicate first-rollout scalar: '+tag)
            row[tag]=points[0]
    values.append(row)
a,b=states
checks=dict(
    old_actor_tensors_exact=all(torch.equal(v,b['actor_state_dict'][n]) for n,v in a['actor_state_dict'].items()),
    original_critic_exact=exact(a['critic_state_dict'],b['critic_state_dict']),
    all_saved_sampling_and_environment_RNG_exact=all(exact(a[k],b[k])
        for k in ('torch_rng_state','cuda_rng_state','env_generator_state')),
    matching_first_rollout_scalar_tags=values[0].keys()==values[1].keys() and len(values[0])==10504,
    all_first_rollout_scalars_exact=values[0]==values[1],
    exact_population_clock_and_command_reference=all(a['infos'][k]==b['infos'][k]
        for k in ('num_envs','control_math_sha256','training_command_reference','teacher_anchor_runtime')))
if not all(checks.values()):
    raise SystemExit(json.dumps(checks,indent=2))
digest=hashlib.sha256(json.dumps(values[0],sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
out.write_text(json.dumps(dict(schema='flat-wheel-from1280-first-rollout-pair-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    initial_state_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
    completed_update_step=1281,first_rollout_scalar_count=len(values[0]),first_rollout_scalar_sha256=digest,
    first_rollout_samples=4096*96,optimizer_updates_by_this_reader=0,native_steps_by_this_reader=0,
    behavior_qualified=False,
    scope='Actual initial state and existing RSL event comparison only. Proves first-rollout equality before the different learning scopes act; it does not prove subsequent closed-loop preservation or task success.'),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),first_rollout_scalar_count=len(values[0])),indent=2))
