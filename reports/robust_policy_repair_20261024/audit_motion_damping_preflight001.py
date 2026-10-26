"""Full recorded-state invariants and real checkpoint tamper rejection."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import time

sys.path[:0] = ['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.motion_target_damping import motion_damping_gate
from wheeled_algo.paired_recovery import wrap

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
source = B/'entry_phase_fit_periodic002/model_probe.pt'
candidate = B/'motion_damping080002/model_probe.pt'
neutral = B/'motion_damping100001/model_probe.pt'
old,oi = load_mean(source)
new,ni = load_mean(candidate)
identity,ii = load_mean(neutral)
weights_exact = all(torch.equal(v,new.state_dict()[k]) and torch.equal(v,identity.state_dict()[k])
                    for k,v in old.state_dict().items())
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation',
             'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation',
             'diagnosis024_entry_phase_fit_fixed/evaluation'):
    path = B/name/'trajectory.npz'
    z = np.load(path)
    frames = z['observations'].reshape(-1,35)
    inactive_count = inactive_exact = neutral_exact = quiet_count = quiet_exact = 0
    wheel_exact = finite = True
    target_peak = active_error_peak = before_power = after_power = 0.
    smooth_samples = 0
    for begin in range(0,len(frames),2048):
        obs = torch.tensor(frames[begin:begin+2048])
        with torch.inference_mode():
            a,b,c = old(obs),new(obs),identity(obs)
            g = motion_damping_gate(obs)[:,0]
        inactive = g==0.
        quiet = (obs[:,0]==0.) & (obs[:,2]==0.)
        inactive_count += int(inactive.sum())
        inactive_exact += int((a[inactive]==b[inactive]).all(-1).sum())
        quiet_count += int(quiet.sum())
        quiet_exact += int((a[quiet]==b[quiet]).all(-1).sum())
        neutral_exact += int((a==c).all(-1).sum())
        wheel_exact &= torch.equal(a[:,4:],b[:,4:])
        finite &= bool(torch.isfinite(b).all())
        target_peak = max(target_peak,float(wrap((b[:,:4]-a[:,:4])*.25).abs().max()))
        if (g>0.).any():
            active_error_peak = max(active_error_peak,float(wrap(b[g>0.,:4]*.25-obs[g>0.,10:14]).abs().max()))
        step = wrap((a[:,:4]-obs[:,22:26])*.25)
        normal = ((g>.999)&(step.abs().amax(-1)<.04)
                  &(wrap(a[:,:4]*.25-obs[:,10:14]).abs().amax(-1)<.05))
        before_power += float(step[normal].square().sum())
        after_power += float(wrap((b[normal,:4]-obs[normal,22:26])*.25).square().sum())
        smooth_samples += int(normal.sum())
    rows.append(dict(trace=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=len(frames),
        inactive_frames=inactive_count,inactive_frames_bit_exact=inactive_exact,
        quiet_frames=quiet_count,quiet_frames_bit_exact=quiet_exact,neutral_frames_bit_exact=neutral_exact,
        wheels_bit_exact=wheel_exact,finite=finite,physical_target_delta_peak_rad=target_peak,
        active_upright_target_error_peak_rad=active_error_peak,normal_smoothing_samples=smooth_samples,
        normal_step_rms_ratio=(after_power/before_power)**.5 if before_power else None))
    print(name,'complete',flush=True)

state = torch.load(candidate,map_location='cpu',weights_only=False)
negative = []
with tempfile.TemporaryDirectory(prefix='motion-damping-tamper-') as temp:
    for kind in ('actor','critic','optimizer','rng','clock','alpha','version','source','helper'):
        bad = deepcopy(state)
        info = bad['infos']
        if kind=='actor':
            next(iter(bad['actor_state_dict'].values())).reshape(-1)[0] += 1.
        elif kind=='critic':
            next(iter(bad['critic_state_dict'].values())).reshape(-1)[0] += 1.
        elif kind=='optimizer':
            bad['optimizer_state_dict']['param_groups'][0]['lr'] *= 2.
        elif kind=='rng':
            info['unrelated_rng_tamper'] = True
        elif kind=='clock':
            info['completed_updates'] += 1
        elif kind=='alpha':
            info['settings']['motion_leg_damping_alpha'] = float('nan')
        elif kind=='version':
            info['settings']['motion_leg_damping_version'] = 'unknown'
        elif kind=='source':
            info['motion_leg_damping_migration']['source_sha256'] = '0'*64
        else:
            info['source_sha256']['src/wheeled_algo/motion_target_damping.py'] = '0'*64
        path = Path(temp)/(kind+'.pt')
        torch.save(bad,path)
        try:
            load_mean(path)
        except ValueError as error:
            negative.append(dict(kind=kind,rejected=True,error=str(error)))
        else:
            negative.append(dict(kind=kind,rejected=False))
checks = dict(all_weights_exact=weights_exact,clock_equal=oi['completed_updates']==ni['completed_updates']==ii['completed_updates']==1501,
    inactive_bit_exact=all(x['inactive_frames']==x['inactive_frames_bit_exact'] for x in rows),
    quiet_bit_exact=all(x['quiet_frames']==x['quiet_frames_bit_exact'] for x in rows),
    neutral_all_bit_exact=all(x['frames']==x['neutral_frames_bit_exact'] for x in rows),
    all_wheels_exact=all(x['wheels_bit_exact'] for x in rows),
    active_error_under_012=all(x['active_upright_target_error_peak_rad']<=.12001 for x in rows),
    bounded_small_edit=all(x['physical_target_delta_peak_rad']<=.03 for x in rows),
    normal_motion_step_ratio=all(x['normal_step_rms_ratio'] is None or x['normal_step_rms_ratio']<=.805 for x in rows),
    finite=all(x['finite'] for x in rows),tamper_rejected=all(x['rejected'] for x in negative))
out = dict(schema='motion-leg-damping-preflight-v1',source=str(source),candidate=str(candidate),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),alpha=.8,
    checks=checks,replays=rows,negative_cases=negative,decision='admit_fixed51_zero_PPO_only' if all(checks.values()) else 'reject_without_native_or_PPO',
    new_PPO=0,actual_experiment_consumed=2603,qualification=None,hardware_started=False,created_unix=time.time())
(R/'motion_damping_preflight001.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('replays','negative_cases')},indent=2),flush=True)
assert all(checks.values())
