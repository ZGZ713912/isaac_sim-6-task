#!/usr/bin/env python3
"""Fit all six coordinated teacher outputs into the existing 396-parameter actor.

Only training-design-seed traces are used. No environment or PPO update occurs.
The closed-form fit is a candidate initializer and requires native validation.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from torch.utils.tensorboard import SummaryWriter
    from flat_speed_prior import progressive_reference
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.speed_reference import condition_reference
    from wheeled_tasks.chassis.flat_base_continuation import digest,encode
    torch.set_num_threads(1)
    data=ROOT/'reports/flat_base_speed_20261020/progressive_probe001'
    report=json.loads((data/'report.json').read_text())
    config=json.loads((data/'contract.json').read_text())
    assert report['status']=='evaluated' and config['evaluation']['seed']==191119
    # Select on design data only. Completion precedes tracking error.
    chosen={};selection={}
    for direction,names,metric in (
            ('reverse',('backward_2','backward_3'),'vx_mae_m_s'),
            ('yaw_positive',('rotate_4',),'yaw_mae_rad_s'),
            ('yaw_negative',('rotate_4_reverse',),'yaw_mae_rad_s')):
        scores=[]
        for arm in report['gains']:
            rows=[v for n,v in report['result']['cases'].items()
                  if n.startswith(f'g{arm:g}_') and any(n.endswith('_'+name)for name in names)]
            full=sum(v['full_horizon_episodes']for v in rows)
            error=sum(v[metric]for v in rows)/len(rows) if all(v['frames']for v in rows) else float('inf')
            scores.append((full,-error,arm))
        full,negative_error,arm=max(scores)
        chosen[direction]=arm
        selection[direction]=dict(arm=arm,full_episodes=full,
            requested_episodes=len(names)*8,error=(-negative_error if np.isfinite(negative_error)else None))
        if direction.startswith('yaw'):
            rows=[v for n,v in report['result']['cases'].items()
                  if n.startswith(f'g{arm:g}_') and any(n.endswith('_'+name)for name in names)]
            assert all(v['full_horizon_episodes']==v['requested_episodes'] and v['frames']
                       and v['yaw_mae_rad_s']<=.1 for v in rows), 'No valid turn teacher; reject stationary substitutes'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
    parent=ROOT/'runs/flat_base_20261020/pilot031_speed_reference1342/model_001342.pt'
    state=torch.load(parent,map_location='cpu',weights_only=True)
    student=mean_from_checkpoint(state,source)
    teacher_state=torch.load(ROOT/'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt',
                             map_location='cpu',weights_only=True)
    teacher=mean_from_checkpoint(teacher_state,source)
    samples=[];provenance=[]
    for directory in (data,ROOT/'reports/flat_base_speed_20261020/lean_probe003'):
        metadata=json.loads((directory/'contract.json').read_text())
        assert metadata['evaluation']['seed']==191119
        z=np.load(directory/'traces.npz');obs=z['observations'];active=z['active'].astype(bool)
        mask=active & (obs[:,:,9]<-np.cos(np.deg2rad(8.)))
        # Keep early balance examples and stable moving states. Failed/inactive
        # samples cannot teach an attractive velocity score after termination.
        selected=obs[mask].copy()
        samples.append(torch.from_numpy(selected))
        provenance.append(dict(directory=str(directory.relative_to(ROOT)),
            trace_sha256=digest(directory/'traces.npz'),report_sha256=digest(directory/'report.json'),frames=len(selected)))
    obs=torch.cat(samples)
    baseline=deepcopy(student)
    with torch.no_grad():
        baseline.speed_residual.weight.zero_();baseline.speed_residual.bias.zero_()
        arm=torch.where(obs[:,0:1]<-.75,chosen['reverse'],
            torch.where(obs[:,2:3]<0.,chosen['yaw_negative'],chosen['yaw_positive']))
        target_actions=teacher(progressive_reference(obs,arm))
        base_actions=baseline(obs)
        bound=baseline.speed_residual_bound
        raw_delta=target_actions-base_actions
        delta=raw_delta.clamp(-bound*.97,bound*.97)
        target=bound*torch.atanh(delta/bound)
        translation=obs[:,:1].abs()/.75>=obs[:,2:3].abs()/1.5
        index=torch.where(translation,(obs[:,:1]<0).long(),2+(obs[:,2:3]<0).long())
        ref=condition_reference(obs,baseline.reference_margin.clamp(.05,1.2)[index],
            baseline.reference_lean_gain.clamp(-.04,.04)[(obs[:,:1]<0).long()])
        features=ref
        for layer in list(baseline.command.base.children())[:-1]:
            features=layer(features)
        center=features.double().mean(0);scale=features.double().std(0).clamp_min(.1)
        x=torch.cat(((features.double()-center)/scale,torch.ones(len(obs),1,dtype=torch.float64)),-1)
        # Give start-up/large-angle states additional weight without demanding
        # zero correction: the teacher must also coordinate those six outputs.
        weights=torch.where(obs[:,9]>-np.cos(np.deg2rad(2.)),4.,1.).double()
        normal=x.T@(weights[:,None]*x)/weights.sum()
        regularizer=torch.eye(65,dtype=torch.float64)*.002;regularizer[-1,-1]=0.
        solution=torch.linalg.solve(normal+regularizer,x.T@(weights[:,None]*target.double())/weights.sum())
        fitted_weight=(solution[:-1]/scale[:,None]).T.float()
        fitted_bias=(solution[-1]-(center[:,None]*solution[:-1]/scale[:,None]).sum(0)).float()
    fixture=TensorDict({'policy':torch.zeros(1,35),'critic':torch.zeros(1,83)},batch_size=[1])
    model=MLPModel(fixture,{'actor':['policy'],'critic':['critic']},'actor',6,hidden_dims=[256,128,64],
        activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.04,'std_type':'scalar'})
    model.mlp=student;model.load_state_dict(state['actor_state_dict'],strict=True)
    owner=dict(zip(dict(model.named_parameters()),state['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    clear=[owner['mlp.speed_residual.weight'],owner['mlp.speed_residual.bias']]
    receipt=dict(schema='flat-speed-coordinated-distillation-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(Path(__file__)),prior_source_sha256=digest(ROOT/'scripts/flat_speed_prior.py'),
        parent_sha256=digest(parent),teacher_parent_sha256=digest(ROOT/'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt'),
        selection=selection,provenance=provenance,data_seed=191119,excluded_comparison_seeds=[191139,191159],
        selection_policy='completion count, then external-command error; design data only',
        frames=len(obs),linear_solves=1,campaign_closed_form_fits=2,new_PPO_updates=0,
        target_clipped_fraction=(raw_delta.abs()>bound*.97).float().mean(0).tolist(),
        optimizer_revision='clear only speed_residual weight/bias Adam; preserve all other state',
        qualified=False,adopted=False,candidates={})
    with torch.no_grad(),SummaryWriter(str(out),filename_suffix='.distillation') as writer:
        for fraction in (.25,.5,1.):
            child=deepcopy(state)
            for key,value in [('weight',fitted_weight),('bias',fitted_bias)]:
                name='mlp.speed_residual.'+key
                child['actor_state_dict'][name].lerp_(value,fraction)
            for key in clear:
                child['optimizer_state_dict']['state'].pop(key,None)
            child['infos']['closed_form_head_fits_total']=2
            child['infos']['supervised_head_fit']=dict(kind='coordinated_six_action_teacher',
                parent_sha256=receipt['parent_sha256'],data_seed=191119,fraction=fraction,
                source_sha256=receipt['source_sha256'],prior_source_sha256=receipt['prior_source_sha256'],
                selection=selection,new_PPO_updates=0,linear_solves=1,optimizer_revision=receipt['optimizer_revision'])
            path=out/f'model_001342_distill{fraction:g}.pt';torch.save(child,path)
            recovered=mean_from_checkpoint(child,source)
            assert all(torch.equal(value,child['actor_state_dict'][name])for name,value in state['actor_state_dict'].items()
                if not name.startswith('mlp.speed_residual.'))
            low=obs[:4096].clone();low[:,0].clamp_(-.75,.75);low[:,2].clamp_(-1.5,1.5)
            assert torch.equal(student(low),recovered(low))
            rmse=(recovered(obs)-target_actions).square().mean(0).sqrt().tolist()
            receipt['candidates'][str(fraction)]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),
                teacher_action_rmse_raw=rmse,low_function_equal=True)
            for i,value in enumerate(rmse):
                writer.add_scalar(f'FlatSpeedDistillation/fraction_{fraction:g}/action_{i}_rmse_raw',value,1342)
    (out/'receipt.json').write_bytes(encode(receipt))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out/'flat_speed_prior.py').write_bytes((ROOT/'scripts/flat_speed_prior.py').read_bytes())
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
