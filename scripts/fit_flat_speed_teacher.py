#!/usr/bin/env python3
"""One closed-form wheel-head fit from initialization data, with birth constraints.

Uses encoder/IMU observations from the training-design seed only. The separate
checkpoint-comparison seed is excluded. No PPO or environment step occurs here.
"""
from copy import deepcopy
from datetime import datetime,timezone
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
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.speed_reference import condition_reference
    from wheeled_tasks.chassis.flat_base_continuation import digest,encode
    torch.set_num_threads(1)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'runs/flat_base_20261020/pilot031_speed_reference1342/model_001342.pt'
    state=torch.load(parent,map_location='cpu',weights_only=True)
    actor=mean_from_checkpoint(state,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx')
    data=ROOT/'reports/flat_base_speed_20261020/lean_probe003'
    provenance=json.loads((data/'report.json').read_text())
    data_contract=json.loads((data/'contract.json').read_text())
    assert provenance['status']=='evaluated' and data_contract['evaluation']['seed']==191119
    assert provenance.get('seed',191119)==191119
    z=np.load(data/'traces.npz');active=z['active'].astype(bool);ticks=z['episode_ticks']
    obs=z['observations'];gravity=obs[:,:,7:10];gyro_xy=obs[:,:,4:6]*2.
    # Early examples preserve the existing launch actions. Later teacher rows
    # are only balanced, slowly tilting states; angular z may be a deliberate spin.
    birth=active & (ticks<=75)
    steady=active & (ticks>200) & (gravity[:,:,2]<-np.cos(np.deg2rad(3.))) & (np.linalg.norm(gyro_xy,axis=-1)<.2)
    selected=birth|steady
    selected_obs=torch.from_numpy(obs[selected].copy())
    is_birth=torch.from_numpy(birth[selected].copy())
    assert selected_obs.shape[0]>10000 and is_birth.any() and (~is_birth).any()
    mean=deepcopy(actor)
    with torch.no_grad():
        mean.speed_residual.weight[4:].zero_();mean.speed_residual.bias[4:].zero_()
        translation=selected_obs[:,:1].abs()/.75>=selected_obs[:,2:3].abs()/1.5
        index=torch.where(translation,(selected_obs[:,:1]<0).long(),2+(selected_obs[:,2:3]<0).long())
        margin=mean.reference_margin.clamp(.05,1.2)[index]
        lean=mean.reference_lean_gain.clamp(-.04,.04)[(selected_obs[:,:1]<0).long()]
        reference=condition_reference(selected_obs,margin,lean)
        features=reference
        for layer in list(mean.command.base.children())[:-1]:
            features=layer(features)
        base=mean(selected_obs)[:,4:]
        turn=selected_obs[:,2:3]*(.4373/2.)
        desired=torch.cat((selected_obs[:,:1]-turn,-selected_obs[:,:1]-turn),-1)/.6
        correction=(desired-base).clamp(-2.85,2.85)
        correction[is_birth]=0.
        target=3.*torch.atanh(correction/3.)
        x=features.double();center=x.mean(0);scale=x.std(0).clamp_min(.1)
        x=torch.cat(((x-center)/scale,torch.ones(len(x),1,dtype=torch.float64)),-1)
        weights=torch.where(is_birth,10.,1.).double()
        normal=x.T@(weights[:,None]*x)/weights.sum()
        regularizer=torch.eye(65,dtype=torch.float64)*.05;regularizer[-1,-1]=0.
        solution=torch.linalg.solve(normal+regularizer,x.T@(weights[:,None]*target.double())/weights.sum())
        fitted_weight=(solution[:-1]/scale[:,None]).T.float()
        fitted_bias=(solution[-1]-(center[:,None]*solution[:-1]/scale[:,None]).sum(0)).float()
    fixture=TensorDict({'policy':torch.zeros(1,35),'critic':torch.zeros(1,83)},batch_size=[1])
    model=MLPModel(fixture,{'actor':['policy'],'critic':['critic']},'actor',6,hidden_dims=[256,128,64],
        activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.04,'std_type':'scalar'})
    model.mlp=actor;model.load_state_dict(state['actor_state_dict'],strict=True)
    names=list(dict(model.named_parameters()));ids=state['optimizer_state_dict']['param_groups'][0]['params']
    assert len(names)==len(ids)
    owner_ids=dict(zip(names,ids,strict=True))
    clear=[owner_ids['mlp.speed_residual.weight'],owner_ids['mlp.speed_residual.bias']]
    receipt=dict(schema='flat-speed-closed-form-teacher-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
        parent_checkpoint_sha256=digest(parent),data_report_sha256=digest(data/'report.json'),
        data_contract_sha256=digest(data/'contract.json'),
        data_trace_sha256=digest(data/'traces.npz'),data_seed=191119,excluded_comparison_seed=191139,
        birth_frames=int(is_birth.sum()),steady_frames=int((~is_birth).sum()),
        birth_weight=10.,ridge_regularization=.05,new_optimizer_updates=0,new_PPO_actor_updates=0,
        linear_solves=1,qualified=False,source_sha256=digest(Path(__file__)),candidates={})
    with torch.no_grad():
        for fraction in (.25,.5,1.):
            child=deepcopy(state)
            for key,new in [('weight',fitted_weight),('bias',fitted_bias)]:
                target_tensor=child['actor_state_dict']['mlp.speed_residual.'+key]
                target_tensor[4:]=target_tensor[4:]*(1-fraction)+new*fraction
            for index in clear:
                child['optimizer_state_dict']['state'].pop(index,None)
            child['infos']=deepcopy(state['infos'])
            child['infos']['closed_form_head_fits_total']=1
            child['infos']['supervised_head_fit']=dict(parent_checkpoint_sha256=digest(parent),
                training_data_report_sha256=receipt['data_report_sha256'],training_data_seed=191119,
                fraction=fraction,linear_solves=1,new_optimizer_updates=0,source_sha256=receipt['source_sha256'],
                optimizer_revision='reset only speed_residual weight/bias Adam; retain other state and counters')
            path=out/f'model_001342_fit{fraction:g}.pt';torch.save(child,path)
            recovered=mean_from_checkpoint(child,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx')
            probe=selected_obs[:4096].clone();probe[:,0].clamp_(-.75,.75);probe[:,2].clamp_(-1.5,1.5)
            assert torch.equal(recovered(probe),actor(probe))
            assert all(torch.equal(v,child['actor_state_dict'][n])for n,v in state['actor_state_dict'].items()
                       if n not in ('mlp.speed_residual.weight','mlp.speed_residual.bias'))
            assert torch.equal(child['actor_state_dict']['mlp.speed_residual.weight'][:4],state['actor_state_dict']['mlp.speed_residual.weight'][:4])
            prediction=recovered(selected_obs)[:,4:]-base
            receipt['candidates'][str(fraction)]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),
                birth_wheel_correction_abs_max_raw=float(prediction[is_birth].abs().max()),
                steady_wheel_target_rmse_raw=float((recovered(selected_obs)[~is_birth,4:]-desired[~is_birth]).square().mean().sqrt()),
                low_function_equal=True,old_weights_critic_std_RNG_counters_retained=True)
    (out/'receipt.json').write_bytes(encode(receipt))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
