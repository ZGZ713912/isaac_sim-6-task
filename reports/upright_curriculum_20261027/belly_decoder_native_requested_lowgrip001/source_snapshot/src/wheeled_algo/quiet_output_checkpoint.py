"""Native PPO on six existing quiet-head output tensors only.

No additional network, optimizer slots, sensor coordinates, gate or action
transform is introduced. The authenticated runtime is rebuilt from its V4
zero-PPO function root.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.function_root_checkpoint import environment_revision

SCHEMA='robust-getup-checkpoint-v7'
VERSION='quiet-output-function-root-v1'
MODULE='src/wheeled_algo/quiet_output_checkpoint.py'
LEARNED_KEYS=tuple('mlp.mean.'+head+'.2.'+suffix for head in ('quiet','quiet_low','quiet_high') for suffix in ('weight','bias'))
ENVIRONMENT_MODULE='src/wheeled_tasks/chassis/robust_getup_env.py'
TRAINING_FIELDS=frozenset(('version','motion_fraction','curriculum_updates','origin_updates',
    'quiet_training','quiet_spectral_cost','quiet_critic_anchor','standing_support_training',
    'quiet_position_weight','quiet_height_probabilities','standing_early_position_weight',
    'standing_drift_weight','standing_drift_speed_scale_m_s'))
STEPS_PER_UPDATE=12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def learned_digest(actor):
    h=hashlib.sha256()
    for key in LEARNED_KEYS:
        value=actor[key].detach().cpu().contiguous()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>=0


def validate_training_settings(settings,root_settings):
    if ({k:v for k,v in settings.items() if k not in TRAINING_FIELDS}
            != {k:v for k,v in root_settings.items() if k not in TRAINING_FIELDS}
            or settings.get('gap_entry_optimizer_layout') is not None):
        raise ValueError('Quiet output training changed runtime function or observation settings')


def optimizer_actor_names(mean):
    # Native SensorConditionedMean owns log_std before its child modules.
    names=['mlp.log_std']+['mlp.mean.'+name for name,p in mean.named_parameters() if p.requires_grad]
    if len(names)!=28 or any(key not in names for key in LEARNED_KEYS):
        raise ValueError('Unexpected authenticated legacy actor optimizer layout')
    return names


def make_progress(source_path,source,actor,count):
    if not integer(count):raise ValueError('Require a nonnegative completed PPO increment')
    start=source['infos']['completed_updates']
    return dict(schema=VERSION,source_checkpoint=str(Path(source_path).resolve()),source_sha256=digest(source_path),
        source_learning_clock=start,new_optimizer_updates=count,completed_learning_clock=start+count,
        learned_state_sha256=learned_digest(actor),scope='six-existing-quiet-output-tensors-only',
        native_Adam_steps_per_update=STEPS_PER_UPDATE)


def initialize_state(source_path,source,module_sha,names):
    prior=source['infos']
    if (prior.get('schema')!='robust-getup-checkpoint-v4' or prior['settings'].get('quiet_stance_authority_version')!='quiet-stance-authority-v1'
            or prior['settings'].get('gap_entry_optimizer_layout') is not None or len(names)!=28 or len(set(names))!=28
            or not all(k in source['actor_state_dict'] for k in names) or not all(k in names for k in LEARNED_KEYS)):
        raise ValueError('Initialize once from an authenticated V4 policy and original optimizer namespace')
    result=deepcopy(source);info=result['infos']
    info.update(schema=SCHEMA,quiet_output_only_tuning=True,quiet_only_tuning=True,quiet_velocity_only_tuning=False,
        motion_output_only_tuning=False,recovery_only_tuning=False,entry_recovery_only_tuning=False,hardware_ready=False)
    info['source_sha256'][MODULE]=module_sha
    info['policy_function_root']=dict(schema=VERSION,checkpoint=str(Path(source_path).resolve()),sha256=digest(source_path),optimizer_actor_parameter_names=list(names))
    info['policy_function_learning']=make_progress(source_path,source,result['actor_state_dict'],0)
    info['policy_function_stage']='optimizer-initialization'
    return result


def tensor_layout(actual,expected):
    return (actual.keys()==expected.keys() and all(isinstance(actual[k],torch.Tensor) and actual[k].shape==v.shape
        and actual[k].dtype==v.dtype and torch.isfinite(actual[k]).all() for k,v in expected.items()))


def validate_state(state,root,source,names):
    info,prior=state['infos'],source['infos'];root_info=root['infos']
    origin=info.get('policy_function_root',{});record=info.get('policy_function_learning',{})
    if (info.get('schema')!=SCHEMA or root_info.get('schema')!='robust-getup-checkpoint-v4' or origin.get('schema')!=VERSION
            or record.get('schema')!=VERSION or record.get('scope')!='six-existing-quiet-output-tensors-only'
            or record.get('native_Adam_steps_per_update')!=STEPS_PER_UPDATE
            or info.get('quiet_output_only_tuning') is not True or info.get('quiet_only_tuning') is not True
            or info.get('quiet_velocity_only_tuning') is not False or info.get('motion_output_only_tuning') is not False or info.get('recovery_only_tuning') is not False or info.get('entry_recovery_only_tuning') is not False
            or info.get('hardware_ready') is not False or origin.get('optimizer_actor_parameter_names')!=names):
        raise ValueError('Require explicit quiet-output function ancestry and optimizer layout')
    if prior.get('schema')==SCHEMA and not exact_state_equal(origin,prior.get('policy_function_root')):
        raise ValueError('Learned checkpoint changed its function ancestry')
    count=record.get('new_optimizer_updates');start=prior['completed_updates']
    if (not integer(count) or not integer(start) or record.get('source_learning_clock')!=start
            or record.get('completed_learning_clock')!=start+count or info.get('completed_updates')!=start+count):
        raise ValueError('Quiet output PPO clocks are inconsistent')
    validate_training_settings(info['settings'],root_info['settings'])
    revision=environment_revision(root_info['source_sha256'],info['source_sha256'],info['settings'])
    if info.get('policy_function_environment_revision')!=revision:
        raise ValueError('Training environment source revision must be explicit and neutral')
    for name,sha in root_info['source_sha256'].items():
        if name.startswith('src/') and name!=ENVIRONMENT_MODULE and info['source_sha256'].get(name)!=sha:
            raise ValueError('Quiet output training changed a function or observation source')
    for key in ('base_checkpoint','base_checkpoint_sha256','bank_sha256','actor_variant'):
        if info.get(key)!=root_info.get(key):raise ValueError('Quiet output training changed policy identity')
    actor=state['actor_state_dict']
    if not tensor_layout(actor,root['actor_state_dict']):raise ValueError('Actor layout or finiteness differs')
    if any(not torch.equal(v,actor[k]) for k,v in root['actor_state_dict'].items() if k not in LEARNED_KEYS):
        raise ValueError('Training changed a frozen actor tensor, feature basis, entry or std')
    if learned_digest(actor)!=record.get('learned_state_sha256'):raise ValueError('Learned output digest differs')
    if not tensor_layout(state['critic_state_dict'],root['critic_state_dict']):raise ValueError('Critic layout or finiteness differs')
    optimizer=state['optimizer_state_dict'];expected=deepcopy(root['optimizer_state_dict'])
    if len(optimizer['param_groups'])!=2:raise ValueError('Optimizer groups differ')
    for actual,before in zip(optimizer['param_groups'],expected['param_groups'],strict=True):
        before['lr']=actual['lr']
    if not exact_state_equal(optimizer['param_groups'],expected['param_groups']) or optimizer['state'].keys()!=expected['state'].keys():
        raise ValueError('Optimizer layout or unreviewed settings changed')
    actor_ids=expected['param_groups'][0]['params'];critic_ids=expected['param_groups'][1]['params']
    if len(actor_ids)!=len(names):raise ValueError('Actor optimizer namespace differs')
    learned_ids={actor_ids[names.index(k)] for k in LEARNED_KEYS}
    for index in actor_ids:
        if index not in learned_ids and not exact_state_equal(optimizer['state'].get(index),expected['state'].get(index)):
            raise ValueError('Frozen actor Adam history changed')
    for index in (*sorted(learned_ids),*critic_ids):
        slot=optimizer['state'].get(index);before=source['optimizer_state_dict']['state'].get(index)
        template=expected['state'].get(index)
        if (not isinstance(slot,dict) or slot.keys()!=template.keys() or not tensor_layout(slot,template)
                or slot['step'].numel()!=1 or slot['step'].item()!=before['step'].item()+STEPS_PER_UPDATE*count):
            raise ValueError('Native learned/critic Adam history has wrong layout or step accounting')
    stage=info.get('policy_function_stage')
    if stage=='optimizer-initialization':
        expected=initialize_state(origin['checkpoint'],root,info['source_sha256'][MODULE],names)
        if not exact_state_equal(state,expected):raise ValueError('Initialization changed saved state or unrelated metadata')
    elif stage=='native-training':
        if prior.get('schema')!=SCHEMA:raise ValueError('Training requires authenticated initialization')
        if count==0:
            before=deepcopy(source['optimizer_state_dict'])
            for a,b in zip(before['param_groups'],optimizer['param_groups'],strict=True):a['lr']=b['lr']
            if (not exact_state_equal(actor,source['actor_state_dict']) or not exact_state_equal(state['critic_state_dict'],source['critic_state_dict'])
                    or not exact_state_equal(before,optimizer)):
                raise ValueError('Zero completed PPO updates changed actor, critic or Adam')
    else:raise ValueError('Unknown quiet output function-root stage')
