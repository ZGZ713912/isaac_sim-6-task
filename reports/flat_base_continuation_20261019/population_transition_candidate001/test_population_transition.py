"""Continuation clock and authentication checks for a population-only change."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest

text=(Path(__file__).with_name('train_flat_base.py')).read_text()
nodes=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef)
       and node.name in ('restored_training_frames','validate_population_transition')]
namespace={}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<population-transition>','exec'),namespace)
frames=namespace['restored_training_frames']; validate=namespace['validate_population_transition']


def example():
    old=dict(allowed_num_envs=[64,128,256,512],default_num_envs=512,num_steps_per_env=96,
             reward=6.,policy_dt=.02,physics_dt=.001,epochs=3)
    new={**old,'allowed_num_envs':[4096],'default_num_envs':4096}
    info=dict(num_envs=512,successful_updates_total=1248,
              source_sha256={'scripts/train_flat_base.py':'old','env.py':'same'})
    sources={**info['source_sha256'],'scripts/train_flat_base.py':'new'}
    transition=dict(schema='flat-training-population-transition-v1',old_num_envs=512,
        new_num_envs=4096,num_steps_per_env=96,inherited_successful_updates=1248,
        inherited_successful_training_frames=61341696,additional_critic_warmup_updates=0)
    return info,old,new,dict(population_transition=transition),sources


def test_exact_declared_population_change_retains_parent_frames():
    info,old,new,revision,sources=example();before=deepcopy((info,old,new,revision,sources))
    assert validate(info,old,new,4096,revision,sources)==revision['population_transition']
    assert (info,old,new,revision,sources)==before
    assert frames(info,old)==61341696


def test_mixed_population_exposure_is_summed_rather_than_retroactively_rescaled():
    info,old,*_=example()
    info.update(num_envs=4096,successful_updates_total=1280,
        training_transitions_total=61341696+32*4096*96,population_transition={'new_num_envs':4096})
    assert frames(info,old)==73924608
    assert frames(info,old)!=1280*4096*96


def test_missing_saved_count_after_population_transition_is_rejected():
    info,old,*_=example();info['population_transition']={'new_num_envs':4096}
    with pytest.raises(ValueError,match='saved cumulative'):frames(info,old)


@pytest.mark.parametrize('key,value',[('reward',7.),('policy_dt',.005),('physics_dt',.005),('epochs',4)])
def test_population_change_cannot_smuggle_learning_changes(key,value):
    info,old,new,revision,sources=example();new[key]=value
    with pytest.raises(ValueError,match='two population'):validate(info,old,new,4096,revision,sources)


@pytest.mark.parametrize('change',['bad_count','new_env_source','missing_entry','rewritten_frames','warmup','wrong_steps'])
def test_mismatched_population_or_provenance_is_rejected(change):
    info,old,new,revision,sources=example();count=4096
    if change=='bad_count':count=2048
    elif change=='new_env_source':sources['env.py']='changed'
    elif change=='missing_entry':sources['scripts/train_flat_base.py']='old'
    elif change=='rewritten_frames':revision['population_transition']['inherited_successful_training_frames']=1248*4096*96
    elif change=='warmup':revision['population_transition']['additional_critic_warmup_updates']=10
    else:revision['population_transition']['num_steps_per_env']=24
    with pytest.raises(ValueError):validate(info,old,new,count,revision,sources)


@pytest.mark.parametrize('value',[-1,1.5,True,float('nan')])
def test_corrupt_saved_frame_count_is_rejected(value):
    info,old,*_=example();info['training_transitions_total']=value
    with pytest.raises(ValueError):frames(info,old)


def test_native_probe_bypasses_optimizer_wrapper_and_has_separate_status():
    tree=ast.parse(text);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    probe=next(n for n in ast.walk(main) if isinstance(n,ast.If)
               and ast.unparse(n.test)=='args.capacity_probe' and any(isinstance(x,ast.Expr)
               and isinstance(x.value,ast.Call) and ast.unparse(x.value.func)=='probe_population_capacity' for x in n.body))
    assert 'runner.alg.update = update' not in ast.unparse(probe.body)
    assert 'runner.alg.update = update' in ast.unparse(probe.orelse)
    assert 'capacity_complete' in ast.unparse(probe.body)
    assert 'runner.learn' not in ast.unparse(probe.body)
