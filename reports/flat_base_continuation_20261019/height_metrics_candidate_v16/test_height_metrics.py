"""CPU checks for the diagnostic candidate; no Isaac imports or optimizer steps."""
import ast
import hashlib
from copy import deepcopy
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
source=(HERE/'train_flat_base.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='FlatHeightCommandMetrics')
namespace={}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<candidate-height-metrics>','exec'),namespace)
Metrics=namespace['FlatHeightCommandMetrics']


class Writer:
    def __init__(self):self.rows={}
    def add_scalar(self,tag,value,step):
        assert tag not in self.rows, 'No duplicate case/window/band/metric tags'
        self.rows[tag]=(float(value),step)


def example():
    metric=Metrics(3,'cpu')
    values=torch.tensor([[1.,11.,21.,31.,0.],[2.,12.,22.,32.,.5],[3.,13.,23.,33.,1.]],dtype=torch.float64)
    metric.observe(torch.tensor([.23,.305,.42],dtype=torch.float64),values,
        torch.tensor([True,True,True]),torch.tensor([False,False,False]))
    metric.observe(torch.tensor([.42,.23,.305],dtype=torch.float64),2*values,
        torch.tensor([True,False,True]),torch.tensor([False,True,False]))
    return metric


def test_height_reclassification_within_rollout_does_not_use_final_lane_height():
    metric=example()
    assert metric.frames[0].sum(-1).tolist()==[1,2,2]
    assert metric.frames[1].sum(-1).tolist()==[1,0,0]
    expected=torch.tensor([[1.,11.,21.,31.,0.],[8.,38.,68.,98.,2.5],[5.,35.,65.,95.,1.]],dtype=torch.float64)
    assert torch.equal(metric.sums[0].sum(1),expected)
    assert metric.sums[1].sum(1).tolist()==[[4.,24.,44.,64.,1.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]


def test_exact_midpoint_boundaries_form_disjoint_complete_bands():
    lo,hi=Metrics.edges_m
    heights=torch.tensor([.23,lo-1e-8,lo,hi-1e-8,hi,.42],dtype=torch.float64)
    metric=Metrics(6,'cpu');valid=torch.ones(6,dtype=torch.bool)
    metric.observe(heights,torch.ones(6,5),valid,~valid)
    assert metric.frames[0].sum(-1).tolist()==[2,2,2]
    assert metric.frames[0].sum(0).tolist()==[1,1,1,1,1,1]


def test_logging_uses_frame_weighted_group_means_and_real_step():
    metric=example();writer=Writer();metric.log(writer,'backward_edge3',torch.ones(3,dtype=torch.bool),1301)
    assert len(writer.rows)==26
    assert writer.rows['FlatBehavior/backward_edge3/height_nominal/sample_frames']==(2.,1301)
    assert writer.rows['FlatBehavior/backward_edge3/height_nominal/vx_mae_m_s']==(4.,1301)
    assert writer.rows['FlatBehavior/backward_edge3/height_high/yaw_mae_rad_s']==(17.5,1301)
    assert all(step==1301 for _,step in writer.rows.values())


def test_empty_birth_strata_record_zero_frames_without_perfect_error_values():
    writer=Writer();example().log(writer,'group',torch.ones(3,dtype=torch.bool),8)
    for band in ('height_nominal','height_high'):
        prefix='FlatBirth/group/'+band
        assert writer.rows[prefix+'/sample_frames']==(0.,8)
        assert not any(t.startswith(prefix+'/') and not t.endswith('/sample_frames') for t in writer.rows)


def test_direct_and_legacy_views_partition_whole_group_samples():
    metric=example();whole,direct,legacy=Writer(),Writer(),Writer()
    metric.log(whole,'all',torch.tensor([True,True,True]),20)
    metric.log(direct,'all/rmcs_direct',torch.tensor([True,False,True]),20)
    metric.log(legacy,'all/legacy_ramp',torch.tensor([False,True,False]),20)
    for window in ('FlatBehavior','FlatBirth'):
        for band in Metrics.bands:
            suffix='/'+band+'/sample_frames'
            w=whole.rows[window+'/all'+suffix][0]
            d=direct.rows[window+'/all/rmcs_direct'+suffix][0]
            l=legacy.rows[window+'/all/legacy_ramp'+suffix][0]
            assert w==d+l
            if w:
                for tag in Metrics.tags:
                    weighted=0.
                    if d:weighted+=d*direct.rows[window+'/all/rmcs_direct/'+band+'/'+tag][0]
                    if l:weighted+=l*legacy.rows[window+'/all/legacy_ramp/'+band+'/'+tag][0]
                    assert weighted/w==whole.rows[window+'/all/'+band+'/'+tag][0]


def test_metrics_do_not_mutate_inputs_or_consume_CPU_RNG():
    heights=torch.tensor([.23,.305,.42]);values=torch.arange(15,dtype=torch.float32).reshape(3,5).requires_grad_()
    valid=torch.tensor([True,False,True]);birth=~valid
    originals=[x.detach().clone() for x in (heights,values,valid,birth)]
    rng=torch.get_rng_state().clone();metric=Metrics(3,'cpu')
    metric.observe(heights,values,valid,birth);metric.log(Writer(),'group',torch.ones(3,dtype=torch.bool),42)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip((heights,values,valid,birth),originals))
    assert not metric.sums.requires_grad and metric.sums.grad_fn is None and values.grad is None


def test_clear_does_not_leak_old_rollout_samples():
    metric=example();metric.clear();writer=Writer();metric.log(writer,'group',torch.ones(3,dtype=torch.bool),9)
    assert not metric.sums.any() and not metric.frames.any()
    assert len(writer.rows)==6 and all(v==(0.,9) for v in writer.rows.values())


def test_candidate_learning_code_matches_frozen_parent_after_removing_only_diagnostics():
    removed=[]
    class Strip(ast.NodeTransformer):
        def visit_ClassDef(self,n):
            if n.name=='FlatHeightCommandMetrics':removed.append('class');return None
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='height_metrics':
                removed.append('init');return None
            target=n.targets[0]
            if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='report' and isinstance(target.slice,ast.Constant) and target.slice.value=='height_command_metrics_contract':
                removed.append('contract');return None
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='height_metrics':
                assert n.value.func.attr in ('observe','log','clear');removed.append(n.value.func.attr);return None
            return self.generic_visit(n)
    candidate=Strip().visit(ast.parse(source));frozen=HERE/'frozen_parent_train_flat_base.py'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='94f54161596455b7f63187b8963373bb84b70caf446ae297af63abc7cc053e68'
    old=ast.parse(frozen.read_text())
    assert sorted(removed)==sorted(['class','init','contract','observe','log','clear'])
    assert ast.dump(candidate,include_attributes=False)==ast.dump(old,include_attributes=False)


def test_tensorboard_roundtrip_keeps_all_case_height_tags(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with SummaryWriter(str(tmp_path)) as writer:
        for case in ('backward_low','backward_edge3'):
            example().log(writer,case,torch.ones(3,dtype=torch.bool),1302)
    events=EventAccumulator(str(tmp_path),size_guidance={'scalars':0});events.Reload()
    tags=events.Tags()['scalars'];assert len(tags)==52
    for case in ('backward_low','backward_edge3'):
        for window in ('FlatBehavior','FlatBirth'):
            for band in Metrics.bands:
                rows=events.Scalars(f'{window}/{case}/{band}/sample_frames')
                assert len(rows)==1 and rows[0].step==1302
    assert Metrics.contract()['height_bin_edges_m']==list(Metrics.edges_m)
