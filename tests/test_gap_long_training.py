import json
from pathlib import Path
import sys

import pytest
import torch

from run_gap_long_training import (Campaign,adaptation,behavior_metrics,stages,verify_restore)


def test_complete_long_schedule_cannot_refund_or_overshoot_the_user_budget():
    schedule=stages(2048)
    assert len(schedule)==8 and sum(s["updates"] for s in schedule)==2048
    assert [s["offset"] for s in schedule]==list(range(0,2048,256))
    assert [s["severity"] for s in schedule]==[.6,.6,.8,.8,1.,1.,1.,1.]
    assert [s["friction_min"] for s in schedule]==[.3]*4+[.2]*4
    for args in ((2047,256),(2048,257),(8192,256),(128,64)):
        with pytest.raises(ValueError):
            stages(*args)


def test_two_behavior_regressions_reduce_lr_and_hold_the_domain_ramp():
    history=[dict(score=.8,training_severity=.6),dict(score=.6,training_severity=.6),
        dict(score=.55,training_severity=.6)]
    assert adaptation(history,1.)==dict(actor_lr=5e-5,severity=.6)
    history.append(dict(score=.9,training_severity=.6))
    assert adaptation(history,1.)==dict(actor_lr=5e-5,severity=1.)
    history.extend([dict(score=.7,training_severity=1.),dict(score=.6,training_severity=1.)])
    assert adaptation(history,1.)["actor_lr"]==2.5e-5


def test_survival_without_recovery_or_standing_cannot_rank_as_a_good_policy(tmp_path):
    rows=[]
    for case,count in (("real_start",12),("stand",21),("forward",12),("left_symmetric",6)):
        rows.extend(dict(name=f"{case}_{i}",behavior_case=case,survived=True,recovered=False,
            tail_stable_fraction=0.,max_leg_speed_rad_s=0.) for i in range(count))
    evaluation=tmp_path/"evaluation"
    evaluation.mkdir()
    (evaluation/"gap_summary.json").write_text(json.dumps(dict(rows=rows)))
    (evaluation/"report.json").write_text(json.dumps(dict(lanes=[dict(r,followup_success=False) for r in rows])))
    quality=tmp_path/"standing.json"
    quality.write_text(json.dumps(dict(episodes=21,pose_and_position_pass=0)))
    result=behavior_metrics(tmp_path,quality)
    assert result["score"]==pytest.approx(.05)
    assert result["entry"]["followup_success"]==0 and not result["formal_acceptance"]
    rows.pop()
    (evaluation/"gap_summary.json").write_text(json.dumps(dict(rows=rows)))
    with pytest.raises(ValueError,match="populations"):
        behavior_metrics(tmp_path,quality)


def state():
    return dict(actor_state_dict={"weight":torch.ones(2)},critic_state_dict={"weight":torch.zeros(2)},
        optimizer_state_dict=dict(state={0:dict(step=torch.tensor(3.),exp_avg=torch.ones(2))},
            param_groups=[dict(name="actor",lr=.0001,params=[0],eps=1e-8)]),
        infos=dict(completed_updates=569,environment_rng=torch.arange(3),domain_rng=torch.arange(4),
            torch_rng=torch.arange(5),cuda_rng=[torch.arange(6)]))


def test_optimizer_rng_restore_allows_only_declared_learning_rate_revision(tmp_path):
    a,b=tmp_path/"a.pt",tmp_path/"b.pt"
    original=state()
    torch.save(original,a)
    changed=state()
    changed["optimizer_state_dict"]["param_groups"][0]["lr"]=3e-5
    torch.save(changed,b)
    verify_restore(a,b,tmp_path/"ok.json")
    changed["optimizer_state_dict"]["state"][0]["step"]+=1
    torch.save(changed,b)
    with pytest.raises(ValueError,match="restore differs"):
        verify_restore(a,b,tmp_path/"bad.json")
    assert not json.loads((tmp_path/"bad.json").read_text())["checks"]["Adam_moments_and_steps"]


def test_zero_exit_incomplete_native_job_is_rejected_and_attempts_are_charged(tmp_path):
    plan=dict(output=str(tmp_path),prior_experiment_consumed=1448,initial_learning_clock=569,
        initial_checkpoint="unused.pt",source_sha256={},bank=str(tmp_path/"bank.json"))
    bank=Path(plan["bank"])
    bank.write_text("{}")
    from train_robust_getup import digest
    plan["bank_sha256"]=digest(bank)
    campaign=Campaign(plan)
    progress=tmp_path/"worker/progress.json"
    script=tmp_path/"worker.py"
    script.write_text("from pathlib import Path\nimport json\np=Path("+repr(str(progress))+")\n"
        "p.parent.mkdir()\np.write_text(json.dumps(dict(status='failed',new_optimizer_updates=0,"
        "new_optimizer_attempts=1,completed_updates=569)))\n")
    with pytest.raises(RuntimeError,match="incomplete"):
        campaign.job("fake_incomplete",[sys.executable,str(script)],progress,parent=tmp_path/"absent.pt")
    assert campaign.state["new_optimizer_attempts"]==1
    assert campaign.state["actual_experiment_consumed"]==1449
    assert campaign.state["new_optimizer_updates"]==0
    with pytest.raises(ValueError,match="already started"):
        Campaign(plan)
