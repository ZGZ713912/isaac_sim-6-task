"""One live learner; cold mastery reviews only before empty native rollouts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


def unchanged(actual, expected):
    if isinstance(expected,torch.Tensor):
        return torch.equal(actual.detach().cpu(),expected.detach().cpu())
    if isinstance(expected,dict):
        return actual.keys() == expected.keys() and all(unchanged(actual[k],v) for k,v in expected.items())
    if isinstance(expected,(list,tuple)):
        return len(actual) == len(expected) and all(unchanged(a,b) for a,b in zip(actual,expected))
    return actual == expected


class GlobalCurriculumHook:
    def __init__(self, runner, course, *, completed, output, stop_requested=lambda:False):
        self.runner,self.course = runner,course
        self.completed,self.output,self.stop_requested = completed,Path(output),stop_requested
        self.native_act = runner.alg.act
        self.hook = self.act
        runner.alg.act = self.hook

    def review(self, completed, *, force=False):
        from evaluate_global_curriculum import evaluate_skill,complex_cases
        from evaluate_recovery_mastery import evaluate as evaluate_recovery
        from train_scratch_standing import write_json,digest
        alg,env = self.runner.alg,self.runner.env
        if alg.storage.step != 0:raise RuntimeError("Global reviews require an empty rollout")
        identities = tuple(id(x) for x in (alg,alg.actor,alg.critic,alg.optimizer,alg.storage))
        saved = [deepcopy(x.state_dict()) for x in (alg.actor,alg.critic,alg.optimizer)]
        jobs = self.course.jobs(completed,force)
        if not jobs:return None
        regular = force or completed % self.course.config["evaluation_interval_updates"] == 0
        directory = self.output/"global_curriculum"/f"update_{completed:05d}"
        directory.mkdir(parents=True,exist_ok=False)
        results,summaries = {},{}
        for skill,level in jobs:
            if skill == "belly":
                result = evaluate_recovery(env,alg.actor,self.course.recovery.config,stop_requested=self.stop_requested)
                decision = self.course.recovery.review(result["cases"],completed)
                summaries[skill] = dict(passed=decision["all_classes_passed"],per_leg_class=decision["per_leg_class"])
                result["decision"] = decision
            elif skill == "complex":
                cases = complex_cases(env);rows = []
                for i in range(0,len(cases),env.num_envs):
                    result = evaluate_recovery(env,alg.actor,self.course.recovery.config,
                        stop_requested=self.stop_requested,cases_override=cases[i:i+env.num_envs])
                    rows.extend(result["cases"])
                result = dict(cases=rows,scope="curriculum_admission_only",new_PPO=0)
                summaries[skill] = self.course.review_complex(rows,completed)
            else:
                result,trace = evaluate_skill(env,alg.actor,self.course,skill,level,stop_requested=self.stop_requested)
                path = directory/f"{skill}_level{level}.npz"
                np.savez_compressed(path,**trace)
                result.update(trace_file=path.name,trace_sha256=digest(path))
                summaries[skill] = self.course.review_skill(skill,level,result["cases"],completed)
            results[skill] = result
        if identities != tuple(id(x) for x in (alg,alg.actor,alg.critic,alg.optimizer,alg.storage)) or any(
                not unchanged(x.state_dict(),s) for x,s in zip((alg.actor,alg.critic,alg.optimizer),saved)) or alg.storage.step != 0:
            raise RuntimeError("Cold global review changed learner weights, Adam or rollout storage")
        decision = self.course.finish_review(summaries,completed,regular=regular)
        report = dict(schema="wheel-leg-global-cold-review-v1",completed_updates=completed,
            decision=decision,results=results,learner_state_unchanged=True,new_PPO=0,
            physics_dt_s=env.dt,policy_dt_s=env.policy_dt,scope="curriculum_admission_only",
            hardware_ready=False,full_capability_qualified=False)
        write_json(directory/"review.json",report)
        writer = self.runner.logger.writer
        if writer is not None:
            for skill,summary in summaries.items():
                writer.add_scalar(f"Curriculum/{skill}/cold_pass",float(summary["passed"]),completed)
            for skill,level in self.course.levels.items():
                writer.add_scalar(f"Curriculum/{skill}/mastered_levels",level,completed)
            writer.add_scalar("Curriculum/domain_index",self.course.domain_index,completed)
            writer.add_scalar("Curriculum/all_nominal_skills_mastered",float(self.course.all_mastered),completed)
            if "belly" in summaries:
                for cls,summary in summaries["belly"]["per_leg_class"].items():
                    writer.add_scalar(f"RecoveryMastery/{cls}/success_rate",summary["success_rate"],completed)
        for name in ("cur_reward_sum","cur_episode_length","cur_ereward_sum","cur_ireward_sum"):
            value = getattr(self.runner.logger,name,None)
            if value is not None:value.zero_()
        env.reset(torch.arange(env.num_envs,device=env.device))
        return report

    def act(self, observations):
        completed = self.completed()
        if self.course.due(completed):
            self.review(completed)
            observations = self.runner.env.get_observations().to(self.runner.device)
        return self.native_act(observations)

    def close(self):
        if self.runner.alg.act is self.hook:self.runner.alg.act = self.native_act
