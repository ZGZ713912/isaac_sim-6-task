"""Cold mastery checks before an empty rollout, retaining one live learner."""
from __future__ import annotations

from pathlib import Path
import torch


class RecoveryMasteryHook:
    def __init__(self, runner, mastery, *, completed, output, stop_requested=lambda:False):
        self.runner,self.mastery = runner,mastery
        self.completed,self.output,self.stop_requested = completed,Path(output),stop_requested
        self.native_act = runner.alg.act
        self.hook = self.act
        runner.alg.act = self.hook

    def act(self, observations):
        completed = self.completed()
        if self.mastery.due(completed):
            if self.runner.alg.storage.step != 0:
                raise RuntimeError("Mastery evaluation requires an empty native rollout")
            from evaluate_recovery_mastery import evaluate
            from train_scratch_standing import write_json
            alg,env = self.runner.alg,self.runner.env
            identities = tuple(id(x) for x in (alg,alg.actor,alg.critic,alg.optimizer,alg.storage))
            before = {k:v.detach().cpu().clone() for k,v in alg.actor.state_dict().items()}
            result = evaluate(env,alg.actor,self.mastery.config,stop_requested=self.stop_requested)
            if identities != tuple(id(x) for x in (alg,alg.actor,alg.critic,alg.optimizer,alg.storage)) or any(
                    not torch.equal(before[k],v.detach().cpu()) for k,v in alg.actor.state_dict().items()):
                raise RuntimeError("Cold mastery evaluation changed learner identity or actor state")
            result["decision"] = self.mastery.review(result["cases"],completed)
            write_json(self.output/"recovery_mastery"/f"update_{completed:05d}.json",result)
            writer = self.runner.logger.writer
            if writer is not None:
                for cls,summary in result["decision"]["per_leg_class"].items():
                    writer.add_scalar(f"RecoveryMastery/{cls}/success_rate",summary["success_rate"],completed)
                    writer.add_scalar(f"RecoveryMastery/{cls}/trials",summary["trials"],completed)
                writer.add_scalar("RecoveryMastery/complex_unlocked",float(self.mastery.unlocked),completed)
            # Evaluation replaces the previous incomplete training episodes.
            for name in ("cur_reward_sum","cur_episode_length","cur_ereward_sum","cur_ireward_sum"):
                value = getattr(self.runner.logger,name,None)
                if value is not None: value.zero_()
            if self.mastery.unlocked:
                env.reset(torch.arange(env.num_envs,device=env.device))
            observations = env.get_observations().to(self.runner.device)
        return self.native_act(observations)

    def close(self):
        if self.runner.alg.act is self.hook:
            self.runner.alg.act = self.native_act
