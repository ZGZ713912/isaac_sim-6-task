"""Training-only backward reset difficulty over already admitted static poses."""
from copy import deepcopy
import math
import random


def contract():
    return dict(schema="wheel-leg-recovery-start-curriculum-v1",
        pose_source="existing_native_admitted_training_static_poses",
        belly_orbit_limits_deg=[20., 40., 70., 180.],
        final_stage="original_all_family_sampler", full_family_practice_fraction=.1,
        window_episodes=64, advance_success_rate=.75,
        success="actual_recovery_plus_existing_3s_stable_hold",
        evaluation_sampling="unchanged", admission_gates="unchanged",
        state_writes="episode_reset_only", teacher=False)


def validate(config):
    if config != contract():
        raise ValueError("Use the complete recovery start curriculum contract")
    return deepcopy(config)


class RecoveryStartCurriculum:
    """Advance only on completed training episodes, never PPO/reward counters.

    Birth-stage attribution prevents an older, easier episode from advancing
    a later stage. The interleaved full-family pool never supplies gate credit.
    No evaluation result or global capability admission is changed here.
    """
    def __init__(self, bank, config, seed):
        self.config = validate(config)
        self.rng = random.Random(seed)
        self.pools = []
        for limit in self.config["belly_orbit_limits_deg"]:
            pool = []
            for i, pose in enumerate(bank["poses"]):
                r = pose.get("recovery", {})
                if (r.get("split") == "train" and r.get("dynamic") is False
                        and r.get("family") == "belly_down"
                        and r.get("native_passive_admitted") is True
                        and r.get("cad_surface_screen_passed") is True
                        and pose.get("admitted_as_training_reset") is True
                        and max(map(abs, r["orbital_pair_deg"])) <= limit):
                    pool.append(i)
            if not pool:
                raise ValueError("Recovery start stage has no admitted training poses")
            self.pools.append(pool)
        self.phase = 0
        self.window = []
        self.trials = [0]*5
        self.successes = [0]*5
        self.advancements = []

    def sample(self, full_sampler):
        if self.phase == len(self.pools):
            return full_sampler(), self.phase
        if self.rng.random() < self.config["full_family_practice_fraction"]:
            return full_sampler(), -1
        return self.rng.choice(self.pools[self.phase]), self.phase

    def observe(self, birth_phases, successes):
        """Consume a whole policy interval atomically, in deterministic lane order."""
        if len(birth_phases) != len(successes):
            raise ValueError("One success flag is required per completed episode")
        if (any(type(p) is not int or not -1 <= p <= 4 for p in birth_phases)
                or any(type(value) is not bool for value in successes)):
            raise ValueError("Need integer birth phases and actual boolean success flags")
        current = self.phase
        credited = [success for phase, success in zip(birth_phases, successes) if phase == current]
        self.trials[current] += len(credited)
        self.successes[current] += sum(credited)
        self.window = (self.window+credited)[-self.config["window_episodes"]:]
        if (current < len(self.pools) and len(self.window) == self.config["window_episodes"]
                and sum(self.window)/len(self.window) >= self.config["advance_success_rate"]):
            self.advancements.append(dict(from_phase=current, trials=len(self.window), successes=sum(self.window)))
            self.phase += 1
            self.window = []
        return dict(phase=current, episodes=len(credited), successes=sum(credited))

    def state_dict(self):
        return deepcopy(dict(schema="wheel-leg-recovery-start-state-v1", config=self.config,
            rng=self.rng.getstate(), phase=self.phase, window=self.window, trials=self.trials,
            successes=self.successes, advancements=self.advancements))

    @staticmethod
    def validate_state(state):
        if not isinstance(state, dict) or set(state) != {
                "schema", "config", "rng", "phase", "window", "trials", "successes", "advancements"}:
            raise ValueError("Incomplete recovery start checkpoint")
        c = validate(state["config"])
        if (state["schema"] != "wheel-leg-recovery-start-state-v1"
                or type(state["phase"]) is not int or not 0 <= state["phase"] <= 4
                or not isinstance(state["window"], list) or len(state["window"]) > c["window_episodes"]
                or any(type(v) is not bool for v in state["window"])
                or any(not isinstance(state[k], list) or len(state[k]) != 5
                    or any(type(v) is not int or v < 0 for v in state[k]) for k in ("trials", "successes"))
                or any(s > n for s, n in zip(state["successes"], state["trials"]))
                or not isinstance(state["advancements"], list)
                or len(state["advancements"]) != state["phase"]):
            raise ValueError("Invalid recovery start checkpoint counters")
        phase = state["phase"]
        if (len(state["window"]) > state["trials"][phase]
                or sum(state["window"]) > state["successes"][phase]
                or any(state["trials"][i] for i in range(phase+1,5))
                or (phase < 4 and len(state["window"]) == c["window_episodes"]
                    and sum(state["window"])/len(state["window"]) >= c["advance_success_rate"])):
            raise ValueError("Recovery start window exceeds observed episodes")
        for i, row in enumerate(state["advancements"]):
            if (not isinstance(row, dict) or set(row) != {"from_phase", "trials", "successes"} or row["from_phase"] != i
                    or row["trials"] != c["window_episodes"]
                    or type(row["successes"]) is not int
                    or not math.ceil(c["advance_success_rate"]*row["trials"]) <= row["successes"] <= row["trials"]
                    or state["trials"][i] < row["trials"] or state["successes"][i] < row["successes"]):
                raise ValueError("Recovery start checkpoint lacks successful advancement evidence")
        try:
            random.Random().setstate(state["rng"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid recovery start RNG state") from error

    def load_state_dict(self, state):
        self.validate_state(state)
        if state["config"] != self.config:
            raise ValueError("Recovery start checkpoint contract differs")
        self.rng.setstate(state["rng"])
        for key in ("phase", "window", "trials", "successes", "advancements"):
            setattr(self, key, deepcopy(state[key]))


def select_start(sampler, completed, *, training):
    return sampler.sample_start(completed) if training else (sampler.sample(completed), -1)


def optional_state(sampler):
    course = getattr(sampler, "start_curriculum", None)
    return course.state_dict() if course is not None else None


def restore_optional(sampler, state):
    if state is not None:
        sampler.start_curriculum.load_state_dict(state)
