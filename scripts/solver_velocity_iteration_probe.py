"""One declared, zero-PPO TGS velocity-iteration counterfactual."""


def configure(config, design, iterations):
    if type(iterations) is not int or iterations not in (1, 4):
        raise ValueError("Velocity probe is bounded to1 or4 iterations")
    selected = design["stages"]["speed"]["roles"]["train"]["contract"]
    if (config["position_iterations"] != 64 or config["velocity_iterations"] != 32
            or selected["position_iterations"] != 64 or selected["velocity_iterations"] != 32
            or config["physics_dt"] != .001 or config["policy_dt"] != .005):
        raise ValueError("Need the authenticated64/32 and200/1000Hz baseline")
    config["velocity_iterations"] = iterations
    selected["velocity_iterations"] = iterations
    return dict(schema="scratch-solver-velocity-counterfactual-v1", position_iterations=64,
        original_velocity_iterations=32, requested_velocity_iterations=iterations,
        physical_dt_s=.001, policy_dt_s=.005, adoption_by_training=False,
        mechanical_guards_unchanged=True, policy_unchanged=True, new_PPO=0,
        scope="Only articulation velocity iteration count changes before construction; frozen actor, "
            "position iterations, timestep, forces, geometry, reset and original strict evaluation remain.")


def assert_observed(physics, contract):
    flags = physics["scene_flags"]
    expected_flags = {
        "physxScene:solverType": "TGS",
        "physxScene:enableExternalForcesEveryIteration": False,
        "physxScene:enableStabilization": False, "physxScene:enableCCD": False,
        "physxScene:minPositionIterationCount": 1, "physxScene:maxPositionIterationCount": 255,
        "physxScene:minVelocityIterationCount": 0, "physxScene:maxVelocityIterationCount": 255,
    }
    expected_iterations = {
        "physxArticulation:solverPositionIterationCount": 64,
        "physxArticulation:solverVelocityIterationCount": contract["requested_velocity_iterations"],
    }
    if (any(flags.get(key) != value for key, value in expected_flags.items())
            or physics["articulation_iterations"] != expected_iterations
            or physics["physical_dt_s"] != .001 or physics["policy_dt_s"] != .005
            or physics["cylinder_convex_approximation"].get("status") != "observed"
            or physics["cylinder_convex_approximation"].get("value") is not False):
        raise ValueError("Observed native physics differs from the declared velocity-only counterfactual")
