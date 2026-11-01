"""Override physical differences while inheriting the reference task config."""
from __future__ import annotations

from copy import deepcopy
import json

from .model import ACTIVE, BUNDLE, JOINT_ALIASES, ROOT
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
import isaaclab.sim as sim
from isaaclab.utils import configclass
from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import (
    WheelbipeV14FlatEnvCfg, WheelbipeV14RoughEnvCfg,
)


def configure(cfg):
    from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202
    asset = FullRobotV202()
    # Construct the same authenticated response components used in our native
    # vehicle probe, independently of the training curriculum.
    _, response = asset.components(1, "cpu")
    friction = asset.leg_friction(1, "cpu")
    native = response.native_parameters()
    aliases = lambda names: [JOINT_ALIASES[n] for n in names]
    legs = [ACTIVE[i] for i in (0,1,3,4)]
    wheels = [ACTIVE[i] for i in (2,5)]
    springs = asset.manifest["spring_joint_names"]
    passive = [n for n in asset.manifest["tree_joint_names"] if n not in (*ACTIVE,*springs)]
    by_name = {joint["name"]:joint for joint in asset.spec["joints"]}
    def actuator(names, cap, damping=0., **kwargs):
        return IdealPDActuatorCfg(joint_names_expr=aliases(names), stiffness=0.,damping=damping,
            effort_limit=cap,effort_limit_sim=cap,
            velocity_limit_sim={JOINT_ALIASES[n]:float(by_name[n]["limit"]["velocity"]) for n in names},**kwargs)
    cfg.robot_cfg = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
        spawn=sim.UsdFileCfg(usd_path=str(BUNDLE/"robot.usda"),activate_contact_sensors=True,
            copy_from_source=True,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,
                angular_damping=0.,max_depenetration_velocity=1.),
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                solver_position_iteration_count=12,solver_velocity_iteration_count=6)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.32),
            joint_pos={JOINT_ALIASES[n]:v for n,v in asset.manifest["nominal_joint_pos"].items()},
            joint_vel={".*":0.}),
        actuators={
            "legs_act":actuator(legs,friction.native_effort_limit,armature=0.),
            "wheel":actuator(wheels,response.native_effort_limit,
                armature={JOINT_ALIASES[n]:native["armature"][n] for n in wheels},
                friction={JOINT_ALIASES[n]:native["frictionloss"][n] for n in wheels},
                dynamic_friction={JOINT_ALIASES[n]:native["dynamic_friction"][n] for n in wheels},
                viscous_friction=0.),
            "legs_inact":actuator(passive,100.,damping=.002,armature=0.,friction=0.,dynamic_friction=0.),
            "spring":actuator(springs,1500.,damping=.002,armature=0.,friction=0.,dynamic_friction=0.),
        },soft_joint_pos_limit_factor=1.)
    cfg.legs_act_name = aliases(legs)
    cfg.legs_inact_name = aliases(passive)
    cfg.wheel_name = aliases(wheels)
    cfg.spring_name = aliases(springs)
    cfg.use_gimbal = False
    cfg.gimbal_yaw_name, cfg.gimbal_pitch_name = [], []
    cfg.gimbal_heading_control_cfg = {"enabled":False}
    cfg.gimbal_spin_translate_cfg = {"enabled":False}
    if hasattr(cfg.commands,"special_modes"):
        cfg.commands.special_modes = {k:v for k,v in cfg.commands.special_modes.items() if not k.startswith("gimbal")}
    cfg.use_act_delay = False  # the calibrated wheel delay is applied once below the policy
    cfg.use_obs_delay = False  # explicit v2 policy-frame age is applied in the adapter
    cfg.spring_settings = dict(mode="measured",constant_force=0.,random_force=None,damping=False)
    cfg.leg_action_scale = .5  # preserve the Wheelbipe action parameterization
    cfg.max_wheel_torque = 4.5
    cfg.max_wheel_vel = 60.
    cfg.default_height_cmd = .305
    cfg.height_range = [.23,.42]
    cfg.leg_length_range = [.16,.36]
    cfg.wheel_motor_z_axis_align_ref_y_offset = .1833
    cfg.scene.replicate_physics = False  # retain the six external constraints per clone
    cfg.sim.dt, cfg.decimation = .005, 4
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_external_forces_every_iteration = False
    cfg.play_wheel_material_debug = False
    cfg.play_wheel_motor_z_axis_align_debug = False
    cfg.debug_obs_alert_print_interval = 1000000
    # The source's arbitrary actuator/spring/friction randomization is replaced
    # by the named groups and amplitudes of the user's v2 contract.
    cfg.events = deepcopy(cfg.events)
    for name,value in vars(cfg.events).items():
        if hasattr(value,"mode") and name != "reset_base":
            setattr(cfg.events,name,None)
    cfg.self_obs_noise_cfg = deepcopy(cfg.self_obs_noise_cfg)
    for value in cfg.self_obs_noise_cfg.values():
        value.noise_cfg.n_min = value.noise_cfg.n_max = 0.
    cfg.terrain.physics_material.static_friction = 1.
    cfg.terrain.physics_material.dynamic_friction = 1.
    cfg.terrain.physics_material.restitution = 0.
    cfg.terrain.physics_material.friction_combine_mode = "multiply"
    for name,body in (("left_wheel_height_scanner","L_link3"),("right_wheel_height_scanner","R_link3")):
        scanner = getattr(cfg,name,None)
        if scanner is not None:
            scanner.prim_path = f"/World/envs/env_.*/Robot/{body}"
    if getattr(cfg,"height_scanner",None) is not None:
        cfg.height_scanner.prim_path = "/World/envs/env_.*/Robot/base_link"
    cfg.port_randomization_profile = "requested"
    cfg.port_randomization_strength = 1.
    cfg.port_randomization_contract = json.loads((ROOT/"contracts/wheel_leg_randomization_study_v2.json").read_text())


@configclass
class WheelbipePortFlatCfg(WheelbipeV14FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure(self)


@configclass
class WheelbipePortRoughCfg(WheelbipeV14RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure(self)
