#!/usr/bin/env python3
"""Interactive, unscored GUI of an authenticated fresh six-motor checkpoint.

Frozen actor, mechanism, feedback and recovery code are read from its archive.
Only reset poses and requested commands are changed by this GUI. No learner,
optimizer, old recovery FSM, state correction or hardware driver is started.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
ROOT = Path(__file__).resolve().parents[1]
from evaluate_fresh_full_suite import authenticate, digest, write_json, verify_actor_chain
from wheel_leg_full_suite import build_cases, configure_protocol


def preview_cases(catalogue):
    choices = {}
    for label, height in (('low', .23), ('stand', .305), ('high', .42)):
        choices[label] = next(deepcopy(c) for c in catalogue if c['capability'] == 'standing'
                             and c['command'][2] == height and c['pitch_deg'] == 0. and c['replica'] == 0)
    for family in ('front', 'back', 'left', 'right', 'inverted', 'compound'):
        choices[family] = next(deepcopy(c) for c in catalogue if c['capability'] == 'initial_recovery'
            and c['requested_spawn_family'] == family and c['followup'] == 'stand'
            and 'symmetric_' in c['name'] and 'asymmetric_' not in c['name'] and c['replica'] == 0)
    return choices


def prepare(args, bundle):
    cfg, task, design = deepcopy(bundle['cfg']), deepcopy(bundle['infos']['settings']), deepcopy(bundle['design'])
    spec = json.loads((ROOT/'contracts/wheel_leg_full_suite_v1.json').read_text())
    choices = preview_cases(build_cases(bundle['bank'], spec, args.seed))
    design['scope'] = 'unscored_single_robot_fresh_policy_GUI'
    design['run_scope'] = 'interactive_unscored'
    design['review_basis'] = dict(source_mechanism_sha256=digest(args.checkpoint.parent/'config/contract.json'),
        checkpoint_sha256=digest(args.checkpoint), only_runtime_override={})
    from scratch_domain_evaluation import configure
    domain, _ = configure(cfg, design, task, ROOT/'contracts/wheel_leg_randomization_study_v2.json', args.profile, 1.)
    overrides = configure_protocol(cfg, design, task, spec, case_count=1, seed=args.seed)
    role = design['stages']['speed']['roles'].pop('train')
    design['stages']['speed']['roles'] = {'playback': role}
    cfg['evaluation']['seed'] = args.seed
    role['contract']['evaluation.seed'] = args.seed
    if 'evaluation' in role['contract']:
        role['contract']['evaluation']['seed'] = args.seed
    # The frozen training mix needs one environment per scene group, which a
    # single-robot preview cannot allocate. This GUI owns its own unscored
    # design table, so pin one flat group in the effective config and in the
    # playback row, and record the override instead of silently diverge.
    preview_groups = [dict(name='preview', fraction=1., terrain=['flat'])]
    cfg['scene_groups'] = deepcopy(preview_groups)
    role['contract']['scene_groups'] = deepcopy(preview_groups)
    design['review_basis']['only_runtime_override'] = dict(scene_groups=preview_groups)
    write_json(args.output/'design.json', design)
    cfg['design_preflight'] = dict(file=str(args.output/'design.json'), sha256=digest(args.output/'design.json'), stage='speed')
    sys.path[:0] = [str(bundle['archive']/'src'), str(bundle['archive']/'scripts')]
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins
    assert Path(inspect.getsourcefile(load_chassis_control)).resolve() == bundle['archive']/'src/wheeled_tasks/chassis/control_contract.py'
    control = load_chassis_control(ROOT/cfg['control_math_source'])
    preflight = design_preflight(cfg, control, ROOT, stage='speed', role='playback', seed=args.seed, num_envs=1, block_index=0)
    validate_isolated_origins(cfg, 1)
    for name, value in (('config.json', cfg), ('task.json', task), ('preflight.json', preflight), ('choices.json', choices)):
        write_json(args.output/name, value)
    # Do not open the RMUC USD here.  ``prepare_field`` imports ``pxr`` and
    # opens a USD stage; importing any pxr module before ``SimulationApp`` is
    # created can load a second OpenUSD build and crash Kit during startup.
    # The actual field extraction is therefore deferred until ``native`` has
    # launched Kit, while all policy/design preflight work remains offline.
    return cfg, task, control, choices, domain, overrides, None


def native(args, bundle, prepared, receipt):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg, task, control, choices, domain, overrides, field = prepared
    receipt['scene_groups_override'] = cfg['scene_groups']
    from isaaclab.app import AppLauncher
    from play_flat_base import prepare_flat_gui_experience
    # play_flat_base prepends this checkout's own src/scripts on import. The
    # authenticated archive must keep precedence, or the actor chain and the
    # frozen task code silently resolve from the editable worktree.
    sys.path[:0] = [str(bundle['archive']/'src'), str(bundle['archive']/'scripts')]
    experience = prepare_flat_gui_experience(args.output)
    launcher = AppLauncher(dict(headless=args.headless, device=args.device, enable_cameras=False,
        experience=experience['file'], **({'visualizer': ['kit']} if not args.headless else {}),
        kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
    env = view = None
    try:
        if args.rmuc:
            # ``pxr`` must be imported only after AppLauncher/SimulationApp.
            # See the corresponding guard in ``prepare`` above.
            from fresh_rmuc_preview import prepare_field
            field = prepare_field(args.field_root, args.output)
            receipt['rmuc'] = field
            write_json(args.output/'receipt.json', receipt)
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.grounded_standing_reset import install as ground
        from wheeled_tasks.chassis.physical_standing_reward import install as physical
        from wheeled_tasks.chassis.teleop import KeyboardCommand
        from wheeled_tasks.chassis.randomization_study import runtime_samples
        from rsl_rl.utils import resolve_class
        cases = [deepcopy(choices[args.scene])]

        class PreviewEnv(RobustGetupEnv):
            def __init__(self, *positional, **keywords):
                self.preview_ready = False
                super().__init__(*positional, **keywords)
                self.preview_ready = True
                self.reset(torch.arange(1, device=self.device))
                ground(self, include_motion=True)

            def reset(self, ids):
                if self.preview_ready:
                    self.fixed_cases = cases
                return super().reset(ids)

        manifest_file = ROOT/cfg['asset_directory']/'manifest.json'
        assert digest(manifest_file) == cfg['asset_manifest_sha256']
        import wheeled_tasks.chassis.isolated_origins as isolation
        original_isolation = isolation.isolate_native_worlds
        def preview_isolation(stage, physics_scene_path, count):
            if count != 1:
                raise ValueError('RMUC preview supports exactly one robot')
            from fresh_rmuc_preview import install_field
            receipt['rmuc_runtime'] = install_field(stage, field, args.output)
            return original_isolation(stage, physics_scene_path, count)
        if field is not None:
            isolation.isolate_native_worlds = preview_isolation
        try:
            env = PreviewEnv(cfg, json.loads(manifest_file.read_text()), control, ROOT,
                stage_name='speed', num_envs=1, device=args.device, seed=args.seed, level=1., run_role='playback',
                render_use_fabric=False, bank=bundle['bank_file'], settings=task)
        finally:
            isolation.isolate_native_worlds = original_isolation
        physical(env, speed_weight=0., contact_weight=0., torque_weight=0.)
        cls, settings = resolve_class(bundle['agent']['actor'])
        receipt['actor_sources'] = [inspect.getsourcefile(item) for item in cls.__mro__[:4]]
        write_json(args.output/'receipt.json', receipt)
        verify_actor_chain(cls, bundle['archive'], bundle['actor_source_chain'])
        actor = cls(env.get_observations(), bundle['agent']['obs_groups'], 'actor', 6, **settings).to(args.device).eval()
        actor.load_state_dict(bundle['state']['actor_state_dict'], strict=True)
        assert env.dt == .001 and env.policy_dt == .005
        receipt.update(status='running', physics_dt_s=env.dt, policy_dt_s=env.policy_dt,
            actual_domain_samples=runtime_samples(env), reset_scene=args.scene)
        write_json(args.output/'receipt.json', receipt)

        class View:
            def __init__(self):
                import carb.input
                import carb.windowing
                import omni.appwindow
                import omni.ui as ui
                self.paused, self.requested_scene = False, None
                self.teleop = KeyboardCommand(2., 6., height_range=(.23, .42))
                self.teleop.height = cases[0]['command'][2]
                self.input = carb.input.acquire_input_interface()
                self.window = omni.appwindow.get_default_app_window()
                self.keyboard = self.window.get_keyboard()
                self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.on_key)
                carb.windowing.acquire_windowing_interface().set_window_title(self.window.get_window(),
                    f'Fresh018 update {receipt["checkpoint_completed_updates"]} | simulation preview')
                self.panel = ui.Window('FRESH SIX-MOTOR RL / UNSCORED', width=470, height=560)
                with self.panel.frame:
                    with ui.VStack(spacing=5):
                        ui.Label(f'Checkpoint: {args.checkpoint.name}\nCompleted updates: {receipt["checkpoint_completed_updates"]} | {args.profile}\n'
                                 'W/S drive | A/D turn | Q/E height | SPACE stop\nP pause | R reset. Click viewport for keyboard focus.', word_wrap=True)
                        self.info = ui.Label('Initializing', height=145, word_wrap=True)
                        with ui.HStack():
                            self.keyboard_mode = ui.CheckBox(); self.keyboard_mode.model.set_value(True)
                            ui.Label('Keyboard mode (disable to use sliders)')
                        self.vx = ui.FloatSlider(min=-3., max=3.)
                        self.yaw = ui.FloatSlider(min=-10.471976, max=10.471976)
                        self.height = ui.FloatSlider(min=.23, max=.42)
                        self.height.model.set_value(cases[0]['command'][2])
                        ui.Label('Sliders: vx [m/s], yaw [rad/s], height [m]')
                        with ui.HStack():
                            ui.Button('STOP', clicked_fn=self.stop)
                            ui.Button('Pause/resume', clicked_fn=self.toggle)
                            ui.Button('Reset scene', clicked_fn=lambda: self.request(args.scene))
                        for row in (('low', 'stand', 'high'), ('front', 'back', 'inverted'), ('left', 'right', 'compound')):
                            with ui.HStack():
                                for scene in row:
                                    ui.Button(scene, clicked_fn=lambda chosen=scene: self.request(chosen))
                        ui.Label('All six motors use the loaded RL policy.\nRecovery retains native passive start and command release.\n30s end/guard pauses in place; R or a scene button resets.\nThis preview is not a capability score.', word_wrap=True)

            def request(self, scene):
                self.requested_scene = scene

            def stop(self):
                self.teleop.stop()
                self.vx.model.set_value(0.); self.yaw.model.set_value(0.)

            def toggle(self):
                self.paused = not self.paused

            def on_key(self, event, *_):
                key, kind = self.teleop.event_fields(event)
                if key is None:
                    return True
                if kind == 'KEY_RELEASE':
                    self.teleop.key(key, False)
                elif key in self.teleop.MOTION_KEYS:
                    self.teleop.key(key, True); self.keyboard_mode.model.set_value(True)
                elif kind == 'KEY_PRESS':
                    if key == 'SPACE': self.stop()
                    elif key == 'P': self.toggle()
                    elif key == 'R': self.request(receipt['reset_scene'])
                return key not in self.teleop.MOTION_KEYS | {'SPACE', 'P', 'R'}

            def command(self):
                if self.keyboard_mode.model.as_bool:
                    command = self.teleop.advance(env.policy_dt)
                    self.vx.model.set_value(command[0]); self.yaw.model.set_value(command[1]); self.height.model.set_value(command[2])
                    return command
                return [self.vx.model.as_float, self.yaw.model.as_float, self.height.model.as_float]

        if not args.headless:
            view = View()
        env.sim.set_camera_view((1.35, -1.5, .9), (0., 0., .25))
        observation = env.get_observations()
        ticks, total_ticks, last_save, last_render = 0, 0, 0., 0.
        peak = 0.
        wall_start = time.monotonic()
        with torch.inference_mode():
            while launcher.app.is_running() and (not args.max_policy_steps or total_ticks < args.max_policy_steps):
                now = time.monotonic()
                if view and view.requested_scene is not None:
                    scene = view.requested_scene; cases[0] = deepcopy(choices[scene])
                    env.reset(torch.arange(1, device=env.device)); observation = env.get_observations()
                    view.stop(); view.teleop.height = cases[0]['command'][2]
                    view.height.model.set_value(view.teleop.height)
                    view.requested_scene = None; view.paused = False
                    ticks, wall_start = 0, now
                    receipt['reset_scene'] = scene
                if view and (view.paused or not env.sim.is_playing()):
                    launcher.app.update(); time.sleep(.02)
                    continue
                command = view.command() if view else cases[0]['command']
                env.pending_command[0] = env.pending_command.new_tensor(command)
                observation = env.get_observations()
                action = actor(observation)
                if action.shape != (1,6) or not torch.isfinite(action).all():
                    raise ValueError('Nonfinite or incompatible policy action')
                observation, _, done, info = env.step(action)
                ticks += 1; total_ticks += 1
                peak = max(peak, float(env.physical_standing_last['dq'].abs().max()))
                d = info['diagnostics']
                pose = env.robot.data.root_link_pose_w.torch[0,:3].cpu().tolist()
                if view and now-last_render >= 1/30.:
                    env.sim.set_camera_view((pose[0]+1.35,pose[1]-1.5,pose[2]+.65), (pose[0],pose[1],pose[2]-.05))
                    env.sim.render(); last_render = now
                    velocity = d['velocity'][0].cpu().tolist()
                    tilt = math.degrees(math.acos(max(-1.,min(1.,-float(d['gravity'][0,2])))))
                    phase = 'PASSIVE' if bool(env.passive[0]) else 'RL RECOVERY' if bool(env.recovery_active[0]) else 'RL CONTROL'
                    view.info.text = (f'{receipt["reset_scene"]} | {phase} | sim {ticks*.005:.2f}s\n'
                        f'Body vx {velocity[0]:+.3f} m/s | tilt {tilt:.2f} deg\n'
                        f'Request vx/yaw/h {command[0]:+.2f}/{command[1]:+.2f}/{command[2]:.3f}\n'
                        f'Actual 1ms leg peak so far: {peak:.2f} rad/s\n'
                        f'Native done: {bool(done[0])} | guard: {bool(d["terminated"][0])}')
                if bool(done[0]):
                    if view: view.paused = True
                    else: break
                if now-last_save >= .5:
                    receipt.update(native_policy_steps=total_ticks, physical_steps=5*total_ticks,
                        last_updated_utc=datetime.now(timezone.utc).isoformat(), paused=bool(view and view.paused),
                        native_done=bool(done[0]), actual_leg_peak_rad_s=peak,
                        root_position=pose, requested_command=command, applied_command=d['commands'][0].cpu().tolist(),
                        passive=bool(env.passive[0]), recovery=bool(env.recovery_active[0]))
                    write_json(args.output/'receipt.json', receipt); last_save = now
                delay = ticks*env.policy_dt-(time.monotonic()-wall_start)
                if delay > 0: time.sleep(min(delay,.02))
        receipt.update(status='closed_unscored', native_policy_steps=total_ticks, physical_steps=5*total_ticks)
    except BaseException:
        receipt.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        write_json(args.output/'receipt.json', receipt)
        if view is not None:
            view.input.unsubscribe_to_keyboard_events(view.keyboard, view.subscription)
        if env is not None: env.close()
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rmuc', action='store_true', help='Use the installed RMUC map in the native level-floor preview frame')
    from fresh_rmuc_preview import DEFAULT_FIELD
    parser.add_argument('--field-root', type=Path, default=DEFAULT_FIELD)
    parser.add_argument('--scene', choices=('low','stand','high','front','back','left','right','inverted','compound'), default='stand')
    parser.add_argument('--profile', choices=('nominal','requested'), default='nominal')
    parser.add_argument('--seed', type=int, default=271301)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-policy-steps', type=int, default=0)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT/'reports/interactive_play'/datetime.now(timezone.utc).strftime('fresh_%Y%m%dT%H%M%S_%fZ')
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = authenticate(args.checkpoint)
    receipt = dict(schema='fresh-unified-interactive-preview-v1', status='preparing', pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint), checkpoint_completed_updates=bundle['infos']['completed_updates'],
        source_archive=str(bundle['archive']), preview_source_sha256=digest(Path(__file__)),
        profile=args.profile, graphical=not args.headless, new_PPO=0, optimizer_constructed=False,
        behavior_qualified=False, hardware_ready=False, native_policy_steps=0)
    prepared = prepare(args, bundle)
    receipt['rmuc'] = prepared[-1]
    write_json(args.output/'receipt.json', receipt)
    if args.preflight_only:
        receipt['status'] = 'CPU_preflight_only'; write_json(args.output/'receipt.json', receipt)
        print(json.dumps(receipt)); return
    native(args, bundle, prepared, receipt)


if __name__ == '__main__':
    main()
