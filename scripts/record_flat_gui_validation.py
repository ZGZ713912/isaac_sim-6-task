#!/usr/bin/env python3
"""Record the authenticated flat ONNX through the existing native WASD GUI."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def validation_cases(steady_seconds=8., phase_seconds=4.):
    """Each case starts from a native reset; targets use the real key handler."""
    if not (0 < steady_seconds <= 30 and 0 < phase_seconds <= 10):
        raise ValueError("Bounded positive case durations are required")
    def phase(seconds, key=None, vx=3., yaw=4.):
        return dict(seconds=seconds, key=key, vx_limit=vx, yaw_limit=yaw)
    rows = [("stand_305mm", .305, [phase(min(4., steady_seconds))]),
        ("forward_05", .305, [phase(min(4., steady_seconds), "W", .5)]),
        ("backward_05", .305, [phase(min(4., steady_seconds), "S", .5)]),
        ("rotate_1", .305, [phase(min(4., steady_seconds), "A", yaw=1.)]),
        ("rotate_1_reverse", .305, [phase(min(4., steady_seconds), "D", yaw=1.)]),
        ("forward_3", .305, [phase(steady_seconds, "W")]),
        ("backward_2p5", .305, [phase(steady_seconds, "S", 2.5)]),
        ("rotate_4", .305, [phase(min(4., steady_seconds), "A")]),
        ("rotate_4_reverse", .305, [phase(min(4., steady_seconds), "D")]),
        ("start_stop_3_to_2p5", .305, [phase(phase_seconds, "W"), phase(phase_seconds),
            phase(phase_seconds, "S", 2.5), phase(phase_seconds)]),
        ("backward_start_stop_2p5_to_3", .305, [phase(phase_seconds, "S", 2.5),
            phase(phase_seconds), phase(phase_seconds, "W"), phase(phase_seconds)]),
        ("yaw_start_stop_positive", .305, [phase(phase_seconds, "A"), phase(phase_seconds),
            phase(phase_seconds, "D"), phase(phase_seconds)]),
        ("yaw_start_stop_negative", .305, [phase(phase_seconds, "D"), phase(phase_seconds),
            phase(phase_seconds, "A"), phase(phase_seconds)]),
        ("stand_230mm", .23, [phase(steady_seconds)]),
        ("stand_420mm", .42, [phase(steady_seconds)])]
    return [dict(name=name, height_m=height, phases=phases,
                 duration_s=sum(p["seconds"] for p in phases)) for name, height, phases in rows]


def phase_index(case, tick, policy_dt):
    elapsed = tick * policy_dt
    end = 0.
    for index, phase in enumerate(case["phases"]):
        end += phase["seconds"]
        if elapsed < end - 1e-10:
            return index
    return len(case["phases"]) - 1


class GuiRecorder:
    def __init__(self, cases, fps=25, terminal_hold_s=1.):
        self.cases, self.fps = cases, fps
        self.terminal_hold_s = terminal_hold_s
        self.finished = False
        self.started = False
        self.index = self.case_ticks = self.frames = self.phase = 0
        self.encoder = self.trace = self.frame_log = None
        self.summary = dict(schema="flat-gui-keyboard-recording-v1", status="not_started",
            run_scope="interactive_unscored", qualification=False, hardware_ready=False,
            command_source="scripted KEY_PRESS/KEY_RELEASE callbacks into the existing PolicyView.on_key and KeyboardCommand.advance; not OS key injection",
            physics_state_replay=False, optimizer_updates=0, video_speed="1x simulation time",
            terminal_hold_seconds=terminal_hold_s, fps=fps, cases=[])

    def key(self, name, kind):
        event = SimpleNamespace(input=SimpleNamespace(name=name), type=SimpleNamespace(name=kind))
        self.view.on_key(event)
        self.summary.setdefault("keyboard_events", []).append(dict(
            case=self.case["name"], case_tick=self.case_ticks, input=name, type=kind))

    def start(self, env, view, report, args):
        from omni.kit.viewport.utility import get_active_viewport
        self.env, self.view, self.output, self.report = env, view, args.output, report
        self.policy_dt = env.policy_dt
        stride = 1 / (env.policy_dt * self.fps)
        if not math.isclose(stride, round(stride), abs_tol=1e-10):
            raise ValueError("Video FPS must divide the policy clock exactly")
        self.stride = round(stride)
        self.viewport = get_active_viewport()
        self.viewport.resolution = (960, 540)
        self.trace = (self.output / "keyboard_trace.jsonl").open("x")
        self.frame_log = (self.output / "video_frames.jsonl").open("x")
        self.summary.update(status="recording", policy_hz=round(1/env.policy_dt),
            physics_hz=round(1/env.dt), policy_sha256=report["policy_sha256"],
            successful_updates_total=report["onnx_metadata"]["successful_updates_total"],
            contact_material=env.startup_report.get("contact_material"),
            keyboard_subscription_active=view.keyboard_subscription is not None)
        self.started = True
        self.start_case()
        # Commands still pass through the actual inherited GUI command function.
        original = view.command
        def command(seconds, dt):
            index = phase_index(self.case, self.case_ticks, dt)
            if index != self.phase:
                self.phase = index
                self.apply_phase()
            view.teleop.height = self.case["height_m"]
            return original(seconds, dt)
        view.command = command
        report["video_validation"] = self.summary
        self.save()

    def start_case(self):
        self.case = self.cases[self.index]
        self.case_ticks, self.phase = 0, 0
        self.view.request_reset()
        self.view.teleop.height = self.case["height_m"]
        self.current = dict(name=self.case["name"], requested_seconds=self.case["duration_s"],
            height_target_m=self.case["height_m"], policy_ticks=0, simulated_seconds=0., frames=0,
            terminal=False, termination_status=None, peak_abs_pitch_deg=0., peak_abs_roll_deg=0.,
            closure_gap_peak_mm=0., first_video_frame=self.frames)
        self.summary["cases"].append(self.current)
        self.encoder_log = (self.output / f"{self.index:02d}_{self.case['name']}.ffmpeg.log").open("xb")
        self.video_path = self.output / f"{self.index:02d}_{self.case['name']}_1x.mp4"
        self.encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "960x540", "-framerate",
            str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "19", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart",
            str(self.video_path)], stdin=subprocess.PIPE, stdout=self.encoder_log, stderr=self.encoder_log)
        self.apply_phase()
        print("GUI_RECORD_CASE " + json.dumps(self.current), flush=True)

    def apply_phase(self):
        for key in sorted(self.view.teleop.keys):
            self.key(key, "KEY_RELEASE")
        self.key("SPACE", "KEY_PRESS")
        phase = self.case["phases"][self.phase]
        self.view.speed_limit.model.set_value(phase["vx_limit"])
        self.view.yaw_limit.model.set_value(phase["yaw_limit"])
        if phase["key"]:
            self.key(phase["key"], "KEY_PRESS")

    def capture(self, sample, poses, terminal_hold=False):
        from omni.kit.viewport.utility import capture_viewport_to_file
        from PIL import Image, ImageDraw, ImageFont
        self.view.update(sample, poses, self.env.model_spec)
        path = self.output / ".capture.png"
        path.unlink(missing_ok=True)
        helper = capture_viewport_to_file(self.viewport, str(path))
        future = asyncio.ensure_future(helper.wait_for_result())
        for _ in range(90):
            self.env.sim.render()
            if future.done():
                future.result()
                break
        else:
            future.cancel()
            raise RuntimeError("Viewport capture did not complete after 90 physics-free renders")
        with Image.open(path) as source:
            frame = source.convert("RGB").resize((960, 540))
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 17)
        draw.rectangle((0, 445, 960, 540), fill=(15, 20, 29))
        label = f"Model {self.summary['successful_updates_total']} | 200Hz policy / 1000Hz physics | {self.case['name']}"
        command, velocity, omega = sample["command"], sample["velocity_m_s"], sample["omega_rad_s"]
        lines = [label,
            f"sim {self.case_ticks*self.policy_dt:.2f}s | keys {','.join(sample['pressed_keys']) or 'STOP'} | vx target/actual {command[0]:+.2f}/{velocity[0]:+.2f} m/s | yaw {command[1]:+.2f}/{omega[2]:+.2f} rad/s",
            f"height {sample['height_m']:.3f}m | pitch {sample['pitch_deg']:+.1f}deg | live RTF {sample['real_time_factor']:.3f} | VIDEO 1x SIM TIME",
            ("TERMINAL - labelled 1s freeze: " + sample["status"]) if terminal_hold else sample["status"]]
        for y, line in zip((450, 472, 494, 516), lines):
            draw.text((12, y), line, font=font, fill=(240, 244, 250))
        repeats = max(1, round(self.terminal_hold_s*self.fps)) if terminal_hold else 1
        data = frame.tobytes()
        for _ in range(repeats):
            self.encoder.stdin.write(data)
            self.frame_log.write(json.dumps(dict(frame=self.frames, case=self.case["name"],
                case_tick=self.case_ticks, sim_time_s=self.case_ticks*self.policy_dt,
                kind="labelled_terminal_freeze" if terminal_hold else "simulation_frame")) + "\n")
            self.frames += 1
            self.current["frames"] += 1
        if self.current["frames"] == 1:
            frame.save(self.output / f"{self.index:02d}_{self.case['name']}_first.png")
        if terminal_hold:
            frame.save(self.output / f"{self.index:02d}_{self.case['name']}_terminal.png")
        frame.save(self.output / "latest_frame.png")
        path.unlink(missing_ok=True)

    def sample(self, sample, poses, terminal):
        self.case_ticks += 1
        self.trace.write(json.dumps(dict(case=self.case["name"], case_tick=self.case_ticks, **sample)) + "\n")
        self.current.update(policy_ticks=self.case_ticks, simulated_seconds=self.case_ticks*self.policy_dt,
            final_vx_m_s=sample["velocity_m_s"][0], final_yaw_rad_s=sample["omega_rad_s"][2],
            final_height_m=sample["height_m"], real_time_factor=sample["real_time_factor"])
        for key, value in [("peak_abs_pitch_deg", abs(sample["pitch_deg"])),
                           ("peak_abs_roll_deg", abs(sample["roll_deg"])),
                           ("closure_gap_peak_mm", sample["gap_mm"])]:
            self.current[key] = max(self.current[key], value)
        end = terminal or self.case_ticks >= round(self.case["duration_s"] / self.policy_dt)
        if self.case_ticks % self.stride == 0 or end:
            self.capture(sample, poses)
        if end:
            self.current.update(terminal=terminal, termination_status=sample["status"] if terminal else None)
            if terminal and self.terminal_hold_s:
                self.capture(sample, poses, terminal_hold=True)
            self.finish_encoder()
            print("GUI_RECORD_CASE_DONE " + json.dumps(self.current), flush=True)
            self.index += 1
            if self.index == len(self.cases):
                self.finished = True
            else:
                self.start_case()
        if self.case_ticks % 200 == 0 or end:
            self.trace.flush()
            self.frame_log.flush()
            self.save()

    def finish_encoder(self):
        if self.encoder is None:
            return
        self.encoder.stdin.close()
        code = self.encoder.wait(timeout=60)
        self.encoder_log.close()
        self.encoder = None
        if code:
            raise RuntimeError(f"Video encoding failed: {self.video_path}")
        self.current.update(video=str(self.video_path), video_duration_s=self.current["frames"]/self.fps)

    def save(self):
        self.summary.update(frames=self.frames, completed_cases=self.index,
            updated_at=datetime.now(timezone.utc).isoformat())
        target = self.output / "recording.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2)+"\n")
        temporary.replace(target)

    def close(self, report):
        if not self.started:
            return
        try:
            self.finish_encoder()
        finally:
            self.trace.close()
            self.frame_log.close()
        self.summary["status"] = "completed" if self.finished else "interrupted"
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT/"models/flat_directional200_1346_preview_20261020")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--steady-seconds", type=float, default=8.)
    parser.add_argument("--phase-seconds", type=float, default=4.)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    cases = validation_cases(args.steady_seconds, args.phase_seconds)
    if args.cases:
        by_name = {c["name"]: c for c in cases}
        if len(set(args.cases)) != len(args.cases) or set(args.cases)-set(by_name):
            parser.error("Cases must be unique known validation names")
        cases = [by_name[name] for name in args.cases]
    controller = GuiRecorder(cases)
    import play_chassis_suite as suite
    import play_flat_speed_preview as playback
    original = suite.main
    suite.main = lambda **kwargs: original(recording_controller=controller, **kwargs)
    sys.argv = [sys.argv[0], "--package", str(args.package), "--output", str(args.output), "--device", args.device]
    return playback.main()


if __name__ == "__main__":
    raise SystemExit(main())
