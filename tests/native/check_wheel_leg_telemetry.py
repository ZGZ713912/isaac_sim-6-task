#!/usr/bin/env python3
"""Compile/run the publisher's synthetic ROS smoke test after build-rmcs.

Run inside the sourced ROS environment. Uses the existing compilation database;
does not reconfigure or rebuild RMCS and never instantiates hardware components.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmcs-root", type=Path, required=True)
    parser.add_argument("--stream-pose", type=Path, help="Synthetic fixture: six API angles; estimator computes references")
    parser.add_argument("--stream-seconds", type=float, default=10.)
    parser.add_argument("--hand-moved", action="store_true", help="Synthetic disarmed leg/body motion, then invalidate estimator references")
    parser.add_argument("--config", type=Path, help="Vehicle YAML parameters for the production estimator/mirror")
    args = parser.parse_args()
    database = args.rmcs_root / "rmcs_ws/build/rmcs_rl/compile_commands.json"
    entry = next(e for e in json.loads(database.read_text())
                 if e["file"].endswith("/visualization/wheel_leg_mirror.cpp"))
    command, original = [], iter(shlex.split(entry["command"]))
    for token in original:
        if token in ("-o", "-c"):
            next(original)
        elif token != "-DNDEBUG":
            command.append(token)
    ros2 = shutil.which("ros2")
    if not ros2:
        parser.error("source the ROS setup before running this test")
    ros_lib = Path(ros2).resolve().parents[1] / "lib"
    source = Path(__file__).with_name("wheel_leg_telemetry_smoke.cpp")
    with tempfile.TemporaryDirectory(prefix="wheel_leg_telemetry_") as temporary:
        executable = Path(temporary) / "smoke"
        # The integration fixture includes the real state estimator. The
        # rmcs_rl publisher itself has no dependency on rmcs_core headers.
        command += ["-I" + str(args.rmcs_root / "rmcs_ws/src/rmcs_core/src"),
                    str(source), "-o", str(executable), f"-L{ros_lib}",
                    f"-Wl,-rpath,{ros_lib}", f"-Wl,-rpath-link,{ros_lib}"]
        command += ["-l" + library for library in (
            "rclcpp", "rcutils", "rcl", "rmw", "class_loader", "rcpputils", "console_bridge",
            "libstatistics_collector", "tracetools",
            "sensor_msgs__rosidl_typesupport_cpp", "geometry_msgs__rosidl_typesupport_cpp",
            "diagnostic_msgs__rosidl_typesupport_cpp", "std_msgs__rosidl_typesupport_cpp",
            "statistics_msgs__rosidl_typesupport_cpp", "rcl_interfaces__rosidl_typesupport_cpp",
            "rosgraph_msgs__rosidl_typesupport_cpp")]
        subprocess.run(command, cwd=entry["directory"], check=True)
        arguments = (["--stream-pose", str(args.stream_pose), "--stream-seconds", str(args.stream_seconds)]
                     if args.stream_pose else [])
        if args.config:
            arguments += ["--ros-args", "--params-file", str(args.config)]
        if args.hand_moved:
            arguments.insert(0, "--hand-moved")
        subprocess.run([str(executable), *arguments], check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
