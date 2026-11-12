"""Wheelbipe adapter for Kaiser's existing file-backed trainctl console.

The shared protocol and UI are unmodified. A separate route keeps the other
project and the historical resident-pause controller intact.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
import uuid

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheeled_algo.trainctl_shared.training_control import (
    PROTOCOL, ManagedSession, TrainingControl, _lease, boot_id, command_lock,
    file_pin, process_identity, read_json, timestamp, verify_pin, write_json,
)


KIND = "wheelbipe-rsl-managed-v1"
PACKAGE = Path(__file__).resolve().parent
RUNTIME_ENVIRONMENT = ("OMNI_KIT_ACCEPT_EULA", "PYTHONNOUSERSITE", "OMP_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")


def _option(arguments, name, default=None):
    values = [arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == name]
    values += [value.split("=", 1)[1] for value in arguments if value.startswith(name + "=")]
    if len(values) > 1:
        raise ValueError(f"Duplicate managed training option: {name}")
    return values[0] if values else default


def _source_files(source, script):
    files = {script}
    deployment = source / "deployment.json"
    if deployment.is_file():
        manifest = read_json(deployment)["files_sha256"]
        for relative, expected in manifest.items():
            path = (source / relative).resolve(strict=True)
            if not path.is_relative_to(source) or file_pin(path)["sha256"] != expected:
                raise ValueError(f"Deployment source changed: {relative}")
            files.add(path)
        files.add(deployment)
    for directory in ("src", "scripts", "contracts", "configs", "model/wheelbipe_port_v202",
                      "model/wheel_leg_v2.0.2/friction_v1/isaac",
                      "third_party/wheelbipe/src", "third_party/wheelbipe/scripts"):
        parent = source / directory
        if parent.is_dir():
            files.update(path.resolve() for path in parent.rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts and path.suffix != ".pyc")
    return sorted(files)


def _controller_files():
    return [PACKAGE / "managed_training.py", PACKAGE / "durable_checkpoint.py",
            *sorted((PACKAGE / "trainctl_shared").glob("*.py")),
            PACKAGE / "trainctl_shared/provenance.json"]


def _bind(child, argv, *, timeout=5.):
    deadline, previous = time.monotonic() + timeout, None
    while time.monotonic() < deadline and child.poll() is None:
        current = process_identity(child.pid)
        if current and current.get("argv") == argv:
            if current == previous:
                return current
            previous = current
        else:
            previous = None
        time.sleep(.002)
    return None


class WheelbipeSession(ManagedSession):
    requires_parent_exit = True

    def __init__(self, root):
        super().__init__(root)
        if self.spec.get("kind") != KIND:
            raise ValueError("This route accepts only an explicitly initialized Wheelbipe session")

    @classmethod
    def create(cls, root, source, python, training_args, *, name=None, environment=None):
        source = Path(source).expanduser().resolve(strict=True)
        script = source / "scripts/train_wheelbipe_port.py"
        if not script.is_file():
            raise ValueError("The source has no Wheelbipe training entry")
        arguments = list(training_args)
        if any(_option(arguments, key) is not None for key in
               ("--output", "--resume", "--checkpoint", "--managed-control-dir")):
            raise ValueError("Output and cold-resume bindings are owned by the managed session")
        if _option(arguments, "--mode", "train") != "train":
            raise ValueError("Only PPO training is managed; preflight/evaluation remain separate")
        updates, envs, seed = (_option(arguments, key) for key in ("--iterations", "--num-envs", "--seed"))
        if updates is None or envs is None or seed is None or int(updates) < 1 or int(envs) < 1:
            raise ValueError("Specify positive --iterations, --num-envs and an explicit --seed")
        executable = shutil.which(str(python))
        if executable is None:
            raise ValueError("Training Python is unavailable")
        runtime_environment = {key: os.environ[key] for key in RUNTIME_ENVIRONMENT if key in os.environ}
        runtime_environment.update(environment or {})
        if any(key not in RUNTIME_ENVIRONMENT or "\0" in value for key, value in runtime_environment.items()):
            raise ValueError("Only the documented simulation runtime environment keys can be frozen")
        continuation = _option(arguments, "--continue-from")
        transfer = None
        if continuation:
            from wheeled_algo.learning_transfer import admission
            transfer = admission(continuation, arguments)
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        (root / "attempts").mkdir()
        (root / "controller/src/wheeled_algo").mkdir(parents=True)
        sources = [file_pin(path) for path in _source_files(source, script)]
        pins, manifests = list(transfer["pins"]) if transfer else [], []
        for index in range(0, len(sources), 400):
            manifest = root / f"source_manifest_{index // 400:03d}.json"
            write_json(manifest, sources[index:index + 400])
            manifests.append(str(manifest))
            pins.append(file_pin(manifest))
        for path in _controller_files():
            target = root / "controller/src/wheeled_algo" / path.relative_to(PACKAGE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            pins.append(file_pin(target))
        controller_init = root / "controller/src/wheeled_algo/__init__.py"
        controller_init.write_text("")
        pins.append(file_pin(controller_init))
        spec = dict(protocol=PROTOCOL, kind=KIND, name=name or root.name,
                    architecture="Wheelbipe35->6 / RSL-PPO", source=str(source), script=str(script),
                    python=executable, supervisor_python=sys.executable,
                    training_args=arguments, updates=int(updates), num_envs=int(envs), seed=int(seed),
                    device=_option(arguments, "--device", "cuda:0"), pins=pins, source_manifests=manifests,
                    runtime_environment=runtime_environment,
                    learning_continuation=transfer,
                    created_boot=boot_id(), created_at=timestamp(),
                    cold_resume_semantics="Learning state and curriculum clock restored; physical episodes/history reset")
        write_json(root / "session.json", spec)
        write_json(root / "session.pin.json", file_pin(root / "session.json"))
        return cls(root)

    def verify_sources(self):
        for pin in self.spec["pins"]:
            verify_pin(pin)
        source = Path(self.spec["source"])
        for name in self.spec["source_manifests"]:
            path = Path(name)
            if path.parent != self.root:
                raise ValueError("Source manifest escapes the session")
            for pin in read_json(path):
                if not Path(pin["path"]).is_relative_to(source):
                    raise ValueError("Training source pin escapes its source root")
                verify_pin(pin)

    def command(self, operation, *, wait_seconds=0.):
        # A person may press stop while the campaign has released the GPU for
        # inference. Keep that request durable so a review cannot undo it.
        if operation in ("pause","stop"):
            with command_lock(self.root):
                state = self.status()
                if not state["active"] and state["phase"]=="stopped":
                    path = self.attempt()/"control/request.json"
                    request = read_json(path)
                    if request["operation"]!=operation:
                        request.update(operation=operation,sequence=request["sequence"]+1,at=timestamp())
                        write_json(path,request,replace=True)
                    return self.status()
        return super().command(operation,wait_seconds=wait_seconds)

    def start(self, *, paused=False, accept_new_boot=False, expected_attempt=None, expected_request_sequence=None):
        with command_lock(self.root), ExitStack() as leases:
            if expected_attempt is not None:
                previous = self.attempt()
                request = read_json(previous/"control/request.json") if previous else None
                if (previous!=Path(expected_attempt) or not request or request["operation"]!="resume"
                        or request["sequence"]!=expected_request_sequence):
                    raise ValueError("Manual control changed during review; automatic continuation refused")
            state = self.status()
            if state["active"] or state["phase"] == "completed":
                return state
            previous = self.attempt()
            expected_boot = read_json(previous / "request.json")["boot_id"] if previous else self.spec["created_boot"]
            if boot_id() != expected_boot and not accept_new_boot:
                raise ValueError("Host boot changed; use explicit --accept-new-boot")
            self.verify_sources()
            completed, active, transitions, resume = 0, 0., 0, None
            if previous:
                final, exited = read_json(previous / "control/result.json"), read_json(previous / "exit.json")
                if (not final or final["summary"].get("phase") != "stopped" or not exited
                        or exited.get("returncode") != 0 or exited.get("identity") != final["summary"].get("identity")
                        or exited.get("token") != final.get("token")):
                    raise ValueError("Unclean attempt cannot resume or roll back to an older checkpoint")
                for pin in final["checkpoint_pins"]:
                    verify_pin(pin)
                completed, active, transitions = (final["summary"][key] for key in
                                                  ("update", "session_active_elapsed_s", "cumulative_transitions"))
                resume = final["summary"]["checkpoint"]
            descriptor = _lease(self.root)
            leases.callback(os.close, descriptor)
            attempt = self.root / "attempts" / f"{len(list((self.root / 'attempts').iterdir())) + 1:04d}-{uuid.uuid4().hex[:12]}"
            (attempt / "control").mkdir(parents=True)
            argv = [self.spec["python"], "-u", self.spec["script"], *self.spec["training_args"],
                    "--output", str(attempt / "train"), "--managed-control-dir", str(attempt / "control")]
            if resume:
                argv += ["--resume", resume]
            binding = dict(protocol=PROTOCOL, token=uuid.uuid4().hex, boot_id=boot_id(),
                           submitted_at=timestamp(), submitted_epoch=time.time(), lease_fd=descriptor,
                           root=str(self.root), attempt=str(attempt), start_update=completed,
                           prior_active_elapsed_s=active, prior_transitions=transitions,
                           target_update=self.spec["updates"], worker_argv=argv, resume_checkpoint=resume)
            write_json(attempt / "request.json", binding)
            write_json(attempt / "control/request.json", dict(token=binding["token"], sequence=0,
                       operation="pause" if paused else "resume", at=timestamp()))
            write_json(self.root / "current.json", dict(attempt=attempt.name), replace=True)
            controller = self.root / "controller/src/wheeled_algo/managed_training.py"
            command = [self.spec["supervisor_python"], str(controller), "_supervise", str(self.root), str(attempt)]
            with (attempt / "parent.log").open("xb") as output:
                child = subprocess.Popen(command, cwd=self.root, stdin=subprocess.DEVNULL, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(descriptor,))
            self.submitted_process = child
            identity = _bind(child, command)
            if identity:
                write_json(attempt / "launch.json", identity)
            return self.status()

    def supervise(self, attempt):
        attempt = Path(attempt).resolve(strict=True)
        if attempt != self.attempt():
            raise ValueError("Supervisor does not own this attempt")
        request = read_json(attempt / "request.json")
        self.verify_sources()
        argv = request["worker_argv"]
        environment = dict(os.environ)
        for key in RUNTIME_ENVIRONMENT:
            environment.pop(key, None)
        environment.update(self.spec["runtime_environment"])
        with (attempt / "worker.log").open("xb") as output:
            child = subprocess.Popen(argv, cwd=self.spec["source"], stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, pass_fds=(request["lease_fd"],),
                                     env=environment)
            identity = _bind(child, argv)
            if identity:
                write_json(attempt / "worker_launch.json", identity)
            code = child.wait()
        write_json(attempt / "exit.json", dict(token=request["token"], boot=request["boot_id"],
                   identity=identity, returncode=code, actual_waitpid=True))
        return code if identity else (code or 2)


class ManagedStop(Exception):
    """A fully completed update has been saved for a normal process exit."""


class _Stop:
    reason = None
    deadline = float("inf")

    def stopped(self):
        return self.reason is not None


class ManagedWorker:
    def __init__(self, directory, output, resume):
        self.control = TrainingControl(directory)
        self.binding = self.control.binding
        self.session = WheelbipeSession(self.binding["root"])
        self.output = Path(output).resolve()
        if self.output != Path(self.binding["attempt"]) / "train":
            raise ValueError("Worker output is outside its newly allocated attempt")
        expected = self.binding.get("resume_checkpoint")
        if (str(Path(resume).resolve()) if resume else None) != expected:
            raise ValueError("Resume input differs from the sealed previous attempt")
        if process_identity(os.getpid())["argv"] != self.binding["worker_argv"]:
            raise ValueError("Actual worker arguments differ from its owned launch")
        self.session.verify_sources()
        self.session_sha256 = file_pin(self.session.root / "session.json")["sha256"]
        self.stop = _Stop()
        self.completed = self.binding["start_update"]
        self.runner = self.env = None
        self.previous_handlers = {}
        self.endpoint_phase = None

    def attach(self, runner, env, resume, continuation=None):
        from wheeled_algo.durable_checkpoint import load_checkpoint

        self.runner, self.env = runner, env
        self.budget = self.session.spec["updates"]
        if runner.num_steps_per_env != 24:
            raise ValueError("Wheelbipe curriculum and checkpoint clock require24 policy steps/update")
        if resume:
            proof = load_checkpoint(resume, runner, env, self.session_sha256, self.budget)
            if proof["completed_updates"] != self.completed:
                raise ValueError("Resume endpoint and parent start counter disagree")
            write_json(self.output / "resume_verification.json", proof)
        else:
            if runner.alg.optimizer.state or self.completed:
                raise ValueError("A fresh managed session requires empty Adam and zero updates")
            if continuation:
                from wheeled_algo.learning_transfer import initialize
                if str(Path(continuation).resolve()) != self.session.spec["learning_continuation"]["checkpoint"]:
                    raise ValueError("Continuation input differs from the pinned session")
                proof = initialize(continuation, runner, env, self.session.spec["training_args"])
                write_json(self.output / "continuation_verification.json", proof)
            else:
                env.set_training_progress(iteration=0)
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda signum, frame: setattr(self.stop, "reason", signal.Signals(signum).name))
        self._boundary()

    def _save(self, suffix):
        import torch
        from wheeled_algo.durable_checkpoint import save_checkpoint

        if torch.device(self.runner.device).type == "cuda":
            torch.cuda.synchronize(self.runner.device)
        path = self.output / "checkpoints" / f"update_{self.completed:08d}_{suffix}.pt"
        return save_checkpoint(path, self.runner, self.env, self.completed, self.session_sha256, self.budget)

    def _boundary(self):
        self.control.state.update(update=self.completed,
            cumulative_transitions=self.binding["prior_transitions"]
                + (self.completed - self.binding["start_update"]) * self.session.spec["num_envs"] * 24)
        if not self.control.boundary(self.stop, lambda sequence: self._save(f"pause_{sequence}")):
            self.endpoint_phase = "stopped"
            raise ManagedStop()

    def after_update(self, completed):
        if completed != self.completed + 1:
            raise ValueError("Managed PPO updates must advance exactly once at each completed log boundary")
        self.completed = completed
        self.control.state.update(update=completed,
            cumulative_transitions=self.binding["prior_transitions"]
                + (completed - self.binding["start_update"]) * self.session.spec["num_envs"] * 24)
        if completed < self.budget:
            self._boundary()

    def review_checkpoint(self):
        """Save a completed update for read-only review without stopping PPO."""
        return self._save("review")

    def commit_endpoint(self, phase):
        if phase not in ("stopped", "completed") or (phase == "completed") != (self.completed == self.budget):
            raise ValueError("Final phase does not match the original update budget")
        checkpoint = self._save("final")
        self.control.finish(dict(status=phase, final_update=self.completed,
            cumulative_transitions=self.control.state["cumulative_transitions"], checkpoint=str(checkpoint),
            stop_reason=self.stop.reason or ("budget_complete" if phase == "completed" else "control_stop")))
        self.endpoint_phase = phase
        return checkpoint

    def close(self):
        for number, previous in self.previous_handlers.items():
            signal.signal(number, previous)


def shared_console():
    """Adapt project routing and receipt checks without editing the shared UI."""
    from wheeled_algo.trainctl_shared import training_cli
    from wheeled_algo.durable_checkpoint import FORMAT

    if getattr(training_cli, "wheelbipe_adapter", False):
        return training_cli
    original_files = training_cli._checkpoint_files
    original_release = training_cli.Release

    def checkpoint_files(state):
        payload, metadata = original_files(state)
        record = read_json(metadata)
        if (record.get("format") != FORMAT or record.get("checkpoint_pins") != [file_pin(payload)]
                or record.get("completed_updates") != state.get("checkpoint_update")
                or record.get("total_updates") != state.get("target_update")):
            raise ValueError("Wheelbipe保存点的格式、载荷pin或预算不一致；未请求退出")
        return payload, metadata

    def gpu_memory():
        executable = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        try:
            result = subprocess.run([executable, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=2.)
            used, total = (int(value) for value in result.stdout.splitlines()[0].split(",")[:2])
            return {"used_mib": used, "total_mib": total} if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    class Release(original_release):
        def __init__(self, session, *, probe=gpu_memory, clock=time.monotonic):
            super().__init__(session, probe=probe, clock=clock)

        def step(self, state):
            if self.stage == "stopping" and not state.get("active"):
                if (state.get("phase") not in ("stopped", "completed")
                        or state.get("phase") == "stopped" and not state.get("checkpoint_resume")):
                    self.stage = "aborted"
                    return False, "进程已退出但完整恢复回执未通过；请检查错误，不能报告保存成功。", "error"
            return super().step(state)

        def report(self, state):
            message = super().report(state)
            old = f"trainctl resume {state.get('name')} --wait 60"
            command = f"trainctl wheelbipe resume {shlex.quote(str(self.session.root))} --wait 60"
            return message.replace(old, command)

    training_cli.open_session = WheelbipeSession
    training_cli.active_alias = lambda: None
    training_cli._checkpoint_files = checkpoint_files
    training_cli._gpu_memory = gpu_memory
    training_cli.Release = Release
    training_cli.wheelbipe_adapter = True
    return training_cli


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_supervise"]:
        return WheelbipeSession(argv[1]).supervise(argv[2])
    if argv[:1] == ["init"]:
        split = argv.index("--") if "--" in argv else len(argv)
        parser = argparse.ArgumentParser(description="Freeze a Wheelbipe session; do not start PPO")
        parser.add_argument("session", type=Path)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--python", required=True)
        parser.add_argument("--name")
        parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                            help="Freeze a simulation runtime variable; current documented variables are captured too")
        args = parser.parse_args(argv[1:split])
        try:
            environment = dict(value.split("=", 1) for value in args.env)
            session = WheelbipeSession.create(args.session, args.source, args.python, argv[split + 1:],
                                             name=args.name, environment=environment)
            print(json.dumps(session.status(), ensure_ascii=False, indent=2))
            return 0
        except (OSError, ValueError, KeyError) as error:
            print(f"trainctl wheelbipe: {error}", file=sys.stderr)
            return 1
    return shared_console().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
