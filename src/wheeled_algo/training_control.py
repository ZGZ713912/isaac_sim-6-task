"""Cooperative, same-process training pause and a dependency-free trainctl TUI.

The control thread never touches a learner, simulator, or random generator.
Only the training thread acknowledges a pause, at an explicit safe point.
This file also runs standalone so the CLI needs no Isaac/PyTorch environment.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid


SCHEMA = "wheel-leg-trainctl-v1"
TERMINAL = {"finished", "failed"}


def _private_directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"trainctl requires a private, owned directory (mode 700): {path}")
    return path


def directories():
    state = Path(os.environ.get("TRAINCTL_STATE_DIR", str(
        Path(os.environ.get("XDG_STATE_HOME", str(Path.home()/".local/state")))/"trainctl")))
    runtime = Path(os.environ.get("TRAINCTL_RUNTIME_DIR", str(
        Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))/f"trainctl-{os.getuid()}")))
    return _private_directory(state), _private_directory(runtime)


def process_identity(pid):
    """Linux PID identity, including reboot and PID reuse protection."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=int(pid), start_ticks=int(fields[19]),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (OSError, ValueError, IndexError):
        return None


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _receive(stream):
    data = bytearray()
    while b"\n" not in data:
        block = stream.recv(4096)
        if not block:
            raise OSError("trainctl connection closed")
        data.extend(block)
        if len(data) > 65536:
            raise ValueError("trainctl message too large")
    return json.loads(data.split(b"\n", 1)[0])


class TrainingControl:
    """Call checkpoint() from the training thread, never from a GPU worker."""

    def __init__(self, name=None, *, synchronize=None):
        state, runtime = directories()
        self.run_id = uuid.uuid4().hex[:12]
        self.socket_path = runtime/f"{self.run_id}.sock"
        self.record = state/f"{self.run_id}.json"
        self.cv = threading.Condition()
        self.synchronize = synchronize or (lambda: None)
        self.started = time.monotonic()
        self.finished = None
        self.paused_at = None
        self.paused_total = 0.
        self.pause_requested = False
        self.stopping = False
        self.closed = False
        self.data = dict(schema=SCHEMA, run_id=self.run_id, name=name or Path(sys.argv[0]).stem,
            identity=process_identity(os.getpid()), cwd=os.getcwd(), argv=list(sys.argv),
            socket=str(self.socket_path), started_at=time.time(), status="initializing",
            phase="initializing", completed_updates=0, pause_count=0, pause_requested=False,
            controller_file=str(Path(__file__).resolve()),
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            self.server.listen(8)
            self.server.settimeout(.2)
            _atomic_json(self.record, self.snapshot())
        except BaseException:
            self.server.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self.thread = threading.Thread(target=self._serve, name="trainctl", daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.cv:
            now = self.finished if self.finished is not None else time.monotonic()
            paused = self.paused_total + (now-self.paused_at if self.paused_at is not None else 0.)
            return dict(self.data, pause_requested=self.pause_requested, paused_seconds=paused,
                        active_seconds=now-self.started-paused, updated_at=time.time())

    def request(self, action):
        with self.cv:
            if action not in ("status", "pause", "resume"):
                raise ValueError(f"Unknown trainctl action: {action}")
            if action != "status":
                if self.closed or self.stopping:
                    raise ValueError("Training is finishing; pause/resume is unavailable")
                self.pause_requested = action == "pause"
                if self.pause_requested and self.data["status"] != "paused":
                    self.data["status"] = "pause_requested"
                elif not self.pause_requested and self.data["status"] != "paused":
                    self.data["status"] = "running"
                # Only the main thread can change paused -> running.
                self.cv.notify_all()
            return self.snapshot()

    def checkpoint(self, phase, *, completed_updates=None):
        """Park here with all learner/rollout/native state still in memory."""
        with self.cv:
            self.data["phase"] = phase
            if completed_updates is not None:
                self.data["completed_updates"] = int(completed_updates)
            if not self.pause_requested or self.stopping:
                self.data["status"] = "stopping" if self.stopping else "running"
                return
        # Drain outstanding CUDA work before claiming that physics has paused.
        self.synchronize()
        with self.cv:
            if self.pause_requested and not self.closed and not self.stopping:
                self.paused_at = time.monotonic()
                self.data["status"] = "paused"
                self.data["pause_count"] += 1
                self.cv.notify_all()
                while self.pause_requested and not self.closed and not self.stopping:
                    self.cv.wait(.2)
                if self.paused_at is not None:
                    self.paused_total += time.monotonic()-self.paused_at
                    self.paused_at = None
            if not self.closed:
                self.data["status"] = "stopping" if self.stopping else "running"

    def wake_for_stop(self):
        """Let the existing trainer's signal handler finish its usual stop path."""
        with self.cv:
            self.stopping = True
            self.pause_requested = False
            self.cv.notify_all()

    def _serve(self):
        last_write = 0.
        while not self.closed:
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                connection = None
            except OSError:
                break
            if connection is not None:
                with connection:
                    connection.settimeout(1.)
                    try:
                        peer_uid = struct.unpack("3i", connection.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))[1]
                        if peer_uid != os.getuid():
                            raise ValueError("trainctl peer UID mismatch")
                        message = _receive(connection)
                        if (message.get("run_id") != self.run_id
                                or message.get("identity") != self.data["identity"]):
                            raise ValueError("Training identity changed")
                        reply = dict(ok=True, job=self.request(message["action"]))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        reply = dict(ok=False, error=str(error))
                    try:
                        connection.sendall((json.dumps(reply)+"\n").encode())
                    except OSError:
                        pass  # A disconnected UI does not control training lifetime.
            if time.monotonic()-last_write >= 1.:
                try:
                    _atomic_json(self.record, self.snapshot())
                except OSError:
                    pass  # A full status disk must not kill or resume a learner.
                last_write = time.monotonic()

    def close(self, status="finished"):
        with self.cv:
            if self.closed:
                return
            if self.paused_at is not None:
                self.paused_total += time.monotonic()-self.paused_at
                self.paused_at = None
            self.data.update(status=status, finished_at=time.time())
            self.finished = time.monotonic()
            self.closed = True
            self.pause_requested = False
            self.cv.notify_all()
        self.server.close()
        self.thread.join(timeout=2.)
        self.socket_path.unlink(missing_ok=True)
        _atomic_json(self.record, self.snapshot())


def rpc(job, action="status", *, timeout=1.):
    if process_identity(job["identity"]["pid"]) != job["identity"]:
        raise ValueError("Training process exited or its PID was reused")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(job["socket"])
        connection.sendall((json.dumps(dict(run_id=job["run_id"], identity=job["identity"],
                                           action=action))+"\n").encode())
        reply = _receive(connection)
    if not reply.get("ok"):
        raise ValueError(reply.get("error", "trainctl request failed"))
    return reply["job"]


def jobs(*, include_finished=False):
    state, _ = directories()
    result = []
    for path in state.glob("*.json"):
        try:
            job = json.loads(path.read_text())
            if job.get("schema") != SCHEMA:
                continue
            if job["status"] not in TERMINAL:
                try:
                    job = rpc(job, timeout=.25)
                except (OSError, ValueError):
                    alive = process_identity(job["identity"]["pid"]) == job["identity"]
                    job = dict(job, status="unreachable" if alive else "exited")
            if include_finished or job["status"] not in (*TERMINAL, "exited"):
                result.append(job)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(result, key=lambda row: row["started_at"])


def select_job(selector, rows):
    matches = [row for row in rows if selector in (row["name"], str(row["identity"]["pid"]))
               or row["run_id"].startswith(selector)]
    if len(matches) != 1:
        raise ValueError(f"Expected one controlled job matching {selector!r}, found {len(matches)}. Use trainctl list.")
    return matches[0]


class _SignalBridge:
    """Preserve trainer handlers, including handlers replaced by a native rebuild."""

    def __init__(self, control):
        self.control, self.handlers = control, {}

    def refresh(self):
        if threading.current_thread() is not threading.main_thread():
            return
        for number in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(number)
            if previous == signal.SIG_IGN:
                continue
            if number in self.handlers and previous is self.handlers[number][1]:
                continue

            def handler(received, frame, previous=previous):
                self.control.wake_for_stop()
                if callable(previous):
                    previous(received, frame)
                elif previous == signal.SIG_DFL:
                    if received == signal.SIGINT:
                        raise KeyboardInterrupt
                    raise SystemExit(128+received)

            self.handlers[number] = (previous, handler)
            signal.signal(number, handler)

    def close(self):
        for number, (previous, handler) in self.handlers.items():
            if signal.getsignal(number) is handler:
                signal.signal(number, previous)


def controlled_learn(runner, learn, control):
    """Wrap final instance hooks, preserving domain/evaluation/audit callbacks."""
    if getattr(runner, "is_distributed", False):
        raise ValueError("trainctl currently supports a single training process, not distributed RSL-RL")
    import torch

    def synchronize():
        device = torch.device(runner.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    previous_sync = control.synchronize
    control.synchronize = synchronize
    native_act, native_update = runner.alg.act, runner.alg.update
    bridge = _SignalBridge(control)
    completed = int(runner.current_learning_iteration)

    def checkpoint(phase):
        bridge.refresh()
        control.checkpoint(phase, completed_updates=completed)

    @functools.wraps(native_act)
    def act(*args, **kwargs):
        checkpoint("sampling / context or evaluation")
        return native_act(*args, **kwargs)

    @functools.wraps(native_update)
    def update(*args, **kwargs):
        nonlocal completed
        checkpoint("before PPO update")
        result = native_update(*args, **kwargs)
        completed += 1
        checkpoint("after PPO update")
        logger = getattr(runner, "logger", None)
        writer = getattr(logger, "writer", None) if logger is not None else getattr(runner, "writer", None)
        if writer is not None:
            snapshot = control.snapshot()
            for tag, field in (("pause_count", "pause_count"), ("paused_seconds", "paused_seconds"),
                               ("active_seconds", "active_seconds")):
                writer.add_scalar(f"TrainControl/{tag}", snapshot[field], completed)
        return result

    runner.alg.act, runner.alg.update = act, update
    try:
        checkpoint("starting learner")
        return learn()
    finally:
        if runner.alg.act is act:
            runner.alg.act = native_act
        if runner.alg.update is update:
            runner.alg.update = native_update
        bridge.close()
        control.synchronize = previous_sync


def learn(runner, *args, name=None, **kwargs):
    """Default project entry, also covering direct `python train_*.py` launches."""
    if getattr(runner.learn, "_trainctl_wrapped", False):
        return runner.learn(*args, **kwargs)
    control = TrainingControl(name)
    status = "failed"
    try:
        result = controlled_learn(runner, lambda: runner.learn(*args, **kwargs), control)
        status = "finished"
        return result
    finally:
        control.close(status)


class _RunnerFinder(importlib.abc.MetaPathFinder):
    """Install after RSL is imported normally, without importing torch before Kit."""

    def __init__(self, control):
        self.control = control
        self.attached = False

    def find_spec(self, fullname, path, target=None):
        if fullname != "rsl_rl.runners.on_policy_runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader, owner = spec.loader, self

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return loader.create_module(spec)

            def exec_module(self, module):
                loader.exec_module(module)
                cls = module.OnPolicyRunner
                native_learn = cls.learn

                @functools.wraps(native_learn)
                def learn(runner, *args, **kwargs):
                    owner.attached = True
                    return controlled_learn(runner, lambda: native_learn(runner, *args, **kwargs), owner.control)

                learn._trainctl_wrapped = True
                cls.learn = learn

        spec.loader = Loader()
        return spec


def _run_script(arguments, name=None):
    if not arguments or not Path(arguments[0]).is_file():
        raise ValueError("Expected a Python training script after the interpreter")
    target = str(Path(arguments[0]).resolve())
    sys.argv = [target, *arguments[1:]]
    # Match `python script.py`; do not inject this controller's checkout into
    # another worktree's imports. Installed and project copies are identical.
    sys.path[0] = str(Path(target).parent)
    control = TrainingControl(name)
    finder = _RunnerFinder(control)
    sys.meta_path.insert(0, finder)
    print(f"[trainctl] {control.run_id} PID={os.getpid()} — trainctl to pause/resume", flush=True)
    status = "failed"
    try:
        runpy.run_path(target, run_name="__main__")
        status = "finished"
    except SystemExit as error:
        if error.code in (None, 0):
            status = "finished"
        raise
    finally:
        sys.meta_path.remove(finder)
        control.close(status)
        if not finder.attached:
            print("[trainctl] No RSL-RL learn() ran; this process had no controllable learner.", file=sys.stderr)


def launch(arguments, name=None):
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if len(arguments) < 2:
        raise ValueError("Use trainctl run -- PYTHON TRAINING_SCRIPT.py [training options]")
    executable = shutil.which(arguments[0])
    if executable is None:
        raise ValueError(f"Python executable/launcher unavailable: {arguments[0]}")
    index = 1
    while index < len(arguments) and arguments[index] in ("-u", "-B", "-O", "-OO"):
        index += 1
    if index == len(arguments) or not Path(arguments[index]).is_file() or not arguments[index].endswith(".py"):
        raise ValueError("trainctl run requires a Python interpreter/launcher followed by a .py script")
    command = [executable, *arguments[1:index], str(Path(__file__).resolve()), "_run"]
    if name:
        command.extend(["--name", name])
    command.extend(["--", *arguments[index:]])
    os.execv(executable, command)


def _duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def tui():
    import curses
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("TUI needs a terminal. Use `ssh -t kaiser ~/.local/bin/trainctl` for a remote TUI.")

    def screen(window):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        window.timeout(500)
        selected_id, selected, message = None, 0, ""
        while True:
            rows = jobs()
            if selected_id:
                selected = next((i for i, row in enumerate(rows) if row["run_id"] == selected_id), selected)
            selected = min(max(0, selected), max(0, len(rows)-1))
            selected_id = rows[selected]["run_id"] if rows else None
            window.erase()
            height, width = window.getmaxyx()

            def put(y, value, attr=0):
                if y < height-1 and width > 1:
                    try:
                        window.addnstr(y, 0, value, width-1, attr)
                    except curses.error:
                        pass

            put(0, f"trainctl — {socket.gethostname()} — same-process pause / resume", curses.A_BOLD)
            put(1, "↑/↓ j/k select   Space toggle   p pause   r resume   q disconnect")
            put(2, "ID            PID      STATE             UPDATES   PAUSED     NAME")
            for i, row in enumerate(rows):
                label = "resuming" if row["status"] == "paused" and not row["pause_requested"] else row["status"]
                put(3+i, f"{row['run_id']}  {row['identity']['pid']:<7}  {label:<17} "
                    f"{row['completed_updates']:>7}   {_duration(row['paused_seconds'])}   {row['name']}",
                    curses.A_REVERSE if i == selected else 0)
            y = min(4+len(rows), max(3, height-7))
            if rows:
                row = rows[selected]
                put(y, f"Phase: {row['phase']}")
                put(y+1, f"Directory: {row['cwd']}")
                put(y+2, f"Active: {_duration(row['active_seconds'])}  Pauses: {row['pause_count']}")
            else:
                put(y, "No controlled training process. Existing uninstrumented jobs cannot be paused here.")
            put(y+3, "Pause is acknowledged at the next safe boundary; GPU memory remains allocated.")
            put(y+4, "Closing this TUI leaves training running or paused exactly as selected.")
            put(height-2, message)
            window.refresh()
            key = window.getch()
            if key in (ord("q"), 27):
                return
            if key in (curses.KEY_UP, ord("k")):
                selected = max(0, selected-1)
                selected_id = rows[selected]["run_id"] if rows else None
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = min(len(rows)-1, selected+1)
                selected_id = rows[selected]["run_id"] if rows else None
            elif rows and key in (ord("p"), ord("r"), ord(" ")):
                row = rows[selected]
                action = "resume" if key == ord("r") or (key == ord(" ") and row["pause_requested"]) else "pause"
                try:
                    reply = rpc(row, action)
                    message = f"{row['name']}: {reply['status']} ({action} requested)"
                except (OSError, ValueError) as error:
                    message = str(error)

    curses.wrapper(screen)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pause/resume a live RSL-RL learner without restarting it.")
    parser.add_argument("--host", help="Open the same command on an SSH host, e.g. kaiser")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("tui", help="Interactive controller (default)")
    listing = commands.add_parser("list", help="List registered jobs")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--json", action="store_true")
    for action in ("status", "pause", "resume"):
        command = commands.add_parser(action)
        command.add_argument("job", help="Job ID/prefix, name or PID")
        command.add_argument("--wait", type=float, default=10., help="Seconds to wait for acknowledgement")
    for action in ("run", "_run"):
        command = commands.add_parser(action, help="Launch a Python training script with control support")
        command.add_argument("--name")
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["wheel-leg"]:
        arguments = arguments[1:]
    args = parser.parse_args(arguments)
    try:
        if args.host:
            if args.host.startswith("-") or not args.host.strip():
                raise ValueError("Invalid SSH host")
            remaining = arguments[:]
            for i, item in enumerate(remaining):
                if item == "--host":
                    del remaining[i:i+2]
                    break
                if item.startswith("--host="):
                    del remaining[i]
                    break
            remote = 'exec "$HOME/.local/bin/trainctl" wheel-leg'+(" "+shlex.join(remaining) if remaining else "")
            return subprocess.call(["ssh", "-t" if args.command in (None, "tui") else "-T", args.host, remote])
        if args.command in (None, "tui"):
            tui()
        elif args.command in ("run", "_run"):
            values = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            if args.command == "run":
                launch(values, args.name)
            else:
                _run_script(values, args.name)
        elif args.command == "list":
            rows = jobs(include_finished=args.all)
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=2))
            else:
                for row in rows:
                    print(f"{row['run_id']}  PID={row['identity']['pid']}  {row['status']:<17} "
                          f"updates={row['completed_updates']}  {row['name']}")
                if not rows:
                    print("No controlled training process. Start future jobs with trainctl run or train_prone_motion.sh.")
        else:
            if args.wait < 0:
                raise ValueError("--wait must be nonnegative")
            job = select_job(args.job, jobs(include_finished=True))
            reply = rpc(job, args.command)
            deadline = time.monotonic()+args.wait
            desired = "paused" if args.command == "pause" else "running"
            while args.command != "status" and reply["status"] != desired and time.monotonic() < deadline:
                if reply["status"] in (*TERMINAL, "stopping"):
                    break
                time.sleep(.05)
                reply = rpc(job)
            print(json.dumps(reply, ensure_ascii=False, indent=2))
            if args.command != "status" and reply["status"] != desired:
                return 2  # A pending request is never reported as an acknowledged pause.
    except (OSError, ValueError, RuntimeError) as error:
        print(f"trainctl: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
