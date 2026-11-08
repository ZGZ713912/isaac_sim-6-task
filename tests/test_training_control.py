"""Real IPC and native CPU PPO tests; never control a user-owned learner."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from types import SimpleNamespace

import pytest

from wheeled_algo import training_control as ctl


ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(ctl.__file__).resolve()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINCTL_STATE_DIR", str(tmp_path/"state"))
    # AF_UNIX paths must stay below 108 bytes, including long pytest paths.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="trainctl-") as runtime:
        monkeypatch.setenv("TRAINCTL_RUNTIME_DIR", runtime)
        yield tmp_path


def until(predicate, timeout=8.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError("Timed out waiting for control acknowledgement")


def test_ack_requires_training_thread_and_resume_keeps_same_process(registry):
    sync = []
    control = ctl.TrainingControl("unit-worker", synchronize=lambda: sync.append(True))
    steps = []
    gate = threading.Event()

    def work():
        gate.wait()
        for i in range(100):
            control.checkpoint("sampling", completed_updates=i)
            steps.append(i)
            time.sleep(.002)

    thread = threading.Thread(target=work)
    try:
        job = ctl.jobs()[0]
        assert ctl.rpc(job, "pause")["status"] == "pause_requested"
        assert control.snapshot()["pause_count"] == 0 and steps == []
        thread.start()
        gate.set()
        paused = until(lambda: (row := ctl.rpc(job))["status"] == "paused" and row)
        before = steps[:]
        time.sleep(.08)
        assert steps == before and paused["completed_updates"] == 0
        assert paused["identity"] == ctl.process_identity(os.getpid())
        assert sync == [True]
        assert ctl.rpc(job, "resume")["pause_requested"] is False
        until(lambda: len(steps) >= 10)
        ctl.rpc(job, "pause")
        until(lambda: ctl.rpc(job)["status"] == "paused")
        assert control.snapshot()["pause_count"] == 2
        ctl.rpc(job, "resume")
        thread.join(5)
        assert not thread.is_alive() and steps == list(range(100))
        assert control.snapshot()["paused_seconds"] >= .08
    finally:
        control.close()
        gate.set()
        if thread.ident:
            thread.join(5)
    assert ctl.jobs() == []
    assert ctl.jobs(include_finished=True)[0]["status"] == "finished"
    assert not control.socket_path.exists()


def test_cancel_pending_pause_never_claims_paused(registry):
    control = ctl.TrainingControl("cancel")
    try:
        job = ctl.jobs()[0]
        assert ctl.rpc(job, "pause")["status"] == "pause_requested"
        assert ctl.rpc(job, "resume")["status"] == "running"
        control.checkpoint("sample")
        assert ctl.rpc(job)["pause_count"] == 0
    finally:
        control.close()


def test_gpu_work_drains_before_ack_and_cancel_during_drain(registry):
    entered, leave = threading.Event(), threading.Event()
    control = ctl.TrainingControl("gpu-drain", synchronize=lambda: (entered.set(), leave.wait(5)))
    try:
        control.request("pause")
        thread = threading.Thread(target=lambda: control.checkpoint("sampling"))
        thread.start()
        assert entered.wait(5)
        assert ctl.rpc(ctl.jobs()[0])["status"] == "pause_requested"
        control.request("resume")
        leave.set()
        thread.join(5)
        assert control.snapshot()["status"] == "running" and control.snapshot()["pause_count"] == 0
    finally:
        leave.set()
        control.close()


def test_stop_handler_wakes_paused_job_without_replacing_existing_stop_semantics(registry):
    received = []
    control = ctl.TrainingControl("stop")
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda number, frame: received.append(number))
    bridge = ctl._SignalBridge(control)
    try:
        bridge.refresh()
        control.request("pause")
        thread = threading.Thread(target=lambda: control.checkpoint("sampling"))
        thread.start()
        until(lambda: control.snapshot()["status"] == "paused")
        # Invoke our own test handler directly; no OS process is signalled.
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        thread.join(5)
        assert received == [signal.SIGTERM] and not thread.is_alive()
        assert control.snapshot()["status"] == "stopping"
        with pytest.raises(ValueError, match="finishing"):
            control.request("pause")
        # A native rebuild may install another trainer handler.
        signal.signal(signal.SIGTERM, lambda number, frame: received.append("rebuilt"))
        bridge.refresh()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert received[-1] == "rebuilt"
    finally:
        bridge.close()
        signal.signal(signal.SIGTERM, previous)
        control.close()


def test_stale_identity_and_ambiguous_name_are_rejected(registry):
    first, second = ctl.TrainingControl("duplicate"), ctl.TrainingControl("duplicate")
    try:
        rows = ctl.jobs()
        with pytest.raises(ValueError, match="found 2"):
            ctl.select_job("duplicate", rows)
        row = ctl.select_job(first.run_id[:8], rows)
        assert row["run_id"] == first.run_id
        bad = dict(row, identity=dict(row["identity"], start_ticks=-1))
        with pytest.raises(ValueError, match="reused"):
            ctl.rpc(bad, "pause")
        bad = dict(row, run_id=second.run_id)
        with pytest.raises(ValueError, match="identity changed"):
            ctl.rpc(bad, "pause")
        assert not first.pause_requested and not second.pause_requested
    finally:
        first.close()
        second.close()


def test_pending_cli_returns_nonzero_and_json_is_honest(registry, capsys):
    control = ctl.TrainingControl("not-at-safe-point")
    try:
        assert ctl.main(["pause", control.run_id, "--wait", "0"]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "pause_requested"
        assert ctl.main(["resume", control.run_id]) == 0
        assert json.loads(capsys.readouterr().out)["pause_count"] == 0
        assert ctl.main(["pause", control.run_id, "--wait", "-1"]) == 1
    finally:
        control.close()


def test_private_registry_refuses_shared_directory(registry, monkeypatch):
    shared = registry/"shared"
    shared.mkdir(mode=0o755)
    monkeypatch.setenv("TRAINCTL_STATE_DIR", str(shared))
    with pytest.raises(RuntimeError, match="private"):
        ctl.TrainingControl("refused")


def test_ssh_forward_preserves_arguments_without_shell_expansion(monkeypatch):
    calls = []
    monkeypatch.setattr(ctl.subprocess, "call", lambda args: calls.append(args) or 0)
    assert ctl.main(["--host", "kaiser", "pause", "a $(touch /tmp/not-executed)"]) == 0
    import shlex
    assert calls[0][:3] == ["ssh", "-T", "kaiser"]
    assert shlex.split(calls[0][3])[2:] == ["wheel-leg", "pause", "a $(touch /tmp/not-executed)"]


def test_existing_ppo_hooks_and_metrics_are_preserved(registry):
    pytest.importorskip("torch")
    events, scalars = [], []
    alg = SimpleNamespace(act=lambda obs: events.append(("act", obs)) or obs,
                          update=lambda: events.append("update") or {"loss": 1.})
    native_act, native_update = alg.act, alg.update
    runner = SimpleNamespace(alg=alg, current_learning_iteration=17, device="cpu",
        logger=SimpleNamespace(writer=SimpleNamespace(add_scalar=lambda *args: scalars.append(args))))
    control = ctl.TrainingControl("hooks")
    try:
        def train():
            assert runner.alg.act("observations") == "observations"
            assert runner.alg.update() == {"loss": 1.}
            return "done"
        assert ctl.controlled_learn(runner, train, control) == "done"
        assert events == [("act", "observations"), "update"]
        assert alg.act is native_act and alg.update is native_update
        assert control.snapshot()["completed_updates"] == 18
        assert len(scalars) == 3 and {row[2] for row in scalars} == {18}
        runner.is_distributed = True
        with pytest.raises(ValueError, match="distributed"):
            ctl.controlled_learn(runner, train, control)
    finally:
        control.close()


def test_resume_accepts_control_only_additions_but_rejects_physics_change(registry):
    from wheeled_algo.prone_motion_resume import authenticate_sources, digest
    plant = registry/"plant.py"
    plant.write_text("unchanged dynamics")
    module = registry/"src/wheeled_algo/training_control.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(MODULE.read_bytes())
    old = {"plant.py": digest(plant)}
    current = dict(old, **{"src/wheeled_algo/training_control.py": digest(module)})
    proof = authenticate_sources({"sources": old}, current, registry)
    assert proof["added_resume_sources"] == ["src/wheeled_algo/training_control.py"]
    plant.write_text("changed dynamics")
    with pytest.raises(ValueError, match="changed training sources"):
        authenticate_sources({"sources": old}, dict(current, **{"plant.py": digest(plant)}), registry)


def test_pause_during_optimizer_waits_until_whole_update_commits(registry):
    pytest.importorskip("torch")
    control = ctl.TrainingControl("optimizer-boundary")
    halfway, finish_update = threading.Event(), threading.Event()
    committed = []

    def update():
        committed.append("first Adam batch")
        halfway.set()
        assert finish_update.wait(5)
        committed.append("last Adam batch")
        return {}

    runner = SimpleNamespace(alg=SimpleNamespace(act=lambda x: x, update=update),
                             current_learning_iteration=0, device="cpu")
    # Capture the call dynamically so the controlled wrapper is exercised.
    thread = threading.Thread(target=lambda: ctl.controlled_learn(
        runner, lambda: runner.alg.update(), control))
    try:
        thread.start()
        assert halfway.wait(5)
        control.request("pause")
        time.sleep(.03)
        assert control.snapshot()["status"] == "pause_requested"
        assert committed == ["first Adam batch"]
        finish_update.set()
        until(lambda: control.snapshot()["status"] == "paused")
        assert committed == ["first Adam batch", "last Adam batch"]
        assert control.snapshot()["completed_updates"] == 1
        control.request("resume")
        thread.join(5)
        assert not thread.is_alive()
    finally:
        finish_update.set()
        control.close()
        thread.join(5)


def test_direct_project_entry_does_not_create_duplicate_controller(registry):
    pytest.importorskip("torch")
    runner = SimpleNamespace(device="cpu", current_learning_iteration=0,
        alg=SimpleNamespace(act=lambda x: x, update=lambda: {}))
    runner.learn = lambda count: [runner.alg.update() for _ in range(count)]
    ctl.learn(runner, 2, name="direct-entry")
    rows = ctl.jobs(include_finished=True)
    assert len(rows) == 1 and rows[0]["completed_updates"] == 2
    called = []

    def already_wrapped(count):
        called.append(count)

    already_wrapped._trainctl_wrapped = True
    runner.learn = already_wrapped
    ctl.learn(runner, 7)
    assert called == [7] and len(ctl.jobs(include_finished=True)) == 1


def test_actual_cpu_ppo_pause_does_not_change_adam_rng_or_rollout(registry):
    torch = pytest.importorskip("torch")
    from test_prone_motion_resume import BASE, runner
    from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
    from wheeled_algo.prone_motion_resume import equal_state

    torch.set_num_threads(2)
    plan = resolve_plan(BASE, num_envs=20, num_steps_per_env=4)
    torch.manual_seed(781)
    uninterrupted = runner(plan)
    uninterrupted.learn(3)
    expected = deepcopy(uninterrupted.alg.save())
    expected_rng = torch.get_rng_state().clone()

    torch.manual_seed(781)
    paused = runner(plan)
    control = ctl.TrainingControl("real-cpu-ppo")
    errors, before, after = [], [], []
    native_act = paused.alg.act
    sampled = 0

    def request_mid_rollout(observations):
        nonlocal sampled
        result = native_act(observations)
        sampled += 1
        if sampled == 2:
            control.request("pause")
        return result

    paused.alg.act = request_mid_rollout

    def observe_and_resume():
        try:
            until(lambda: control.snapshot()["status"] == "paused")
            before.append((deepcopy(paused.alg.save()), torch.get_rng_state().clone(), paused.alg.storage.step))
            time.sleep(.08)
            after.append((deepcopy(paused.alg.save()), torch.get_rng_state().clone(), paused.alg.storage.step))
        except BaseException as error:
            errors.append(error)
        finally:
            control.request("resume")

    observer = threading.Thread(target=observe_and_resume)
    observer.start()
    try:
        ctl.controlled_learn(paused, lambda: paused.learn(3), control)
        observer.join(10)
        assert not errors and not observer.is_alive()
        assert before[0][2] == after[0][2] == 2
        assert equal_state(before[0][0], after[0][0])
        assert torch.equal(before[0][1], after[0][1])
        assert equal_state(paused.alg.save(), expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert control.snapshot()["pause_count"] == 1
    finally:
        control.close()
        observer.join(10)


def _dummy_rsl(registry):
    package = registry/"fake/rsl_rl/runners"
    package.mkdir(parents=True)
    (package.parent/"__init__.py").write_text("")
    (package/"__init__.py").write_text("from .on_policy_runner import OnPolicyRunner\n")
    (package/"on_policy_runner.py").write_text('''
from types import SimpleNamespace
import time
class OnPolicyRunner:
    def __init__(self):
        self.device = "cpu"
        self.current_learning_iteration = 0
        self.alg = SimpleNamespace(act=lambda x: x, update=lambda: {})
    def learn(self):
        for i in range(10000):
            self.alg.act(None)
            self.alg.update()
            time.sleep(.01)
''')
    script = registry/"fake/train.py"
    script.write_text("from rsl_rl.runners import OnPolicyRunner\nOnPolicyRunner().learn()\n")
    return script


def test_subprocess_bootstrap_and_actual_tui_disconnect(registry):
    pytest.importorskip("torch")
    script = _dummy_rsl(registry)
    # This is a private dummy learner only. No user process is controlled.
    worker = subprocess.Popen([sys.executable, str(MODULE), "run", "--name", "test-only", "--",
        sys.executable, "-u", str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ui = None
    master = slave = None
    captured = bytearray()
    try:
        job = until(lambda: next((r for r in ctl.jobs() if r["name"] == "test-only"
                                  and r["completed_updates"] >= 2), None))
        assert job["identity"]["pid"] == worker.pid
        master, slave = pty.openpty()
        import fcntl
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))
        original_terminal = termios.tcgetattr(slave)
        env = dict(os.environ, TERM="xterm-256color")
        ui = subprocess.Popen([sys.executable, str(MODULE)], stdin=slave, stdout=slave, stderr=slave, env=env)

        def read_screen():
            if select.select([master], [], [], .05)[0]:
                captured.extend(os.read(master, 65536))
            return b"Space toggle" in captured

        until(read_screen)
        os.write(master, b"p")
        until(lambda: ctl.rpc(job)["status"] == "paused")
        count = ctl.rpc(job)["completed_updates"]
        os.write(master, b"q")
        assert ui.wait(5) == 0
        assert termios.tcgetattr(slave) == original_terminal
        time.sleep(.1)
        assert ctl.rpc(job)["status"] == "paused" and ctl.rpc(job)["completed_updates"] == count
        # Reconnect from another CLI and resume the very same process.
        reply = subprocess.run([sys.executable, str(MODULE), "resume", job["run_id"]],
            text=True, capture_output=True, timeout=8)
        assert reply.returncode == 0, reply.stderr
        assert json.loads(reply.stdout)["identity"]["pid"] == worker.pid
        until(lambda: ctl.rpc(job)["completed_updates"] > count)
    finally:
        if ui is not None and ui.poll() is None:
            ui.terminate()
            ui.wait(5)
        if master is not None:
            os.close(master)
            os.close(slave)
        if worker.poll() is None:
            worker.terminate()  # Only the subprocess created by this test.
        _, stderr = worker.communicate(timeout=8)
    row = ctl.jobs(include_finished=True)[0]
    assert row["status"] == "failed" and b"Traceback" not in stderr


def test_installer_is_stable_and_will_not_replace_unrelated_tool(registry):
    spec = importlib.util.spec_from_file_location("install_trainctl", ROOT/"scripts/install_trainctl.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    executable = installer.install(registry/"install")
    command = subprocess.run([str(executable), "list", "--json"], capture_output=True, text=True)
    assert command.returncode == 0 and json.loads(command.stdout) == []
    assert (registry/"install/share/wheel-leg-trainctl/trainctl.py").read_bytes() == MODULE.read_bytes()
    installer.install(registry/"install")  # An update of our own tool is allowed.
    executable.write_text("#!/bin/sh\necho unrelated\n")
    with pytest.raises(ValueError, match="unrelated"):
        installer.install(registry/"install")


def test_existing_remote_console_keeps_its_commands_and_bytes(registry):
    spec = importlib.util.spec_from_file_location("install_trainctl", ROOT/"scripts/install_trainctl.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    prefix = registry/"shared-install"
    executable = prefix/"bin/trainctl"
    executable.parent.mkdir(parents=True)
    original = b'#!/bin/sh\n# -m transformer_rl.common.training_cli\nprintf "legacy:%s\\n" "$*"\n'
    executable.write_bytes(original)
    executable.chmod(0o755)
    installer.install(prefix, share_existing=True)
    saved = list((prefix/"share/wheel-leg-trainctl").glob("original-trainctl-*.sh"))
    assert len(saved) == 1 and saved[0].read_bytes() == original
    old = subprocess.run([str(executable), "watch", "existing-session"], text=True, capture_output=True)
    assert old.returncode == 0 and old.stdout.strip() == "legacy:watch existing-session"
    new = subprocess.run([str(executable), "wheel-leg", "list", "--json"], text=True, capture_output=True)
    assert new.returncode == 0 and json.loads(new.stdout) == []
    installer.install(prefix)  # Reinstallation retains the legacy route.
    again = subprocess.run([str(executable), "status", "existing-session"], text=True, capture_output=True)
    assert again.stdout.strip() == "legacy:status existing-session"
    assert saved[0].read_bytes() == original
