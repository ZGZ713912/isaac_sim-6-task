"""Compare the actual native task to CPU-prepared settings during startup."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from iteration045_runner import write


def preflight(parent):
    from train_robust_getup import prepare
    with tempfile.TemporaryDirectory(prefix="position058-task-") as folder:
        args = SimpleNamespace(output=Path(folder)/"preflight",resume=parent,
            bank=ROOT/"reports/unified_robust_20261021/pose_bank002.json",no_domains=False,
            curriculum_updates=128,quiet_leg_alpha=.25,quiet_wheel_alpha=.65,domain_scale=.35,
            quiet_only=True,standing_support_training=True,early_standing_position=True,standing_drift_objective=True,
            quiet_exploration_fraction=.06,quiet_wheel_exploration_fraction=.35,
            num_envs=128,rollout_steps=128,mode="train",friction="high",friction_min=.3,seed=211329,
            isolated_origins=True)
        _,settings = prepare(args)
    assert settings["quiet_exploration_fraction"] == .06 and settings["quiet_wheel_exploration_fraction"] == .35
    assert settings["quiet_velocity_feedback_mode"] == "standing_displacement"
    result = dict(schema="position058-cpu-task-preflight-v1",settings=settings,
        settings_sha256=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),new_macro_updates=0)
    write(REPORTS/"iteration058_preflight.json",result)
    return settings


def checked_native(name,command,output,expected):
    if output.exists():
        raise ValueError("Refuse to overwrite native output: "+str(output))
    receipt = dict(schema="position058-checked-native-process-v1",command=command,status="starting",
                   started=time.time(),returncode=None,task_checked_before_training=False)
    with (REPORTS/(name+".log")).open("w") as log:
        child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(pid=child.pid,status="running")
        write(REPORTS/(name+"_process.json"),receipt)
        try:
            while child.poll() is None:
                contract = output/"task_contract.json"
                if not receipt["task_checked_before_training"] and contract.exists():
                    actual = json.loads(contract.read_text())["settings"]
                    if actual != expected:
                        changed = {k:dict(expected=expected.get(k),actual=actual.get(k))
                                   for k in set(expected)|set(actual) if expected.get(k) != actual.get(k)}
                        raise ValueError("Actual native task disagrees with CPU preflight: "+str(changed))
                    receipt["task_checked_before_training"] = True
                    guard = dict(schema="position058-actual-task-guard-v1",pid=child.pid,actual_settings_exact=True,
                        settings_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),
                        actual_leg_fraction=actual["quiet_exploration_fraction"],
                        actual_wheel_fraction=actual["quiet_wheel_exploration_fraction"],
                        model_initial_exists_at_check=(output/"model_initial.pt").exists(),checked_at=time.time())
                    write(REPORTS/"iteration058_task_guard.json",guard)
                    write(REPORTS/(name+"_process.json"),receipt)
                time.sleep(.25)
        except BaseException:
            if child.poll() is None:
                child.terminate()
            child.wait()
            receipt.update(status="rejected_during_startup",returncode=child.returncode,finished=time.time())
            write(REPORTS/(name+"_process.json"),receipt)
            raise
        code = child.wait()
    progress = json.loads((output/"progress.json").read_text())
    receipt.update(returncode=code,native_status=progress.get("status"),finished=time.time(),
        status="completed" if code == 0 and progress.get("status") == "completed" and receipt["task_checked_before_training"] else "failed")
    write(REPORTS/(name+"_process.json"),receipt)
    if receipt["status"] != "completed":
        raise RuntimeError(str(receipt))
    return progress
