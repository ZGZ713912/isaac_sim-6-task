"""Run the complete release gate suite against the staged clock fix."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
out=HERE/'cpu001.json'
if out.exists():
    raise SystemExit('Preserve completed clock candidate proof')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binding=json.loads((HERE/'parent_binding.json').read_text())
if (sha(ROOT/'scripts/export_flat_base.py')!=binding['parent_export_sha256']
        or sha(HERE/'export_flat_base.py')!=binding['candidate_export_sha256']):
    raise SystemExit('Parent or staged candidate changed')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
spec=importlib.util.spec_from_file_location('export_flat_base',HERE/'export_flat_base.py')
candidate=importlib.util.module_from_spec(spec)
sys.modules['export_flat_base']=candidate
spec.loader.exec_module(candidate)
candidate.ROOT=ROOT

class Results:
    def __init__(self):
        self.passed=[]
        self.failed=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' and report.passed:
            self.passed.append(report.nodeid)
        if report.failed:
            self.failed.append(report.nodeid+'/'+report.when)

result=Results()
start=time.monotonic()
code=pytest.main(['-q',str(ROOT/'tests/test_flat_release_evidence.py'),str(HERE/'test_release_clock.py')],plugins=[result])
out.write_text(json.dumps(dict(schema='flat-release-clock-candidate-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=code==0 and not result.failed,
    exit_code=int(code),tests_passed=len(result.passed),duration_s=time.monotonic()-start,
    passed_tests=result.passed,failed_tests=result.failed,
    parent_export_sha256=binding['parent_export_sha256'],candidate_export_sha256=sha(HERE/'export_flat_base.py'),
    verifier_sha256=sha(Path(__file__)),new_test_sha256=sha(HERE/'test_release_clock.py'),
    existing_release_test_sha256=sha(ROOT/'tests/test_flat_release_evidence.py'),
    adopted=False,new_optimizer_updates=0,new_native_steps=0,behavior_qualified=False,
    scope='CPU validator paths and actual checkpoint-plan binding only. All deployment cohorts and numerical acceptance thresholds remain required.'),indent=2,allow_nan=False)+'\n')
raise SystemExit(int(code))
