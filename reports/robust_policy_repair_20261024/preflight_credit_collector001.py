"""Check actual runner ordering and import the exact collector without main."""
import ast
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
sys.path[:0]=['src','scripts',str(Path(__file__).parent)]
from rsl_rl.runners import OnPolicyRunner
from train_robust_getup import digest,write_json

R=Path('reports/robust_policy_repair_20261024')
source=textwrap.dedent(inspect.getsource(OnPolicyRunner.learn))
tree=ast.parse(source)
calls={}
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('compute_returns','update'):
        calls.setdefault(node.func.attr,[]).append(node.lineno)
checks=dict(native_compute_returns_before_update=max(calls['compute_returns'])<min(calls['update']))
wrapper=R/'collect_entry_credit_rollout001.py'
sys.argv=[str(wrapper),'--diagnostic-rollout-only','--output','/tmp/credit-import-only']
tree=ast.parse(wrapper.read_text());tree.body=tree.body[:-1]
namespace={'__file__':str(wrapper.resolve()),'__name__':'credit_import_fixture'}
exec(compile(tree,str(wrapper),'exec'),namespace)
checks.update(collector_imported=True,main_not_called=not Path('/tmp/credit-import-only').exists())
out=dict(schema='zero-PPO-collector-preflight-v1',checks=checks,native_call_lines=calls,
    actual_runner_path=inspect.getsourcefile(OnPolicyRunner),actual_runner_sha256=digest(Path(inspect.getsourcefile(OnPolicyRunner))),
    collector_sha256=digest(wrapper),new_PPO=0,simulator_started=False,created_unix=time.time())
write_json(R/'credit_collector_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values())
