"""Full RMCS default-height static-recall coverage after its basic diagnostic is sealed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
output = RUN/'static_recall1280_default_envelope001'
log_path = RUN/'static_recall1280_default_envelope001.log'
state_path = RUN/'static_recall1280_envelope_queue001.json'
contract_path = HERE/'static_recall1280_envelope_contract001.json'
if any(p.exists() for p in (output, log_path, state_path, contract_path)):
    raise SystemExit('Preserve earlier diagnostic attempts and partial outputs')
previous_path = RUN/'static_recall1280_diagnostic_queue001.json'
previous = read(previous_path)
seal_path = HERE/'static_recall1280_diagnostic_review001.json'
seal = read(seal_path)
if (previous['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(previous_path)
        or any(Path('/proc/'+str(previous[k])).exists() for k in ('pid', 'worker_pid'))):
    raise SystemExit('Wait for completed, sealed basic diagnostic and every owner to exit')
candidate = HERE/'static_recall_candidate001'
cpu_path = candidate/'actual1280_cpu002.json'
ast_path = candidate/'evaluator_ast_cpu001.json'
cpu, ast = read(cpu_path), read(ast_path)
checkpoint = RUN/'pilot025/model_001280.pt'
recalled = RUN/'pilot023/model_001248.pt'
evaluator = candidate/'evaluate.py'
helper = candidate/'static_recall.py'
if (not cpu['passed'] or not ast['passed'] or sha(checkpoint) != cpu['parent_checkpoint_sha256']
        or sha(recalled) != cpu['recalled_checkpoint_sha256']
        or sha(evaluator) != cpu['evaluator_sha256'] or sha(helper) != cpu['helper_sha256']):
    raise SystemExit('Require the exact CPU-reviewed1280/static1248 diagnostic composition')
hashes = dict(previous['source_sha256'])
for path in (evaluator, helper, Path(__file__)):
    hashes[str(path.relative_to(ROOT))] = sha(path)
if any(sha(ROOT/name) != value for name,value in hashes.items()):
    raise SystemExit('Reviewed execution source changed')
contract = dict(schema='flat-static-recall1280-full-envelope-diagnostic-contract-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), basic_diagnostic_review_sha256=sha(seal_path),
    cpu_proof_sha256=sha(cpu_path), evaluator_proof_sha256=sha(ast_path), source_sha256=hashes,
    checkpoint_sha256=sha(checkpoint), recalled_checkpoint_sha256=sha(recalled),
    seed=191029, optimizer_updates=0,
    unchanged='All22 original RMCS default-height cases, thresholds, horizons, reset seed and50Hz/1000Hz clocks.',
    changed='Only diagnostic feedback composition: zero commands recall1248; full moving endpoint preserves1280.',
    behavior_qualified=False, subsequent_training_queued=False, heldout_seed_used=False, hardware_motion_started=False)
contract_path.write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
state = dict(schema='flat-static-recall1280-full-envelope-diagnostic-queue-v1', pid=os.getpid(), status='starting',
    started_at=datetime.now(timezone.utc).isoformat(), contract_sha256=sha(contract_path),
    source_sha256=hashes, checkpoint_sha256=sha(checkpoint), recalled_checkpoint_sha256=sha(recalled),
    optimizer_updates=0, subsequent_training_queued=False, heldout_seed_used=False, hardware_motion_started=False)


def save():
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    temporary.replace(state_path)


try:
    command = [sys.executable, str(evaluator), '--checkpoint', str(checkpoint), '--deployment-pd',
        '--seed', '191029', '--rmcs-command-reference', '--packaged-default-height', '--envelope', '--output', str(output)]
    with log_path.open('x') as log:
        worker = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES'},
            stdout=log, stderr=subprocess.STDOUT)
        state.update(status='running', worker_pid=worker.pid)
        save()
        state['exit_code'] = worker.wait()
    if state['exit_code']:
        raise RuntimeError('Static recall diagnostic failed; no later child launched')
    if any(sha(ROOT/name) != value for name,value in hashes.items()):
        raise ValueError('Frozen execution source changed')
    data = read(output/'evaluation.json')
    if data['status'] != 'evaluated' or not data['result']['diagnostic_only']:
        raise ValueError('Diagnostic identity or forced nonqualification changed')
    state.update(status='complete', evaluation_sha256=sha(output/'evaluation.json'),
        traces_sha256=sha(output/'traces.npz'), finished_at=datetime.now(timezone.utc).isoformat())
except Exception:
    state.update(status='failed', error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
    raise
finally:
    save()
