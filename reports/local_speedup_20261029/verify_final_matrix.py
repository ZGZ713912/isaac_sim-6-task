"""Verify original-event timing and complete fixed-action native traces."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path('/home/noir/.codex/worktrees/prone-motion-long/isaac_wheeled_rl_train')
OUT = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/local_speedup_20261029')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_case(name):
    directory = OUT/name
    result = json.loads((directory/'result.json').read_text())
    assert result['status'] == 'completed' and not result['source_mismatches_at_exit']
    assert not result['checkpoint_reads'] and not result['new_PPO']
    assert not result['optimizer_attempts'] and not result['actor_or_critic_created']
    runtime = result['runtime']
    assert (runtime['physics_dt'],runtime['policy_dt'],runtime['decimation'],runtime['num_envs'],
            runtime['position_iterations'],runtime['velocity_iterations']) == (.001,.005,5,256,64,32)
    events = EventAccumulator(str(directory/'events'),size_guidance={'scalars':0}).Reload()
    samples = events.Scalars('PerformanceProbe/policy_step_s')
    means = events.Scalars('PerformanceProbe/policy_step_mean_s')
    assert len(samples) == result['measured_policy_steps'] == 192
    assert [v.step for v in samples] == list(range(192))
    times = np.array([v.value for v in samples])
    assert np.allclose(times,result['step_times_s'],atol=1e-8,rtol=1e-7)
    assert len(means) == 1 and means[0].step == 192
    assert abs(means[0].value-result['step_mean_s']) < 1e-7
    trace = directory/'trace.npz'
    assert sha(trace) == result['trace_sha256']
    return result,dict(name=name,result=str(directory/'result.json'),result_sha256=sha(directory/'result.json'),
        trace=str(trace),trace_sha256=sha(trace),event_files=[dict(path=str(p),sha256=sha(p))
        for p in sorted((directory/'events').glob('events.out.tfevents.*'))],
        event_sample_count=len(samples),step_mean_s=float(times.mean()),
        step_median_s=float(np.median(times)),step_p90_s=float(np.quantile(times,.9)),
        pid=result['pid'],start_ticks=result['start_ticks'],runtime=runtime,
        source_file_count=len(result['source_hashes']),source_mismatches_at_exit=[])

comparisons = []
for domain,before,after in [
    ('nominal','final_eager_nominal001','final_graph_nominal002'),
    ('requested_1_training_mix','final_eager_requested001','final_graph_requested001')]:
    rb,b = load_case(before)
    ra,a = load_case(after)
    with np.load(OUT/before/'trace.npz') as zb, np.load(OUT/after/'trace.npz') as za:
        assert set(zb.files) == set(za.files)
        checked = []
        for key in sorted(zb.files):
            x,y = zb[key],za[key]
            assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes(),key
            checked.append(dict(key=key,shape=list(x.shape),dtype=str(x.dtype),bitwise_equal=True))
        reset_count = int(za['done'].sum())
        disabled = int(za['diag/passive'].sum())
        total_samples = int(za['diag/passive'].size)
    common = set(rb['source_hashes']) & set(ra['source_hashes'])
    changed = sorted(k for k in common if rb['source_hashes'][k] != ra['source_hashes'][k])
    assert changed == sorted([
        'docs/PRONE_MOTION_TRAINING_20261028.md','scripts/train_prone_motion.py',
        'src/wheeled_tasks/chassis/encoder_spring_feedforward.py',
        'src/wheeled_tasks/chassis/prone_motion_env.py',
        'src/wheeled_tasks/chassis/training_gas_spring.py'])
    comparisons.append(dict(domain=domain,before=b,after=a,trace_policy_steps=256,
        trace_physics_steps=1280,arrays_checked=checked,all_arrays_bitwise_equal=True,
        done_events_in_trace=reset_count,disabled_lane_samples=disabled,
        active_lane_samples=total_samples-disabled,common_source_differences=changed,
        source_paths_added=sorted(set(ra['source_hashes'])-set(rb['source_hashes'])),
        source_paths_removed=sorted(set(rb['source_hashes'])-set(ra['source_hashes'])),
        sampling_time_reduction_percent=100*(1-a['step_mean_s']/b['step_mean_s']),
        sampling_throughput_multiplier=b['step_mean_s']/a['step_mean_s']))

native = json.loads((OUT/'native_probe001/progress.json').read_text())
assert native['status'] == 'native_probe_completed_unqualified' and native['all_native_checks_pass']
assert native['new_PPO'] == native['completed_updates'] == native['optimizer_attempts'] == 0
assert not native['checkpoint_reads']
assert len(native['native_probes']) == 2
for p in native['native_probes']:
    assert len(p['checks']) == 29 and all(p['checks'].values()) and p['new_PPO'] == 0
assert not native['source_mismatches_at_exit']
startup_files = list((OUT/'native_probe001/contexts').glob('*/native_startup.json'))
assert len(startup_files) == 2
for path in startup_files:
    assert json.loads(path.read_text())['performance_backend']['name'] == 'cuda_graph'
gpu_pids = {int(v) for v in subprocess.check_output(
    ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()}
cleanup = []
for name in ('baseline001','graph001','instrument001','threads4_001','graph002',
             'final_graph_nominal001','final_eager_nominal001','final_graph_nominal002',
             'final_eager_requested001','final_graph_requested001','native_probe001'):
    path = OUT/name/('progress.json' if name == 'native_probe001' else 'result.json')
    record = json.loads(path.read_text())
    proc = Path('/proc')/str(record['pid'])
    same_identity = (proc.exists() and int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
                     == record['start_ticks'])
    assert not same_identity and record['pid'] not in gpu_pids
    cleanup.append(dict(name=name,pid=record['pid'],start_ticks=record['start_ticks'],
        original_process_alive=False,in_gpu_compute_list=False))

acceptance = json.loads((ROOT/'contracts/wheel_leg_acceptance_requirements_v2.json').read_text())
capabilities = [dict(id=c['id'],requirement=c['requirement'],status='not_revalidated',
    reason='Zero-PPO constant-action performance/wiring probe; no learned policy evaluated',passed=False)
    for c in acceptance['capabilities']]
assert len(capabilities) == 13
receipt = dict(schema='prone-motion-local-speedup-v1',
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='local_native_equivalence_and_performance_verified',
    source_commit=native['committed_source']['commit'],baseline_commit='0c197fddce0d823770703ee61467dc2efd0c99ad',
    formal_frozen_source=native['committed_source'],
    benchmark_script_sha256=sha(ROOT/'scripts/benchmark_prone_motion.py'),
    new_PPO=0,optimizer_attempts=0,actor_or_critic_created=False,checkpoint_reads=[],
    server_modified=False,server_training_stopped=False,server_speedup_measured=False,
    local_training_started=False,local_tensorboard_server_started=False,
    comparisons=comparisons,native_probe=dict(path=str(OUT/'native_probe001/progress.json'),
        sha256=sha(OUT/'native_probe001/progress.json'),contexts=2,checks_per_context=29,all_pass=True),
    tests=dict(path=str(OUT/'tests002.log'),sha256=sha(OUT/'tests002.log'),passed=110,skipped=1,
        failed=0,earlier_test_fixture_failure_preserved=str(OUT/'tests001.log'),
        skipped_vendor_module_rechecked=dict(path=str(OUT/'tests003_vendor.log'),
            sha256=sha(OUT/'tests003_vendor.log'),passed=13,skipped=0,failed=0,
            ISAACLAB_SOURCE_ROOT='/home/noir/Documents/workspace/IsaacLab/source'),
        total_distinct_tests_passed=123,unresolved_skips=0),
    owned_native_process_cleanup=cleanup,
    capabilities=capabilities,full_capability_qualified=False,hardware_ready=False,
    limits=['Bounded serial same-machine timings, not sustained PPO throughput or convergence.',
        'Desktop GPU is shared; timing variance remains.',
        'Requested1 is the declared training mix with25 percent nominal rehearsal, not a held-out all-random evaluation.',
        'Do not apply these timings directly to the4090 server; its frozen learner was untouched.',
        'All13 long-term capabilities remain unqualified by this performance-only test.'])
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],comparisons=[dict(domain=x['domain'],
    before_ms=1000*x['before']['step_mean_s'],after_ms=1000*x['after']['step_mean_s'],
    reduction_percent=x['sampling_time_reduction_percent'],arrays=len(x['arrays_checked']),
    reset_events=x['done_events_in_trace']) for x in comparisons]),indent=2))
