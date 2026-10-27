"""Seal completed same017PT/seed domain and rejected early-enable diagnostics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')
NAMES=dict(nominal='fresh017_peak_native_nominal002',requested='fresh017_peak_native_requested002',
    short_passive_nominal='fresh017_peak_native_nominal_shortpassive001')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    results,inputs={},{}
    for label,name in NAMES.items():
        p=BASE/REL/name;r=read(p/'receipt.json');s=read(p/'result.json')
        assert r['status']=='completed_diagnostic_unqualified' and not Path(f"/proc/{r['pid']}").exists()
        assert all(sha(ROOT/k)==v==sha(p/'source_snapshot'/k) for k,v in r['source_sha256'].items())
        results[label]=dict(global_phases=[x for x in s['groups'] if x['role']=='all'],
            native_engineering=s['native_engineering'],pid=r['pid'],native_policy_steps=400,physical_steps=2000)
        inputs[label]={str(f):sha(f) for f in (p/'receipt.json',p/'result.json',p/'native_peak_trace.npz',
            p/'cases.json',p/'task_contract.json')}
    a,b=[BASE/REL/NAMES[k] for k in ('nominal','short_passive_nominal')]
    ta,tb=[read(p/'task_contract.json') for p in (a,b)]
    # Each output owns its derived JSON path; its actual domain bytes must match.
    assert sha(Path(ta['randomization_study']['input_file']))==sha(Path(tb['randomization_study']['input_file']))
    ta['randomization_study']['input_file']=tb['randomization_study']['input_file']='<same-content-derived-study>'
    changed=[k for k in ta if ta[k]!=tb[k]]
    assert set(changed)=={'passive_min_seconds','passive_max_seconds','passive_hold_seconds'}
    z1,z2=[np.load(p/'native_peak_trace.npz') for p in (a,b)]
    first={k:np.array_equal(z1[k][0],z2[k][0]) for k in z1.files};assert all(first.values()),first
    assert (a/'cases.json').read_bytes()==(b/'cases.json').read_bytes()
    out=dict(schema='fresh017-native-peak-and-passive-comparison-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=read(a/'receipt.json')['checkpoint_sha256'],
        seed=271301,case_count=198,independent_native_runs=3,new_native_policy_steps=1200,physical_steps=6000,
        new_PPO=0,optimizer_constructed=False,inputs=inputs,results=results,
        only_passive_task_settings_changed=changed,derived_domain_bytes_equal=True,
        all_original_first5ms_fields_equal=first,short_passive_adopted=False,requested_short_passive_run=False,
        decision='Reject5ms candidate after nominal active peak46.951->58.301rad/s,200samples>=50and10impact-guard cases. Enable also changes native guard phase eligibility, so guard count alone is not proof of larger force. Whole valid peak also rises; preserve original startup.',
        domain_interpretation='Same017PT requested active-recovery peak71.070 vs46.951 nominal;2impact guards vs0. Passive36.534 vs35.469; ordinary11.300 vs7.597. Phases separated, individual domain causes not isolated.',
        producer_sha256=sha(Path(__file__)),short_window_not_qualification=True,qualified=False,hardware_ready=False)
    for folder in (ROOT/REL,BASE/REL):
        p=folder/'fresh017_peak_startup_comparison001.json';assert not p.exists()
        p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(paired_first_interval_fields=len(first),short_passive_adopted=False)))


if __name__=='__main__':main()
