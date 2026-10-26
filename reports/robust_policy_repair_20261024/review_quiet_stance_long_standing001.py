"""Apply the unchanged two-seed30s acceptance; preserve every failed item."""
from pathlib import Path
import json,time,sys
sys.path[:0]=['src','scripts']
from export_robust_getup import assess_extended_standing,digest
from train_robust_getup import write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
queue=json.loads((R/'diagnosis_queue038.json').read_text());assert queue['status']=='completed'
plan=json.loads((R/'diagnosis_plan038.json').read_text());assert all(digest(n)==h for n,h in plan['source_sha256'].items())
protocol=json.loads((R/'quiet_stance_standing_protocol001.json').read_text());assert digest(protocol['formal_protocol'])==protocol['formal_protocol_sha256']
formal=json.loads(Path(protocol['formal_protocol']).read_text());root=Path(plan['jobs'][1]['output'])
evaluations=[root/str(seed)/'robust' for seed in formal['long_standing_confirmation']['required_seeds']]
review=assess_extended_standing(evaluations,protocol['checkpoint_sha256'],formal)
development=json.loads((R/'quiet_stance050_development_standing001.json').read_text())
out=dict(schema='quiet-stance-original-long-standing-review-v1',checkpoint=protocol['checkpoint'],checkpoint_sha256=protocol['checkpoint_sha256'],
 formal_protocol=protocol['formal_protocol'],formal_protocol_sha256=protocol['formal_protocol_sha256'],standing=review,
 development_standing_pass=development['pose_and_position_pass'],development_total=development['episodes'],
 frozen_inputs=len(plan['source_sha256']),frozen_sources_exact=True,new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,
 qualification=None,hardware_started=False,decision='long_standing_pass_only_other_gates_remain' if review['adequate'] else 'reject_robust_qualification',created_unix=time.time())
write_json(R/'quiet_stance_long_standing_decision001.json',out)
print(json.dumps(out,indent=2));print('Formal adequate:',review['adequate'],'failures:',len(review['failures']))
p=json.loads((R/'progress.json').read_text());p.update(status='quiet_stance_long_standing_reviewed',active_native_queue=None,active_diagnosis_queue=None,active_supervisor=None,active_job=None,
 latest_research_decision=out['decision'],new_PPO_admitted=0,updated_unix=time.time());write_json(R/'progress.json',p)
