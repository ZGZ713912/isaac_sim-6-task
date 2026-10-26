"""One explicitly labelled native mean-function diagnostic, zero PPO."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
index=sys.argv.index('--retained-quiet-authority');floor=float(sys.argv[index+1]);del sys.argv[index:index+2]
if floor!=.5:raise ValueError('Only the predeclared .5 counterfactual is admitted')
if '--sampled-policy' in sys.argv:raise ValueError('Mean diagnostic only; native sampling is unchanged')
from quiet_gyro_authority_probe001 import policy_authority,VERSION
import evaluate_gap_robustness as evaluate
native_write=evaluate.write_json
metadata=dict(schema=VERSION,retained_quiet_authority=floor,
    intervention='Policy-only quiet gyro release retains half of otherwise removed authority; request/tilt fade and all recovery/motor/contact guards stay native.',
    observer_changed=False,distribution_changed=False,checkpoint_weights_changed=False,
    eligible_for_checkpoint_mean_qualification=False,wrapper_sha256=evaluate.digest(Path(__file__)),
    probe_sha256=evaluate.digest(Path(__file__).with_name('quiet_gyro_authority_probe001.py')))
def write(path,value):
    if Path(path).name in ('task_contract.json','progress.json'):
        value.setdefault('evaluation_overrides',{})['quiet_gyro_authority_probe']=metadata
        value['qualification']=None
    if Path(path).name=='progress.json' and 'source_sha256' in value:
        output=Path(path).parent
        for file in (Path(__file__),Path(__file__).with_name('quiet_gyro_authority_probe001.py')):
            relative=file.relative_to(ROOT);target=output/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())
            value['source_sha256'][str(relative)]=evaluate.digest(file)
    return native_write(path,value)
evaluate.write_json=write
with policy_authority(floor):evaluate.main()
