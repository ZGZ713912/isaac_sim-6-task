"""Nominal three-height mechanics candidate, never learned policy weights."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
RUN=BASE/'runs/scratch_standing_20261026/pilot010_requested_domains_retry001'
REPORT=BASE/'reports/robust_policy_repair_20261024'
OUT=REPORT/'three_height_static_prior002'
OUT.mkdir(exist_ok=False)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

cfg=json.loads((RUN/'config/contract.json').read_text())
manifest_file=WORK/cfg['asset_directory']/'manifest.json'
assert sha(manifest_file)==cfg['asset_manifest_sha256']
manifest=json.loads(manifest_file.read_text())
bank_file=RUN/'pose_bank.json'
bank=json.loads(bank_file.read_text())
assert bank['asset_manifest_sha256']==cfg['asset_manifest_sha256']
spec=importlib.util.spec_from_file_location('height_candidate_static_builder',WORK/'scripts/recovery_static_targets.py')
builder_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder_module)
builder=builder_module.RecoveryStaticTargetBuilder(manifest_file.parent,cfg['asset_manifest_sha256'])
heights=[.23,.305,.42]
targets=builder.build_targets(heights,leg_kp=cfg['v5_control']['leg_kp'])
assert targets['asset_manifest_sha256']==cfg['asset_manifest_sha256']
write(OUT/'targets.json',targets)
nominal=np.array([manifest['nominal_joint_pos'][name] for name in builder.controls])
rows=[]
for height,pose_name in zip(heights,['stand_low','stand','stand_high'],strict=True):
    target=targets['targets'][f'{height:g}']
    pose=bank['poses'][bank['diagnostic_height_starts']['mapping'][pose_name]]
    errors=[abs(target['full_joint_positions'][name]-pose['full_joint_positions'][name]) for name in builder.names]
    geometry=np.array(target['prepare_reference']['geometry_control6_rad'])
    delta=np.arctan2(np.sin(geometry-nominal),np.cos(geometry-nominal))/.25
    offset=np.array(target['prepare_reference']['pd_offset_control6_rad'])/.25
    biased=delta+offset
    biased[[2,5]]=0.
    rows.append(dict(height_m=height,pose_index=bank['diagnostic_height_starts']['mapping'][pose_name],
        maximum_joint_difference_from_current_bank_rad=max(errors),
        root_height_difference_from_current_bank_m=abs(target['root_position_m'][2]-pose['root_position_m'][2]),
        static_torque_control_nm=target['prepare_reference']['hold_torque_nm'],
        biased_action_policy_order=biased[[0,1,3,4,2,5]].tolist(),
        action_within_native_clip=bool(np.abs(biased).max()<cfg['v5_control']['action_clip']),
        static_leg_torque_within_native_cap=bool(np.abs(target['prepare_reference']['hold_torque_nm']).max()<40.)))
matching=all(r['maximum_joint_difference_from_current_bank_rad']<2e-6 and
    r['root_height_difference_from_current_bank_m']<2e-6 and r['action_within_native_clip'] and
    r['static_leg_torque_within_native_cap'] for r in rows)
z=(np.array(heights)-.305)/.1
basis=np.column_stack((np.ones(3),z,z*z))
coeff=np.linalg.solve(basis,np.array([row['biased_action_policy_order'][:4] for row in rows]))
grid=(np.linspace(.23,.42,1001)-.305)/.1
predicted=np.column_stack((np.ones(len(grid)),grid,grid*grid))@coeff
candidate=dict(coordinate='z=(requested_height_m-0.305)/0.1, before empirical normalization',
    basis=['1','z','z_squared'],coefficient_rows_by_basis_policy_legs=coeff.tolist(),
    maximum_anchor_residual_raw=float(np.abs(basis@coeff-np.array([r['biased_action_policy_order'][:4] for r in rows])).max()),
    sampled_interval_max_abs_raw=float(np.abs(predicted).max()),
    sampled_interval_inside_action_clip=bool(np.abs(predicted).max()<cfg['v5_control']['action_clip']),
    future_parameterization='If native-validated, use only to initialize ordinary trainable actor coefficients. No fixed runtime lookup, teacher, leg freeze or scripted recovery.',
    adopted=False)
old=REPORT/'three_height_static_prior001.json'
old_manifest=json.loads(old.read_text())['asset_manifest_sha256']
assert old_manifest!=cfg['asset_manifest_sha256']
result=dict(schema='fresh-three-height-static-initialization-candidate-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(),status='CPU_mechanics_match_unqualified' if matching else 'CPU_geometry_mismatch_rejected',
    current_manifest_sha256=cfg['asset_manifest_sha256'],bank_sha256=sha(bank_file),
    mechanism_config_sha256=sha(RUN/'config/contract.json'),targets_sha256=sha(OUT/'targets.json'),
    rows=rows,candidate=candidate,all_three_reset_geometries_match=matching,
    rejected_default_asset_calculation=dict(path=str(old),sha256=sha(old),manifest_sha256=old_manifest,
        reason='Legacy tool CLI defaults to stop105 asset, not the currently trained flat_base107. Preserved and excluded; constructor now receives exact current asset/manifest.'),
    dynamic_validation_completed=False,randomized_domain_validation_completed=False,actor_constructed=False,
    new_PPO=0,new_native_steps=0,training_changed=False,hardware_ready=False,
    producer_sha256=sha(Path(__file__)))
write(OUT/'result.json',result)
(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status=result['status'],rows=rows,candidate=candidate)))
