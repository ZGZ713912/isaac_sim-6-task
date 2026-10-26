"""Additional diagnostic reset poses; retain every historical acceptance pose.

Only closed-chain geometry and COM alignment are used. No static PD target,
spring-force compensation, control trajectory or new actor is generated.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from build_robust_getup_bank import Builder
from recovery_static_targets import RecoveryStaticTargetBuilder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Paired107StaticGeometry(RecoveryStaticTargetBuilder):
    validate_q = Builder.validate_q


def main():
    r = Path(__file__).resolve().parent
    source = r / 'pose_bank_rmcs003.json'
    target = r / 'pose_bank_height_matched073.json'
    assert not target.exists(), 'Never overwrite a diagnostic reset bank'
    original = json.loads(source.read_text())
    bundle = ROOT / original['asset_directory']
    assert sha(bundle / 'manifest.json') == original['asset_manifest_sha256']
    assert sha(bundle / 'model_spec.json') == original['model_spec_sha256']
    builder = Paired107StaticGeometry(bundle, original['asset_manifest_sha256'])
    bank = deepcopy(original)
    mapping, evidence = {}, []
    for name, height in (('stand_low', .23), ('stand', .305), ('stand_high', .42)):
        knee = brentq(lambda beta: builder.balanced_pose(beta)['height_m'] - height,
            40.001, 106.999, xtol=1e-8)
        pose = builder.balanced_pose(knee)
        validation = builder.validate_q(pose['q'])
        assert abs(pose['height_m'] - height) < 1e-8
        assert pose['minimum_ground_clearance_m'] >= -1e-8
        mapping[name] = len(bank['poses'])
        row = dict(name='diagnostic_height_matched_' + name, family='upright',
            roll_deg=float(np.rad2deg(pose['root_roll_rad'])), pitch_deg=0.,
            root_position_m=[0., 0., float(pose['height_m'])],
            root_quaternion_xyzw=Rotation.from_matrix(pose['rotation']).as_quat().tolist(),
            full_joint_positions={k: float(v) for k, v in pose['q'].items()},
            min_collision_clearance_m=float(pose['minimum_ground_clearance_m']),
            leg_shape='height_matched_balanced_closed_chain', requested_height_m=height,
            diagnostic_reset_only=True, **validation)
        bank['poses'].append(row)
        evidence.append(dict(name=name, pose_index=mapping[name], height_m=float(pose['height_m']),
            knee_deg=validation['knee_deg'], gap_m=validation['gap_m'],
            com_x_minus_wheel_support_m=float(pose['com'][0] - pose['wheels'][:, 0].mean()),
            min_collision_clearance_m=pose['minimum_ground_clearance_m']))
    assert bank['poses'][:len(original['poses'])] == original['poses']
    bank['diagnostic_height_starts'] = dict(source_bank=str(source.relative_to(ROOT)),
        source_sha256=sha(source), original_poses_and_indices_exact=True,
        mapping=mapping, original_acceptance_unchanged=True,
        scope='Additional diagnostic; no policy/PD/reward/termination change. All state writes at episode reset only.')
    target.write_text(json.dumps(bank, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    receipt = dict(schema='height-matched-closed-chain-diagnostic-reset-v1', bank=str(target.relative_to(ROOT)),
        bank_sha256=sha(target), source_bank_sha256=sha(source), old_pose_count=len(original['poses']),
        original_poses_and_indices_exact=True, appended=len(mapping), rows=evidence,
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(),
            ROOT / 'scripts/recovery_static_targets.py', ROOT / 'scripts/build_robust_getup_bank.py',
            ROOT / 'scripts/build_v6_recovery_profiles.py', builder.solver_path,
            bundle / 'tools/v6_mechanism.py', bundle / 'model_spec.json', bundle / 'manifest.json']},
        static_PD_compensation_used=False, new_PPO=0, qualification=None)
    (r / 'height_matched_starts073_geometry001.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
