"""Read-only FK and exact mass-corner audit of the live018 standing births."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE / 'runs/fresh_unified_20261027/pilot018_velocity_quartic_physical_speed_mixed1536'
sys.path.insert(0, str(ROOT / 'scripts'))
from recovery_static_targets import RecoveryStaticTargetBuilder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fractional_bound(mass, low, high, coordinates, maximize):
    """Independent check: the root of sum(m*(x-r)) selects every mass corner."""
    left, right = float(coordinates.min()), float(coordinates.max())
    for _ in range(100):
        center = (left + right) / 2
        above = coordinates >= center
        factors = np.where(above if maximize else ~above, high, low)
        residual = float(np.sum(mass * factors * (coordinates - center)))
        if residual > 0:
            left = center
        else:
            right = center
    return (left + right) / 2


def main():
    target = REL / 'fresh018_reset_support_audit001.json'
    if (ROOT / target).exists() or (BASE / target).exists():
        raise ValueError('Preserve the original geometry audit')
    contract_path, bank_path = RUN / 'config/contract.json', RUN / 'pose_bank.json'
    contract, bank = (json.loads(p.read_text()) for p in (contract_path, bank_path))
    bundle = ROOT / contract['asset_directory']
    assert bank['asset_manifest_sha256'] == contract['asset_manifest_sha256']
    builder = RecoveryStaticTargetBuilder(bundle, contract['asset_manifest_sha256'])
    bodies = builder.spec['bodies']
    names = [body['name'] for body in bodies]
    mass = np.array([body['mass'] for body in bodies])
    cfg = contract['dynamics_randomization']
    ranges = [cfg['base_mass_scale'] if name == cfg['base_body'] else
              cfg['wheel_mass_scale'] if name in cfg['wheel_bodies'] else
              cfg['leg_mass_scale'] for name in names]
    low, high = np.array(ranges).T
    base_id = names.index(cfg['base_body'])
    vertices = np.arange(1 << len(bodies), dtype=np.uint32)
    denominator = np.zeros(len(vertices))
    base_vertex_mass = None
    for index in range(len(bodies)):
        factors = low[index] + (high[index] - low[index]) * ((vertices >> index) & 1)
        value = mass[index] * factors
        denominator += value
        if index == base_id:
            base_vertex_mass = value
    rows = []
    for key in ('stand_low', 'stand', 'stand_high'):
        index = bank['diagnostic_height_starts']['mapping'][key]
        pose = bank['poses'][index]
        q = pose['full_joint_positions']
        rotation = Rotation.from_quat(pose['root_quaternion_xyzw']).as_matrix()
        frames = builder.fk(builder.spec, q)
        positions = np.array([rotation @ (frames[b['name']][:3, :3] @ b['com'] +
                              frames[b['name']][:3, 3]) for b in bodies])
        center, wheels = builder.mass_and_wheel_centers(q, rotation)
        assert np.max(np.abs(center - (mass[:, None] * positions).sum(0) / mass.sum())) < 1e-12
        support_x = float(wheels[:, 0].mean())
        nominal_error = float(center[0] - support_x)
        assert abs(nominal_error) < 1e-8
        numerator = np.zeros(len(vertices))
        for body_index in range(len(bodies)):
            factors = low[body_index] + (high[body_index] - low[body_index]) * ((vertices >> body_index) & 1)
            numerator += mass[body_index] * factors * (positions[body_index, 0] - support_x)
        projected_axes = (rotation @ frames[cfg['base_body']][:3, :3])[0]
        offset_products = projected_axes[:, None] * np.array([cfg['base_com_offset_m'][a] for a in 'xyz'])
        offset_low = float(offset_products.min(1).sum())
        offset_high = float(offset_products.max(1).sum())
        bounds = [float(np.min((numerator + base_vertex_mass * offset_low) / denominator)),
                  float(np.max((numerator + base_vertex_mass * offset_high) / denominator))]
        checked = []
        for offset, maximize in ((offset_low, False), (offset_high, True)):
            coordinates = positions[:, 0].copy() - support_x
            coordinates[base_id] += offset
            checked.append(fractional_bound(mass, low, high, coordinates, maximize))
        assert np.max(np.abs(np.array(bounds) - checked)) < 1e-12
        closure = float(np.linalg.norm(builder.closure_error(builder.spec, q).reshape(-1, 3), axis=1).max())
        assert closure < 1e-8
        rows.append(dict(pose_index=index, standing_case=key,
            requested_height_m=pose['root_position_m'][2], total_nominal_mass_kg=float(mass.sum()),
            nominal_COM_from_root_m=center.tolist(), wheel_centers_from_root_m=wheels.tolist(),
            nominal_COM_x_minus_mean_wheel_x_m=nominal_error,
            closure_gap_m=closure, base_fraction_of_total_mass=float(mass[base_id] / mass.sum()),
            all_independent_mass_corners=1 << len(bodies),
            requested_mass_and_base_COM_x_error_bounds_m=bounds,
            independent_fractional_check_bounds_m=checked))
    result = dict(schema='fresh018-reset-static-support-audit-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),
        asset_manifest_sha256=contract['asset_manifest_sha256'], pose_bank_sha256=sha(bank_path),
        live_contract_sha256=sha(contract_path), actual_dynamics_ranges=cfg, rows=rows,
        nominal_geometries_COM_balanced=True, gross_nominal_reset_COM_mismatch_found=False,
        interpretation='Nominal reset geometry is balanced. Domain endpoints quantify static COM shifts only; no causal attribution of training degradation, controller feasibility, dynamic stability or hardware envelope.',
        tensor_inertia_affects_this_static_COM_calculation=False,
        learned_policy_read=False, new_PPO=0, new_native_steps=0,
        training_changed=False, randomization_changed=False, hardware_ready=False,
        producer_sha256=sha(Path(__file__)),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
            ROOT/'scripts/recovery_static_targets.py', ROOT/'scripts/build_v6_recovery_profiles.py',
            ROOT/'scripts/build_v6_height_lookup.py', bundle/'model_spec.json', bundle/'manifest.json',
            bundle/'tools/v6_mechanism.py', builder.solver_path)})
    for root in (ROOT, BASE):
        (root / target).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (BASE / REL / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(nominal_balanced=True, rows=rows)))


if __name__ == '__main__':
    main()
