"""Read-only manufacturer/prior audit with signed work on recorded 1ms samples.

No-load speeds are reference points, never state clamps or measured bounds on
externally backdriven motion. This script does not infer a torque-speed curve.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/fresh_unified_20261027/pilot017_dense_pose_physical_speed_mixed1536'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    output = ROOT/REL/(Path(__file__).stem+'.json')
    if output.exists():
        raise ValueError('Keep all previous reports')
    manufacturer = BASE/REL/'manufacturer_dm_j8009_2025'
    source = read(manufacturer/'source.json')
    assert sha(Path(source['path'])) == source['sha256']
    assert (manufacturer/'page17.png').is_file()
    control_path = ROOT/'contracts/flat_base_200hz_control_v1.json'
    control = read(control_path)
    config = read(RUN/'config/contract.json')
    startup = read(RUN/'startup_report.json')['actuator_response']
    assert control['actuators']['leg']['model'] == 'DM-J8009-2EC'
    assert startup['native_properties_verified']
    assert startup['config'] == config['actuator_response']
    leg_order = (0, 1, 3, 4)
    for key in ('armature', 'frictionloss', 'damping'):
        assert all(startup['native_first_environment'][key][i] == 0. for i in leg_order)
    assert all(startup['config']['time_constant_s'][i] == 0. for i in leg_order)
    previous = read(ROOT/REL/'fresh016_native_peak_comparison001.json')
    rows = []
    for profile in ('nominal', 'requested'):
        directory = BASE/REL/('fresh016_peak_native_'+profile+'001')
        path = directory/'native_peak_trace.npz'
        expected, = [item for item in previous['runs'] if item['profile'] == profile]
        assert sha(path) == expected['trace_sha256']
        before = sha(path)
        with np.load(path) as arrays:
            dq, torque = arrays['dq'], arrays['torque']
            assert dq.shape == torque.shape == (400, 5, 198, 4)
            assert np.isfinite(dq).all() and np.isfinite(torque).all()
            phases = dict(passive_placement=arrays['passive'],
                active_recovery=~arrays['passive'] & arrays['recovery'],
                ordinary_active_control=~arrays['passive'] & ~arrays['recovery'])
            phase_sum = sum(mask.astype(np.int8) for mask in phases.values())
            assert np.all(phase_sum == 1)
            power = dq.astype(np.float64)*torque.astype(np.float64)
            phase_rows = []
            for name, mask in phases.items():
                valid = np.broadcast_to((mask & arrays['valid'])[:, None, :, None], dq.shape)
                for reference, rpm in (('24V_published_no_load', 168.),
                        ('48V_published_no_load', 335.), ('existing_50rad_s_diagnostic', 50.*30./math.pi)):
                    threshold = rpm*math.pi/30.
                    high = valid & (np.abs(dq) >= threshold)
                    counts = dict(positive_motor_work_samples=int(np.count_nonzero(high & (power > 1e-8))),
                        braking_motor_work_samples=int(np.count_nonzero(high & (power < -1e-8))),
                        zero_motor_work_samples=int(np.count_nonzero(high & (np.abs(power) <= 1e-8))))
                    assert sum(counts.values()) == int(high.sum())
                    phase_rows.append(dict(phase=name, reference=reference,
                        reference_speed_rad_s=threshold, motor_samples=int(high.sum()), **counts))
        assert sha(path) == before
        rows.append(dict(profile=profile, trace_sha256=before, counts=phase_rows))
    result = dict(schema='leg-motor-model-gap-read-only-audit-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),
        producer_sha256=sha(Path(__file__)), manufacturer=dict(**source,
            rendered_page_sha256=sha(manufacturer/'page17.png'),
            complete_relevant_page_visually_inspected=True,
            rated_torque_nm=20., peak_torque_nm=40., rated_speed_rpm_at24V=100.,
            rated_speed_rpm_at48V=200., no_load_speed_rpm_at24V=168.,
            no_load_speed_rpm_at48V=335., supply_range_v=[24.,48.],
            performance_plots='Both plots are labelled100RPM load tests, not a full voltage-speed-torque boundary',
            real_supply_voltage_known=False, rotor_inertia_published_on_this_page=False,
            full_torque_speed_envelope_identified=False),
        current_native_leg_prior=dict(control_contract_sha256=sha(control_path),
            run_config_sha256=sha(RUN/'config/contract.json'),
            startup_report_sha256=sha(RUN/'startup_report.json'),
            actuator_response=config['actuator_response'],
            native_first_environment=startup['native_first_environment'],
            leg_additional_armature_zero=True, leg_native_friction_and_damping_zero=True,
            total_link_inertia_zero=False, peak_torque_cap_nm=40.,
            identified_torque_speed_envelope=False, hardware_identified=False),
        signed_work_by_1ms_motor_sample=rows,
        scope='Same exact016 stopped policy, seed271301, first2s of198 cases per profile. Signed motor work is simultaneous tau*dq, not net mechanical energy or an attribution of017 training extrema. Published no-load speed is not an admissibility test for passive/backdriven motion. Positive work above that reference exposes the constant-cap modelling assumption but does not identify the missing real motor envelope.',
        new_native_steps=0, new_PPO=0, model_parameters_changed=False,
        curve_adopted=False, policy_qualified=False, hardware_ready=False)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    for base in (ROOT, BASE):
        (base/REL/output.name).write_text(text)
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(manufacturer_sha256=source['sha256'], counts=rows)))


if __name__ == '__main__':
    main()
