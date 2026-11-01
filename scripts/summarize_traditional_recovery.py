#!/usr/bin/env python3
"""Inventory native recovery evidence without upgrading partial successes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from simulate_traditional_self_righting import ROOT, digest, save


def initial_leg_state(spec, state):
    if state is None:
        return {}
    from wheeled_tasks.chassis.side_fall_study import orbital_angles
    q=state['full_joint_positions'];dq=state['full_joint_velocities']
    zero=spec['passive_knee_zero_inner_rad']
    return dict(controlled_initial_leg_orbit_deg=orbital_angles(spec,q),
        controlled_initial_knee_deg=[math.degrees(q['L_joint2']+zero['L_joint2']),
            math.degrees(zero['R_jonit2']-q['R_jonit2'])],
        controlled_initial_leg_speed_peak_rad_s=max(abs(dq[n])for n in
            ('L_joint1','LL_joint1','R_joint1','RR_joint1')),
        controlled_initial_gimbal_yaw_pitch_deg=[math.degrees(q[n])for n in ('M_joint1','M_joint2')],
        controlled_initial_com_speed_m_s=math.hypot(*state['root_com_velocity_world'][:3]),
        controlled_initial_gyro_norm_rad_s=math.hypot(*state['gyro_body_rad_s']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing inventories; select a new output')
    runs = []
    for path in sorted(args.root.glob('native*/receipt.json')):
        receipt = json.loads(path.read_text())
        closure = path.parent/'closure.json'
        status = receipt['status']
        if closure.exists():
            status = 'stopped_invalid_fixture'
        elif status == 'running':
            status = 'incomplete_receipt_not_success'
        records = []
        controlled={row['name']:row for row in receipt.get('controlled_initial_states',[])}
        spec=json.loads((ROOT/receipt['asset_directory']/'model_spec.json').read_text())if controlled else None
        cases = {row['name']: row for row in receipt['cases']}
        for row in receipt['records']:
            case = cases[row['name']]
            x, y, z, w = case['quaternion_xyzw']
            gravity_y = -2.*(y*z+x*w)
            tilt = math.degrees(math.acos(max(-1., min(1., 1.-2.*(x*x+y*y)))))
            live=controlled.get(row['name'])
            strict_tilt=live['tilt_deg']if live else tilt
            strict_gy=live['gravity_body'][1]if live else gravity_y
            records.append(dict(name=row['name'], source_family=case['family'],
                initial_leg_combination=case.get('legs'),
                initial_tilt_deg=tilt,controlled_initial_tilt_deg=live['tilt_deg']if live else None,
                post_settle_initial_state_recorded=live is not None,
                strict_side_start=abs(strict_gy)>.98 and 80.<strict_tilt<100.,
                first_completion_s=row['completion_seconds'],
                sequence_completed=row.get('recovery_sequence_completed'),
                observation_end_passed=row['passed'], fault=row['fault'],
                final_stable_s=row['final_stable_seconds'],
                actual_speed_peak_rad_s=max(row['leg_speed_peak_rad_s']),
                position_increment_rate_peak_rad_s=max(row['leg_position_increment_rate_peak_rad_s'])
                    if row.get('leg_position_increment_rate_peak_rad_s') else None,
                loop_gap_peak_m=row['loop_gap_peak_m'],stalled_flip_waypoints=row.get('stalled_flip_waypoints'),
                maximum_single_body_ground_force_n=max(row['contact_force_peak_n'])
                    if row.get('contact_force_peak_n')else None,
                maximum_single_body_ground_load_integral_10ms_ns=max(row['ground_load_impulse_peak_10ms_n_s'])
                    if row.get('ground_load_impulse_peak_10ms_n_s')else None,
                **initial_leg_state(spec,live)))
        trace = path.parent/'trace.npz'
        snapshot_valid = all((path.parent/'sources'/Path(name).name).is_file()
            and digest(path.parent/'sources'/Path(name).name)==sha for name,sha in receipt['sources'].items())
        corrected_root_filter=bool(receipt.get('self_collision_enabled') and
            receipt.get('root_pair_filter_policy','').startswith('No articulation-root pair targets'))
        contact_scope=('corrected root-filter policy recorded; trajectory and CAD admission remain separate'
            if corrected_root_filter else 'unqualified root-level pair filtering or disabled self-contact')
        guard_checks=[]
        check_paths=set(path.parent.glob('gimbal_guard_coverage*.json'))|set(path.parent.glob('cooked_guard_coverage*.json'))
        for check_path in sorted(check_paths):
            check=json.loads(check_path.read_text())
            inputs=check.get('inputs')
            guard_checks.append(dict(path=str(check_path),sha256=digest(check_path),
                all_vertices_contained=check['all_vertices_contained'],
                input_hashes_match=all((ROOT/Path(p)).exists()and digest(ROOT/Path(p))==expected
                    for p,expected in inputs.items())if inputs else None,
                input_hashes_recorded=bool(inputs),
                maximum_outward_distance_from_original_cad_convex_hull_m=max(
                    (r.get('maximum_outward_distance_from_original_cad_convex_hull_m',0.)for r in check['records']),default=None)))
        runs.append(dict(run=path.parent.name, status=status, receipt_sha256=digest(path),
            closure=str(closure)if closure.exists()else None, error=receipt.get('error'),
            case_count=len(receipt['cases']), completed_records=len(records),
            first_completion_count=sum(r['first_completion_s']is not None for r in records),
            observation_end_pass_count=sum(r['observation_end_passed']for r in records),
            self_collision=receipt.get('self_collision_enabled'),
            contact_fixture_scope=contact_scope,
            root_pair_filter_policy=receipt.get('root_pair_filter_policy'),
            effective_pair_filter_relations_recorded='effective_pair_filter_relations' in receipt,
            collision_geometry=receipt.get('collision_geometry','model-spec'),
            geometry_fidelity_scope=('Coarse 30-plane guards can protrude 28..35mm; debug only'
                if receipt.get('collision_geometry')=='cad-gimbal-guards'else
                'Tight guards bound protrusion from the CAD convex hull, not filling of CAD concavities; trajectory audit remains required'
                if receipt.get('collision_geometry')=='cad-gimbal-tight-guards'else
                'Component guards require corresponding-piece cooked coverage; no bound on within-component concavity filling'
                if receipt.get('collision_geometry')=='cad-gimbal-component-guards'else 'Per-run CAD/native geometry review required'),
            base_ground_vertex_guards=receipt.get('base_ground_vertex_guards',False),
            gimbal_guard_coverage_checks=guard_checks,
            model_spec_sha256=receipt['model_spec_sha256'],sources=receipt['sources'],
            physics_dt_s=receipt.get('physics_dt_s'),controller_dt_s=receipt.get('controller_dt_s'),
            trace_hash_valid=bool(trace.exists()and receipt.get('trace_sha256')==digest(trace)),
            source_snapshot_hashes_valid=snapshot_valid,
            records=records))
    reviews = []
    for path in sorted(args.root.glob('review*/review.json')):
        review = json.loads(path.read_text())
        reviews.append(dict(path=str(path),sha256=digest(path),cases=review['cases'],
            cad_screen=review.get('cad_screen'),scope=review.get('cad_scope')))
    result = dict(schema='traditional-recovery-evidence-inventory/v1',source_sha256=digest(__file__),
        required_case_qualification=False,hardware_deployment_ready=False,ppo_updates=0,hardware_motion=False,
        scope='Observed finite recovery windows and engineering checks; no complete traditional self-righting or hardware qualification',
        speed_requirement_rad_s=3.14,
        force_scope='Maximum of per-body resultant ground force and moving 10ms load integral. '
            'Includes support load; individual maxima need not occur simultaneously. '
            'No collision-event, deformation, material damage or hardware force qualification.',
        blockers=[
            'Corrected self-contact trials have finite successes, but strict left/right side recovery is not robustly covered.',
            'Random passive gimbal configurations and independent front/rear leg combinations are not fully covered.',
            'Finite prone, inverted and side candidates pass sampled CAD screens, but required poses remain uncovered and some earlier apparent successes have CAD intersections.',
            'Earlier root-level collision masks invalidated apparent successes; corrected SDF and convex fixtures still disagree.',
            'Some CAD-grounded poses are not proven passive equilibria; measured endpoints may retain residual motion.',
            'PhysX speed governor is not proof of hardware or controller-only actual speed enforcement.',
            'Sustained standing and complete v202 traditional-control handoff remain unverified.'],
        fixture_history=dict(native001_to_native014='Missing passive numerical damping/wheel hull correction; debug only',
            native015_and_native018='Implicit actuator configured without an authored DriveAPI; invalid motor implementation',
            native016='Trace exists but NumPy-bool receipt serialization failed; never a completed receipt',
            native019_onwards='Explicit angular DriveAPI in implicit mode; later source/settings remain per run',
            native030_to_native032='Nonphysical self-contact ejection; explicitly stopped, closure records retained',
            native039_to_native069='Research filters on articulation-root base_link unintentionally suppress unrelated link contact; debug only',
            native070='Corrected filters on deliberately intersecting positive-control start; not a recovery trial',
            native071_onwards='No articulation-root pair targets; optional base/gas filters at collider level; source and contact representation remain per run'),
        contact_controls=[dict(path=str(p),sha256=digest(p),
            all_controls_passed=json.loads(p.read_text()).get('all_controls_passed'))
            for p in sorted(args.root.glob('contact_control*/receipt.json'))],
        runs=runs,reviews=reviews)
    save(args.output,result)
    print(json.dumps(dict(runs=len(runs),reviews=len(reviews),required_case_qualification=False)))


if __name__=='__main__':
    main()
