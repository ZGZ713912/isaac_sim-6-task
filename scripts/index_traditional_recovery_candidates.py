#!/usr/bin/env python3
"""Index finite recovery examples without combining them into qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulate_traditional_self_righting import ROOT,digest,save
from summarize_traditional_recovery import initial_leg_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous candidate indices')
    selected=[('prone',legs,126,'prone_'+legs+'__prone_normalize109',126)
        for legs in ('ff','rr','fr','rf')]
    selected += [('inverted',legs,125,'inverted_'+legs+'__inverted_long_plant001',125)
        for legs in ('ff','fr','rf')]
    selected += [('inverted','rr',127,'inverted_rr__inverted_right_first_d-1_held-90',127),
        ('strict_left_measured','measured',128,'side_candidate_004__serial_clearance_upper_d1_k65',128)]
    rows=[];inputs={str(ROOT/'scripts'/name):digest(ROOT/'scripts'/name)for name in
        ('index_traditional_recovery_candidates.py','summarize_traditional_recovery.py',
         'simulate_traditional_self_righting.py')}
    for family,legs,native,name,review in selected:
        run=args.root/f'native{native:03}'
        receipt_path=run/'receipt.json';review_path=args.root/f'review{review:03}'/'review.json'
        receipt=json.loads(receipt_path.read_text());audit=json.loads(review_path.read_text())
        for path in (receipt_path,review_path):inputs[str(path)]=digest(path)
        if receipt['status']!='completed' or digest(run/'trace.npz')!=receipt['trace_sha256']:
            raise ValueError('Missing final unchanged native evidence: '+name)
        if audit['native_trace_sha256']!=receipt['trace_sha256'] or not audit.get('full_window'):
            raise ValueError('Need matching full-window CAD review: '+name)
        if not all((run/'sources'/Path(p).name).is_file()
            and digest(run/'sources'/Path(p).name)==sha for p,sha in receipt['sources'].items()):
            raise ValueError('Changed or missing native source snapshot: '+name)
        row=next(r for r in receipt['records']if r['name']==name)
        cad=next(r for r in audit['cad_screen']if r['case']==name)
        if not row['passed'] or cad['flagged']:
            raise ValueError('Selected example failed a finite observation or CAD check: '+name)
        live=next(r for r in receipt['controlled_initial_states']if r['name']==name)
        case=next(r for r in receipt['cases']if r['name']==name)
        spec=json.loads((ROOT/receipt['asset_directory']/'model_spec.json').read_text())
        rows.append(dict(family=family,placement_leg_combination=legs,native_run=run.name,case=name,
            first_completion_seconds=row['completion_seconds'],observation_window_passed=True,
            requested_control_seconds=receipt['requested_control_seconds'],
            passive_settle_seconds=receipt['requested_passive_settle_seconds'],
            controlled_initial_tilt_deg=live['tilt_deg'],**initial_leg_state(spec,live),
            native_leg_speed_peak_rad_s=max(row['leg_speed_peak_rad_s']),
            position_increment_rate_peak_rad_s=max(row['leg_position_increment_rate_peak_rad_s']),
            loop_gap_peak_m=row['loop_gap_peak_m'],
            maximum_single_body_ground_force_n=max(row['contact_force_peak_n']),
            maximum_single_body_ground_load_integral_10ms_ns=max(row['ground_load_impulse_peak_10ms_n_s']),
            sampled_cad_screen=cad,collision_geometry=receipt['collision_geometry'],
            physics_dt_s=receipt['physics_dt_s'],controller_dt_s=receipt['controller_dt_s'],
            model_spec_sha256=receipt['model_spec_sha256'],sources=receipt['sources'],
            case_configuration=case))
    negative=[]
    for number in (108,118,129,130):
        path=args.root/f'native{number:03}'/'receipt.json'
        receipt=json.loads(path.read_text());inputs[str(path)]=digest(path)
        if receipt['status']!='completed':raise ValueError('Wait for final negative/control evidence')
        negative.append(dict(run=path.parent.name,passive_settle_seconds=receipt['requested_passive_settle_seconds'],
            case_count=len(receipt['cases']),observation_end_pass_count=sum(r['passed']for r in receipt['records']),
            cases=[dict(name=r['name'],passed=r['passed'],fault=r['fault'],
                first_completion_seconds=r['completion_seconds'])for r in receipt['records']]))
    matrix=[dict(family=family,placement_leg_combination=legs,
        indexed_examples=[r['case']for r in rows if r['family']==family and r['placement_leg_combination']==legs],
        required_case_qualified=False)for family in ('prone','inverted','strict_left','strict_right')
        for legs in ('ff','rr','fr','rf')]
    save(args.output,dict(schema='traditional-recovery-finite-candidate-index/v1',
        source_sha256=digest(__file__),inputs=inputs,examples=rows,required_matrix=matrix,
        layout_and_settle_controls=negative,combined_controller_tested=False,
        required_case_qualification=False,hardware_deployment_ready=False,
        force_scope='Per-body resultant ground force and moving 10ms load integral include support load; '
            'no collision-event, material damage or hardware force qualification.',
        scope='Research examples with finite full-window behavior and sampled CAD checks. '
            'Different programs and environment layouts are not a unified controller. '
            'Strict measured endpoints do not supply missing front/rear combinations. '
            'The required matrix lists only this selection, not all historical trials.'))
    print(json.dumps(dict(examples=len(rows),required_case_qualification=False)))


if __name__=='__main__':main()
