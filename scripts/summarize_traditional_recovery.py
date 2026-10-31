#!/usr/bin/env python3
"""Inventory native recovery evidence without upgrading partial successes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from simulate_traditional_self_righting import digest, save


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
        cases = {row['name']: row for row in receipt['cases']}
        for row in receipt['records']:
            case = cases[row['name']]
            x, y, z, w = case['quaternion_xyzw']
            gravity_y = -2.*(y*z+x*w)
            tilt = math.degrees(math.acos(max(-1., min(1., 1.-2.*(x*x+y*y)))))
            records.append(dict(name=row['name'], source_family=case['family'],
                initial_tilt_deg=tilt, strict_side_start=abs(gravity_y)>.98 and 80.<tilt<100.,
                first_completion_s=row['completion_seconds'],
                sequence_completed=row.get('recovery_sequence_completed'),
                observation_end_passed=row['passed'], fault=row['fault'],
                final_stable_s=row['final_stable_seconds'],
                actual_speed_peak_rad_s=max(row['leg_speed_peak_rad_s']),
                loop_gap_peak_m=row['loop_gap_peak_m']))
        trace = path.parent/'trace.npz'
        snapshot_valid = all((path.parent/'sources'/Path(name).name).is_file()
            and digest(path.parent/'sources'/Path(name).name)==sha for name,sha in receipt['sources'].items())
        runs.append(dict(run=path.parent.name, status=status, receipt_sha256=digest(path),
            closure=str(closure)if closure.exists()else None, error=receipt.get('error'),
            case_count=len(receipt['cases']), completed_records=len(records),
            first_completion_count=sum(r['first_completion_s']is not None for r in records),
            observation_end_pass_count=sum(r['observation_end_passed']for r in records),
            self_collision=receipt.get('self_collision_enabled'),
            collision_geometry=receipt.get('collision_geometry','model-spec'),
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
        blockers=[
            'Strict side-lying recovery has not succeeded in the tested bounded leg programs.',
            'Random passive gimbal configurations and independent front/rear leg combinations are not fully covered.',
            'CAD surface intersections invalidate some apparent recovery successes.',
            'Native self-contact collision envelopes are not validated; replacement probes remain unqualified.',
            'PhysX speed governor is not proof of hardware or controller-only actual speed enforcement.',
            'Sustained standing and complete v202 traditional-control handoff remain unverified.'],
        fixture_history=dict(native001_to_native014='Missing passive numerical damping/wheel hull correction; debug only',
            native015_and_native018='Implicit actuator configured without an authored DriveAPI; invalid motor implementation',
            native016='Trace exists but NumPy-bool receipt serialization failed; never a completed receipt',
            native019_onwards='Explicit angular DriveAPI in implicit mode; later source/settings remain per run',
            native030_to_native032='Nonphysical self-contact ejection; explicitly stopped, closure records retained'),
        runs=runs,reviews=reviews)
    save(args.output,result)
    print(json.dumps(dict(runs=len(runs),reviews=len(reviews),required_case_qualification=False)))


if __name__=='__main__':
    main()
