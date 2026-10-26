"""Locate every lost-load millisecond without changing the standing predicate."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

R = Path('reports/robust_policy_repair_20261024')
analyses = ['contact058_strict_grounded002.json', 'contact065_strict_trained019.json',
            'contact068_strict_trained020.json']
windows = [('first20ms', 0, 20), ('20to200ms', 20, 200), ('200ms_to2s', 200, 2000),
           ('2to4s', 2000, 4000), ('4to8s', 4000, 8000)]
runs = []
for name in analyses:
    analysis = json.loads((R / name).read_text())
    folder = Path(analysis['run']) / 'evaluation'
    meta = json.loads((folder / 'native_wheel_contact_1ms.json').read_text())
    source = folder / 'native_wheel_contact_1ms.npz'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['sha256']
    data = np.load(source)
    present = (np.linalg.norm(data['force'], axis=-1) > 2.).all(-1)
    gap = (data['clearance'] > .0001).any(-1)
    present, gap = [x.reshape(-1, x.shape[-1]) for x in (present, gap)]
    assert present.shape[0] == 8000
    rows = []
    writer = SummaryWriter(str(folder / 'metrics'), filename_suffix='.support-timing-v1')
    for i, case in enumerate(analysis['rows']):
        missing = ~present[:, i]
        bins = {key: int(missing[lo:hi].sum()) for key, lo, hi in windows}
        row = dict(name=case['name'], missing_support_ms=int(missing.sum()), windows_ms=bins,
            missing_load_and_resolved_gap_ms=int((missing & gap[:, i]).sum()),
            missing_load_without_resolved_gap_ms=int((missing & ~gap[:, i]).sum()),
            resolved_gap_total_ms=int(gap[:, i].sum()))
        assert sum(bins.values()) == row['missing_support_ms']
        for key, value in {**bins, **{k: v for k, v in row.items() if isinstance(v, int)}}.items():
            writer.add_scalar('StrictStandingTiming/' + case['name'] + '/' + key, value, analysis['learning_clock'])
        rows.append(row)
    writer.close()
    totals = {key: sum(row['windows_ms'][key] for row in rows) for key, _, _ in windows}
    total = sum(row['missing_support_ms'] for row in rows)
    run = dict(analysis=name, run=analysis['run'], source_sha256=meta['sha256'], rows=rows,
        total_missing_support_ms=total, windows_ms=totals,
        fraction_lost_load_after2s=sum(totals[key] for key in ('2to4s', '4to8s')) / max(1, total),
        missing_load_and_resolved_gap_ms=sum(row['missing_load_and_resolved_gap_ms'] for row in rows))
    runs.append(run)
    print(json.dumps({k: v for k, v in run.items() if k not in ('rows', 'source_sha256')}))
out = dict(schema='native-support-loss-timing-v1', source='Authenticated native1ms force and analytic wheel geometry',
    force_threshold_n=2., geometry_reporting_resolution_m=.0001, no_grace=True,
    runs=runs, interpretation='A lack of a resolved gap never excuses missing load; continuous support still fails.',
    new_PPO=0, qualification=None)
(R / 'support_loss_timing021.json').write_text(json.dumps(out, indent=2) + '\n')
