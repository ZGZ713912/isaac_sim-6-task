"""Export the fixed physical comparison from existing evaluation TensorBoard events."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

R = Path('reports/robust_policy_repair_20261024')
inputs = [('contact058_strict_grounded002.json', 'Initial c025'),
          ('contact065_strict_trained019.json', '019 dense load/gap'),
          ('contact068_strict_trained020.json', '020 + support breaks')]
fig, axes = plt.subplots(3, 1, figsize=(13, 9), layout='constrained', sharex=True)
receipts = []
for j, (name, label) in enumerate(inputs):
    analysis = json.loads((R / name).read_text())
    folder = Path(analysis['run']) / 'evaluation/metrics'
    files = sorted(folder.glob('events.out.tfevents.*'))
    acc = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
    case_names = [r['name'] for r in analysis['rows']]
    values = [[], [], []]
    exported = {}
    for case in case_names:
        for i, suffix in enumerate(('whole/dual_support_fraction', 'tail_leg_motion_3_95hz_rms_rad',
                                    'tail4s/maximum_clearance_m')):
            tag = 'StrictStandingEvaluation/' + case + '/' + suffix
            samples = [x for x in acc.Scalars(tag) if x.step == analysis['learning_clock']]
            assert samples, tag
            v = samples[-1].value
            exported[tag] = dict(step=samples[-1].step, value=v)
            values[i].append((1. - v) * analysis['seconds'] * 1000. if i == 0 else v * 1000.)
    x = np.arange(len(case_names)) + (j - 1) * .25
    for axis, points in zip(axes, values, strict=True):
        axis.bar(x, points, width=.24, label=label)
    receipts.append(dict(analysis=name, run=analysis['run'], records=exported,
        events={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
for axis, title in zip(axes, ('Total missing dual-wheel support (ms / 8 s)',
                              'Late leg motion, 3-95 Hz RMS (mrad)',
                              'Late maximum wheel clearance (mm)'), strict=True):
    axis.set_ylabel(title)
    axis.grid(axis='y', alpha=.2)
axes[0].legend(ncol=3)
axes[2].axhline(.1, color='red', linestyle='--', linewidth=1, label='0.1 mm reporting resolution')
axes[2].legend()
short = ['clean', 'pitch+', 'pitch-', 'roll+', 'roll-', 'wobble', 'combined']
axes[2].set_xticks(range(21), ['low/' + x for x in short] + ['mid/' + x for x in short]
                    + ['high/' + x for x in short], rotation=65, ha='right')
fig.suptitle('Same 21 standing cases / seed / full DR / native 1 ms. Continuous support requires zero missing ms.')
target = R / 'standing019_020_physical_comparison.png'
fig.savefig(target, dpi=160)
plt.close(fig)
receipt = dict(schema='native-evaluation-TensorBoard-export-v1', sources=receipts, figure=str(target),
    figure_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    exact_acceptance_source='Native1ms NPZ and strict analysis JSON; plots use float32 event scalars.',
    qualification=None, hardware_started=False)
(R / 'standing019_020_physical_comparison.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(target)
