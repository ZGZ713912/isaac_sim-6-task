"""Export only authenticated RSL event values, checking diagnostic receipts."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).parent
OUTPUT = HERE / 'scratch_feedback_gain007_plot001.png'
RECEIPT = OUTPUT.with_suffix('.receipt.json')
assert not OUTPUT.exists() and not RECEIPT.exists()
native = json.loads((HERE / 'scratch_trajectory_feedback007_read001.json').read_text())
timeline = json.loads((HERE / 'scratch_feedback_gain007_timeline001.json').read_text())
sources = {}


def read(directory, hashes):
    event, = Path(directory).glob('events.out.tfevents.*')
    digest = hashlib.sha256(event.read_bytes()).hexdigest()
    assert digest in hashes.values() and len(hashes) == 1
    sources[str(event.resolve())] = digest
    accumulator = EventAccumulator(str(event), size_guidance={'scalars': 0}).Reload()
    values = {}
    for tag in accumulator.Tags()['scalars']:
        rows = accumulator.Scalars(tag)
        values[tag] = {row.step: row.value for row in rows}
        assert len(rows) == len(values[tag])
    return values


actual = read(native['tensorboard']['directory'], native['tensorboard']['events_sha256'])
history = {name: read(item['directory'], item['files_sha256']) for name, item in timeline['tensorboard'].items()}
keys = ['leg_command_gain_max_nm_per_rad_s', 'wheel_command_gain_max_nm_per_rad_s']
checks = 0
for row in native['rows']:
    prefix = f"DiagnosticScratchLocalFeedback200Hz/scale1.00/pitch{row['pitch_deg']:+.2f}/"
    step = round(row['decision_time_s'] / .005)
    expected = [max(row['leg_command_gain_symmetric_eigenvalues']), max(row['wheel_command_gain_symmetric_eigenvalues'])]
    for key, value in zip(keys, expected):
        assert abs(actual[prefix + key][step] - value) < 2e-7
        checks += 1
for row in timeline['rows']:
    for key in keys:
        assert abs(history[row['run_name']]['DiagnosticScratchNominalFeedback200Hz/' + key][row['completed_updates']] - row[key]) < 2e-7
        checks += 1

figure, axes = plt.subplots(2, 2, figsize=(12, 7.2))
colors = ['#2674b5', '#dc793a', '#43996b', '#965da6']
for axis_index, key in enumerate(keys):
    left, right = axes[axis_index]
    for color, pitch in zip(colors, native['requested_pitch_deg']):
        prefix = f'DiagnosticScratchLocalFeedback200Hz/scale1.00/pitch{pitch:+.2f}/'
        series = actual[prefix + key]
        left.plot([step * .005 for step in sorted(series)], [series[step] for step in sorted(series)],
                  marker='o', markersize=3, color=color, label=f'{pitch:+.2f} deg')
    for name, color, label in [('pilot007_contact_cost_zero', '#2674b5', '007'),
                               ('pilot008_leg_target_rate020', '#dc793a', '008 (97 only)')]:
        series = history[name]['DiagnosticScratchNominalFeedback200Hz/' + key]
        right.plot(sorted(series), [series[step] for step in sorted(series)],
                   marker='o', markersize=5, color=color, label=label)
    for axis in (left, right):
        axis.axhline(0., color='#555555', linestyle='--', linewidth=1.)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    left.set_ylabel(('Legs' if axis_index == 0 else 'Wheels') + ': max symmetric gain\n[N m / (rad/s)]')
    left.set_xlabel('Original decision time [s]')
    right.set_xlabel('Completed fresh PPO updates')
axes[0, 0].set_title('007 at recorded standing states (frozen 512-update actor)')
axes[0, 1].set_title('Same ideal middle-height input at saved checkpoints')
axes[0, 0].legend(ncol=2, fontsize=9)
axes[0, 1].legend(fontsize=9)
figure.suptitle('Encoder feedback can counteract the nominal PD damping', y=.985, fontsize=14)
figure.text(.5, .035, 'Positive gain: a velocity perturbation direction adds aligned command torque. Other observations stay fixed.',
            ha='center', fontsize=9)
figure.text(.5, .013, 'CPU derivatives from authenticated RSL events; these are not closed-loop poles or proof of the first contact-loss cause.',
            ha='center', fontsize=9)
figure.tight_layout(rect=(0., .055, 1., .955))
figure.savefig(OUTPUT, dpi=160)
plt.close(figure)
RECEIPT.write_text(json.dumps(dict(
    schema='scratch-feedback-event-plot-v1', authenticated_event_files_sha256=sources,
    every_plotted_value_checked_against_diagnostic_receipts=True, checked_values=checks,
    plot=str(OUTPUT.resolve()), plot_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_PPO=0, native_steps=0, behavior_qualification=False, hardware_ready=False,
), indent=2) + '\n')
print(json.dumps(dict(plot=str(OUTPUT.resolve()), checked_values=checks)))
