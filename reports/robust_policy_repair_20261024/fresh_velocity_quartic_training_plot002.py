"""Export the first128 original RSL updates of017/018 without a TB server."""
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')


def main():
    bindings = [('017: quadratic leg speed', 'fresh_unified017_live192_007.json', '#777777'),
                ('018: body velocity + quartic cost', 'fresh_unified018_live128_005.json', '#0072B2')]
    metrics = [('DomainAdmission/nominal/tilt_mean_deg', 'Nominal tilt (deg)', 1., False),
               ('DomainAdmission/nominal/planar_speed_rms_m_s', 'Nominal planar speed RMS (m/s)', 1., False),
               ('DomainAdmission/nominal/episode_end_events', 'Episode ends / nominal car / s', 200. / 68, False),
               ('ScratchStandingExtrema/peak_leg_speed_rad_s', 'True 1 ms leg peak, all 256 cars (rad/s)', 1., True)]
    output = ROOT / REL / 'fresh_velocity_quartic_first128_002'
    if any(output.with_suffix(s).exists() for s in ('.json', '.png', '.svg')):
        raise ValueError('Preserve the original plot')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    sources = []
    for label, name, color in bindings:
        prefix = json.loads((ROOT / REL / name).read_text())['event_prefix']
        with open(prefix['path'], 'rb') as stream:
            raw = stream.read(prefix['bytes'])
        assert hashlib.sha256(raw).hexdigest() == prefix['sha256']
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / Path(prefix['path']).name
            snapshot.write_bytes(raw)
            events = EventAccumulator(str(snapshot), size_guidance={'scalars': 0}).Reload()
            for ax, (tag, title, scale, peak) in zip(axes.flat, metrics, strict=True):
                data = {point.step: point.value for point in events.Scalars(tag)}
                assert all(i in data for i in range(128))
                windows = [[data[j] for j in range(i, i + 16)] for i in range(0, 128, 16)]
                y = [(max(row) if peak else sum(row) / len(row)) * scale for row in windows]
                ax.plot([8.5 + 16 * i for i in range(8)], y, 'o-', label=label,
                        color=color, linewidth=1.8, markersize=4)
                ax.set_title(title, fontsize=10)
                ax.grid(alpha=.2)
                ax.set_xlabel('Completed fresh PPO updates')
        sources.append(dict(label=label, source_report=name, event_prefix=prefix))
    fig.suptitle('Fresh017 vs 018: first 128 updates (16-update windows)', fontsize=14, y=.98)
    fig.text(.5, .925, 'Same initialization, seed, 17 domain arrays and 3-height curriculum; nominal-only PPO in both.',
             ha='center', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .055), ncol=2, frameon=False)
    fig.text(.5, .025, 'Failure resets remain in training means. These curves do not establish continuous standing or robustness.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(top=.85, bottom=.19, hspace=.42, wspace=.3)
    for suffix in ('.png', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    receipt = dict(schema='fresh-objective-training-plot-v1', sources=sources, updates=[1, 128], window=16,
        data_source='Original RSL TensorBoard event byte prefixes, SHA verified',
        nominal_car_count=68, policy_hz=200, full_physical_car_count=256, behavior_qualified=False,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={s: hashlib.sha256(output.with_suffix(s).read_bytes()).hexdigest() for s in ('.png', '.svg')})
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for suffix in ('.png', '.svg', '.json'):
        (BASE / REL / output.with_suffix(suffix).name).write_bytes(output.with_suffix(suffix).read_bytes())
    (BASE / REL / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
