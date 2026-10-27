"""Compare the authenticated original017/018 RSL prefixes through update320.

No simulator, model, optimizer or TensorBoard server is constructed. Standing
tags preserve their actual changing population, including unadmitted lanes.
"""
import hashlib
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
STANDING = 'FreshTasks/standing/'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    output = ROOT / REL / 'fresh018_admission320_read001'
    if any(output.with_suffix(s).exists() for s in ('.json', '.png', '.svg')):
        raise ValueError('Keep every original report')
    settings = [('017', 'fresh_unified017_live320_011.json', '#777777'),
                ('018', 'fresh_unified018_live320_011.json', '#0072B2')]
    metrics = [(STANDING + 'tilt_mean_deg', 'Actual standing roles: tilt (deg)', False),
               (STANDING + 'planar_speed_rms_m_s', 'Actual standing roles: speed RMS (m/s)', False),
               (STANDING + 'episode_end_events', 'Standing ends / policy step (200 Hz)', False),
               ('ScratchStandingExtrema/peak_leg_speed_rad_s', 'All roles/cars: true 1 ms leg peak (rad/s)', True)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    sources, rows = [], {}
    for name, report_name, color in settings:
        report_path = ROOT / REL / report_name
        original = report_path.read_bytes()
        report = json.loads(original)
        assert report['all_checks_pass'] and all(report['checks'].values())
        assert report['extrema']['complete_RSL_updates'] >= 320
        prefix = report['event_prefix']
        with open(prefix['path'], 'rb') as stream:
            raw = stream.read(prefix['bytes'])
        assert len(raw) == prefix['bytes'] and digest(raw) == prefix['sha256']
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / Path(prefix['path']).name
            snapshot.write_bytes(raw)
            events = EventAccumulator(str(snapshot), size_guidance={'scalars': 0}).Reload()
            selected = [tag for tag, _, _ in metrics] + [STANDING + 'active_lanes', STANDING + 'height_mae_m']
            data = {tag: {point.step: point.value for point in events.Scalars(tag)} for tag in selected}
        assert all(all(i in values and math.isfinite(values[i]) for i in range(320)) for values in data.values())
        windows = []
        for first in range(0, 320, 32):
            values = {tag: sum(data[tag][i] for i in range(first, first + 32)) / 32 for tag in selected}
            saved = report['pure_standing_and_objective_windows'][first // 32]
            assert saved['updates'] == [first + 1, first + 32]
            for tag in selected:
                if tag.startswith(STANDING):
                    assert values[tag] == saved['pure_standing_and_objective'][tag]
            windows.append(dict(updates=[first + 1, first + 32], standing={
                tag.removeprefix(STANDING): value for tag, value in values.items() if tag.startswith(STANDING)},
                all_roles_peak_rad_s=max(data[metrics[-1][0]][i] for i in range(first, first + 32))))
        rows[name] = windows
        x = [16.5 + 32 * i for i in range(10)]
        for ax, (tag, title, peak) in zip(axes.flat, metrics, strict=True):
            y = [max(data[tag][i] for i in range(first, first + 32)) if peak else
                 sum(data[tag][i] for i in range(first, first + 32)) / 32 for first in range(0, 320, 32)]
            ax.plot(x, y, 'o-', color=color, label=name, linewidth=1.8, markersize=4)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Completed fresh PPO updates')
            ax.axvline(192, linestyle='--', linewidth=.9, color='#333333', alpha=.65)
            ax.set_xlim(0, 330)
            ax.grid(alpha=.2)
        assert report_path.read_bytes() == original
        sources.append(dict(run=name, report=str(report_path), report_sha256=digest(original), event_prefix=prefix))
    fig.suptitle('Fresh017 vs 018: foundation and domain admission (32-update windows)', fontsize=13, y=.98)
    fig.text(.5, .925, 'Dashed line: task exposure / DR admission start at192. All188 random lanes first learn at321.',
             ha='center', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
    fig.text(.5, .045, 'Standing means include failure resets and not-yet-admitted random cars; population changes as tasks enter.',
             ha='center', fontsize=9)
    fig.text(.5, .022, 'Same seed / initialization /17 initial domain arrays. Training diagnostics do not qualify continuous standing.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(top=.85, bottom=.2, hspace=.4, wspace=.28)
    for suffix in ('.png', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    receipt = dict(schema='fresh018-admission320-RSL-comparison-v1', sources=sources,
        updates=[1, 320], window=32, rows=rows,
        data_source='Original RSL TensorBoard SHA-authenticated byte prefixes; no independent curve data source',
        full_random_admission_first_update=321, random_lane_count=188, nominal_lane_count=68,
        same_initialization_seed_and_initial17_domain_arrays=True,
        scope='Standing roles across the actual full population, including unadmitted random lanes; failure resets retained. Maxima include passive and recovery samples.',
        new_native_steps=0, new_PPO=0, optimizer_constructed=False, behavior_qualified=False,
        hardware_ready=False, producer_sha256=digest(Path(__file__).read_bytes()),
        artifacts={s: digest(output.with_suffix(s).read_bytes()) for s in ('.png', '.svg')})
    output.with_suffix('.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for suffix in ('.png', '.svg', '.json'):
        (BASE / REL / output.with_suffix(suffix).name).write_bytes(output.with_suffix(suffix).read_bytes())
    (BASE / REL / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'result': str(output.with_suffix('.json')), 'last_windows': {k: v[-1] for k, v in rows.items()}}))


if __name__ == '__main__':
    main()
