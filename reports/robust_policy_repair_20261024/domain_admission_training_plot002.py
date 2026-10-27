"""Export paired-index training diagnostics from authenticated RSL prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/robust_policy_repair_20261024'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prefix(path, size, expected):
    if hashlib.sha256(Path(path).read_bytes()[:size]).hexdigest() != expected:
        raise ValueError('Actual RSL event prefix differs: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission-report', type=Path, required=True)
    parser.add_argument('--last-update', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 32 <= args.last_update <= 512:
        raise ValueError('Need a complete window through at most the declared512 updates')
    if any(args.output.with_suffix(suffix).exists() for suffix in ('.png', '.svg', '.json')):
        raise ValueError('Preserve the earlier figure and provenance')
    baseline = REPORT / 'scratch_training007_009_window032_final512_read007.json'
    old, new = json.loads(baseline.read_text()), json.loads(args.admission_report.read_text())
    source = old['sources']['pilot009_leg_velocity_separated']
    verify_prefix(source['path'], source['observed_prefix_bytes'], source['observed_prefix_sha256'])
    prefix = new['event_prefix']
    verify_prefix(prefix['path'], prefix['bytes'], prefix['sha256'])
    rows = [window for window in new['windows'] if window['fresh_PPO_update_numbers_inclusive'][1] <= args.last_update]
    expected = [[start+1, start+32] for start in range(0, args.last_update, 32)]
    if args.last_update % 32 or [row['fresh_PPO_update_numbers_inclusive'] for row in rows] != expected:
        raise ValueError('Need complete contiguous32-update windows through the declared end')
    old_rows = old['windows'][:len(rows)]
    if [row['PPO_update_numbers_inclusive'] for row in old_rows] != expected:
        raise ValueError('Historical native windows differ')
    x = [(first+last)/2 for first, last in expected]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, metric, label in zip(axes, ('planar_speed_rms_m_s', 'tilt_mean_deg'),
            ('Planar speed RMS [m/s]', 'True tilt mean [deg]'), strict=True):
        old_y = [row['runs']['pilot009_leg_velocity_separated']['PrecisionStanding/'+metric] for row in old_rows]
        ax.plot(x, old_y, 'o-', color='#25805b', label='009: all nominal (256 lanes)')
        for group, color, text in (('nominal', '#2459ad', '013: nominal sensors/actuators/rigid (79)'),
                                   ('randomized', '#bd482e', '013: randomized physical domains (177)')):
            ax.plot(x, [row['groups'][group][metric] for row in rows], 'o-', color=color, label=text)
        ax.set(xlabel='Fresh PPO update (32-update window mean)', ylabel=label, xlim=(0, args.last_update))
        if metric == 'planar_speed_rms_m_s':
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0.)
        for first, last, shade in ((0,192,'#e7edf5'), (192,320,'#fff0d4'), (320,args.last_update,'#e6f2e8')):
            if first < args.last_update:
                ax.axvspan(first,min(last,args.last_update),color=shade,zorder=-1)
        for boundary in (192,320):
            if boundary < args.last_update:
                ax.axvline(boundary,color='#6d7377',linestyle='--',linewidth=.8)
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .135), ncol=1, frameon=False, fontsize=9)
    fig.suptitle(f'Training through update{args.last_update}; cold behavior evaluation remains required', fontsize=13)
    fig.text(.5,.875,'0-192: nominal-only PPO    |    193-320: gradual admission    |    321+: all256 lanes',ha='center',fontsize=9)
    fig.text(.5, .035, '013 nominal lanes retain requested randomized ground; 009 uses fixed ground.\n'
        'Population, contact and learning-path differences prevent a single-variable causal conclusion.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .29, 1, .87))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        fig.savefig(args.output.with_suffix(suffix), dpi=150)
        if suffix == '.svg':
            svg = args.output.with_suffix(suffix)
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    result = dict(schema='RSL-domain-admission-training-figure-v1',
        last_complete_update=args.last_update, phase_boundaries=[192,320], speed_axis='logarithmic', source_reports={str(baseline): digest(baseline),
            str(args.admission_report): digest(args.admission_report)},
        actual_event_prefixes_verified=True, raw_source='RSL TensorBoard events only; no independent curve source',
        SVG_lines_trimmed_for_source_control=True,
        figure_sha256={suffix: digest(args.output.with_suffix(suffix)) for suffix in ('.png', '.svg')},
        producer_sha256=digest(Path(__file__)), new_PPO=0, native_steps=0, qualified=False, hardware_ready=False)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
