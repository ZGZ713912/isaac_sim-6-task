"""Plot actual RSL event prefixes of matched fresh foundations, with reset counts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUNS = dict(without_scoped_failure=BASE/'runs/fresh_unified_20261027/pilot015_requested_mixed1536',
    with_scoped_failure=BASE/'runs/fresh_unified_20261027/pilot016_scoped_failure_requested_mixed1536')
METRICS = ('tilt_mean_deg', 'height_mae_m', 'planar_speed_rms_m_s', 'episode_end_events')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    if args.updates < 8 or args.updates > 192 or args.updates % 8:
        raise ValueError('Use complete8-update windows within the declared nominal-only foundation')
    data, provenance = {}, {}
    for label, run in RUNS.items():
        progress = read(run/'progress.json')
        if progress['completed_updates'] < args.updates:
            raise ValueError('Wait for the actual completed native updates')
        event, = run.glob('events.out.tfevents.*')
        raw = event.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            frozen = Path(temporary)/event.name
            frozen.write_bytes(raw)
            events = EventAccumulator(str(frozen), size_guidance={'scalars': 0}).Reload()
            data[label] = {}
            for metric in METRICS:
                points = events.Scalars('DomainAdmission/nominal/'+metric)[:args.updates]
                if [p.step for p in points] != list(range(args.updates)):
                    raise ValueError('Need the complete actual RSL prefix: '+metric)
                data[label][metric] = np.array([p.value for p in points])
        sources = read(run/'source_hashes.json')
        if any(sha(run/'source_snapshot'/name) != value for name, value in sources.items()):
            raise ValueError('Archived learner source changed')
        if label == 'with_scoped_failure' and any(sha(ROOT/name) != value for name, value in sources.items()):
            raise ValueError('Active learner source changed')
        provenance[label] = dict(run=str(run), event=str(event), bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(), complete_updates_used=args.updates,
            source_count=len(sources), committed_source=progress['committed_source']['commit'])
    starts = [read(run/'initialization.json') for run in RUNS.values()]
    draws = [read(run/'randomization_study.initial_samples.json') for run in RUNS.values()]
    assert all(starts[0][role+'_tensor_sha256'] == starts[1][role+'_tensor_sha256'] for role in ('actor', 'critic'))
    assert draws[0]['samples'] == draws[1]['samples']
    nominal = len(draws[0]['samples']['enabled'])-sum(draws[0]['samples']['enabled'])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    colors = dict(without_scoped_failure='#bf4b36', with_scoped_failure='#167d9a')
    labels = dict(without_scoped_failure='015: original native termination',
        with_scoped_failure='016: pure-standing failure reset + cost')
    scales = (1., 1000., 1., 1./(.005*nominal))
    titles = ('True tilt before reset', 'True height error before reset',
        'True planar speed under exploration', 'Episode ends per nominal robot per second')
    units = ('deg', 'mm', 'm/s', '1/s')
    for axis, metric, scale, title, unit in zip(axes.flat, METRICS, scales, titles, units, strict=True):
        for label in RUNS:
            values = data[label][metric]*scale
            axis.plot(np.arange(1, args.updates+1), values, color=colors[label], alpha=.2, linewidth=1.)
            axis.plot(np.arange(8, args.updates+1, 8)-3.5, values.reshape(-1, 8).mean(1),
                color=colors[label], linewidth=2.2, marker='o', markersize=4, label=labels[label])
        axis.set(title=title, ylabel=unit, xlabel='Completed fresh PPO updates')
        axis.set_ylim(bottom=0.)
        axis.grid(alpha=.2)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Fresh three-height foundation: actual stochastic training diagnostics\n'
        'Same seed, initial parameters and domain draws; mean reductions include failure resets', fontsize=13)
    target = BASE/REL/args.output_name
    if target.with_suffix('.json').exists():
        raise ValueError('Preserve the prior plotted event prefix')
    for suffix in ('.png', '.svg'):
        fig.savefig(target.with_suffix(suffix), dpi=160)
        if suffix == '.svg':
            path = target.with_suffix(suffix)
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        (ROOT/REL/target.with_suffix(suffix).name).write_bytes(target.with_suffix(suffix).read_bytes())
    output = dict(schema='fresh-scoped-failure-training-plot-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
        provenance=provenance, window_updates=8, nominal_lanes=nominal,
        same_initial_actor_critic_and_domain_draws=True,
        final8_update_means={label:{metric:float(values[-8:].mean()) for metric, values in row.items()}
            for label, row in data.items()},
        images={suffix:sha(target.with_suffix(suffix)) for suffix in ('.png', '.svg')},
        interpretation='Failure termination changes the trajectory distribution. These curves measure training signal and '
            'reset frequency, not uninterrupted standing, fixed contact/drift acceptance or a unique cause.',
        new_PPO=0, native_steps=0, qualified=False, hardware_ready=False, producer_sha256=sha(Path(__file__)))
    for directory in (ROOT/REL, BASE/REL):
        (directory/target.with_suffix('.json').name).write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(output['final8_update_means']))


if __name__ == '__main__':
    main()
