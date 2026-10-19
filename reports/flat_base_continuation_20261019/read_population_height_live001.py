"""Bound a live training-event snapshot at the last flushed completed update."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
trial=ROOT/'runs/flat_base_20261019/pilot024'
info=json.loads((trial/'progress.json').read_text())
height=info['height_command_metrics_contract'];checks={};height_rows=0
reader=EventAccumulator(str(trial),size_guidance={'scalars':0});reader.Reload()
flushed=[x.step for x in reader.Scalars('FlatTransfer/actor_updates_total')]
completed=min(info['successful_updates_total'],max(flushed))
output=HERE/f'feedback_population_height_live{completed}.json'
if output.exists():raise SystemExit('Preserve the completed snapshot')
if True:
    events = EventAccumulator(str(trial),size_guidance={'scalars':0});events.Reload()
    tags = set(events.Tags()['scalars'])
    cache = {}
    def scalar(tag):
        if tag not in cache:
            cache[tag] = {point.step:point.value for point in events.Scalars(tag)}
        return cache[tag]
    expected = list(range(1249,completed+1))
    height_checks = []
    for namespace in ('FlatBehavior','FlatBirth'):
        for group,population in info['training_command_reference']['population'].items():
            views = [group]+([group+'/rmcs_direct'] if population['direct'] else [])+([group+'/legacy_ramp'] if population['ramped'] else [])
            for view in views:
                prefix = namespace+'/'+view
                whole = scalar(prefix+'/sample_frames')
                bins = [scalar(prefix+'/'+band+'/sample_frames') for band in height['bands']]
                height_checks.append(sorted(whole) == expected and all(sorted(values) == expected for values in bins))
                for count in expected:
                    total = whole[count]
                    height_checks.append(sum(values[count] for values in bins) == total)
                    for metric in ('vx_mae_m_s','yaw_mae_rad_s','height_mae_m','planar_speed_mean_m_s','terminated_frame_fraction'):
                        aggregate_tag = prefix+'/'+metric
                        all_values = scalar(aggregate_tag) if aggregate_tag in tags else {}
                        weighted = 0.
                        for band,frames in zip(height['bands'],bins,strict=True):
                            tag = prefix+'/'+band+'/'+metric
                            values = scalar(tag) if tag in tags else {}
                            if frames[count]:
                                height_checks.append(count in values and math.isfinite(values.get(count,float('nan'))))
                                weighted += frames[count]*values.get(count,float('nan'))
                            else:
                                height_checks.append(count not in values)
                        height_checks.append(math.isclose(weighted/total,all_values.get(count,float('nan')),rel_tol=3e-7,abs_tol=3e-8)
                            if total else count not in all_values)
                        height_rows += 1
    checks['all_flushed_prefix_rollout_frame_clock_points'] = all(scalar('FlatTransfer/training_transitions_total')[n] == 61341696+(n-1248)*4096*96 for n in expected)
    checks['all_height_counts_and_weighted_errors_match_existing_logger'] = bool(height_checks) and all(height_checks)

if not all(checks.values()):raise SystemExit('Height strata do not conserve their existing aggregate metrics')
rows={}
for group in ('backward_low','backward_edge3'):
    for mode in ('','/rmcs_direct','/legacy_ramp'):
        for band in height['bands']:
            prefix='FlatBehavior/'+group+mode+'/'+band
            frames=scalar(prefix+'/sample_frames');values=scalar(prefix+'/vx_mae_m_s')
            valid=[n for n in expected if frames.get(n,0)]
            count=sum(frames[n] for n in valid)
            rows[group+mode+'/'+band]=dict(policy_frames=count,
                frame_weighted_vx_mae_m_s=sum(frames[n]*values[n] for n in valid)/count if count else None,
                values_by_update={n:values[n] for n in valid})
output.write_text(json.dumps(dict(schema='flat-live-training-height-conservation-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),completed_update=completed,
    custom_update_steps=expected,passed=True,checks=checks,height_metric_rows_checked=height_rows,
    sampled_training_rows=rows,new_native_steps=0,new_optimizer_updates=0,
    behavior_qualified=False,scope='Flushed prefix of the running trial only; includes reset transients and varying population exposure'),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed=completed,passed=True,height_metric_rows_checked=height_rows,output=str(output))))
