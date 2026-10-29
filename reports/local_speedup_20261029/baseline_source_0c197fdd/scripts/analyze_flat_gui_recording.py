#!/usr/bin/env python3
"""Verify GUI frame clocks, retain failed cases, and assemble a 1x video."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_dwell(mask, count):
    run = 0
    for index, valid in enumerate(mask):
        run = run + 1 if valid else 0
        if run >= count:
            return index-count+1
    return None


def response_metrics(rows, dt=.005):
    """Measure each keypress phase, including physical command-slew latency."""
    phases=[]
    for index,row in enumerate(rows):
        key=tuple(row['pressed_keys'])
        if not phases or phases[-1]['key']!=key:
            phases.append(dict(key=key,rows=[],previous=rows[index-1] if index else None))
        phases[-1]['rows'].append(row)
    result=[]
    for phase in phases:
        data=phase['rows']; key=phase['key']; first=data[0]['case_tick']
        if key and set(key)&{'W','S','A','D'}:
            axis='vx' if set(key)&{'W','S'} else 'yaw'
            command_index=0 if axis=='vx' else 1
            target=data[-1]['requested_command'][command_index]
            actual=np.array([r['velocity_m_s'][0] if axis=='vx' else r['omega_rad_s'][2] for r in data])
            entry=first_dwell(actual*np.sign(target)>=.9*abs(target),round(.1/dt))
            result.append(dict(kind='launch',axis=axis,keys=list(key),target=target,
                phase_start_s=(first-1)*dt,response90_s=None if entry is None else (entry+1)*dt,
                censored=entry is None))
        elif result:
            speed=np.array([np.linalg.norm(r['velocity_m_s'][:2]) for r in data])
            yaw=np.array([abs(r['omega_rad_s'][2]) for r in data])
            entry=first_dwell((speed<=.05)&(yaw<=.1),round(.25/dt))
            position=np.array([r['position_m'][:2] for r in data])
            if phase['previous'] is not None:
                position=np.vstack((phase['previous']['position_m'][:2],position))
            distance=None if entry is None else float(np.linalg.norm(np.diff(position[:entry+2],axis=0),axis=1).sum())
            tail=max(1,round(.5/dt))
            result.append(dict(kind='stop',phase_start_s=(first-1)*dt,
                settle_time_s=None if entry is None else (entry+1)*dt,stop_path_m=distance,
                censored=entry is None,final_planar_mean_m_s=float(speed[-tail:].mean()),
                final_yaw_abs_mean_rad_s=float(yaw[-tail:].mean())))
    return result


def plot_events(out, cases):
    """All plotted time series come from the existing TensorBoard event log."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    events=EventAccumulator(str(out),size_guidance={'scalars':0});events.Reload()
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True)
    groups=[(['vx_command_m_s','vx_m_s'],'vx (m/s)'),
        (['yaw_command_rad_s','yaw_rad_s'],'yaw (rad/s)'),
        (['height_command_m','height_m'],'height (m)'),
        (['pitch_deg','roll_deg'],'attitude (deg)')]
    for axis,(tags,label) in zip(axes,groups):
        for tag in tags:
            values=events.Scalars('Playback/'+tag)
            axis.plot([p.step/200 for p in values],[p.value for p in values],label=tag,linewidth=1.2)
        axis.set_ylabel(label);axis.grid(alpha=.25);axis.legend(loc='upper right',fontsize=8)
    start=0.
    for index,case in enumerate(cases):
        for axis in axes:axis.axvline(start,color='gray',linewidth=.5,alpha=.5)
        axes[0].text(start,.99 if index%2==0 else .87,case['name'],transform=axes[0].get_xaxis_transform(),
            rotation=65,va='bottom',ha='left',fontsize=6)
        start+=case['simulated_seconds']
    axes[-1].set_xlabel('Cumulative simulated time (s); native resets separate cases')
    fig.suptitle('1346 ONNX | 200Hz policy / 1000Hz physics | GUI keyboard command slew retained',y=.99)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(out/'gui_tensorboard_metrics.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--stopped-by-user',type=Path,
        help='Verified stop receipt; retain the interrupted last clip and disclose incomplete coverage')
    args=parser.parse_args(); out=args.input.resolve()
    recording=json.loads((out/'recording.json').read_text())
    report=json.loads((out/'report.json').read_text())
    stopped=args.stopped_by_user is not None
    stop_receipt=None
    if stopped:
        stop_receipt=json.loads(args.stopped_by_user.read_text())
        if (stop_receipt.get('schema')!='user-requested-play-stop-v1'
                or str(out) not in stop_receipt.get('cmdline','')):
            raise ValueError('Stop receipt does not identify this native recording')
        proc=Path(f'/proc/{stop_receipt["pid"]}/stat')
        if proc.exists():
            fields=proc.read_text().rsplit(')',1)[1].split()
            if int(fields[19])==stop_receipt['start_ticks']:
                raise ValueError('The recorded play process is still alive')
    if not stopped and (recording['status']!='completed' or report['status']!='completed'):
        raise ValueError('Require a completed native GUI recording; preserve interrupted evidence')
    if not stopped and recording['completed_cases']!=len(recording['cases']):
        raise ValueError('Incomplete case denominator')
    if stopped and recording['completed_cases'] not in (len(recording['cases']),len(recording['cases'])-1):
        raise ValueError('More than one unfinished native case')
    if (recording['policy_hz'],recording['physics_hz'])!=(200,1000):
        raise ValueError('Policy/physics clock changed')
    if report['startup_report']['contact_material']['native_readback']['usd_collision_bindings_verified'] is not True:
        raise ValueError('Native friction material readback required')
    trace=defaultdict(list)
    with (out/'keyboard_trace.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);trace[row['case']].append(row)
    frame_map=defaultdict(list)
    with (out/'video_frames.jsonl').open() as stream:
        for index,line in enumerate(stream):
            row=json.loads(line)
            if row['frame']!=index or row['native_physics_steps']!=5*row['native_policy_steps']:
                raise ValueError('Frame or physical feedback clock differs')
            frame_map[row['case']].append(row)
    checks=[]; concat=[]; total_frames=0; total_ticks=0; total_manifest_frames=0
    for case_index,original_case in enumerate(recording['cases']):
        case=dict(original_case)
        name=case['name']; rows=trace[name]; frames=frame_map[name]
        partial=stopped and case_index>=recording['completed_cases']
        if partial:
            case.update(policy_ticks=len(rows),simulated_seconds=len(rows)/200,
                video=str(out/f'{case_index:02d}_{name}_1x.mp4'),user_stopped_partial=True,
                final_vx_m_s=rows[-1]['velocity_m_s'][0],final_yaw_rad_s=rows[-1]['omega_rad_s'][2],
                final_height_m=rows[-1]['height_m'],real_time_factor=rows[-1]['real_time_factor'])
        if len(rows)!=case['policy_ticks'] or [r['case_tick'] for r in rows]!=list(range(1,len(rows)+1)):
            raise ValueError('Missing physical policy samples: '+name)
        if rows[-1]['status'].startswith('PAUSED:')!=case['terminal']:
            raise ValueError('Terminal failure was hidden: '+name)
        sampled=[f['case_tick'] for f in frames if f['kind']=='simulation_frame']
        expected=list(range(8,len(rows)+1,8))
        if len(rows)%8 and not partial:expected.append(len(rows))
        if sampled!=(expected[:len(sampled)] if partial else expected):
            raise ValueError('Dropped or repeated simulation video frames: '+name)
        holds=sum(f['kind']=='labelled_terminal_freeze' for f in frames)
        if holds!=(round(recording['terminal_hold_seconds']*25) if case['terminal'] else 0):
            raise ValueError('Unlabelled video time padding: '+name)
        video=Path(case['video'])
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
            'stream=nb_frames,duration,r_frame_rate,width,height','-of','json',str(video)]))['streams'][0]
        encoded_frames=int(probe['nb_frames'])
        expected_encoded=len(expected) if partial else len(frames)
        if (encoded_frames!=expected_encoded or encoded_frames<len(frames) or probe['r_frame_rate']!='25/1'
                or not math.isclose(float(probe['duration']),encoded_frames/25,abs_tol=.0001)):
            raise ValueError('Encoded time differs from the simulation frame clock: '+name)
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
        tail=rows[-min(200,len(rows)):]
        row=dict(case,frames=encoded_frames,video_duration_s=encoded_frames/25,
            video_sha256=sha(video),encoded=probe,
            last_second_vx_m_s=float(np.mean([r['velocity_m_s'][0] for r in tail])),
            last_second_yaw_rad_s=float(np.mean([r['omega_rad_s'][2] for r in tail])),
            last_second_height_m=float(np.mean([r['height_m'] for r in tail])),
            response=response_metrics(rows),frame_clock_verified=encoded_frames==len(frames),
            manifest_verified_frames=len(frames),unflushed_manifest_tail_frames=encoded_frames-len(frames),
            physics_feedback_steps=5*len(rows),labelled_terminal_freeze_frames=holds,
            max_vx_command_acceleration_m_s2=max(abs(r['command_acceleration_vx_yaw'][0]) for r in rows),
            max_yaw_command_acceleration_rad_s2=max(abs(r['command_acceleration_vx_yaw'][1]) for r in rows),
            inference_ms_median=float(np.median([r['inference_ms'] for r in rows])))
        checks.append(row);total_frames+=encoded_frames;total_manifest_frames+=len(frames);total_ticks+=len(rows)
        concat.append("file '"+str(video).replace("'","'\\''")+"'")
    if total_manifest_frames!=recording['frames']:
        raise ValueError('Video frame denominator changed')
    (out/'concat.txt').write_text('\n'.join(concat)+'\n')
    video=out/'1346_gui_validation_1x.mp4'
    subprocess.run(['ffmpeg','-v','error','-n','-f','concat','-safe','0','-i',str(out/'concat.txt'),
        '-c','copy','-movflags','+faststart',str(video)],check=True)
    combined=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'stream=nb_frames,duration,r_frame_rate','-of','json',str(video)]))['streams'][0]
    if (int(combined['nb_frames'])!=total_frames or combined['r_frame_rate']!='25/1'
            or not math.isclose(float(combined['duration']),total_frames/25,abs_tol=.0001)):
        raise ValueError('Compilation lost video frames')
    result=dict(schema='flat-gui-recording-verification-v1',status='verified',
        policy_sha256=recording['policy_sha256'],successful_updates_total=recording['successful_updates_total'],
        policy_hz=200,physics_hz=1000,formal_updates_consumed=0,qualification=False,hardware_ready=False,
        scope='Native ONNX GUI and scripted WASD callbacks with original command slew. Different from the fixed raw-stop admission protocol.',
        ground_material=recording['contact_material']['ground_static_friction_per_environment'],
        wheel_material=recording['contact_material']['wheel_static_friction'],
        recording_stopped_by_user=stopped,stop_receipt=stop_receipt,
        completed_cases=recording['completed_cases'],
        partial_cases=sum(bool(r.get('user_stopped_partial')) for r in checks),
        cases=checks,total_cases=len(checks),completed_without_native_terminal=sum(
            not r['terminal'] and not r.get('user_stopped_partial') for r in checks),
        video=str(video),video_sha256=sha(video),encoded=combined,simulation_seconds=total_ticks/200,
        total_frames=total_frames,manifest_verified_frames=total_manifest_frames,
        unflushed_manifest_tail_frames=total_frames-total_manifest_frames,
        tail_clock_note='Interrupted final clip has encoded frames beyond the flushed frame manifest; nominal cadence is consistent with the recorded policy trace. These tail frames are not independently manifest-verified.' if total_frames!=total_manifest_frames else None,
        native_policy_steps=total_ticks,native_physics_feedback_steps=5*total_ticks,
        final_live_real_time_factor=rows[-1]['real_time_factor'],
        traces_sha256=sha(out/'keyboard_trace.jsonl'),frame_manifest_sha256=sha(out/'video_frames.jsonl'),
        recording_sha256=sha(out/'recording.json'),native_report_sha256=sha(out/'report.json'))
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(out),filename_suffix='.gui_verification') as writer:
        for index,row in enumerate(checks):
            for metric in ['last_second_vx_m_s','last_second_yaw_rad_s','last_second_height_m',
                           'peak_abs_pitch_deg','closure_gap_peak_mm','real_time_factor','inference_ms_median']:
                writer.add_scalar(f'GUIValidation/{row["name"]}/{metric}',row[metric],index)
            writer.add_scalar(f'GUIValidation/{row["name"]}/native_terminal',float(row['terminal']),index)
    plot_events(out,checks)
    entries=[]
    for row in checks:
        entries.append(f'<tr><td><a href="{html.escape(Path(row["video"]).name)}">{html.escape(row["name"])}</a></td>'
            f'<td>{"用户中止（未跑完）" if row.get("user_stopped_partial") else "终止："+html.escape(row["termination_status"]) if row["terminal"] else "跑满"}</td>'
            f'<td>{row["simulated_seconds"]:.2f}</td><td>{row["last_second_vx_m_s"]:+.3f}</td>'
            f'<td>{row["last_second_yaw_rad_s"]:+.3f}</td><td>{row["peak_abs_pitch_deg"]:.2f}</td></tr>')
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>1346 GUI验证视频</title>'
        '<style>body{background:#101827;color:#e9eef6;font:16px system-ui;max-width:1100px;margin:32px auto}a{color:#69caff}'
        'video{width:100%}td,th{padding:10px;border-bottom:1px solid #334155;text-align:left}table{width:100%;border-collapse:collapse}</style>'
        '<h1>1346号模型 · 200Hz · Isaac Sim GUI</h1><p>视频按仿真1倍速，25FPS；失败处额外冻结1秒并明确标注。'
        '这是原键盘指令斜率下的GUI演示；不等同于突然停车请求的正式评测，也不授予实车资格。</p>'
        +('<p>用户要求停止play，已停止。最后一段仅录制部分；230mm与420mm高度项目未运行。最后3个编码帧因退出未刷新帧清单，保留视频并在回执中披露。</p>' if stopped else '')+
        '<video controls preload="metadata" src="1346_gui_validation_1x.mp4"></video>'
        '<p><a href="gui_tensorboard_metrics.png">从TensorBoard event导出的速度、旋转、高度与姿态曲线</a></p><table><tr><th>动作与分段视频</th>'
        '<th>原生结果</th><th>仿真秒</th><th>末1秒 vx m/s</th><th>末1秒 yaw rad/s</th><th>俯仰峰值 °</th></tr>'
        +''.join(entries)+'</table><p><a href="verification.json">完整数据与SHA回执</a></p></html>')
    print(json.dumps({k:result[k] for k in ['status','video','total_cases','completed_without_native_terminal',
        'simulation_seconds','total_frames','final_live_real_time_factor']},ensure_ascii=False))


if __name__=='__main__':
    main()
