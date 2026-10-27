"""Plot existing RSL prefixes only; training resets are never behavior passes."""
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')


def main():
    bindings=[('016: scoped failure','fresh_unified016_stopped_closure001.json','RSL_event','#777777'),
        ('017: dense pose + physical speed','fresh_unified017_live64_003.json','event_prefix','#0072B2')]
    metrics=[('DomainAdmission/nominal/tilt_mean_deg','Nominal tilt (deg)',1.,False),
        ('DomainAdmission/nominal/planar_speed_rms_m_s','Nominal planar speed RMS (m/s)',1.,False),
        ('DomainAdmission/nominal/episode_end_events','Mean episode ends / nominal car / s',200./68,False),
        ('ScratchStandingExtrema/peak_leg_speed_rad_s','True1ms leg peak, all256 cars (rad/s)',1.,True)]
    output=ROOT/REL/'fresh_objective_first64_001'
    if output.with_suffix('.json').exists():raise ValueError('Preserve the first figure and its bindings')
    fig,axes=plt.subplots(2,2,figsize=(11,7.6));metadata=[]
    for label,name,key,color in bindings:
        report=json.loads((ROOT/REL/name).read_text());prefix=report[key]
        with open(prefix['path'],'rb') as stream:raw=stream.read(prefix['bytes'])
        assert hashlib.sha256(raw).hexdigest()==prefix['sha256']
        with tempfile.TemporaryDirectory() as temporary:
            snapshot=Path(temporary)/Path(prefix['path']).name;snapshot.write_bytes(raw)
            ev=EventAccumulator(str(snapshot),size_guidance={'scalars':0}).Reload()
            for ax,(tag,title,scale,peak) in zip(axes.flat,metrics,strict=True):
                data={p.step:p.value for p in ev.Scalars(tag)}
                assert all(i in data for i in range(64))
                windows=[[data[j] for j in range(i,i+8)] for i in range(0,64,8)]
                y=[(max(row) if peak else sum(row)/len(row))*scale for row in windows]
                ax.plot([4.5+8*i for i in range(8)],y,'o-',label=label,color=color,linewidth=1.8,markersize=4)
                ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.set_xlabel('Completed fresh PPO updates')
        metadata.append(dict(label=label,source_report=name,event_prefix=prefix))
    fig.suptitle('Fresh016 vs017: first64 updates (8-update windows)',fontsize=14,y=.98)
    fig.text(.5,.925,'Same initial actor/critic, seed,17 domain arrays and3-height curriculum; nominal-only PPO in both.',ha='center',fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.text(.5,.025,'Training contains failure resets. Lower mean tilt does not establish continuous standing or robustness.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.85,bottom=.19,hspace=.42,wspace=.3)
    for suffix in ('.png','.svg'):fig.savefig(output.with_suffix(suffix),dpi=160)
    plt.close(fig)
    result=dict(schema='fresh-objective-training-plot-v1',sources=metadata,updates=[1,64],window=8,
        data_source='Original RSL TensorBoard event byte prefixes, SHA verified',
        nominal_car_count=68,policy_hz=200,full_physical_car_count=256,behavior_qualified=False,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={s:hashlib.sha256(output.with_suffix(s).read_bytes()).hexdigest() for s in ('.png','.svg')})
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    for suffix in ('.png','.svg','.json'):(BASE/REL/output.with_suffix(suffix).name).write_bytes(output.with_suffix(suffix).read_bytes())
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(output.with_suffix('.png'))


if __name__=='__main__':main()
