"""Read-only comparison of current headless defaults with explicit output-off."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path('/home/noir/.codex/worktrees/prone-motion-long/isaac_wheeled_rl_train')
OUT=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/fabric_output_20261029')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
names=('stock_nominal001','off_nominal001')
cases=[]
gpu_pids={int(v) for v in subprocess.check_output(
    ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()}
for name in names:
    folder=OUT/name
    r=json.loads((folder/'result.json').read_text())
    assert r['status']=='completed' and not r['source_mismatches_at_exit']
    assert r['new_PPO']==r['optimizer_attempts']==0 and not r['actor_or_critic_created']
    assert not r['checkpoint_reads']
    assert r['fabric_after_environment']['/physics/fabricUpdateTransformations'] is False
    assert r['fabric_after_environment']['/physics/fabricUpdateVelocities'] is False
    assert r['fabric_after_environment']['/physics/suppressReadback'] is True
    assert r['fabric_after_environment']['/isaaclab/fabric_enabled'] is True
    assert r['runtime']['physics_dt']==.001 and r['runtime']['policy_dt']==.005
    assert r['runtime']['decimation']==5 and r['runtime']['num_envs']==256
    assert r['runtime']['position_iterations']==64 and r['runtime']['velocity_iterations']==32
    proc=Path('/proc')/str(r['pid'])
    same_identity=proc.exists() and int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])==r['start_ticks']
    assert not same_identity and r['pid'] not in gpu_pids
    assert sha(folder/'trace.npz')==r['trace_sha256']
    events=EventAccumulator(str(folder/'events'),size_guidance={'scalars':0}).Reload()
    values=events.Scalars('PerformanceProbe/policy_step_s')
    assert len(values)==128 and [x.step for x in values]==list(range(128))
    assert np.allclose([v.value for v in values],r['step_times_s'],atol=1e-8,rtol=1e-7)
    cases.append(dict(name=name,result_sha256=sha(folder/'result.json'),trace_sha256=r['trace_sha256'],
        benchmark_script_sha256=r['benchmark_script_sha256'],
        effective_settings=r['fabric_after_environment'],runtime=r['runtime'],
        source_hashes=r['source_hashes'],source_mismatches_at_exit=[],
        event_files=[dict(path=str(p),sha256=sha(p)) for p in (folder/'events').glob('events.out.tfevents.*')],
        event_mean_s=float(np.mean([v.value for v in values])),
        process_cleanup=dict(pid=r['pid'],start_ticks=r['start_ticks'],same_identity_alive=False,
            in_gpu_compute_list=False)))
assert cases[0]['effective_settings']==cases[1]['effective_settings']
assert cases[0]['source_hashes']==cases[1]['source_hashes']
checks=[]
with np.load(OUT/names[0]/'trace.npz') as a,np.load(OUT/names[1]/'trace.npz') as b:
    assert set(a.files)==set(b.files)
    for key in sorted(a.files):
        x,y=a[key],b[key]
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),key
        checks.append(dict(key=key,shape=list(x.shape),dtype=str(x.dtype),bitwise_equal=True))
    counts=dict(reset_done_events=int(b['done'].sum()),disabled_lane_samples=int(b['diag/passive'].sum()))
acceptance=json.loads((ROOT/'contracts/wheel_leg_acceptance_requirements_v2.json').read_text())
timings=[dict(policy_hz=200,rollout_steps=128,physics_hz=1000,physics_steps=640,simulated_seconds=.64,
              samples=32768),
         dict(policy_hz=100,rollout_steps=128,physics_hz=1000,physics_steps=1280,simulated_seconds=1.28,
              samples=32768),
         dict(policy_hz=100,rollout_steps=64,physics_hz=1000,physics_steps=640,simulated_seconds=.64,
              samples=16384)]
receipt=dict(schema='headless-fabric-output-verification-v1',
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    conclusion='Current local headless defaults already disable both transform and velocity publishing; explicit off changes no effective setting.',
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    cases=cases,all_trace_arrays_bitwise_equal=True,trace_arrays=checks,
    trace_policy_steps=256,trace_physics_steps=1280,trace_counts=counts,
    policy_frequency_comparison=dict(kind='Exact clock/sample count, not100Hz native or PPO benchmark',
        cases=timings,production_frequency_changed=False,production_100hz_supported=False),
    new_PPO=0,optimizer_attempts=0,checkpoint_reads=[],actor_or_critic_created=False,
    source_changed=False,server_modified=False,server_live_settings_verified=False,
    local_tensorboard_server_started=False,hardware_ready=False,full_capability_qualified=False,
    capabilities=[dict(id=c['id'],requirement=c['requirement'],status='not_revalidated',passed=False)
                  for c in acceptance['capabilities']],
    notes=['Timing variation between identical effective settings is not evidence of speedup.',
        'Do not disable the Fabric subsystem or turn on USD readback for this headless tensor workload.',
        'GUI playback, cameras and other rendering consumers require their normal transform publication.',
        'Current100Hz would require a new control contract, time-aware discount/reward/history handling and deployment validation.',
        'The server uses a different Isaac version; its live settings were not read or changed this turn.'])
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
rows='\n'.join(f"| {c['name']} | {1000*c['event_mean_s']:.3f} | false | false |"
               for c in cases)
caps='\n'.join(f"| {c['id']} | 本轮未重新验收，未授予通过 |" for c in receipt['capabilities'])
report=f'''# 无界面Fabric输出与100Hz策略频率检查

本机当前冻结源`{receipt['source_commit']}`的无界面训练默认已经关闭姿态和速度发布。显式再次关闭两项不改变有效配置，不能作为额外提速。服务器当前learner没有停止、修改或启动新进程；本轮未重连服务器，其Isaac版本不同，不能将本机标志读回当作服务器读回。

## 原生读回与对照

| 本机256环境 / 原200Hz策略 | 128步计时均值ms/策略步 | fabricUpdateTransformations | fabricUpdateVelocities |
| --- | ---: | --- | --- |
{rows}

这两个进程的有效设置完全相同。计时取原TensorBoard event的`PerformanceProbe/policy_step_s`，不是独立绘图数据源；相同配置之间的波动不解释为优化收益。两次均为1ms物理、5ms策略、decimation5、64/32求解迭代，256步原始轨迹、16步额外warmup及128步计时。各126组轨迹数组全部逐位一致，含观测、奖励、命令、根/关节状态、每1ms物理诊断以及done；含{counts['reset_done_events']}次done事件及{counts['disabled_lane_samples']}个失能lane样本。0PPO、0actor/critic、0旧checkpoint读取。两进程已按PID/start_ticks及GPU列表确认退出，源码摘要无变化。

启动后其他有效设置：`fabricUseGPUInterop=true`、`suppressReadback=true`、`isaaclab/fabric_enabled=true`。这里保留Fabric子系统，只关闭无界面任务不用的发布。不能将`use_fabric=False`当等价优化：它可能启用USD同步，或破坏GUI显示链路。本机IsaacLab AppLauncher已将无渲染姿态输出关闭，速度输出默认也为false；环境构建后复核两项仍false。底层根/连杆状态读取由PhysX张量视图`get_root_transforms`/`get_link_transforms`提供；上述原生轨迹对照独立验证了观测不依赖再次开启发布。没有评测GUI、相机或学习策略，不能转授它们的资格。

## 策略200Hz改100Hz

在底层物理/PD保持1000Hz时，每个动作保持从5ms改10ms。精确时钟计算：

| 策略频率 | 每轮步数 | 仿真秒/环境 | 每轮串行物理步 | 256环境总样本 |
| --- | ---: | ---: | ---: | ---: |
| 200Hz | 128 | 0.64 | 640 | 32768 |
| 100Hz | 128 | 1.28 | 1280 | 32768 |
| 100Hz | 64 | 0.64 | 640 | 16384 |

保持128步会让占大头的物理/低层控制工作翻倍，不能预期每轮变快。保持0.64秒而用64步，则物理工作不变，仅减少策略推理、观测、奖励及策略频率诊断等调用，训练样本也减半；不能宣称相同10000轮具有相同训练量。这是时钟分析，本轮没有创建100Hz控制合同或执行100Hz原生/PPO探针。

100Hz的策略计算和实车推理负担较小，但新目标每10ms而非5ms刷新。1kHz低层PD仍逐步读反馈；RL层对失衡、刹车、快速自旋的修正更稀疏，是否能保持目标必须行为验证。2m/s时每次目标更新间隔内行进约2cm，而200Hz约1cm；100rpm时约6°而非3°。这些数字是运动学间隔，不能单独判定性能好坏。

正式采用100Hz还需同步控制合同/decimation、以秒为单位的回合和课程、动作变化成本与参考步长、折扣/GAE、按policy-tick表示的传感延迟、导出/实车推理频率等。当前新轮控制合同要求200Hz，不能只改`policy_dt`绕过校验。不会为本次速度讨论修改正式合同、暂停训练或续训旧模型。

## 全部长期能力

本轮仅性能与接线，不授予策略行为或实车资格；接触口径、原地起立约10cm和实际完成后完整30秒均保留。全部13项逐项保留：

| 能力 | 本轮结论 |
| --- | --- |
{caps}

机器可读来源、TensorBoard event和逐数组核验见[receipt.json](receipt.json)。[NVIDIA官方说明](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html#physics-simulation-optimizations)要求姿态和速度发布同时关闭，且只适用于不消费显示输出的无界面GPU张量任务。Context7当前工具不可用，本轮使用安装源码及官方文档核对。
'''
(OUT/'REVIEW.md').write_text(report)
print(json.dumps(dict(conclusion=receipt['conclusion'],trace_arrays=len(checks),timing_ms=[c['event_mean_s']*1000 for c in cases]),indent=2))
