# 平地策略的低摩擦材质和渐进自起

## 当前地面范围：0.2–0.5

用户最新指示将地面材质改为每个环境均匀随机0.2–0.5，静摩擦和动摩擦相同；
轮子保持0.1，机身保持0.25。当前入口为
`contracts/flat_base_finetune_ground_0p2_0p5_v1.json`，
其`average`轮地接触系数为0.15–0.30。
此前0.2–0.3和包含0.1的三档地面属于下面保留的历史实验，已不作为当前训练范围。
新范围的基线目录是`runs/flat_base_20261019/ground025_parent1328_basic001/`。
静站分支的后续小训说明见[静站适应实验](FLAT_STATIC_GRIP_20261019.md)。

## 历史材质与自起证据

本次使用已封存的 `pilot028_wheel_rate1328/model_001328.pt`，SHA-256
`5e2320c50b8fd7982b278dfb4d21cc4763cf12be3cf10193bbf630c648bf672b`。
策略频率保持50 Hz、物理/PD保持1000 Hz，自起参考保持200 Hz。
本次所有原生运行均为固定策略回放，新增PPO更新为0。

## 两阶段材质

轮子、机身和地面分别绑定物理材质。静摩擦和动摩擦先使用相同值。

| 阶段 | 轮子材质 | 地面材质 | average组合后的轮地摩擦 |
| --- | --- | --- | --- |
| 低摩擦适应 | 0.1 | 每个环境随机0.2–0.3 | 0.15–0.20 |
| 随机摩擦 | 0.1 | 0.1、随机0.2–0.3、随机0.3–0.4，各1/3采样概率 | 0.10、0.15–0.20、0.20–0.25 |

机身碰撞材质单独设为0.25，这是本次候选的建模值；不代表实测机身摩擦。
轮子采用用户所给范围的下限0.1。地面系数是材质输入值，未按有效摩擦倒算或与轮子系数相乘。
组合公式见[PhysX官方说明](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html)。

两个入口分别为：

- `contracts/flat_base_finetune_low_friction_v1.json`
- `contracts/flat_base_finetune_random_friction_v1.json`

地面在场景建立时逐环境采样，同一环境的整块平地保持一致；episode reset不更换材质。
随机阶段采用三档混合抽样，实际人口并非强制精确各1/3。
材质使用独立随机数生成器，配置、样本和有效接触系数写入启动回执。
原生启动核对USD碰撞绑定及PhysX机器人形状的摩擦系数。
地面检查目前是导入场景后的USD静态碰撞材质读回，不宣称读到了PhysX地面材质张量。

## 低摩擦平地证据

`runs/flat_base_20261019/low_friction1328_basic001/evaluation.json` 已完成。
32/32轨迹跑满，8项基础行为通过7项。唯一失败项是23 cm静站，最大漂移
从原材质的1.353 cm增至3.149 cm，超过既定3 cm门限。
前后0.5 m/s的速度MAE为0.028916/0.028085 m/s；未降低验收门限。

该回放中的实际地面材质范围为0.200411–0.291896，轮地有效摩擦为0.150206–0.195948。
64个轮碰撞、896个机身碰撞及32个地面碰撞的USD材质绑定检查通过；
PhysX机器人形状读回包含每台2个0.1轮材质和28个0.25机身材质。

基础固定回放可用以下命令重做，输出目录必须不存在：

```sh
OMNI_KIT_ACCEPT_EULA=YES /home/noir/miniconda3/envs/isaacsim6/bin/python scripts/evaluate_flat_base.py \
  --plan contracts/flat_base_finetune_low_friction_v1.json \
  --checkpoint runs/flat_base_20261019/pilot028_wheel_rate1328/model_001328.pt \
  --output runs/flat_base_20261019/low_friction_recheck001
```

切换随机阶段时显式选择 `flat_base_finetune_random_friction_v1.json`。
这两份计划不自动启动优化器或串联长训。仍需认证父checkpoint、源码、计划迁移和实际行为证据。
原1328分支只训练高速/停车的轮速头，静止且轮缘速度低于0.4 m/s时此头门控为零；
仅替换材质不会让该分支获得低位静站的学习能力。因此这两份材质计划是场景候选，
不能据此声称低摩擦适应训练已经完成。

## 渐进自起候选

`slow_recovery.py` 是显式启用的候选，历史冻结合同及默认快速流程不切换。
流程为收腿、收腿姿态停留、沿传感器选择的方向转腿、双轮支撑后缓慢推起、
轮子辅助平衡、200 ms力矩交接，再由RL连续稳定1秒并完成后续静站。
该路径是脚本辅助接管，不是策略已经学会纯RL自起。

正反方向共享阶段速度上限。FK、PD和动作历史共用连续编码器角度；
同一腿的髋/辅助电机共用等价绕转，不独立取最短角。
收腿停留阶段不沿用原来约2.2 rad的反向落脚目标。
推起目标根据重力方向对齐，并选择与已完成旋转相邻的整腿等价绕转。
轮子辅助需满足传感器支撑条件、倾角门和250 ms爬升；实际力矩仍受原电机限制。

原生对照均使用同一父策略和低摩擦计划：

| 流程 | 完成自起与后续静站 | 补充说明 |
| --- | --- | --- |
| 旧流程 | 21/32 | 仍有脚本失败和过早放行 |
| 慢流程v1 | 8/32 | 6种倒地姿态全失败，轮子辅助0帧 |
| 慢流程v2 | 原8种姿态7/32；另加两方向立即接管0/8 | 辅助已触发，正向推起停在倒置姿态 |
| 慢流程v3 | 原8种姿态12/32；另加两方向立即接管0/8 | 趴地轮子朝上的4/4成功，侧倒仍在捕获阶段超时 |
| 慢流程v4 | 原8种姿态4/32；另加两方向立即接管0/8 | 降低轮子力矩未改善起身，不采纳该组增益 |
| 慢流程v5 | 原8种姿态7/32；另加两方向立即接管1/8 | 分开方向补偿与伸腿限速仍未解决整体自起 |

v2轨迹显示：约105°时，FK假设轮子着地得到根高度约0.12 m，
但已知机身碰撞几何要求根高度至少约0.245 m。该姿态不可能以轮子支撑且不穿地，
不能仅以FK高度合理、IMU加速度接近重力作为触地证明。
v3增加机身碰撞顶点检查：候选轮地高度必须满足机身不会穿地的必要条件。
这个条件只使用封存机身几何、编码器FK和机身重力方向，不读取仿真真实高度或接触。
它仍是必要条件，不能单独证明双轮触地。

v3又定位到捕获阶段仍使用原8 Nm/rad轮子平衡增益，目标速度峰约49 rad/s；
当前模型总质量13.886 kg，在轮地有效摩擦0.15时，均匀双轮承重的单轮静态牵引力矩
量级仅为0.613 Nm。此估算不能替代瞬时载荷与打滑测量。
v4曾将脚本轮子请求单独限为0.5 Nm，平衡增益降为2 Nm/rad和0.4 Nm·s/rad，
保持原生电机上限和RL动作限制；推起阶段保持伸腿，倾角小于25°后才转入捕获。
此实验失败，不能单凭静态牵引估算决定整个动态平衡控制器的力矩上限。
v5回到v3的轮子控制参数，改为分开限制整腿方向和腿长：
整腿方向随IMU姿态补偿最多3 rad/s，腿长对应的成对差分仍按站立0.8、推起1 rad/s缓慢改变。
单个根电机参考速度上界是两项之和，推起时至多4 rad/s；不能把它全部标成0.8 rad/s慢速。
所有候选仍保持8秒脚本预算、200 ms交接和1秒稳定门，没有放宽失败判定。

结论：材质分离已实现，但慢自起尚未达到要求。v5仅保留为可复现实验，
不替换默认流程，也不作为恢复课程或长训的准入证据。
当前仍有控制阶段衔接和支撑判断的问题，不能把这些固定控制失败归因于PPO轮数不足。
CPU相关测试159项通过，仅证明上述合同、数学边界和控制/学习归属的实现检查通过。

旧三档随机材质也完成了2秒原生诊断，32个环境的三档人口为12/9/11；
材质绑定和机器人PhysX形状读回通过，不作为行为达标证据。
`random_friction1328_material001`因评测器把`full`误当成具体架构消融而失败，
修复无操作分支并通过4项回归检查后，`material002`实际完成；原失败记录保留。
上述历史阶段的封存回执为
`reports/flat_base_continuation_20261019/low_friction_candidate001/review001.json`。

几何文件和每次源快照、预检、JSON、NPZ及TensorBoard event保存在
`reports/flat_base_continuation_20261019/low_friction_candidate001/` 和
`runs/flat_base_20261019/low_friction_recovery_*`。
事件步数绑定父策略的1328次成功更新；没有启动本机TensorBoard服务。

原生自起诊断命令：

```sh
OMNI_KIT_ACCEPT_EULA=YES /home/noir/miniconda3/envs/isaacsim6/bin/python scripts/probe_flat_recovery.py \
  --plan contracts/flat_base_finetune_low_friction_v1.json \
  --checkpoint runs/flat_base_20261019/pilot028_wheel_rate1328/model_001328.pt \
  --profiles reports/flat_base_continuation_20261019/low_friction_candidate001/recovery_profiles_with_shell.json \
  --lookup reports/flat_base_continuation_20261019/low_friction_candidate001/height_lookup.json \
  --variant slow --include-immediate-control \
  --output runs/flat_base_20261019/low_friction_recovery_recheck001
```

每个场景四条首次episode。额外的±90°场景在reset后立即启动控制，
用于覆盖两种旋转方向；常规倒地场景保留原1秒无主动驱动力矩的落地过程。
所有状态注入仅用于episode初始化。此诊断不授予长训、功能验收或实车资格。
