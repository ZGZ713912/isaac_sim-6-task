# v4 标准训练内的多构型翻正与自起

开发分支：`dev/leg-friction-v202-20261031`。当前默认任务与正式入口为 `WheelLeg-ProneMotion-v4`，合同 `contracts/wheel_leg_prone_motion_v4.json` 已直接融合多姿态自起。整个训练仍是一次 `learn(10000)`：30%自起环境改用多初态，其余70%的角色、指令渐进、域随机化和总预算保持。自起不追加另一段训练，也不创建第二个learner或optimizer。

原有标准v4合同逐字节保存在 `contracts/wheel_leg_prone_motion_v4_standard_20261031.json`，供历史复现与旧checkpoint兼容验证。之前实现的v5入口保留为开发历史；新训练直接使用v4。Play以checkpoint内嵌的原始resolved plan为准，不会把新恢复语义强加到旧v4模型。

## 训练行为

同一个六轴 RL 策略连续执行“侧躺/倒扣等状态 → 正向可恢复区域 → 稳定站立”。先只学习正向倒地的四种腿构型，四类分别通过留出初态评测后才加入困难初态，随后始终保留正向自起复习。翻正和起身之间不重置姿态、速度、动作历史或总用时，不指定翻转方向、固定折腿角或电机轨迹。翻正后的腿可以为双前、双后、左前右后、左后右前，直接从实际状态继续自起。

下面的比例只作用于v4原有30%自起环境。设四种腿组合全部通过的更新数为U；未通过时，不会因训练到第2000轮等固定时间点而解锁：

| 阶段 | 正向趴地（四种腿组合） | 其他六类倒地合计 |
| --- | ---: | ---: |
| 尚未通过四类评测 | 100% | 0% |
| 通过后前2500次更新 | 60% | 40% |
| 通过后第2500次更新起 | 40% | 60% |

默认每250次更新，在空rollout边界进行一次确定性冷评测：每种腿组合2个留出构型×2个heading，共16条独立试验；每类至少4条，成功率均须达到100%，完成自起后还须连续站稳3秒。成功率、评测间隔和最少试验数可配置。评测不调用optimizer，也不增加PPO轮数；环境恢复训练采样并刷新观测，actor、critic、Adam和rollout storage身份保持。门槛仅授予课程准入，不代替30秒保持及完整能力验收。若在10000轮内仍未通过，复杂倒地保持锁定，结果明确记为未掌握，不延长或另开正式训练。

v4其他课程仍共用同一更新时钟：0–999主要基础自起与站立，1000开始增加运动/高度请求，5000达到完整高度、6000达到±2m/s、7000达到±100rpm；原域随机化时序保留。上述时间点是预定采样课程，不是已经训练达标的事件。

用户提出的全局次序建议为：URDF名义姿态静站→四种腿组合正向自起→复杂倒地与低速直行/启停→低速转弯及自旋→对应高速动作。高度另按静止变高→低速变高→转弯/自旋变高→高速组合渐进，并持续复习已学技能。这个全局站立优先、逐技能评测解锁的安排尚未替换上述原v4运动时序；当前已落实的是自起内部四类评测门槛。

新初态库 `model/recovery_v202_20261031/pose_bank.json` 包含原有4个参考初态，以及7类车身状态×4类前后腿组合的连续角度采样。每格6个训练样本、2个独立留出样本；另外有24个正向动态起点，完整保存闭链一致的关节速度和身体速度，共252条。它是有限的可行初态库，不是全部静态平衡的穷举。全车 heading 在训练 reset 时随机，云台 yaw 随编译时连续采样的构型保存；失能阶段所有8个电机力矩为零。

编译器从完整新机构求解闭链，验证所有关节限位、气簧行程和地面间隙，并用原始未简化CAD三角面筛查新增非相邻接触。相邻/名义装配接触豁免明确保存在库中。初态在地面上方约2mm温和放置，允许短时失能后的残余运动；不声称每条都是静态平衡。24个动态起点的速度通过闭链Jacobian投影得到。

腿摩擦、物理气簧与气簧补偿、轮子real2sim、原有碰撞材料和零摩擦导向滚轮均沿用v4绑定。采集研究临时使用的非轮摩擦0.7不迁移到训练。30%自起、25%静站和全部前后±2m/s、双向100rpm自旋、四向弧线与高度角色继续保留。

## 时间与冲击

恢复奖励以未完成时间成本为主，辅以有界进度势函数和弱冲击/用力/动作变化代价。权重是首轮工程候选，需要实际训练后用总时间、成功率与冲击联合比较。自起失败的代价覆盖主动恢复时间预算，避免提前失败比继续恢复更有利。取消半圈/秒统一限速，继续保留力矩、膝限位、气簧行程、闭链保护及既有20N·s工程冲击护栏。共享后段的速度参数只用于判断是否进入已覆盖的接续状态区域，并不限制电机速度。

新增 `RecoveryImpact` 在每个1ms物理子步统计底盘、云台、腿、轮子的对地法向合力峰值和固定20ms窗口冲量。先逐刚体积分，再汇总各部件组峰值，窗口跨策略步和翻正/起身边界连续维护。失能碰撞不污染主动控制窗口。正常承重和冲击共同作为风险代理记录，轮子不豁免。

分组 `force_limits_n` 和 `impulse_limits_ns` 默认均为null，因为尚无结构耐受载荷依据；一旦显式配置，超限为失败，不能由速度奖励抵消。已有20N·s护栏是继承的工程判据，也不是新辨识的损伤阈值。当前统计为平地对地接触的刚体合力，不是应力或损坏概率；原生自碰撞响应仍关闭，内部接触仅有初态CAD筛查，不能据此宣称运动全程免自碰撞。

计时从允许电机恢复开始，直到连续稳定站立完成，另记录首次翻正进入后段时间、完整腿角/残余速度和后续起身时间。翻正事件每回合至多一次，不能靠反复侧翻刷奖励。最终评价总时间、成功率和整段冲击；30秒静止保持继续从实际自起完成事件之后统计。

## 命令行

配置预检不启动Isaac或PPO：

```sh
scripts/train_prone_motion.sh --task WheelLeg-ProneMotion-v4 --preflight-only --run_name recovery_v4_preflight
```

后续新训练命令示例（本次实施不运行它）：

```sh
scripts/train_prone_motion.sh --task WheelLeg-ProneMotion-v4 --num_envs 256 --max_iterations 10000 --seed 105 --save_interval 250 --domain_randomization curriculum --device cuda:0 --headless --run_name recovery_v4_seed105
```

当前v4强制`--recovery-curriculum staged`，不能用`all`绕过四类门槛；`all`仅保留历史未设门槛任务的兼容语义。`--recovery-gate-success-rate`、`--recovery-gate-every`、`--recovery-gate-min-trials`配置门槛；试验数不可超过留出构型与heading能提供的独立试验数。`--recovery-impact-window-ms`指定整数物理步的窗口。分部件阈值通过`--recovery-force-limits-n`和`--recovery-impulse-limits-ns`传入含`chassis/gimbal/legs/wheels`四个键的JSON对象，null表示尚未指定；单位分别N、N·s，数值应来自结构依据。

Play支持 `--scene prone --fallen-family roof_down --leg-class front_rear` 等组合，只读融合后v4或原型v5的checkpoint；旧checkpoint继续使用其原任务与初态。正式训练始终全新actor、critic、空Adam，不提供checkpoint、teacher或donor加载入口。

用户另行授权的续训验证通过独立脚本`scripts/validate_recovery_continuation.py`执行，默认`--updates 0`只做四类前置冷评测；先逐值验证actor、critic与Adam恢复，四类通过才允许复杂倒地续训。`--allow-belly-pretraining`明确允许未通过时继续练正向；验证预算必须小于10000，所有产物标记`continuation_validation_only`且不能进入最终正式训练。`--allow-friction-migration`仅允许已有迁移回执认证的旧无腿摩擦v202父模型进入新摩擦环境，并要求重新评测；其他机构/观测变化均拒绝。配置预检加`--preflight-only`，不启动Isaac或PPO。

## 验证与数据

`scripts/probe_multi_pose_recovery.py --output <新目录> --device cpu --wiring-suite` 独立执行原生PhysX初态/接线检查，默认遍历全部252条初态及名义/训练requested两组分布，不构造策略、learner或optimizer，不向现有训练进程发送信号或修改其源码。它检查完整状态读回、初始闭链、实际失能电机零力矩、摩擦只加一次和新增指标；`--wiring-suite` 另外检查训练入口的40项原生接线判据。GPU检查使用 `--device cuda:0 --performance-backend cuda_graph`。短时测试动作不代表学会翻正或自起；requested仍保留训练规定的25%名义样本份额，不是全随机化留出行为评测。

上一版v5原型的164项定向测试及CPU/GPU接线检查回执保存在 `reports/multi_pose_recovery_20261031/`，属于该版源码的证据。融合后的v4验证另存 `reports/multi_pose_v4_integration_20261031/`；不将旧回执当作改动后的实测结果。固定测试动作触发的终止全部保存在逐初态回执中，不能将接线通过解释为所有初态均能完成自起。

早期 `native001` 的重复reset推理张量上下文异常已修复；`native003` 将原有站立初态的双轮接地修正误判为位姿读回错误，已修正检查口径，最终回执另行重跑。离线编译器从保留的标准v4快照构造初态，避免从当前已经融合的v4输出库重复追加生成；编译器与库SHA均重新绑定。

固定冷评测保留原22个能力场景，并增加56个留出恢复初态，共78个场景；评测环境数需要至少78。每个落面×腿组合分别保留分母、失败/未完成和成功子集的时间中位数/P95，记录翻正和总时间以及分组力/冲量，不能用成功子集快轨迹掩盖失败。

全部训练指标经既有RSL-RL logger进入TensorBoard，新增 `Recovery/<family>/<leg_class>/...`、`Recovery/impact/<part>/...`、`RecoveryMastery/<leg_class>/success_rate`和`RecoveryMastery/complex_unlocked`。事件tag使用每策略步事件数，时间tag保留完成事件时间之和及事件分母。没有新增本机TensorBoard服务。

重编译初态需要离线几何依赖numpy/scipy/trimesh/python-fcl，训练本身不需要python-fcl。编译器拒绝覆盖已有库；有意重新生成时应另存新路径、更新合同SHA并重新做原生检查。原始CAD、脚本和生成库哈希均进入源冻结清单。

本轮实施与原生接线检查不授予策略能力资格。全部13项长期能力继续按 `contracts/wheel_leg_acceptance_requirements_v2.json` 逐项验收，未测项目保持未测，实车资格保持未授予。
