# 本机 200 Hz 基础运动命令复习候选

本文件定义本机 v5-2176 → v6 的一次命名全状态接续。当前模型八场景仅反向转向通过，前进 +0.5 m/s 实速 −0.0109 m/s，反向 −0.5 m/s 实速 −0.0509 m/s，三档高度仍集中在 0.31 m。长训门关闭。

原课表在 360 更新已达到 ±5 m/s、±18.85 rad/s。credit_probe001 在 2176 时钟进行了 12×96 帧、512 环境的原生 Gaussian/GAE 采样，0 PPO，actor/critic/17 Adam 均不变。跨 1 s warmup 的高难命令（|vx|>1 或 |yaw|>4）占样本 25.7524%、advantage 平方和 73.3731%；基础命令（|vx|≤0.5 且 |yaw|≤1）占 59.1808%/18.3337%。这是描述性证据，不能证明降低难度一定改善学习，也不能把同一环境时间样本当独立重复。

唯一训练变化：新命名的 locomotion-difficulty-local-rehearsal-v1 保留原 0/120/240/360 节点、domain_actor_updates 时钟、origin=0、采样时缩放算法和逐节点运动高度范围，仅把各节点 vx/yaw 上限设为 0.5 m/s、1 rad/s。不做命令限幅堆积，原采样分布按原机制缩放。全部场景、组数/人口/概率/高度采样、静站高度锚点、启停相位和固定评测保持。基础跟踪正常后，扩大命令范围需要另行声明，不能把本候选说成完整高速任务。

精确接续 pilot011 最终 checkpoint/合同/seal/completion，绑定原 pilot011 结果和 credit_diagnosis001 的 SHA。只允许 locomotion_difficulty 及明确的 plan/design/budget 引用变化，其他合同字段逐项相同。旧 v1 难度及 v4→v5 出生迁移保持；新规则不能让旧合同冒用，v6 禁止从零启动，首次命名块最多 128 更新。actor/critic/17 Adam/学习 std、CPU/CUDA/环境 RNG、学习/课程/消费时钟完整恢复，无 critic 预热。RNG 恢复后创建新 episode，不宣称模拟器中途重放。

保留已消费 3264 = 1088 旧实验 + 当前谱系 2176；谱系上限 18912、全部实验 20000。128 完成应谱系 2304、全部消费 3392、余 16608；任何失败不退款。保留 solver64/32、3 mm 门、200 Hz 物理/策略、PD160/2.5、全部力矩/响应/DR、非名义出生首帧100 ms渐入、conditioning mask、capture1.5s、奖励/entropy0/原生可学习 Gaussian 和其他课程。

工程准入依次为：针对性 CPU 与负例、独立 target design、512 原生 0PPO 采样。原生对照同一2176checkpoint/RNG：源 v5 的初态与 Gaussian 首帧应精确复现 credit_probe001；候选 v6 的出生 pose/q/dq/profile 及 Gaussian 标准噪声应相同，仅可见命令及其后策略/动力学可不同。验证实际命令包络≤0.5/1、原高度范围及 mask，actor/critic/Adam 不变。候选的 0PPO 信号变化只授工程短训，不授行为资格。

通过后从已提交冻结源码追加128真实PPO，独立initial_state逐值核验actor/critic/17Adam与父模型相同（Adam43520），首PPO核验有限且推进（Adam43540，全部时钟2177）。完成且确认退出后，以原八场景/种子/时长/1s warmup/阈值对原初始、v5-2176、新v6-2304复测，保存全部失败与traces；旧候选必须复现。原RSL event比较匹配窗口的六轴std、实付成本、回合秒数及按结束回合/环境秒归一的机械保护。只有实际站立和前后跟踪正常、机械无系统异常后才可长训。
