# WheelBipe 平地与跳跃混合课程

`foundation-jump-v1` 在 `foundation-v2` 的 v2.0.2 物理模型、50 Hz 策略、200 Hz 物理反馈、六轴动作和 35/78 维 actor/critic 接口上加入主动起跳与腾空落地。原 `foundation-v2` 入口继续保留。机器可读定义为 `contracts/wheelbipe_foundation_jump_v1.json`。

第三方来源清单中三份说明保存在 `docs/third_party/wheelbipe/`。`provenance.json` 的 `documentation_locations` 只改存放位置，仍校验原始快照 SHA-256，不跳过文档或 Python 源文件的认证。

## 样本配额

|任务|环境比例|
|---|---:|
|站立|15%|
|前进|15%|
|后退|15%|
|连续变高|10%|
|转向|5%|
|冲刺|5%|
|双方向自旋|5%|
|主动跳跃|20%|
|腾空落地|10%|

最大余数法一次分配固定环境组，至少需要 20 个环境。每次重置保留组别；命令采样和特殊模式不能再次覆盖配额。前进和后退组各有一半速度采样机会留给 0.1–0.5 m/s 低速域，其他速度沿用原阶段上限。站立组跨高度取样并每次保持 10 秒静止指令，运动指令保持 5 秒，变高组使用 0.25 Hz 连续高度命令。转向按两个独立符号覆盖四种弧线；冲刺组交替训练双向冲刺与零速制动，自旋训练两个方向。自旋从原后期课程范围逐步达到 100 rpm 附近及以上。

每组中仍会出现重置、失效、准备和接稳样本。20% 主动跳跃环境比例不等于 20% 离地瞬间样本；`Mixed/jump/request_active_envs` 与物理事件计数专门报告请求与实际起跳。普通组继续使用原多姿态地面重置课程；这不代表已通过完整自起验收。

## 请求与奖励

主动跳跃由计时请求给出下蹲、弹道蹬伸参考和目标顶点。腾空组只在 reset 提高出生位置。运行中六轴均由策略控制，保留原 PD、气簧、轮端动态、限幅、摩擦和随机化；没有起跳外力或状态强制写入。

已有七个控制槽位保持维度。普通任务的槽位数值保持原语义；跳跃请求使用 `jump_request`、目标顶点和请求计时。真实接触、离地、支撑和成功状态仅用于奖励与诊断，不能推进 actor 的命令时钟。轮地间隙由模型中两只圆柱的真实原点、方向、半径和厚度计算。

普通任务全部保留原奖励。跳跃请求窗口内替换固定站高和抑制竖直速度的冲突项，其余姿态、XY/yaw、能耗、动作平滑、安全终止和不可达腿目标代价保留。新增密集奖励乘 0.5：理想高度项上限 2/s，蹬伸速度项上限 3/s，飞行轮位置项上限 2/s。密集项按原 logger 乘一次 policy dt。主动离地 +0.5、首次落地 +0.25、接稳成功 +1.5、超时 −1.5 为单次事件；先除以 policy dt 再进入原计费路径，避免按步长重复缩放。成功须有真实离地、目标顶点、双轮支撑及连续接稳。

这些尺度是可审查的工程初值，尚无学习收敛或梯度干扰结论。保持原 PPO 的全局优势标准化，记录各组实际 rollout 数、奖励/回报/优势 RMS。应结合 TensorBoard 及独立行为回放判断，不能凭原始 reward 大小宣布配比最优。

## 入口与验证

训练入口仍为 `scripts/train_wheelbipe_port.py --training-profile foundation-jump-v1`，并通过现有 WheelBipe managed trainctl 创建独立会话。`--checkpoint` 仍限只读评测，`--resume` 仍校验同课程完整学习状态。普通启动从全新 actor、critic 和空 Adam 开始。

明确授权的跨课程试验另用 `--continue-from FULL_DURABLE_CHECKPOINT --motion-update-offset 20000`，并由当前冻结源中的 `src/wheeled_algo/managed_training.py init` 创建新会话。入口只接受完成 20000 更新的 foundation-v2 完整学习状态；校验原会话、检查点及 sidecar pin，恢复 actor、critic、Adam、学习率、归一化器和全局随机状态，逐项验证指纹。目标必须保留原 1024 环境及指定物理/域随机化参数，不能使用模型权重文件冒充完整续训。

新增预算与日志从零计数，报告另记来源的 20000 更新。运动课程采用 20000 更新偏移；跳跃难度从新增更新零开始。物理 episode、历史和混合课程独立随机流重新初始化。后续同会话冷恢复仍由严格的 `--resume` 保存和恢复这些新课程状态，不再次迁移旧模型。每个保存间隔另存完整 `update_*_review.pt`，可只读评测而不暂停训练。

`contracts/wheelbipe_jump_trial_v1.json` 定义试验的配对退化容差和跳跃成功判据。七组固定旧命令之外，三个 seed 的混合评测覆盖普通任务；主动跳跃和腾空出生分别按完整 episode 统计。通过只允许启动新的从零工程实验，不授予完整十三项能力或硬件资格。缺少证据、起跳不足、接稳不足或旧任务退化时，不能自动转入从零训练；人工暂停/停止优先。

`scripts/audit_wheelbipe_behavior.py` 支持该 profile，固定旧命令回放时关闭混合采样，便于与 `foundation-v2` 对照。`scripts/validate_wheelbipe_mixed.py --checkpoint PATH --output NEW_DIRECTORY` 在真实仿真中运行零 PPO 的混合课程、记录配额、状态有限性、奖励和物理事件。它不会创建 optimizer。`MixedRollout/<task>/samples` 是 PPO rollout 张量的真实数量；`finite_physics_samples_per_step` 只诊断物理数值有效性。

短时间零 PPO 验证不证明策略已经学会跳跃，也不替代 `contracts/wheel_leg_acceptance_requirements_v2.json` 的逐项验收。旧模型通常尚未学会新跳跃上下文。
