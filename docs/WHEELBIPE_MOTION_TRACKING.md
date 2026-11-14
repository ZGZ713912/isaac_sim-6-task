# V10 前后速度跟踪与驻停

`foundation-jump-v10` 使用 `contracts/wheelbipe_foundation_jump_v10.json`，开发分支按 `V10-motion-tracking` 命名。它基于 V9，保留 14 种任务的配额、课程时钟、地形、跳跃序列、real2sim v3.2、TGS 64/0、200 Hz 物理反馈、50 Hz 策略、腿 PD 80/1.5 和执行器边界。正常实验从全新 actor、critic 与空 Adam 开始，总预算仍为 20000 个 PPO 更新。

本版本修改普通有符号平移目标、actor 的时间上下文和方向诊断。配置、数学测试和原生接线通过均不表示训练后的前进、倒退或驻停已通过行为验收；增加环境数或更新次数也不提供这种保证。

## 普通平移目标

速度 `vx` 是世界水平速度投影到归一化的车体水平朝向；正方向与原命令一致。它只用于奖励、诊断和独立评测，不作为 actor 输入。误差 `e = vx - vx_cmd`，新奖励为每秒密度：

```text
motion_tracking = exp(-(e / 0.20 m/s)^2)               weight +2
motion_error    = sqrt(1 + (e / 0.25 m/s)^2) - 1       weight -1
```

两项对正反方向对称。误差成本在远离目标时仍有斜率，不使用原有 ±1 m/s 误差截断，也没有名为 sigma 实际却乘在误差上的隐含缩放。原入口统一乘一次 policy dt；这些项内部不再乘 dt。原 `stand_still_lin_vel`、V9 位置保持和位置漂移奖励继续生效。

普通工况在出生 0.3 s 后使用新项，同时将旧 `track_lin_vel_xy` 和 `track_lin_vel_xy_square` 置零，避免重复计费。出生宽限、有效跳跃/空降请求及 V9 下台阶穿越窗口保持旧平移项，且不支付新项；其跳跃、落地和下台阶专用奖励沿用 V9。`stand_still_deadzone_threshold` 在 V10 收紧到 `1e-6 m/s`，避免将非零的小速度要求当作停车惩罚。

## 175/78/6 观测与动作合同

actor 输入为五个现有 35D 帧，顺序为 `[t, t-20ms, t-40ms, t-60ms, t-80ms]`，展平后为 **175D**。每帧继续包含命令、高度要求、带既有噪声/延迟的 IMU、关节位置和速度、上帧动作、既有控制上下文；裁剪和缩放在入历史之前完成。历史为网络提供轮腿与姿态的时间变化，不能将其宣称为已经验证的车体速度估计器。

critic 仍为 **78D**，动作仍为 **6D**，网络隐藏层继续为 256/128/64、ELU。真实平移速度、位置锚点、接触、地形和球状态不能进入 actor 历史。父类的辅助 `policy_hist` 保持原单帧配置，实际 actor 读取 `policy`。

历史每个策略 tick 只推进一次，重复读取观测不追加时间；单个 lane 复位后，用该 lane 第一帧有效观测填满五帧，不保留旧回合历史。原进程驻留暂停自然保留历史；显式冷恢复沿用原合同，重新开始物理回合及观测历史。

V1–V9 的 35D 检查点不能初始化或恢复成 V10，不复制权重、不补零或重复输入冒充迁移。训练、只读评测和 ONNX 导出按 profile 检查输入维度。V10 ONNX 输入为已经预处理并堆叠的 175D 张量，部署端须按相同 20 ms 时钟、帧顺序和复位填充规则构造历史。导出合同记录这些规则；不自动修改实车 RMCS。

DirectRLEnv 在 done/reward 后先复位完成的环境，再计算返回观测；历史清除与首帧填充遵循该顺序。[Isaac Lab Direct 工作流说明](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)

## TensorBoard 与独立检查

原训练 event 中新增 `Motion/<stand|forward|reverse>/<low|mid|high>/*`。高度分组为 `<0.27 m`、`[0.27,0.37) m`、`>=0.37 m`，按高度命令划分；方向按当前有符号命令划分。记录项包含：

- `sample_envs`：当前有效 lane 数；零表示该组没有测量，不能算作零误差通过。
- `command_mean_m_s`、`actual_mean_m_s`、`signed_error_mean_m_s`、`error_abs_mean_m_s`、`wrong_direction_fraction`。
- `signed_error_sum_m_s`、`error_abs_sum_m_s`：跨 rollout/更新汇总时，以这些分子除以 `sample_envs` 的同窗口和或均值，得到按样本数正确加权的误差，避免空组/变动人数造成抵消。
- `wheel_target_proxy_mean_m_s`、`wheel_speed_proxy_mean_m_s`：轮速目标和轮速测量的有符号折算，用于区分策略目标偏差和执行器跟踪偏差。它们是诊断代理，不能替代车体速度或直接等同于打滑。

上述训练诊断在自动复位之前采样，排除出生宽限、专用请求/下台阶窗口与数值安全复位样本。奖励支付项继续由原 RSL-RL logger 写入同一训练 event。

只读 `motion` 套件包含 39 格：高度 0.23/0.305/0.42 m × 指令 0、±0.05、±0.1、±0.25、±0.5、±1、±2.7 m/s。每格固定指令运行 10 s，并分别报告实际速度、有符号偏差、MAE、反向运动占比、驻停漂移、姿态及失败；默认 nominal 和 requested strength=1 分开测试。动作转换、变高、自旋、跳跃及地形保持原回归/retention 入口，39 格平移套件不能代替完整长期验收清单。

```sh
python scripts/train_wheelbipe_port.py --mode preflight \
  --training-profile foundation-jump-v10 --iterations 20000 \
  --num-envs 64 --steps 150 --physics-hz 200 --headless \
  --output .codex/report/<run>/preflight

python scripts/evaluate_wheelbipe_scenarios.py --suite motion \
  --training-profile foundation-jump-v10 --checkpoint <V10-checkpoint> \
  --output .codex/report/<run>/motion
```

正式训练先提交、冻结源码并使用 `trainctl wheelbipe` 管理；本版本的实现或验证不自动启动正式训练，也不热改既有运行。`scripts/watch_wheelbipe_retention.py --training-profile foundation-jump-v10` 支持相同 ABI 的独立检查点回归，监测与 TensorBoard 不承担训练生命周期。
