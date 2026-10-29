# 成功平地策略的 ONNX 权重恢复

输入文件：`/home/noir/Downloads/wheel_leg_flat_base.onnx`。
SHA-256：`a00d13df1d4113fe949c2a6a15b508e5c491fd1a5e6881570bf025a8d3c29c29`。
用户确认策略频率为 **50 Hz**，即 20 ms；频率不在 ONNX 图中。

## 已恢复的内容

- 输入 `obs`：float32，`[batch,35]`；输出 `actions`：float32，`[batch,6]`。
- 完整 actor 均值网络：`35 → 256 → 128 → 64 → 6`，三层 ELU，最后一层线性。
- 四组 weight/bias 共 8 个张量、50,758 个参数，全部保留原始 float32 数值。
- 图中没有归一化或动作缩放节点；不能据此排除图外预处理和后处理。
- **文件没有 critic、探索标准差、Adam 状态、训练步数或 reward 定义。**

恢复产物在 `models/wheel_leg_flat_base_20261018/`：

| 文件 | 用途 |
| --- | --- |
| `source.onnx` | 原文件的字节相同副本 |
| `actor_mean.pt` | 可训练的 actor 均值 state_dict；不是完整 PPO checkpoint |
| `actor_mean.torchscript.pt` | 可直接推理的恢复网络 |
| `inspection.json` | 网络结构、逐张量摘要、独立引擎数值对照 |
| `rsl_import_verification.json` | 实际权重装入本机 RSL actor 的验证 |
| `interface_audit.json` | 本机参考接口与现有训练环境的差异 |
| `reference_deployment.yaml` | 本机部署配置的快照，作为候选接口证据 |
| `context_sensitivity.json` | 假设传感器快照下，对最后一维输入变化的网络响应 |

ONNX Runtime 与 PyTorch 对照共 1,624 条输入。单位尺度随机输入最大绝对误差为
`5.7220458984375e-6`；大尺度压力输入的最大误差为 `3.662109375e-4`，使用随输入尺度变化的
float32 容差。所有恢复参数逐张量精确相等；这是网络数值验证，不是仿真行为验收。
本机 stock RSL `MLPModel` 加载这 8 个张量后也通过同样的引擎对照，新的探索参数及 RNG 未被修改。

## 观测和动作的候选含义

ONNX 只声明张量形状，不能从权重唯一反推出每个数字的物理含义。
本机 RMCS 的 `wheel-leg-infantry-rl.yaml` 与该文件同时匹配以下两个 metadata 字段：

- `leg_winding_version = paired_branch107_v1`
- `leg_target_model_spec_sha256 = 0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997`

但该 YAML 所指 ONNX 与用户文件字节不同，所以它是**待行为核验的候选接口**，不能当作已认证的原训练配置。
按这份配置，输入索引为：

| obs 索引，从 0 开始 | 候选含义 |
| --- | --- |
| 0–2 | 前进、横移、yaw 速度命令 |
| 3 | 高度命令（m）× 5 |
| 4–6 | 机体角速度（rad/s）× 0.5 |
| 7–9 | 机体系投影重力 |
| 10–13 | 左髋、左膝驱动、右髋、右膝驱动相对名义角的位置 |
| 14–15 | 两轮位置槽，固定为 0 |
| 16–21 | 四腿驱动和两轮速度 × 0.1 |
| 22–27 | 上一帧的 6 个动作 |
| 28–34 | 平地固定上下文 `[1,0,0,0,0,0,0]` |

候选动作顺序为左髋、左膝驱动、右髋、右膝驱动、左轮、右轮。
四腿输出乘 `0.25 rad` 后限幅到 `[-0.75,0.75] rad`，作为名义角偏移；两轮乘 `10 rad/s`
后限幅到 `[-90,90] rad/s`。名义腿角为 `[-0.42,0.13742282595395358,0.42,-0.1374155762580851] rad`。
部署侧还存在 paired-107 winding 与 hinge 约束，不能仅复制六个输出尺度就认定控制等价。

## 与当前本机训练的差异

当前 `pilot020` 仍是 200 Hz policy、200 Hz physics，使用
`model/纯底盘_v6_232mm_stop105/urdf`，model spec SHA 为
`7429f6e59369e590e0fd96cba8b45fd60c362a0231c4bcc2e80a4c4688eadd88`，与 ONNX 声明的模型不同。
50 Hz policy 本身不决定物理步长或 PD 频率，这些不能从 ONNX 恢复。

此外，当前 `build_manual35()` 将 `reference.elapsed.clamp(0,5)` 写入 obs[34]，
与候选部署配置的平地固定 0 不同。假设的静止传感器快照中，仅改变这一维即可令某个原始动作相差约
`0.998`；这说明输入含义必须对齐，不证明哪一份外部配置属于原始训练。

## 续训接入方式与完成边界

`src/wheeled_algo/onnx_actor_import.py` 的 `load_actor_mean()` 可以把恢复的均值参数严格装入
现有 RSL MLP actor，拒绝层结构、激活、输入归一化、参数集合和形状不匹配的目标。
它保留由新训练显式初始化的探索参数，不构造 critic 或 optimizer，也不伪造原训练轮数。

恢复命令（输出目录必须不存在）：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python scripts/recover_chassis_actor.py \
  /home/noir/Downloads/wheel_leg_flat_base.onnx \
  --output models/wheel_leg_flat_base_20261018 --policy-hz 50
```

后续训练应建立一条新的 actor 迁移谱系：保持 50 Hz，对齐观测/动作和机器人坐标及约束，
先以原 actor 均值在仿真中重放静站、前后移动、转向和启停。确认原有平地能力保留后，
先固定 actor 估计新的 critic，再以较小学习率和受控探索开始 PPO，保留原平地任务复习。
缺失的原 critic/Adam 无法作为“完整状态恢复”；当前恢复包也不会被伪装成旧的 fullseal checkpoint。

本轮没有启动仿真、PPO 或实车。6 项定向 CPU 测试通过，覆盖权重恢复、跨引擎输出、反向传播、
真实 RSL actor 导入，以及错误图结构/归一化/参数集合的拒绝。
