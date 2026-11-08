# Wheelbipe V14.2 与 v2.0.2 训练入口

本入口将冻结的 Wheelbipe V14.2 环境、奖励和 PPO 配置连接到本仓库的
v2.0.2 整车 CAD、实测气簧与执行器模型。代码位于 `main`；训练入口是
`scripts/train_wheelbipe_port.py`，使用 `--training-profile foundation-v2` 选择正式课程。

## 源码与职责

- `third_party/wheelbipe/`：原项目源码快照、许可证与逐文件 SHA-256 来源清单。
  `authenticate_source()` 校验后才将该目录加入 Python 导入路径。
- `src/wheeled_tasks/wheelbipe_port/`：关节/刚体名称映射、非对称闭链复位、
  机械边界内的腿目标投影、物理气簧、轮响应和命名域随机化适配。
- `contracts/wheelbipe_foundation_repair_v2.json`：奖励覆盖、课程阶段与固定预算。
- `contracts/wheel_leg_randomization_study_v2.json`：随机化分组、范围和固定参数。
- `model/wheelbipe_port_v202/`：复位运动学查表、碰撞支撑范围与本地平地资产。
- `src/wheeled_algo/managed_training.py` 和 `durable_checkpoint.py`：受管训练、完整学习状态保存、
  原预算恢复和进程身份认证。控制方式见 [trainctl](TRAINCTL.md)。

Kaiser 原训练的源码标识为 `a603a7c9a5b4fa6d94e7b6846c944a42cdba8391`；
仓库历史清理后的对应提交为 `a29ca83460728f5b552f9ad51de3fd87f4bb6252`。
该代码来源标识与检查点的已完成更新数是不同概念。

## 环境与控制合同

本入口要求 Isaac Sim 5.1、RSL-RL **3.0.1**，不能与其他入口使用的
RSL-RL 5.5.1 配置混用。运行前通过 `--help` 查看当前支持参数。

| 项目 | foundation-v2 配置 |
|---|---|
| actor / critic / 动作 | 35 / 78 / 6 |
| 策略频率 | 50 Hz |
| 物理与逐物理步反馈 PD | 200 Hz，显式 `--physics-hz 200` |
| actor 与 critic 隐层 | 256、128、64，ELU |
| 腿位置动作尺度 / 轮速度动作尺度 | 0.5 rad / 10 rad/s |
| 腿 Kp / Kd | 60 N·m/rad / 2 N·m/(rad/s)，显式参数 |
| 每个主动腿轴附加惯量 | 0.015795 kg·m² |
| 腿 / 轮电气限矩 | 40 / 4.5 N·m |
| PhysX 位置 / 速度迭代 | 64 / 32 |
| 物理气簧 / 电机气簧抵消前馈 | 保留 / 关闭 |
| PPO rollout / epoch / minibatch | 24 / 5 / 4 |
| gamma / lambda / 初始学习率 | 0.99 / 0.95 / 0.0001 |

actor 输入包含命令、高度、IMU、腿位置、置零的轮位置、关节速度、上次动作及控制模式。
该版本没有驻停累计位移输入。所有六个功能轴持续由策略控制。
奖励中的力矩和功率按电机电气输出计费，物理气簧和被动摩擦独立施加。

当前模型资产遵循[统一碰撞几何](CHASSIS_COLLISION_GEOMETRY.md)。
历史检查点的严格复现还须使用其冻结资产、执行器参数和合同；源码一致本身不证明物理轨迹等价。

## 新训练

先提交并冻结已验证源码，在同一冻结目录完成配置检查和原生短测，再启动正式训练。
每次使用新的输出目录，常用参数通过 CLI 指定。以下命令展示与 foundation-v2 匹配的配置：

```bash
trainctl run --name wheelbipe-new -- /path/to/isaac51/bin/python -u \
  scripts/train_wheelbipe_port.py \
  --mode train --task flat --num-envs 1024 --iterations 20000 \
  --save-interval 500 --seed 119 --device cuda:0 --headless \
  --training-profile foundation-v2 --physics-hz 200 \
  --leg-armature 0.015795 --leg-kp 60 --leg-kd 2 \
  --leg-target-projection-penalty 1 \
  --solver-position-iterations 64 --solver-velocity-iterations 32 \
  --no-external-forces-every-iteration \
  --randomization-profile requested --randomization-strength 1 \
  --output .codex/artifacts/wheelbipe-new
```

预检使用 `--mode preflight`，通过 `--steps` 指定短测长度；它不执行 PPO。
小规模工程准入不能替代正式人口的原生验证或行为验收。

正式固定预算编排入口为 `scripts/run_wheelbipe_repair_campaign.py`。
它要求同冻结源码、同人口的工程准入 receipt 和 ONNX parity 合同，
创建受管会话，并在预算结束后执行导出与诊断。用其 `--help` 查询参数。
受管会话已有控制目录时，保持其原生控制协议，不再嵌套第二层进程控制器。

## 学习状态与评测

普通新训练从全新 actor、critic、空 Adam 开始。`--checkpoint` 仅用于
`--mode evaluate`；显式 `--resume` 用于同合同中断恢复，使用新输出目录，
保持原总预算。磁盘恢复保留学习、随机和课程状态，不能声称恢复了原仿真物理现场。

`foundation-v2` 根据已完成更新数推进：0、2000、6000、12000；
四阶段静止采样比例分别为 0.5、0.35、0.3、0.25，始终覆盖 0.23–0.42 m 高度。
行为诊断不自动暂停、退出或缩短 learner 预算；手动控制与预算自然完成保留。

评测与导出入口为 `scripts/audit_wheelbipe_behavior.py` 和
`scripts/export_wheelbipe_port_onnx.py`。训练标量统一写入现有 TensorBoard event；
检查点、导出包、诊断 receipt、日志和图表放在 `.codex/artifacts/` 或 `.codex/report/`。
Kaiser 的 TensorBoard 服务在远端读取 event，本机只做 SSH 转发。

能力验收按 [轮腿能力要求](WHEEL_LEG_ACCEPTANCE.md) 逐项执行。
工程预检、ONNX 一致性与训练完成均不直接授予行为或实车资格。
