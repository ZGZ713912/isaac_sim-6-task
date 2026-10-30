# v4 整车训练配置

默认 `WheelLeg-ProneMotion-v4` 使用 `model/wheel_leg_v2.0.2/friction_v1/training`，继承 2.0.2 的全部 21 个刚体、21.44174 kg 质量、惯量、碰撞与六个闭链约束，并加入2026-10-31辨识的腿电机动摩擦初值。原 `model/wheel_leg_v2.0.2/training` 和原父资产冻结保留，供既有运行继续使用；原始 CAD 资产和历史纯底盘 v1/v2/v3 配置保留。只训练全新 actor、critic 和空 Adam；没有旧权重、teacher 或 donor。动摩擦适用范围、未辨识参数及数值实现见 [接入说明](LEG_MOTOR_FRICTION_20261031.md)。

2026-10-31 已把默认整车训练链同步到用户确认的 **40°–110° 大小腿机械内夹角**，电机置零参考为 **110°**。父资产及可动云台派生资产的 USD、机械限位、编码器和气簧几何、主动轴相对角软限位、出生姿态库、站立里程计、训练合同和源文件归档均已重新绑定。原始气簧力节点保留，不把限位修改称作重新辨识。旧 40°–107° 回执只保留历史身份，当前检查记录在 `reports/full_vehicle_limits110_20261031/`。

云台 yaw 无位置限位，pitch 正方向为上仰，机械限位下 30°、上 35°。yaw 每个 episode 独立采样 −π..π rad/s 目标速度，以 π rad/s² 平滑进入；pitch 从零位开始，在 −30°..35°间正弦扫描，周期均匀采样 8..16 s、初始方向随机。两轴采用 wheelbipe V14 的控制结构及参数：yaw 速度 P=0.5、力矩帽 2 N·m；pitch 位置 P=20、D=0.5、力矩帽 10 N·m。参考实现原本固定 pitch，本任务为覆盖俯仰运动而增加平滑扫描。这些是仿真扰动参数，不是已辨识的实车云台电机参数。

云台在每个 1 ms 物理步通过真实关节力矩运动，反作用由 PhysX 计算；不在运行中写关节位置或速度。被动出生阶段和 finished 状态两轴电机力矩为零，运动时钟暂停。底盘仍由 RL 六路动作控制，不增加底盘脚本控制。actor 保持 35D，包含 v4 的静止轮位移与有效标记；critic 为 87D，读取新增两轴的位置/速度，yaw 位置按周期包裹。

CAD 的导轮并未独立建成可转动关节；其碰撞与底板合并为 `base_link` 的 8 个包络。按简化要求，整个这组底盘碰撞采用静/动摩擦 0、组合模式 min，确保与非零地面配对仍为零。驱动轮保留 0.7，其余连杆和云台保留原体摩擦。没有把全车或驱动轮都设成零摩擦。

新几何表重新绑定整车模型，气弹簧保持原测量力曲线及全车安装几何；轮子继续使用原辨识总惯量减去整车 CAD 轮惯量，延迟只加一次。新姿态库重新求解 230/305/420 mm 轮支撑、整车 COM 平衡出生与 103°标准趴地出生；只在 reset 使用，不构成运行中的参考轨迹。域随机化范围、100 Hz 策略/1 kHz 反馈 PD、六动作、原课程与预算保留。

构建入口 `scripts/build_full_vehicle_training.py` 拒绝覆盖已有目录；父资产全部文件与派生配方独立校验。训练源归档包括被引用的父 USD、网格和碰撞，而非仅归档轻量 USD 引用层。

共享环境增加了显式整车分支，因此原气弹簧证据对 `env.py` 的旧源码绑定另存为 `evidence_full_vehicle_integration_20261030.json`，并调整统一基础配置及设计中的绑定。原 `evidence.json`、原生结果和拟合字节保留，新文件引用其 SHA，明确只继承既有组件证据；新整车运行接线另由本次原生检查证明，不把旧测试称为可动云台已验证。

正常训练命令如下；脚本先执行独立的零 PPO 原生检查，通过后才进入一次连续训练：

```bash
cd /home/noir/Documents/workspace/isaac_wheeled_rl_train
bash scripts/train_prone_motion.sh --task WheelLeg-ProneMotion-v4 \
  --python /home/noir/miniconda3/envs/isaacsim6/bin/python \
  --num_envs 256 --max_iterations 10000 --save_interval 250 \
  --domain_randomization curriculum --device cuda:0 --headless
```

需要训练窗口时改为 `--no-headless`；输出目录默认自动使用新时间戳，不覆盖或续训旧运行。仅做接线检查：

```bash
bash scripts/train_prone_motion.sh --task WheelLeg-ProneMotion-v4 \
  --python /home/noir/miniconda3/envs/isaacsim6/bin/python \
  --num_envs 80 --device cuda:0 --headless --native-probe \
  --output reports/full_vehicle_limits110_20261031/training_native001
```

此命令仅做原生接线检查，0 PPO；启动长期训练仍是另一次明确操作。当前 Kaiser v3 按用户要求继续保留，未停止或替换。TensorBoard 新 tag 位于 `Gimbal/`，其余静止里程计、奖励与行为指标仍进入原 RSL-RL event logger；不启动本机 TensorBoard。

重现这次限位迁移时，先运行 `scripts/sync_full_vehicle_knee_limits.py --asset-only`，随后执行新父资产的 `scripts/probe_full_robot_v202.py` 并归档匹配模型身份的组件回执，再运行 `scripts/sync_full_vehicle_knee_limits.py --training-only --component-validation <新回执路径>`。普通构建器仍拒绝覆盖既有输出；迁移入口在重建训练目录时暂存旧目录，失败会恢复。历史回执、测量数据及旧 checkpoint 不修改。

本次同步已通过 180 项集成测试。新父资产在 41°、75°、109° 各执行 1000 个 GPU 物理步，并读取确认 40°–110° 原生关节限位；整车 v4 训练入口在名义域和全强度随机化域分别通过 41 项原生接线检查。检查绑定实现提交 `0c67f0b6`，PPO 更新、优化器尝试和 checkpoint 读取均为零，退出时源文件无差异。完整摘要及文件哈希见 [verification.json](../reports/full_vehicle_limits110_20261031/verification.json)。

工程检查不能证明已学会整车静站、自起、高速或旋转，也不构成实车资格。训练及后续冷启动行为评测仍从零开始，并完整记录未测的既有能力。
