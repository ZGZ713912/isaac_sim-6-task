# 混合腾空与蹬腿跳跃训练

独立分支 `dev/wheel-leg-jump` 从统一主线 `52a13b96` 创建。入口是
`scripts/train_jump_motion.sh`，任务是 `WheelLeg-JumpMotion-v1`，合同是
`contracts/wheel_leg_jump_motion_v1.json`。本分支使用同一完整 v2.0.2 机械模型、
腿摩擦、气弹簧与补偿、轮响应、100 Hz 策略、200 Hz 物理与反馈 PD、64/32 求解器。
新训练从全新 actor/critic/空 Adam 开始。

参考 SCUTRobotLab `wheeled-legged_RL`，提交
`0b62556b34977cc629fb5767e1cda9daa0d4bf36` 的 wheelbipe V14 腾空重置配置、
`airborne.py` 与 `jump_takeoff.py`（MIT；原文件哈希在新合同内）。移植的是混合重置、
弹道蹬伸参考、收腿/轮心对齐、落地缓冲奖励与接触滞回设计；参数按本车模型单独定义。

每10个环境中6个保持原课程分配，2个空中出生，2个从接地站立发起主动跳跃。
原课程的站立/自起/移动能力解锁保持原门槛，跳跃课程独立出现。
跳跃样本持续6秒，空中高度与目标顶点随前3000次已完成 PPO 更新逐渐增大。
空中出生只发生在 reset；所有实际物理间隔仍由同一个六轴 actor 控制。
状态机只改变指令、奖励和终止判定，没有外力助跳、脚本力矩、关节目标接管或步内位姿写入。

主动跳跃经过等待稳定、下蹲、蹬伸、实测飞行、落地和连续稳定。
起跳必须双轮轮缘净空超过8mm、竖直支撑小于车重10%，连续15ms，且车体世界竖直速度
至少0.2m/s；空中重置不会计入主动起跳。落地至少记录30ms真实飞行与10ms接触，
成功还要求双轮连续稳定0.3秒、目标顶点误差不低于目标25mm、无机械/碰撞失败。
原闭链、机械行程、电机限幅、冲量和空间边界保护保留。
普通站立的高身高/竖直运动奖励仅在跳跃样本中替换为跳跃目标，不会把合法腾空判为站立掉落。

输入仍为35维。预留槽28/29/30/32/33/34提供普通/空中/跳跃模式、参考竖直速度、
目标顶点增量和指令时钟；31仍为站立里程计有效位。跳跃样本的14/15/31清零，
避免空中轮自转产生站立位置误差。没有向 actor 增加模拟器高度、速度、接触或净空真值。
这是新的指令语义，已有策略不能静默迁移到这个任务。

```bash
# CPU配置检查
scripts/train_jump_motion.sh --preflight-only --output /tmp/jump_preflight

# 原生两域0-PPO检查，完整证据通过后才能启动训练
scripts/train_jump_motion.sh --native-probe --output /tmp/jump_native

# 正式新训练；所有常用参数沿用统一入口CLI
scripts/train_jump_motion.sh --num-envs 256 --max-iterations 10000 \
  --seed 261103 --save-interval 250 --headless --run-name mixed_jump_seed261103

# 原进程暂停/继续；使用独立tmux保持训练生命周期
trainctl --host kaiser
```

入口默认接入 trainctl。TensorBoard 使用既有 RSL-RL event，新增 `Airborne/*`、
`Jump/*` 与 `JumpReward/*`。事件 tag 单位是每策略步事件数，速度 m/s、净空/高度 m、
力 N、时间 s；TensorBoard step 是已完成 PPO 更新（一次参数更新，不是一次 episode）。
`progress.json` 的 `jump_training` 保存从本次训练开始累计的空中出生、请求、主动起跳、
落地与两种成功计数。混合 episode 长度不能作为站立生存时间。

原生测试与首 PPO 审计只确认接线和优化器工作。腾空、起跳高度、缓冲、连续落地成功率及
全部既有能力仍须分场景实测，未测不算通过。实车跳跃没有执行或获资格。

只读播放使用本任务自己的检查点：

```bash
bash scripts/play_prone_motion.sh --checkpoint runs/jump_motion/RUN/model_00250.pt \
  --scene jump --num-envs 8 --seconds 6 --headless
bash scripts/play_prone_motion.sh --checkpoint runs/jump_motion/RUN/model_00250.pt \
  --scene airborne --num-envs 8 --seconds 6 --headless
```

本次分支开发不授予旧能力的新验收：R01标准自起、R02多姿态自起、S01静站、
M01前后直行、M02自旋、M03弧线、M04驻停、H01静态变高、H02移动变高、
H03转向/自旋变高、B01扰动恢复、Q01噪声下腿稳定、D01全域随机化均待独立测定。
