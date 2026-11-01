# 中断训练恢复

2026-11-01 用户明确要求恢复本机意外退出的训练。普通训练仍默认从全新
actor、critic、空 Adam 开始；只有显式 `--resume` 才加载中断训练的完整状态。
历史合同中的禁止续训文字描述旧授权，不代表 PT 缺少优化器状态。2026-11-01
统一分支新增的默认 v4 使用物理与反馈 PD 200 Hz、策略 100 Hz，包含多姿态
自起和全局评测解锁；这与旧 1 kHz 运行是不同的物理配置。

```bash
bash scripts/train_prone_motion.sh \
  --python /home/noir/miniconda3/envs/isaacsim6/bin/python \
  --task WheelLeg-ProneMotion-v4 \
  --resume /absolute/path/to/original/run/model_04000.pt \
  --output /absolute/path/to/new/run --headless
```

从检查点原目录读取完整课程和原始运行参数。恢复 actor、critic、双方的
observation normalizer、探索分布、Adam 动量及学习率、完成轮数、课程/域随机化
进度和已保存的 RNG。全局课程恢复逐技能/逐档准入、复习评测记录、四类
自起门槛、复杂倒地及 DR 准入阶段，在创建首次训练 reset 前装回。姿态库
抽样 RNG、逐类轮换游标和 reset 代次同样接续。初态逐张量核对后才允许 PPO；第一轮继续检查真实梯度和
Adam 更新。课程总预算保持 10000，因此 4000 轮检查点只再运行 6000 轮。
TensorBoard step 从 4000 接续，写到新目录，原记录与 PT 不覆盖。

`--max_iterations` 表示原计划总预算，不能填剩余轮数；与原计划冲突的环境数、
预算、seed、学习率、DR、采样步数、物理时钟会拒绝。省略采样/性能后端时继承
原运行；显式改为不同后端同样拒绝。物理模型、观测、奖励和课程必须与
原 manifest 一致；只允许恢复入口及其测试、文档发生变化。外壳仍先进行独立
零 PPO 的原生检查，检查过程只读 JSON，不读取 PT。

恢复需使用生成该检查点的原始合同和源码。不能用当前默认的 200 Hz 配置
加载旧 1 kHz 运行，也不能把旧的按轮数课程迁移成新全局解锁课程。旧版本
若未保存新课程、抽样器等状态，新入口会拒绝猜测或重置这些状态。已有旧
恢复任务继续使用自己的冻结工作树；本次分支整合不改变正在运行的进程。

旧 PT 没有 PhysX 状态和未完成 rollout，恢复时 episode 重新初始化；这不会
把课程轮数或已学参数清零。旧 PT 也未存 Python/NumPy RNG，因此不能宣称
逐物理帧重放。新增检查点同时保存这两个 RNG，但仍不保存原生物理状态。
意外退出后未落盘的更新无法恢复，应以最后完整 PT 的轮数为准。
