# 本机 V6 从零训练

使用 Conda `isaacsim6`。入口为 `scripts/train_local_chassis.py`，计划为
`contracts/v6_local_scratch_v1.json`。

```bash
source /home/noir/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim6
cd /home/noir/Documents/workspace/isaac_wheeled_rl_train
python scripts/train_local_chassis.py --num-envs 512 --updates 20000 --max-runtime-seconds 604800
```

该命令建立独立本地实验，随机初始化 actor、critic 和 optimizer。训练范围是
平地站立、230–430 mm 高度、前后平移、旋转、联合运动及启停，沿用原生
speed 配方的 135 个场景组。最大课程目标为 5 m/s、3 圈/s；目标范围不是能力结论。
后续 terrain、jump、stepjump 需要独立的本地接续合同。

奖励继承当前 v26/v27 speed 的 `scut_v14_flat_v5`，包括
`v6-dense-visible35-v1`、驻停速度成本和姿态项。计划分别固定原生 v18 worker、
独立设计表以及 v27 reward 参照的 SHA-256。原生 v18 与 v27 静态 worker 的
有效配置逐项一致，仅设计／续训绑定不同；准备器再次核对这一事实。

保留 35D actor、81D critic、6D 动作、200 Hz 物理／反馈 PD、50 Hz 策略、
64/32 求解器迭代、3 mm 闭链保护、轮响应模型和原生 reward 支付。当前分支的
通信模型和已有训练问题仍适用，见 `TRAINING_PAUSE_REVIEW_20261011.md`。

本地差异是：512 并行，24 步 rollout，4 个 minibatch（各 3072 样本）；
独立 20000 更新上限与零历史消费；三个课程 origin 从 0 开始；首更新即保存
checkpoint，其后沿用 500 更新快照间隔。允许的并行数为 256、512、1024，
各自生成并绑定完整独立设计表。原作者的合同、学习状态和预算均保留原样。

每次运行使用新的输出目录 `runs/v6_local_scratch/<UTC时间>-<随机后缀>/`，
包含 TensorBoard event、startup、progress、completion 和 checkpoint。
TensorBoard event 由现有 RSL-RL logger 写入。遵循仓库约定，不自动启动本机
TensorBoard 服务或占用已有 6006/6007。

启动 PPO 前要求训练代码与计划已提交且 tracked 文件干净。启动器从该 commit
生成并解压 Git 源码归档，在冻结目录中运行；
`runs/local_scratch_sources/<运行编号>/launch_source.json` 记录 commit 与归档
SHA-256。后续工作区编辑不会改变该运行的源码。

只做预检：

```bash
python scripts/train_local_chassis.py --preflight-only
```

原生场景／reward 工程检查（零 PPO 更新）：

```bash
python scripts/train_local_chassis.py --num-envs 512 --updates 1 --validate-scene 24
```

此入口专用于新实验，不接受作者 checkpoint、旧消费或旧课程时钟。当前命令
以全新初始化开始；同一实验的后续恢复需要使用其本地封存合同、checkpoint
和预算记录，不能通过再次运行上述从零命令冒充续训。

## 2026-10-16 本机验证

CPU 验证覆盖原生合同／时序回归，以及本地独立预算、来源拒绝和源码归档。
512 环境真实执行 24 策略步／每环境 96 物理步，runtime preflight 与 4 次
新反馈 PD 读回通过。随后从提交 `e90fc0a5ab7216251b9ab81c0f5bc5415818b1af`
冻结源码，在本机完成 2 次 PPO 更新，parent_updates=0；actor 的 9 个、critic
的 8 个张量在两次更新间均改变，17 组 Adam 状态已写入 checkpoint。
TensorBoard event 含 157 个 scalar tag，loss 有限。

验证回执见 `reports/local_scratch_20261016/verification.json`。两次 PPO 更新的
迭代总时间约 28 秒，第二次约 14.90 秒；短窗不作为长期吞吐承诺。命令中的
20000 是更新上限，604800 是 7 天墙钟上限，先达到者结束。当前只完成短测，
没有启动 20000 次长训，也不由短测推断策略能力。
