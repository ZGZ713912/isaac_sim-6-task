# 本机 200 Hz 短训与长训

2026-10-17 用户授权在本机完成短训，检查效果；正常后长训，有问题先诊断并迭代。
本实验继承 `contracts/v6_local_scratch_200hz_v2.json`，独立于已停止的 Kaiser 课程。
不启动本机 TensorBoard 服务，曲线写入现有 RSL-RL TensorBoard event。

## 执行顺序

1. 512 环境、200 Hz 策略／物理，从零完成首段 64 次 PPO；这与此前两更新 smoke 分开。
2. 用相同固定种子、固定初态，对比初始模型与短训模型。检查 230/305/430 mm 站立、
   前后 0.5 m/s、正反 1 rad/s、启停；保持原固定场景的时长和判据，记录所有失败。
3. 检查 loss/权重/Adam 有限、消费和样本计数、实际 5 ms 时序、闭链与力矩保护，
   以及行为指标的趋势。短训不要求已掌握 5 m/s、3 圈/s 或后续地形／跳跃能力。
4. 若数值或控制链有缺陷，先保留证据并修复；若只是初学能力不足，完整恢复同一
   actor、critic、Adam、RNG 和课程时钟，再累计短训至 128/256 更新观察，不能重置消费。
5. 训练链稳定、基础行为出现可复核的学习进展且无系统性机械保护异常后，接续同一
   实验至其冻结累计上限。总预算 20000 更新包含所有失败短训消费，墙钟最长 7 天。

## 验证边界

固定场景成绩、训练中的随机动作与工程稳定性分别报告；保留原始验收阈值，不能以
reward 改善代替行为通过。继续学习的准入不授予最终技能、机械或实车部署资格。
若尚无可复核的基础行为进展，继续短训诊断，不自动把启动成功当作效果正常。

每段从已提交源码生成冻结归档，记录 commit/SHA、进程身份、checkpoint 和预算。
修复及结果分别提交，并推送到用户指定的 upstream 仓库，保留主工作区已暂存内容。

首段运行目录中的 `20261018` 是实验标识；真实启动时间为 2026-10-17 UTC，
所有完成和进程回执使用实际 UTC 时间。

续训命令的 `--updates` 表示追加更新数，必须使用同一实验已完成区块的
`model_final.pt`；自动恢复 actor/critic/Adam/RNG 及累计消费，不能重新从零计费。
本机 8 场景诊断使用原生 v18 独立 train 表中的 230/305/430 mm、前后 0.5、
正反 1 rad/s 和启停原始场景；4 副本、种子、时长和阈值不变。它不是完整课程验收。

## 本机入口

```bash
conda activate isaacsim6
python scripts/train_local_chassis.py --num-envs 512 --updates 64 --run-dir runs/my_pilot
python scripts/evaluate_local_chassis.py --checkpoint runs/my_pilot/initial_state/model_final.pt --checkpoint runs/my_pilot/model_final.pt --output runs/my_pilot_eval
# 评测和工程读回通过后，v2 的 64 次短训通过后，追加到本轮 19936（加旧失败 64，总计 20000）：
python scripts/train_local_chassis.py --num-envs 512 --resume runs/my_pilot/model_final.pt --updates 19872 --max-runtime-seconds 604800 --run-dir runs/my_long
```

评测 `LocalEvaluation/<case>/<metric>` 写入各候选训练目录原有 TensorBoard
event；step 是该 checkpoint 的累计更新数。JSON 保留完整机器判定及 SHA。

## 第一次短训与修订

v1 实际完成 64 PPO 更新。actor/critic/Adam 全部有限且参数改变，但确定性
8 场景共 32 回合全部在约 0.7 秒因非轮接触终止，0/8 通过；不准入长训。
随机训练累计有 1953 次闭链终止，峰值 6.823 mm；确定性考试峰值均不足
0.35 mm，两者不能混作同一种证据。详见 `reports/local_training_20261017/pilot001.json`。

v2 单独从相同种子重新初始化，将 Gaussian 初始 std 从 1.0 改为 0.2，先做
128 更新的对照。该修改是待验证的探索幅度假设；不宣称已定位唯一根因。
保持原始 reward、5 ms 时序、折扣、GAE、课程、闭链和力矩门。v1 的 64 更新
不退款，v2 剩余上限 19936；两条学习谱系分开记录，不把重置称为恢复。

只读检查当前阶段（不启动 TensorBoard 服务）：

```bash
python scripts/inspect_local_chassis.py --run-dir runs/my_pilot
```

该命令读取现有 event，回合长度同时换算为秒；有限性检查限定为每个选定 tag
最多 200 个保留样本。PPO 梯度裁剪已核对本机 5.5.1 源码：actor 与 critic
分别裁剪，不能归因为 critic 梯度占用了 actor 的裁剪额度。
