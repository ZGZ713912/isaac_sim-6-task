# Wheelbipe V9：水平车身与混合地形课程

`foundation-jump-v9` 使用 `contracts/wheelbipe_foundation_jump_v9.json`。它从全新 actor、critic 和空 Adam 开始，总预算 20000 个 PPO 更新；默认策略 50 Hz、逐物理步反馈 PD 200 Hz。35/78/6 的 actor、critic、action 接口与 V8 相同。机械、真实气簧、real2sim v3.2、腿 PD 80/1.5 和 V8 跳跃序列保持原合同。

固定 lane 同时练习 stand、forward、reverse、height、turn、dash、spin、jump、airborne、platform、slope、step_down、bump、balls。配额分别为 9/8/8/5/5/3/5/13/5/11/11/8/6/3%。每个回合只重采样模式和参数，不重新分配任务。解锁后的低难度继续参与复习。

| 内容 | 解锁更新 | 满难度更新 | 目标 |
|---|---:|---:|---|
| 坡面 | 2500 | 8500 | 固定 3/5/7/9/10/12/12/12° 的 lane |
| 坡面自旋 | 2500 | 9000 | 30→100 rpm，两方向，保留低转速 |
| 下台阶 | 3000 | 9000 | 固定 10/15/20/20/25/25/25/25 cm 的 lane |
| 小台阶 | 10000 | 14000 | 固定 3/4/5/6/7/7 cm 条带 |
| 坡面跳跃 | 11000 | 15000 | 可跳坡角 4→12°，最多 30% slope lane |
| 小球 | 12000 | 15000 | 面积覆盖率 10→30%，保留稀疏复习 |

15000–20000 全混合巩固。V8 原有 phases、平台解锁、移动跳、12000/14000 的特殊命令里程碑按绝对更新号保留。常规训练不传 `--course-updates`；显式缩放实验会同时缩放所有新旧更新里程碑。

地形、接触、质心和球的位置只用于复位、奖励、成功判定与诊断。它们不能进入 actor 或推进命令时钟。下台阶和小台阶只给速度命令；坡面跳跃使用既有显式 jump request/apex/clock 协议。全局 roll/pitch 水平项使用仿真真实姿态，复位后 0.3 秒宽限；airborne 首次双轮支撑后另有 0.3 秒宽限。位置保持在零平移命令持续 0.3 秒后锁定水平锚点。

坡面和台阶几何均位于所属环境 prim 下，在碰撞过滤前生成。静态形状四元数为 wxyz；PhysX ball tensor view 的位姿为 xyzw。视图初始化按实际 prim paths 建立 lane/球索引映射，不依赖枚举顺序。坡面的高度参考是两轮下方平面的均值；下台阶使用块内支撑高度。小台阶始终以主地面为高度参考，使腿吸收条带起伏。

小球直径 17 mm、质量 3 g，接触材料摩擦按 lane 取 0.2/0.35/0.5，restitution 0.2。默认训练区域 1.2 m，评测 2.0 m。球数按 `ceil(coverage*side²/(pi*radius²))` 计算，1.2 m 满覆盖为 1904 球；布局保留至少 17.5 mm 中心距。球是自由刚体，回合复位时使用 PhysX 批量写位姿和零速度。性能降级顺序为区域 1.0 m、balls 配额 1.5%、球位置迭代 4；每次必须写入新冻结合同并保留评测区域，不修改运行中会话。

所有奖励都是每秒密度，由原 reward 入口乘一次 policy dt 和权重。事件先除以 policy dt。下台阶窗口抑制高度跟踪和竖直速度的旧项；坡面跳跃窗口使用专门水平项，避免叠加 slope_level。完成后的失败会撤销回合成功统计。完整验收使用合同参数及只读套件，不能用预检、reward 或训练完成代替行为验收。

使用 `scripts/train_wheelbipe_port.py --mode preflight --training-profile foundation-jump-v9 --iterations 20000 --num-envs 64 --physics-hz 200 --solver-position-iterations 64 --solver-velocity-iterations 32 --headless --output .codex/report/<run>/preflight` 检查原生接线。

`--mode benchmark --iterations 50 --benchmark-course-update 20000` 从零构造诊断 PPO，并固定满难度课程时钟；`--benchmark-without-balls` 只允许此模式。比较相同机器、环境数的 V8、V9 无球和 V9 含球，使用 TensorBoard 的 `Perf/iteration_time_s` 后 30 更新均值。正式训练必须从已验证提交 `git archive` 冻结，scoped deployment manifest 小于 256 KB，然后用 `WheelbipeSession.create` 和 `trainctl wheelbipe` 管理。

只读评测入口 `scripts/evaluate_wheelbipe_scenarios.py --suite slope_spin|slope_still|step_down|bump|slope_jump|balls|regression|retention --training-profile foundation-jump-v9 --checkpoint <file> --output .codex/report/<run>/<suite>`。常规验收每格至少 64 回合，分别报告 nominal 和 requested strength=1，报告 Wilson 95% 下界。未完成场景、缺失指标、部分网格均不能称通过。小球评测使用独立进程，每次 16 个目标 lane。

TensorBoard 使用原训练 event 中的 `Level/<task>/*`、`Hold/<task>/*`、`Mixed/<task>/*`、`MixedRollout/<task>/advantage_rms`。每 2000 更新保存 review checkpoint。独立 `scripts/watch_wheelbipe_retention.py --host kaiser --session <remote-session> --output .codex/report/<run>/retention` 将检查点取回本机，运行 16 env/格、两个 seed 的 retention，只写独立评测 event 和 `retention.csv`。监测进程和远端 TensorBoard 不影响训练生命周期。成功率连续两次下降 15 个百分点、漂移/误差变差 50%、倾角增加 2°，或任务 advantage RMS 超过中位数 3 倍持续 500 更新时输出诊断告警，不自动暂停或改课。

暂停使用 `trainctl wheelbipe pause <session>`，等待训练线程在更新边界确认；继续使用 `trainctl wheelbipe resume <session>`。原 PID、GPU、Adam、rollout、仿真和随机状态保留。release 是另一个保存退出并释放 GPU 的操作，冷恢复会重置物理回合，不能当作原进程暂停验证。
