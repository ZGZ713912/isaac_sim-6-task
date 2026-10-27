# 标准趴地训练：普通静止站立为主要奖励

更新于 `2026-10-27T14:33:07.122184+00:00`。当前真实训练为 `pilot007_quiet_standing_dr025_u128_e256`，
状态 `training`，已完成 **94/128** 次更新；尚无稳定站立成功证据。
主回执为 [receipt.json](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/receipt.json)。

用户最新纠正已落实：重点检查正常静止站立奖励，停止继续试初始缩腿偏置。
真实103°失能趴地reset、42°/±0.6rad的固定可训练初始bias保持不变。

## 当前主目标与保留条件

P0直接调用U0的 `upright_locomotion.reward_terms`，命令为零前后速度、零yaw、305mm。
普通静站的10项高度、姿态、速度/yaw跟踪、连续误差与动作/腿速/力矩成本逐项复用。
取消持续的收腿、轮心对准、部分抬升密度奖励及8/s硬稳定瞬时加分。
原路径势函数只保留gain0.1（原3），gamma与PPO相同。

额外保留宽高度误差平方成本4/s、实际完成前时间成本1/s，联合安静站立最高4/s：
必须同时接近305mm、水平IMU、低线速度/角速度，且5个1ms样本双轮同时有载、没有车体支撑。
高度40mm、水平5°、速度0.08m/s与角速度0.5rad/s是连续奖励的工程尺度，未新增硬验收线。
2N只作载荷门控和诊断，不能据此判明显离地。倒扣另有连续成本，避免重力xy为零的漏洞。

六轴始终学习四个腿位置和两个绝对轮速目标；没有固定腿、定时伸腿或脚本/FSM接管。
起立前底盘触地允许，起立全过程约10cm内无位置成本，超出才连续增加；
0.5m仍只是训练范围保护。实际连续1s站稳后才记录完成，完成后再触地/翻倒保留失败，
固定冷测为10s内实际完成起立并在其后完整30s站立，不移动起点隐藏后续失败。

## 实际运行与验证

- 冻结源：`b48a5139e9e0ffdf37cc214b36de7fe80930c1d3`，275个Python源逐字节匹配。
- 源归档SHA-256：`55a67b0a298534a042b996e48ad9442569f4a1e3af83e633b8d720095419b160`。
- PID `2074081` / start_ticks `9273247`；256环境、rollout256、seed261005。
- 新actor、新critic、空Adam；训练检查点读取为0、无旧teacher/donor。首轮12次真实Adam更新、
  9个actor和8个critic参数张量变化；四腿和双轮的6行输出权重及bias均变化。
- v2 requested强度0.25，腿P/D与气弹簧系数固定1、IMU晃动关闭；
  完整强度、各独立分组、原低摩擦与独立种子矩阵尚未验收。
- 9个相关模块85项CPU检查通过，1项旧metadata测试因旧model fixture缺失而明确排除。
  早期负测试和检查器失败均保留，未引入旧模型使测试通过。
- 新源64环境、600策略/3000物理步，18项奖励实付及原生接线检查通过、0终止、0PPO。
  这只能证明接线和支付，不能算学会起立。
- TensorBoard新增逐项实付reward、主动段高度/倾角；仍不启动本机TensorBoard服务。
  每5分钟后台检查同一运行，仅在实质变化或完成时通知，最终据固定冷测报告效果。

合成状态目标检查（非行为、未声称动力学可达）：安静趴地密度约0/s；
215mm/18°半抬升约−0.81/s；305mm水平双轮安静支撑在实际完成前9.5/s、完成后10.5/s。
305mm但仍以0.3m/s滑动约3.61/s，转速1rad/s约4.59/s。这里未计势函数、完成事件或终止成本。

## 已封存代际

| 代际 | 实际完整更新 | 结果 |
|---|---:|---|
| pilot001 | 4 | 正常停止，未续训 |
| pilot002 | 16 | 冷测0/21起立 |
| pilot003 | 64 | 冷测0/21起立，仅收腿 |
| pilot004 | 166 | 正常停止，冷测0/21起立 |
| pilot005 | 128 | 正常完成，0/21起立、0/21站立；瞬时300mm后落回约216mm/17° |
| pilot006 | 53 | 按本次用户纠正一次SIGTERM正常封存，53轮稳定样本为0；只读冷测0/21起立、0/21站立基线 |
| pilot007 | 94 / 128 | 当前全新训练，不能声称已站稳 |

之前各代共消费431更新，加当前94更新共525；不退款、不续训。
pilot006停止模型SHA-256 `6666423bbe13ce838865bd3639b516cff8a2726543af5b24f7bac33e6ea43d3c`，
进程已退出、17个Adam状态均636步、13项停止完整性检查通过。
初次只读检查器误把观测归一化buffer算作参数，修正计数后通过，未修改模型。
旧只读冷测工具假定所有actor都在本地冻结源内，拒绝官方MLPModel；失败回执保留。
已仅修复该外部检查器并认证当前官方RSL5.5.1安装RECORD，单独保存依赖源码用于只读复核；
这不改变pilot007的275个训练源或其PPO，不把当前依赖副本冒充旧训练时已冻结的包。

旧53轮模型只读冷测已完成：21/21均10s未完成起立，0场机械/边界等终止，
21/21在10cm全过程范围内。原始高度峰值308.06mm，
1s后高度中位数208.25mm、倾角中位数17.35°，
全过程最大XY偏移5.15cm。没有实际起立完成事件，不能对其起立后接地或30s站立授予资格。
这仍是同起点名义复制，不是21个独立随机种子。详见 [旧代冷测摘要](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/pilot006_final_behavior_summary001.json)。

## 全部长期要求

| 能力ID | 本代状态 |
|---|---|
| initial_recovery | 未验收通过 |
| moving_fall_recovery | 未验收通过 |
| standing | 未验收通过 |
| low_speed_straight | 未验收通过 |
| high_speed_straight | 未验收通过 |
| stationary_spin | 未验收通过 |
| four_direction_arcs | 未验收通过 |
| standing_height_change | 未验收通过 |
| moving_height_change | 未验收通过 |
| spinning_height_change | 未验收通过 |
| high_speed_braking | 未验收通过 |
| smoothness | 未验收通过 |
| robustness | 未验收通过 |

仍按原 [长期要求](/home/noir/Documents/workspace/isaac_wheeled_rl_train/docs/WHEEL_LEG_ACCEPTANCE.md) 和
[机器可读v2合同](/home/noir/Documents/workspace/isaac_wheeled_rl_train/contracts/wheel_leg_acceptance_requirements_v2.json) 逐项验收。
P0只覆盖一个标准趴地家族；未测试的倒地类型、高速/弧线/变高/自旋等不得算通过，
各代最好分项不能拼成整体资格。本轮没有实车动作或实车资格证据。
