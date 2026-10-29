# 无界面Fabric输出与100Hz策略频率检查

本机当前冻结源`f4996e0dc8aec96b6944c85a7bef6bbd2b6097a8`的无界面训练默认已经关闭姿态和速度发布。显式再次关闭两项不改变有效配置，不能作为额外提速。服务器当前learner没有停止、修改或启动新进程；本轮未重连服务器，其Isaac版本不同，不能将本机标志读回当作服务器读回。

## 原生读回与对照

| 本机256环境 / 原200Hz策略 | 128步计时均值ms/策略步 | fabricUpdateTransformations | fabricUpdateVelocities |
| --- | ---: | --- | --- |
| stock_nominal001 | 142.879 | false | false |
| off_nominal001 | 136.638 | false | false |

这两个进程的有效设置完全相同。计时取原TensorBoard event的`PerformanceProbe/policy_step_s`，不是独立绘图数据源；相同配置之间的波动不解释为优化收益。两次均为1ms物理、5ms策略、decimation5、64/32求解迭代，256步原始轨迹、16步额外warmup及128步计时。各126组轨迹数组全部逐位一致，含观测、奖励、命令、根/关节状态、每1ms物理诊断以及done；含74次done事件及3588个失能lane样本。0PPO、0actor/critic、0旧checkpoint读取。两进程已按PID/start_ticks及GPU列表确认退出，源码摘要无变化。

启动后其他有效设置：`fabricUseGPUInterop=true`、`suppressReadback=true`、`isaaclab/fabric_enabled=true`。这里保留Fabric子系统，只关闭无界面任务不用的发布。不能将`use_fabric=False`当等价优化：它可能启用USD同步，或破坏GUI显示链路。本机IsaacLab AppLauncher已将无渲染姿态输出关闭，速度输出默认也为false；环境构建后复核两项仍false。底层根/连杆状态读取由PhysX张量视图`get_root_transforms`/`get_link_transforms`提供；上述原生轨迹对照独立验证了观测不依赖再次开启发布。没有评测GUI、相机或学习策略，不能转授它们的资格。

## 策略200Hz改100Hz

在底层物理/PD保持1000Hz时，每个动作保持从5ms改10ms。精确时钟计算：

| 策略频率 | 每轮步数 | 仿真秒/环境 | 每轮串行物理步 | 256环境总样本 |
| --- | ---: | ---: | ---: | ---: |
| 200Hz | 128 | 0.64 | 640 | 32768 |
| 100Hz | 128 | 1.28 | 1280 | 32768 |
| 100Hz | 64 | 0.64 | 640 | 16384 |

保持128步会让占大头的物理/低层控制工作翻倍，不能预期每轮变快。保持0.64秒而用64步，则物理工作不变，仅减少策略推理、观测、奖励及策略频率诊断等调用，训练样本也减半；不能宣称相同10000轮具有相同训练量。这是时钟分析，本轮没有创建100Hz控制合同或执行100Hz原生/PPO探针。

100Hz的策略计算和实车推理负担较小，但新目标每10ms而非5ms刷新。1kHz低层PD仍逐步读反馈；RL层对失衡、刹车、快速自旋的修正更稀疏，是否能保持目标必须行为验证。2m/s时每次目标更新间隔内行进约2cm，而200Hz约1cm；100rpm时约6°而非3°。这些数字是运动学间隔，不能单独判定性能好坏。

正式采用100Hz还需同步控制合同/decimation、以秒为单位的回合和课程、动作变化成本与参考步长、折扣/GAE、按policy-tick表示的传感延迟、导出/实车推理频率等。当前新轮控制合同要求200Hz，不能只改`policy_dt`绕过校验。不会为本次速度讨论修改正式合同、暂停训练或续训旧模型。

## 全部长期能力

本轮仅性能与接线，不授予策略行为或实车资格；接触口径、原地起立约10cm和实际完成后完整30秒均保留。全部13项逐项保留：

| 能力 | 本轮结论 |
| --- | --- |
| initial_recovery | 本轮未重新验收，未授予通过 |
| moving_fall_recovery | 本轮未重新验收，未授予通过 |
| standing | 本轮未重新验收，未授予通过 |
| low_speed_straight | 本轮未重新验收，未授予通过 |
| high_speed_straight | 本轮未重新验收，未授予通过 |
| stationary_spin | 本轮未重新验收，未授予通过 |
| four_direction_arcs | 本轮未重新验收，未授予通过 |
| standing_height_change | 本轮未重新验收，未授予通过 |
| moving_height_change | 本轮未重新验收，未授予通过 |
| spinning_height_change | 本轮未重新验收，未授予通过 |
| high_speed_braking | 本轮未重新验收，未授予通过 |
| smoothness | 本轮未重新验收，未授予通过 |
| robustness | 本轮未重新验收，未授予通过 |

机器可读来源、TensorBoard event和逐数组核验见[receipt.json](receipt.json)。[NVIDIA官方说明](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html#physics-simulation-optimizations)要求姿态和速度发布同时关闭，且只适用于不消费显示输出的无界面GPU张量任务。Context7当前工具不可用，本轮使用安装源码及官方文档核对。
