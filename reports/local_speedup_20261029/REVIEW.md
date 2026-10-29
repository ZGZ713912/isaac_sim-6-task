# 本机训练采样提速验证

已在RTX4060Ti16GB / i5-12600KF本机实现并完成原生验证。新启动默认启用CUDA执行图；名义域和完整强度训练随机域的采样耗时见下表。本轮0PPO，不创建actor/critic、不读取旧模型；服务器运行中的冻结learner未修改、未停止。结果不代表策略已经学会站立、自起或运动。

| 同条件本机采样 | 原版ms/策略步 | 新版ms/策略步 | 耗时减少 | 吞吐倍率 |
| --- | ---: | ---: | ---: | ---: |
| 名义域 | 158.308 | 147.826 | 6.62% | 1.071× |
| requested强度1训练混合域 | 170.737 | 159.121 | 6.80% | 1.073× |

时间逐点取自各运行的原TensorBoard event：`PerformanceProbe/policy_step_s`，192点/运行；step0–191，每点包含同一个256环境批次的5个物理步。逐点event与原JSON计时核验一致。两个域分别保持相同seed、固定六轴小动作、256策略步轨迹预热、额外16步warmup、192步计时；关闭分析器计时。每段256策略步轨迹包含1280次物理推进，前后126组数组全部逐位一致，包括初态、观测、奖励、done、根/关节状态、命令和逐1ms物理诊断。每个域的全部数组核验、done计数及失能/启用样本见[receipt.json](receipt.json)。requested强度1使用合同原有25%名义复习，不能作为全随机held-out行为评测。

这是串行有界固定动作采样比较，没有实跑10000次PPO、没有计入策略前后向优化。桌面进程共享GPU，存在计时波动。不能把本机的加速百分比、绝对耗时或128步乘积直接当4090服务器的完整训练时间，亦不承诺训练耗时减半。此前同32步轨迹的96步短测约137.64→129.53ms；长轨迹计时阶段不同，最终以本表同长度对照为准，保留全部短测数据。

## 实现与不变量

- 冻结新版来源：`f4996e0dc8aec96b6944c85a7bef6bbd2b6097a8`；原版来源：`0c197fddce0d823770703ee61467dc2efd0c99ad`。正式原生探针321份来源逐字节匹配提交，归档SHA见回执；所有测量退出时来源均未变化。
- 编码器气簧补偿、编码器腿几何、物理气簧曲线使用固定形状CUDA执行图，回放原Torch内核；不使用编译融合或混合精度。每次调用保留非有限输入检查，输出复制到独立存储，避免下一步回放覆盖上一帧记录。
- 几何/支撑结果只在同一原生读取epoch与reset代际内复用；reset/物理推进后重新计算，域切换重建环境及执行图。合并逐角色指标的样本存在判断，原tag及选中样本归约保持。
- 物理1000Hz、策略200Hz、decimation5、64/32求解迭代、256环境、128步rollout、10000连续更新、250次存档、原课程/reward/控制/域随机化全部保留。未冻结腿、未降诊断频率，左右轮辨识和左右同款气簧模型不变。
- 剖析显示PhysX推进约占本机采样时间六成，是当前主要开销；4线程对照与默认线程没有稳定收益，未采纳线程数更改。工程优化无法据此保证数量级加速。

## 验证与来源

- 定向单元测试：先110通过、1模块因默认vendor目录不存在跳过；随后指定实际安装的IsaacLab源码重跑该模块，13/13通过，共123项不同测试通过，无未解决跳过。GPU逐位等价测试真实运行。此前CPU测试夹具直接导入原生环境时缺Kit的PhysxSchema，修正为提取实际方法后通过；失败日志保留`tests001.log`，通过日志`tests002.log`、`tests003_vendor.log`。
- 训练shell真实入口`--native-probe`：名义域29/29、requested强度1域29/29，0更新、0优化器消费、0checkpoint读取。检查实际奖励实付、气簧曲线/总限幅、左右轮惯量/延迟/损耗、1kHz六轴采样、失能零力矩与掩码、逐角色时长、35D/83D观测及域重建后新观测。结果：[native_probe001/progress.json](native_probe001/progress.json)。本测试不等于行为资格。
- 完整逐位轨迹与event独立核验：[receipt.json](receipt.json)，脚本[verify_final_matrix.py](verify_final_matrix.py)。所有静态图若以后需要，必须从这些TensorBoard event导出；本轮未启动TensorBoard服务。
- 最终名义对照：`final_eager_nominal001` / `final_graph_nominal002`；随机对照：`final_eager_requested001` / `final_graph_requested001`。各目录含原始`trace.npz`、`events/`、`native_startup.json`、`result.json`与逐点计时。
- 原始基线封存在`baseline_source_0c197fdd/`；不是可恢复learner。全部探针只用常量动作，不能作为teacher/donor。
- 全部11个本轮原生探针/基准进程均已按PID和start_ticks确认原身份退出，GPU计算进程列表无残留；原始桌面进程未改动，退出核验见回执。

## 使用

本机新启动直接使用原命令；加速默认启用。先跑独立0PPO原生检查，然后一个learner连续训练10000次，每250次保存完整检查点：

```bash
bash /home/noir/.codex/worktrees/prone-motion-long/isaac_wheeled_rl_train/scripts/train_prone_motion.sh \
  --output /home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/prone_motion_20261029/fast001
```

目录必须未存在；此报告没有执行以上PPO命令。需要排查执行图兼容时附加`--performance-backend eager`。正在运行的服务器冻结learner不会自动使用这些本机修改；采用新版须新目录、全新actor/critic/空Adam，禁止停止后续训旧模型。本轮没有同步或启动服务器新任务。

本机独立性能复验入口：使用Isaac环境Python执行`/home/noir/.codex/worktrees/prone-motion-long/isaac_wheeled_rl_train/scripts/benchmark_prone_motion.py --output <新目录> --backend cuda_graph`；比较时串行运行并保持环境、seed、轨迹与计时长度一致。`eager`只关闭执行图，当前版本仍包含安全的同epoch缓存与指标同步优化；复现此表全部改动前的基线要使用已封存的原版来源。

## 全部长期能力保留

本轮性能探针不授予任何行为资格，不替代历史或当前策略的验收。标准趴地起立、约10cm全程位移、完成后完整30秒、接触/IMU/速度/平顺均需真实策略另验；单次连续≤2ms失载只诊断，不能把≤2N直接称明显离地，不能后移起点遮盖再摔倒。

| 能力ID | 原有目标 | 本轮结论 |
| --- | --- | --- |
| initial_recovery | 初始倒地恢复 | 本轮未重新验收；未授予通过 |
| moving_fall_recovery | 运动中摔倒恢复 | 本轮未重新验收；未授予通过 |
| standing | 低/中/高静站 | 本轮未重新验收；未授予通过 |
| low_speed_straight | 正反低速直行 | 本轮未重新验收；未授予通过 |
| high_speed_straight | 正反高速直行 | 本轮未重新验收；未授予通过 |
| stationary_spin | 双向原地自旋80–100rpm | 本轮未重新验收；未授予通过 |
| four_direction_arcs | 前后左右四方向转弯 | 本轮未重新验收；未授予通过 |
| standing_height_change | 静止变高 | 本轮未重新验收；未授予通过 |
| moving_height_change | 运动变高 | 本轮未重新验收；未授予通过 |
| spinning_height_change | 自旋变高 | 本轮未重新验收；未授予通过 |
| high_speed_braking | 高速急停 | 本轮未重新验收；未授予通过 |
| smoothness | 全任务平顺 | 本轮未重新验收；未授予通过 |
| robustness | 随机域鲁棒性 | 本轮未重新验收；未授予通过 |

CUDA执行图的固定地址、静态输入和独立输出存储做法参照[PyTorch2.11官方文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。Context7在当前工具集中不可用，本轮以安装源码和官方文档核对API；速度与轨迹结论全部来自本机实测。
