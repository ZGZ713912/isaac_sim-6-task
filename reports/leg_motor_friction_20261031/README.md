# v2.0.2 腿电机摩擦 real2sim 接入回执（2026-10-31）

实现位于独立分支 `dev/leg-friction-v202`。该分支的默认 `FullRobotV202()`、整车组件探针以及 `WheelLeg-ProneMotion-v4` 已选择带腿电机动摩擦的 v2.0.2 派生资产。原训练目录不合入本次代码，现有用户训练继续原模型；本次未启动或恢复训练。

| 项目 | 结果 |
|---|---|
| 左/右 hip 库仑平台初值 | 2.019506006848738 N·m，主动输出轴 |
| 左/右 knee 库仑平台初值 | 2.047246309400147 N·m，主动输出轴 |
| 残余中点偏置 | 不施加；沿用报告的正反向半差结果 |
| 右侧来源 | 用户授权的左侧近似，非右侧独立辨识 |
| 黏性、静摩擦脱离阈值、Stribeck 特征速度 | 全部 `null`，不从低速运动摩擦虚构静摩擦峰值 |
| 数值平滑 | 0.05 rad/s 线性速度带，明示未辨识；零速为零 |
| 机械与其他组件 | 21体、21.44174kg、6闭链、40°–110°限位、110°零点、气簧与轮子辨识数据保留 |
| 新训练初始化 | 新 actor、critic、空 Adam；无 checkpoint/teacher/donor |

具体应用方向、执行顺序、数值选择与适用边界见 [实现说明](../../docs/LEG_MOTOR_FRICTION_20261031.md)。未知域采用固定库仑初值是显式先验，不能称作全角度/全速度辨识完成。未把本次摩擦接入控制前馈或实车控制器。

验证已完成：

- **132项定向测试通过**：参数来源、左右轴映射、两方向耗散、零速行为、篡改拒绝、冻结旧资产、v4整车配置与v3兼容、编码器气簧补偿、轮子动态和里程计。完整输出见 [tests.log](tests.log)。
- **CPU PhysX名义域、全强度随机化域各40项接线检查通过**。每域80个环境、每环境1500个1ms物理步，覆盖既有1kHz PD、气簧、轮子和云台接线。新增检查核对四腿轴摩擦恰好施加一次、非正摩擦功率、零速零摩擦、原生关节摩擦读回为零、40N·m电机力矩与独立被动力矩相加后与执行器读回一致，以及TensorBoard tag。见 [native002/report.json](native002/report.json)。
- 训练CLI最终配置预检通过，冻结**513个来源文件**，`new_PPO=0`、checkpoint读取为空。见 [cli_preflight002/preflight.json](cli_preflight002/preflight.json)。这次预检不启动GPU PhysX；本次原生执行验证使用CPU。
- 原训练PID3732465及其启动身份保持，501个冻结来源文件SHA全部一致；核对快照时仍为 `training`、已完成292次原训练更新。信号、重启、本次变更采纳均为0，详见 [live_training_unchanged.json](live_training_unchanged.json)。这292次来自原用户训练，不是本次PPO消费。
- 实现已提交为 `4686624f`，原生训练归档函数实际检查全部513个来源与该提交逐字节一致，并成功生成来源tar及SHA；回执见 [source_revision.json](committed_archive001/source_revision.json)。tar保留在本工作区，按目录内规则不重复提交81MiB归档；可由回执中的提交和冻结来源列表重建。

首次独立探针适配器 `native001` 因未绑定既有 `nominal_action` 所需的模块级 `torch` 变量失败，已保留失败回执和日志。修正适配器后，`native002` 两域全部通过。CPU后端产生了重复的 PhysX 材质索引警告；原始日志无损压缩保存，解压SHA与原文件匹配，见 [log_integrity.json](log_integrity.json)。未通过隐藏警告改变仿真设置。

所有13项长期行为能力均记作**本次未测**，不以接线检查当作通过。逐项要求及 `passed:null` 见 [capabilities.json](capabilities.json)：初始自起、运动中翻倒恢复、三高度静站、低速直行、高速直行、双向定点自旋、四方向弧线、站立变高、运动变高、自旋变高、高速急停、平顺性和留出鲁棒性。无策略行为、动态摩擦辨识、CAN或实车验收结论。

复现新目录的构建时，在未生成 `model/dm_leg_friction_20261031` 与 `model/wheel_leg_v2.0.2/friction_v1` 的干净副本中运行：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python scripts/build_leg_friction_v202.py \
  --source /home/noir/Documents/workspace/RMCS/reports/dm_leg_friction_20261031
```

独立CPU原生验证命令如下，输出目录须不存在；不构建learner或optimizer：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python scripts/probe_leg_motor_friction.py \
  --output reports/leg_motor_friction_20261031/recheck --num-envs 80
```

训练默认选择新摩擦模型，但真正启动前仍执行原有的GPU原生预检和已有learner检查。当前用户训练没有停止，也没有因本次接入被替换。
