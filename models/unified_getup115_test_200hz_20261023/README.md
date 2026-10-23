# 115统一策略实车试验包

用户于2026-10-23明确要求导出供自己实车试验。本包是**实验候选**，不是已通过完整自起和实车验收的策略。当前正式行为门与大训练门保持关闭。建议首先评估305mm中位静站及低速直行/弧线的实车减振表现。

`policy.onnx`：float32 `obs[1,35] → actions[1,6]`，确定性均值，200Hz，history=1。来源PT SHA `11c4451ab73c4935051fc0187a5fc3a70a3534cee67f22d393500f623bd9f121`；ONNX SHA `48d4880f2b928512be54bea0e4b24f09fc3c47057c3c36dba87066ab34d7e443`。模型包含079恢复分支及110静站分支，没有本轮新增PPO。

## 已知效果及边界

- 115的两个随机种子、三高度各16次，共96次静站全部完成30秒。地面材料摩擦范围0.3–0.5、研究域随机化倍率0.35；这些是仿真假设，不是实车已辨识参数或实际摩擦下界。
- 首种子低/中/高位最差位移3.926/3.026/4.819cm；新种子6.592/3.586/5.679cm。实际腿/轮3–95Hz高频检查通过既有基线非退化标准，高位末10秒最坏平均偏移约1.85/1.73cm；有限窗口不能保证永远站稳或实车无抖动。
- 新种子低位超过原5cm门，因此115后续140随机完整套件、两个名义配对套件和静止跌倒生命周期没有运行。保留正式`needs_iteration`结论。
- 最新115单车GUI左侧对称腿姿在0.815秒保护终止，没有起身。079父恢复分支已有倒扣/侧倒成功证据，但强右推名义测试0/16；不能把父模型成功率直接称为115实车成功率。
- 115与079在201066条构造运动输入、13020条既有真实非quiet输入上动作逐位一致，这是离线保持证据，不是115新闭环运动验收。自碰撞关闭，实车尚未测试。

完整原回执在`evidence/`，合同在`deployment_contract.json`。

## 实车obs必须同步修改

仍是35维，**零起始索引14、15**由原空置通道改为`20 × 相对站立位移(m)`。不是直接输入多圈轮角。

| 槽位 | 内容 |
| --- | --- |
| 0–2 | 机体系vx、vy、yaw指令 |
| 3 | 目标高度(m) ×5 |
| 4–6 | 机体系角速度(rad/s) ×0.5 |
| 7–9 | 机体系单位重力方向 |
| 10–13 | 四腿角相对名义值，atan2(sin,cos)绕回±π |
| 14–15 | 相对站立轮坐标位移(m) ×20 |
| 16–21 | 六轴角速度(rad/s) ×0.1 |
| 22–27 | 上一帧实际进入PD计算的raw action |
| 28–34 | 固定[1,0,0,0,0,0,0] |

策略关节顺序：左髋、左辅助腿轴、右髋、右辅助腿轴、左轮、右轮。它不同于控制层将左轮排在第3个的顺序。名义角为`[-.42,.13742282595395358,.42,-.1374155762580851,0,0]`。

位移预处理使用连续轮侧角度、腿角和IMU重力及冻结闭链几何表，150ms低通，±0.25m限幅。满足静止请求及姿态条件后捕获参考；reset、进入RL、离开静止请求和严重倾倒时清参考；几何无效时输出零并保留参考。50ms捕获比较按训练float32时钟，首次可用在55ms。请使用配套C++实现，不用真值XY、不自行替换成原始轮角积分或两维恒零。

模型及运行配置绑定观测合同SHA `db0fc6121c73121dcd39eef0524ecfeab4b3aa69aa7f5700d4205499af2402a9`。旧模型与旧YAML继续保留。

## 动作与使能

四腿raw范围±16，乘0.25为相对名义角的偏移；通过`paired_branch107_v1`连续角目标解码和闭链约束后执行。两轮raw范围±9，乘10为轮侧rad/s。不是把这六个值都发成位置目标，也不能沿用旧raw腿±3裁剪。

策略200Hz，host PD1000Hz，腿Kp/Kd=160/2.5，腿40N·m、轮4.5N·m力矩帽。ONNX内已有静站平滑；不要叠加额外动作低通。运行端保留模型标定、传感器新鲜度、机械和力矩保护。启用前六轴零力矩；进入RL时用测得的可行成对腿角播种历史，轮历史零，此后记录实际应用raw action。模式2握手且保持零力矩，模式3直接六轴RL，不运行旧转腿脚本。动作等待/反馈陈旧时由既有保护停止输出。

组合命令同训练限制`abs(vx*yaw) ≤ 2.06m/s²`：超过时保留yaw、降低vx。位置反馈仅在静站门起作用，不能认为任何微小非零指令都会立即关闭静站门。

## 对应RMCS代码

已实现并编译的隔离工作区：

`/home/noir/Documents/workspace/RMCS/.codex/worktrees/unified-standing-v3`

分支`codex/unified-standing-v3`，父提交`9abd68b75d393bf4c383aff095f67bce5be71ba5`；`rmcs_rl`子模块提交`795729c1d4e200e635a9fe85b7708241bf62f6ff`。主RMCS checkout没有同步这些修改，**不能只将本ONNX替换进主checkout的旧配置**。

`WheelLegController`负责位移预处理、完整观测帧、已应用动作历史、使能/保护和host PD；`RlBridge`只负责完整帧路由和线程推理。详细说明见`runtime/observation_runtime_v3.md`，其中“候选模型缺省”描述的是085接口实现时的状态，本次测试模型由本包另行提供。

包内`runtime/rmcs_rl-child.patch`与`runtime/rmcs-parent.patch`供审阅；同名`.bundle`为增量Git包，保留上述精确提交ID，分别要求已有子模块`22ac9e5`及父仓库`5a7867ec`。迁移到另一台机器时先在子模块仓库fetch child bundle的`HEAD`，再在父仓库fetch parent bundle的`HEAD`；使用独立分支/工作区检出对应提交，先核对各自base和子模块gitlink。`git am`重建的commit ID可能不同，不能据此宣称满足本包的精确提交绑定。不同base需正常审阅合并，不能直接覆盖车辆标定。

本轮实读主RMCS存在较多未提交的控制器重构/文件删除，本包没有覆盖或回退这些改动。请使用上述已验证的独立工作区；包内补丁不是对当前主checkout无条件覆盖的指令。

## 安装与启动

以下安装命令仅复制新文件，不启动硬件。在宿主机执行（已具备上述对应工作区时）：

```bash
trial_dir=/home/noir/Documents/workspace/isaac_wheeled_rl_train/models/unified_getup115_test_200hz_20261023
rmcs_trial=/home/noir/Documents/workspace/RMCS/.codex/worktrees/unified-standing-v3
python "$trial_dir/verify_export.py" "$trial_dir"
cp "$trial_dir/policy.onnx" "$rmcs_trial/rmcs_ws/src/rmcs_rl/models/wheel_leg/unified115_test.onnx"
cp "$trial_dir/wheel-leg-infantry-unified115-test.yaml" "$rmcs_trial/rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-unified115-test.yaml"
```

离线校验需要NumPy、ONNX Runtime、PyYAML；本机可用`/home/noir/miniconda3/envs/isaacsim6/bin/python`替代`python`。当前shell的系统Python没有这些依赖。

在现有RMCS开发容器中构建和加载**此工作区**：

```bash
export RMCS_PATH=/workspaces/RMCS/.codex/worktrees/unified-standing-v3
build-rmcs --packages-select rmcs_rl rmcs_core rmcs_bringup
source "$RMCS_PATH/rmcs_ws/install/setup.bash"
```

由用户在实际车辆环境主动启动：

```bash
ros2 launch rmcs_bringup rmcs.launch.py robot:=wheel-leg-infantry-unified115-test
```

此配置保留对应工作区现有标定与传感器/力矩保护；若车辆使用另一组机械标定，应先核对，不能用文件复制替代标定匹配。此入口是全RL，改变的是接管方式，不会让尚未学会的翻倒姿态自动变可靠。

## 本次离线验证

- ONNX结构检查通过；46465条输入Torch/ORT最大raw误差2.861023e-6，包括115两组静站、079动态场景输入和35137条有效C++完整帧。动态旧输入仅用于图保真。
- 新图实际经生产`RlBridge`、控制器和内存executor通过17项gtest，没有跳过。倒扣/侧身进入、复位、陈旧反馈停止、序列化及已应用历史均核验；这里没有物理动力学。
- 50Hz、缺观测hash、错误history、旧obs路由、错误动作倍率、脚本恢复、错误host PD七个负例被拒绝。
- 单线程本机CPU推理p50约0.224ms、p99约0.262ms，不是车辆控制周期保证。
- 首次内存测试的临时YAML重写丢失`"nan"`引号，ROS在加载模型前拒绝。只修复临时测试配置的序列化后002通过；原失败日志保留，打包运行配置始终保留正确引号。

`python verify_export.py`用于离线复验；`sha256sum -c SHA256SUMS`校验包内文件。本轮没有启动CAN或任何实车进程。
