# 实车状态驱动的 Isaac Sim 三维镜像

镜像直接用实车反馈摆放训练模型，检查电机方向、零点、IMU 坐标、闭链几何和左右对应关系。默认加载 `model/wheel_leg_v2.0.2/friction_v1/training` 的完整车体与视觉网格；GUI 中没有物理步进、控制器或策略推理。

```text
实车 WheelLeg 硬件反馈 + WheelLegStateEstimator
  → rmcs_rl::WheelLegMirrorComponent，默认 100 Hz
  → ROS 2 /wheel_leg/telemetry/*
  → foxglove_bridge WebSocket
  → 本机按名称映射、独立闭链解算
  → Isaac Sim GUI，默认最高 60 FPS
```

## 快速使用

在实车重新构建并部署 RMCS。`wheel-leg-infantry-traditional`、`wheel-leg-infantry-prone-v4`、`wheel-leg-infantry-wheelbipe` 和 `wheel-leg-infantry-wheelbipe-recovery` 配置已加入遥测组件，默认启用。沿用所选配置原来的 RMCS 启动方式和使能流程。构建及遥测参数见 RMCS 的 `docs/zh-cn/wheel_leg_live_mirror.md`。

组件位于 `rmcs_rl/src/visualization/`，按普通 RMCS 插件在车辆 YAML 中启用：

```yaml
rmcs_executor:
  ros__parameters:
    components:
      # 追加到原组件列表
      - rmcs_rl::WheelLegMirrorComponent -> wheel_leg_mirror

wheel_leg_mirror:
  ros__parameters:
    enabled: true
    update_rate: 100.0
```

默认启用、100 Hz，因此默认使用时参数块可以省略。组件从状态估计器读取实际标定身份与零点；GUI、模型几何重建仍运行在本机。

`wheel-leg-infantry-wheelbipe-recovery.yaml` 已开启该组件。`wheel_leg_mirror.enabled` 控制的是遥测，和底盘/电机是否使能无关。实车按原流程失能、保持传感器供电后，手动搬动车体和腿部仍会持续显示。镜像会标注 `DISABLED (feedback still live)`、DM ready 和估计器状态。

在实车的同一 ROS 2 环境启动桥接：

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

默认端口为 8765。这个启动命令来自 [Foxglove ROS 2 官方说明](https://docs.foxglove.dev/docs/getting-started/frameworks/ros2)。Foxglove 桌面端可以同时连接，查看原始话题。镜像接收器仅发送订阅请求。

在本机项目根目录运行：

```bash
# 模型、映射、依赖预检，不启动 GUI，也不连接实车
scripts/view_wheel_leg_live.sh --url ws://192.168.1.10:8765 --preflight-only

# 实时查看，并记录每个完整遥测快照
scripts/view_wheel_leg_live.sh --url ws://192.168.1.10:8765 --record session.jsonl

# 离线回放；不需要实车、ROS 安装或 Foxglove bridge
scripts/view_wheel_leg_live.sh --replay session.jsonl

# 合成静态姿态，检查本机模型与 GUI；界面标注 SYNTHETIC DEMO
scripts/view_wheel_leg_live.sh --demo
```

将示例 IP 换为实车 IP。本机需要安装带 GUI/RTX 扩展的 Isaac Sim、NumPy、SciPy，以及 `rosbags` 和 `websockets`。后两项可装在实际运行 GUI 的 Python 环境中：

```bash
/home/noir/miniconda3/envs/isaaclab/bin/python -m pip install \
  -r /home/noir/Documents/workspace/isaac_wheeled_rl_train/scripts/requirements-wheel-leg-mirror.txt
```

脚本沿用项目的 Isaac Python 选择规则。当默认环境缺少渲染扩展缓存时，会尝试本机已有完整 `isaaclab` 环境，并打印所选解释器。可以用 `--python /绝对路径/bin/python` 或 `WHEELED_RL_PYTHON` 明确指定，显式选择不会被覆盖。它不改变训练脚本的环境选择。

镜像复用 RMCS 的 `rmcs_ws/src/rmcs_rl/tool/visualization/wheel_leg_telemetry.py`，默认 RMCS 与本项目位于同一个父目录；其他布局使用 `--rmcs-root /绝对路径/RMCS`。

## 数据名称如何对应模型

唯一映射表为 `contracts/wheel_leg_live_mirror_v1.json`。数组顺序不参与名称匹配：接收端先根据 `JointState.name` 找到每个轴，再按映射表找到模型关节。

`q_model = sign × q_api + offset_rad`。`q_api` 是 RMCS 硬件层输出的轴角，已经包含电机方向与减速比转换。

| 遥测名称 | 模型关节名称 | sign | offset_rad | 物理含义 |
|---|---|---:|---:|---|
| `left_hip_joint` | `L_joint1` | -1 | 1.6 | 左髋主动轴 |
| `left_knee_joint` | `LL_joint1` | -1 | 3.0282761689183353 | 左侧主动传动曲柄 |
| `right_hip_joint` | `R_joint1` | -1 | -1.6 | 右髋主动轴 |
| `right_knee_joint` | `RR_joint1` | -1 | -3.0283102010168 | 右侧主动传动曲柄 |
| `left_wheel` | `L_joint3` | +1 | 0 | 左轮输出轴 |
| `right_wheel` | `R_joint3` | +1 | 0 | 右轮输出轴 |

尤其注意 `left_knee_joint` / `right_knee_joint` 是主动轴，不能直接赋给被动膝关节 `L_joint2` / `R_jonit2`。右膝模型名称中的 `jonit` 是现有资产的实际拼写，镜像沿用原名。

四个腿轴分别处理多圈编码器，再以声明的左右装配分支求相对角；轮轴同样保留原始多圈值。画面上的旋转可按整圈等价显示，记录中的原始值和模型坐标保持多圈信息。轮子的 15.62 减速比已经在 RMCS 硬件层处理，镜像不会再次除以它。

IMU 采用右手坐标系：X 前、Y 左、Z 上。`imu_body` 中的朝向表示机体到惯性世界的旋转，安装外参已由硬件层处理；ROS 字段为 `x,y,z,w`，镜像内部为 `w,x,y,z`。镜像只转换一次四元数顺序，不再次施加安装变换。

映射文件分别绑定当前训练资产、视觉网格和 RMCS 标定来源的 SHA-256。标定的历史源模型 hash 为 `7f62e0b4…`，当前含独立云台的训练资产 hash 为 `8a9d7595…`，这两个身份分别检查。修改模型或标定后，应审查并更新映射及其记录，而不能直接跳过校验。

## 独立解算与 GUI

`src/wheeled_algo/wheel_leg_mirror.py` 直接读取训练模型的关节树、关节限位和六个闭链锚点约束，以六个测得的主动轴固定输入，求解其余被动关节。它不调用 RMCS 的被动膝查表或腿长解算。

GUI 显示：

- 完整车体、随车体旋转的红 X / 绿 Y / 蓝 Z 坐标轴。
- 六轴 API 角、映射后的模型角、API 角速度和名称对应表。
- IMU 四元数、机体角速度、重力方向差、传感器反馈年龄。
- 底盘使能、DM ready、RMCS 估计器有效性；这些仅为标注，不阻止测量姿态更新。
- 每侧独立计算的膝内角、腿长、轮心在机体系的位置，以及相对 RMCS 结果的误差。
- 闭链锚点残差和模型坐标转换误差；运行 JSON 中保留完整各轴误差。

默认固定世界位置 `(0,0,0.65)` m，IMU 更新车体朝向。用 `--base-position X Y Z` 修改显示锚点。高度、水平位置和脚轮接地关系不是测量结果；不应从画面离地量反推真实车体高度。没有云台反馈，因此云台保持标注的名义零位。

画面没有重力、接触、碰撞、倾角或角速度门控。悬空、倒置、手持及快速运动均按返回的朝向显示，不施加扶正或落地约束。名义膝角范围及模型关节限位只产生诊断提示，不限制解算输入；镜像不会夹紧实测角度。机构锚点的闭合关系仍需满足，否则无法确定被动杆件姿态。

显示线程只渲染最近完整快照，接收线程可按发布频率记录全部完整快照；GUI 频率不等于传感器采样率。`--fps 30` 可降低显示负担。

## 时间戳、失效与重连

发布器在同一执行器周期读取所有接口，五个话题使用同一个 ROS 时间戳及 `chassis_body` frame。CAN 与 IMU 的实际采集异步，消息还保留七个传感器的单独反馈年龄，不能把共同发布时间解释成硬件严格同步采样。

接收端仅合并时间戳完全相等的五个话题；缺少任一话题不会拼接旧帧。以下情况会保持最后有效姿态并标红：

- 超过 `--stale-after`（默认 0.2 s）没有收到完整快照，或连接断开。
- 单个传感器反馈缺失或年龄超过 40 ms。
- 标定、零点、消息 frame 不一致，数值无效，或闭链残差超过 2 µm。

`feedback_fresh=false` 或 `state_valid=false` 本身不会冻结显示。只要原始编码器、IMU 和各自反馈年龄合格，镜像继续独立重建，状态标注为 `LIVE / DIAGNOSTIC`。估计器无效时，不显示其膝角、腿长、模型坐标和重力对照差，避免把旧参考误当真值。重力对照即使明显不一致，也只显示差值，不修正或拒绝当前姿态。

手动加速可能影响实车已有 IMU 融合器的输出精度；镜像按其实际四元数显示，没有额外的加速度/重力融合。原有实车执行器保护与控制逻辑未因可视化而改变。

接收器在断线后自动重连，重新发现话题与订阅 ID。它不清零姿态或外推运动。界面中的 receive age 是本机收到完整快照之后经过的单调时间；未同步两台机器时，它不能表示端到端网络延迟。回放结束显示 `REPLAY END` 并保持末帧。

## 记录、回放和诊断文件

JSONL 第一行为模型/映射身份，随后每行包括原始遥测、RMCS 参考结果、传感器年龄、重置计数、源时间戳及本机相对接收时间。新记录采用排他创建，不覆盖已有文件。回放首先检查模型和映射身份，再按相对接收时间播放；它复现已收到的状态序列及其中的停顿。

```bash
# 半速、循环回放
scripts/view_wheel_leg_live.sh --replay session.jsonl --speed 0.5 --loop

# 有限时长检查，输出最后状态、渲染统计与一张视口截图
scripts/view_wheel_leg_live.sh --demo --seconds 10 \
  --output /tmp/wheel-leg-mirror-check \
  --screenshot /tmp/wheel-leg-mirror-check/viewport.png
```

`--output` 和截图目标同样要求为新路径。`runtime.json` 含最后有效几何状态和当前接收状态，`summary.json` 含本次接收/渲染/拒绝计数；它们用于镜像诊断，不参与训练指标记录。

## 核对步骤

1. 在已知静态姿态比较实物与模型的大腿方向、被动膝内角和轮心位置，并保留尺量或照片证据。
2. 逐轴变化，再让同侧双主动轴共同变化，检查左右、符号和闭链传动。
3. 对比机体俯仰、横滚和转向，检查坐标轴与 IMU 四元数方向。
4. 对比跨圈前后姿态，断开并恢复网络，确认冻结标红、重连恢复和离线复现。

这些操作检查接口与几何约定。实际执行仍需按照既有实车操作流程安排。镜像不能证明摩擦、气弹簧、力矩响应或策略表现已经匹配实车。

## 软件检查入口

在有 NumPy、SciPy、pytest、rosbags、websockets 的 Python 环境运行：

```bash
PYTHONPATH=src python -m pytest -q tests/test_wheel_leg_live_mirror.py
```

ROS 发布器的独立软件检查位于 `tests/native/`。先按 RMCS 约定完成 `build-rmcs`，再在同一个已 source 的 ROS 构建环境运行：

```bash
python3 /home/noir/Documents/workspace/isaac_wheeled_rl_train/tests/native/check_wheel_leg_telemetry.py \
  --rmcs-root /home/noir/Documents/workspace/RMCS \
  --config /home/noir/Documents/workspace/RMCS/rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-wheelbipe-recovery.yaml
```

该检查复用 `rmcs_rl` 构建生成的编译参数和选定 YAML 的估计器/镜像参数，向生产状态估计器绑定合成硬件反馈，再通过 `WheelLegMirrorComponent` 发布并订阅五个 ROS 话题，检查真实标定接口接线、发布节流、同帧时间戳、无效参考、重置计数，以及底盘失能、DM 未就绪时姿态仍持续变化。它不创建硬件组件或 CAN 连接。
