# v202 传统自起：实现及实际验证边界

本轮已实现与传统项目共用的 C++ 自起核心、DM 速度伺服接入和独立原生 Isaac Sim 测试入口。**目前是未准入的候选，尚未完成全部倒地姿态的正确快速自起。** 严格侧躺仍未解决，随机云台下倒扣覆盖不完整，部分轨迹存在 CAD 相交；不能以少数起身片段代替整个要求通过。

工作区：

| 用途 | 工作树与分支 |
|---|---|
| 独立 Isaac 实验 | `/home/noir/.codex/worktrees/9085/isaac_wheeled_rl_train`，`dev/traditional-recovery-v202`，基于腿摩擦分支提交 `75e1b5b6dac063fd002d5c14dbca009c3943bbdb` |
| 用户指定传统项目 | `/home/noir/.codex/worktrees/rmcs-traditional-recovery/RMCS`，`dev/wheel_leg_tradition`，基于 `61d19a0c` |

原训练工作树及原 RMCS 工作树未修改；用户自己的 v4 learner PID 3732465 保持运行。本实验 CPU/headless，0 次 PPO、0 次实车/CAN 启动，无本地 TensorBoard 服务。这里只写现有 TensorBoard event 格式，不改监控端口或远端训练。

## 使用的机构和控制方法

使用认证的 `wheel_leg_v2.0.2/friction_v1/training`：21 刚体、20 树关节、六个外部闭链约束、两个气弹簧滑块、总质量 21.44174 kg。保留最新腿动摩擦、气弹簧物理力与对应前馈、轮 real2sim 惯量/被动损耗/响应。云台 yaw/pitch 全程无电机驱动；支持随机 yaw。没有把云台固定，也没有删除气弹簧来换取成功。

非轮材质临时静/动摩擦 0.7、`max` 合成，地面 0.5，因此有效非轮接触系数为 0.7；轮材质仍 0.7、`average`，与地面有效 0.6。源训练 USD 不改写。轮碰撞在独立场景中使用 64 边凸包，避免已有圆柱接触表示的穿地负例。被动旋转关节和气弹簧滑块保留原生研究中的 0.002 数值阻尼，不把它称作新辨识的电机摩擦。

`scripts/simulate_traditional_self_righting.py` 编译并加载 RMCS 的 `SelfRightingCore`（ABI v6），同时保存源文件快照、哈希、原生全关节/刚体轨迹及 event。核心执行 7 翻正 → 3 收腿 → 2 将轮放到身下 → 1 伸腿 → 4 接稳 → 5 完成。四个关节以实际 CAD 输出轴角度规划，左右腿前后组合分别处理。默认参考巡航 1.5 rad/s，参考加速度 3 rad/s²，主动电机总力矩 40 N·m。

速度外环及积分前馈对应传统 DM 的 `Kp=0, Kd<=5` MIT 速度伺服。Isaac 隐式驱动以 `v_cmd + feedforward/Kd` 表达同一速度伺服方程；这个带前馈的数学偏置不是额外允许的物理转速。原生 motor force 上限 40 N·m，辨识被动摩擦另行施加。明确添加四轴 angular DriveAPI，不能仅设置 tensor damping 就假定被动研究 USD 存在电机驱动。

四轴实际速度和参考速度分别记录，不截断实测值。当前可起身样本依赖 PhysX 2.8 rad/s 原生轴限速，同时核心在接近 3.14 时制动、超过 3.140001 时锁存故障。去掉原生限速的对照出现实际超速；原生限速在接触冲量下也不是严格的硬限速。**尚未证明仅凭传统控制器或 DM 固件可在实车满足实际速度最高 3.14 rad/s。**

## 可复现入口

```bash
cd /home/noir/.codex/worktrees/9085/isaac_wheeled_rl_train
source scripts/resolve_isaac_python.sh
wheeled_prepare_native_loader
/home/noir/miniconda3/envs/isaacsim6/bin/python \
  scripts/simulate_traditional_self_righting.py \
  --rmcs /home/noir/.codex/worktrees/rmcs-traditional-recovery/RMCS \
  --output reports/traditional_self_righting_20261031/new-run \
  --families prone inverted left right front rear \
  --leg-combinations ff rr fr rf --seconds 30 --random-gimbal \
  --flip-asymmetry 75 --balance-kp 35 --balance-kd 5 --balance-kv 5.2 \
  --pitch-trim-gain .3
```

`ff/rr/fr/rf` 按左/右腿依次表示前/后。`--speed`、`--acceleration`、`--dt`、`--control-dt`、腿形、域材质及控制参数均由 CLI 指定；`--prepare-only` 只生成起点和配置。`--variants` 读取有名称的参数对照，批次保留每个环境的配置。隐式电机模式要求同一批次的速度增益一致，避免控制器增益与原生 DriveAPI 不一致。

上述组合起点是几何放置候选，**随机 yaw 可能与某些腿形构成初始 CAD 相交，不能将全部组合都当作物理可实现的倒地姿态。** 判定前需检查 CAD。复放此前采集的真实被动端点可改用：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python \
  scripts/simulate_traditional_self_righting.py \
  --output reports/traditional_self_righting_20261031/new-measured-run \
  --starts reports/side_fall_leg_friction_20261031/final/side_pose_candidates.json \
  --seconds 30 --flip-asymmetry 75 --balance-kv 5.2 --pitch-trim-gain .3
```

复放校验车型哈希、20 关节键及数值，保留全部端点残余关节速度和根 COM 速度；没有人为归零让初态更容易。候选库的 `static_equilibrium_proven=false` 继续有效。库中的 family 是原始分类元数据，具体起姿必须查看根四元数/实际重力方向；例如 `side_candidate_007` 的初始倾角约 31°，它起身不能证明严格 90° 侧躺成功。

## 计量与判定

自起序列完成：Capture/Complete 阶段连续 0.5 s 同时满足倾角 <8°、基座高度 305±30 mm、角速度模 <0.5 rad/s、水平速度模 <0.12 m/s、两轮名义表面离地绝对值 ≤3 mm、腿轴速度绝对值 <0.6 rad/s、无故障；此前实际四轴速度 ≤3.140001 rad/s、六闭链最大误差 ≤3 mm。

`recovery_sequence_completed` 只表示上述首个连续窗口。`passed` 还要求整个控制观察窗无故障/超速/闭链拒绝，并在观察末尾再次有连续 0.5 s 稳定窗口。后者不能解释成整个 20/30 秒从未失稳，更不能代替长期轮接地、静站漂移等验收项。CAD 自相交、真实碰撞动力学、硬件映射及完整传统交接另行判定。

单位为轴 rad/s、位置 m、角度 deg、力 N、时间 s。接触力 ≤2 N 只作诊断，不作为轮离地判据。峰值接触力不能直接换算机械损伤或材料安全界限。起身和翻正段不计作完成后的站立接地验收。

## 已获得的证据

- `offline002.json`：2048 组含多圈编码器角度的 C++/Torch 气弹簧补偿对照，最大误差 `2.3093e-14 N·m`；非法编码器及实际超速故障检查通过。只证明离线接口与计算一致。
- RMCS `build005`：7 包构建、2 个 CTest 通过。覆盖四组合的理想/滞后跟踪参考连续性、报文量化、故障/复位，不是车辆起身证明。
- `native028`：1 ms，六方向×四腿组合、随机 yaw，24 次、每次 30 s。正向趴地四组合首次完成分别约 3.583/3.875/3.553/4.132 s；其中后三个通过上述观察末尾判据，ff 未通过。倒扣及严格左右侧躺均未完成；部分前/后倒地有完成窗口。整组未通过。
- `native029`：15 个真实被动端点复放，7 个获得首个自起完成窗口，3 个通过观察末尾判据。严格左右侧躺仍失败，不能把倾斜 31° 的成功样本算作严格侧躺。
- `native024/025`：固定云台初始 yaw 的旧候选出现倒扣四腿组合的成功片段，部分通过观察末尾判据；源和参数各不相同，不能拼成统一控制器通过证据。
- `review028b`：对 `prone_rr/inverted_rr/left_rr` 真实逐刚体位姿，以 0.2 s 间隔筛查至首完成或首故障，共 214 帧，18 帧出现新 CAD 相交。部分相交已存在于几何起点，部分出现在运动中。此筛查不简化 CAD 三角网格；图片简化仅影响渲染。即使某段行为判据通过，也不能机械准入。
- `native030/031/032`：非相邻原生自碰撞重测发生内部碰撞包络数值发散，停止这些自建进程，保留日志、源和独立 `closure.json`。Kit 的 SIGINT 直接退出码为 0，原 `receipt.status=running` 是历史快照，不是仍在运行或成功。这些是碰撞模型无效负例，不是合格轨迹。
- `native033`：固定 yaw、0.5 ms 物理步长、1 ms 控制、正趴/倒扣各四组合。正趴四组合约 3.033–4.903 s 出现完成窗口，仅 fr 通过观察末尾判据；倒扣四组未完成，fr 全窗含故障后速度达到约 3.468 rad/s。半步长结果不能替代随机 yaw 覆盖或机械准入。
- `review029`：对四个真实端点的起身片段做 0.1 s CAD 表面抽查，188 帧中 5 帧有新增腿—云台相交，四个片段各有至少一次标记。真实端点也必须检查运动过程，不能仅凭初态通过筛查就认为轨迹可用。
- `native034/036`：临时由原始 CAD 生成新的凸分解（每刚体最多 64 凸包、64 顶点/凸包），保留质量、惯量、闭链与电机模型，开启非相邻自接触。单个 `side_candidate_009` 能数值推进，但分别在收腿目标 75°/48° 的阶段未到达目标，故障 5，没有自起完成。
- `native035/037/038`：用七个真实端点检查新凸分解，仍在起动后约毫秒量级出现 >10 mm 闭链误差，自动拒绝并保存 partial trace。研究性排除单腿内部碰撞对、将 CAD 接触偏置从 2 mm 降到 0.02 mm 都未消除这一问题。没有把这些修改写回训练资产，也没有据此放宽闭链门；新的碰撞表示仍未验证。

最新结果见 `reports/traditional_self_righting_20261031/summary002.json`（38 次原生尝试、两个 CAD 回放筛查）；初版 `summary.json` 和每轮原件保留，不覆盖失败数据或以最佳片段覆盖失败姿态。002 额外核对每轮源文件快照是否存在并与回执哈希一致；早期仅保留哈希而没有快照的调试不能称为完整源码可复现。

```bash
# 离线渲染实际刚体轨迹并以原始 CAD 做有限时刻筛查
PYTHONPATH=/tmp/fallen_pose_analysis_deps \
  /home/noir/miniconda3/envs/isaacsim6/bin/python scripts/review_traditional_recovery.py \
  --run reports/traditional_self_righting_20261031/native029 \
  --output /tmp/recovery-review-new --cases side_candidate_009 --audit --audit-period .1
# 不将不同源/参数的最佳片段拼成全姿态通过
/home/noir/miniconda3/envs/isaacsim6/bin/python scripts/summarize_traditional_recovery.py \
  --root reports/traditional_self_righting_20261031 --output /tmp/recovery-inventory-new.json
```

离线 CAD 分析依赖放在临时目录，未安装进正在训练的环境。`--collision-geometry cad-decomposition`、`--filter-leg-assembly` 仅用于研究碰撞表示，当前都有原生负例，不能作为默认可用资产。最新 runner 对异常物理状态中止并保存局部轨迹，对 SIGINT/SIGTERM 明确记录中断及非零退出，修复 Kit 默认信号处理留下 running 快照的问题。

## 当前障碍

严格 90° 侧躺时，当前结构的腿/轮与地面仍有约 61–64 mm 间隙。`side_support001.json` 在每腿 4176 构型和云台 2016 构型的有限几何网格中，即使将云台极值作为乐观包络，COM 仍未越过当前底盘支撑面，距离最近边界约 5.65/10.98 mm。该结果不是动态自起不可能的数学证明；它说明仅靠缓慢改变腿角不能假定会自然翻正。已测试的有限惯性摆腿候选未完成侧躺翻正。

源资产的保守凸包存在机构内部伪接触；原生自碰撞关闭时又不能防止腿穿越自由云台。下一轮必须先修复并验证内部自接触碰撞表示，剔除相交初态，在此基础上设计单侧惯性动作或获得允许后研究云台辅助。倒扣需要按云台和两腿构型调整翻正后收腿/支撑顺序；持续站稳及完整传统控制交接也尚未完成。当前配置保持关闭，`hardware_deployment_ready=false`。
