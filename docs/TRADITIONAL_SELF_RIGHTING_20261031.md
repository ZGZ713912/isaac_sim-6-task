# v202 传统自起：实现及实际验证边界

本轮已实现与传统项目共用的 C++ 自起核心、DM 速度伺服接入和独立原生 Isaac Sim 测试入口。**目前是未准入的候选，尚未完成全部倒地姿态的正确快速自起。** 修正碰撞过滤后，正趴及个别严格侧躺端点出现完成窗口；随机云台、倒扣及腿前后组合尚未全面覆盖，不能以少数起身片段代替整个要求通过。

工作区：

| 用途 | 工作树与分支 |
|---|---|
| 独立 Isaac 实验 | `/home/noir/.codex/worktrees/9085/isaac_wheeled_rl_train`，`dev/traditional-recovery-v202`，基于腿摩擦分支提交 `75e1b5b6dac063fd002d5c14dbca009c3943bbdb` |
| 用户指定传统项目 | `/home/noir/.codex/worktrees/rmcs-traditional-recovery/RMCS`，`dev/wheel_leg_tradition`，基于 `61d19a0c` |

原训练工作树及原 RMCS 工作树未修改；用户自己的 v4 learner PID 3732465 保持运行。本实验 CPU/headless，0 次 PPO、0 次实车/CAN 启动，无本地 TensorBoard 服务。这里只写现有 TensorBoard event 格式，不改监控端口或远端训练。

## 本次继续工作的关键修正

独立实验入口曾把 `FilteredPairsAPI` 写在 articulation root `base_link`，或把其他连杆的排除目标指向该根。这样会把整条关节树与目标连杆过滤，导致腿—云台接触也被意外屏蔽。**旧轨迹成功窗口均保留为调试记录，不能用作机械通过证据。** 此问题仅在独立测试入口中修正，没有改写用户训练资产。

`contact_control007` 用简单关节树做七组原生对照：开启/关闭自碰撞、排除兄弟连杆、根指向连杆、连杆指向根、根下具体 collider 指向 collider，以及独立关节驱动力检查。全部对照符合预期。`native067/068` 在旧配置下自碰撞开关轨迹完全相同；`native070` 对同一个故意相交的 CAD 快照采用修正过滤后，立即读到了腿—云台接触力。后者是相交正控，不是有效自起初态。

`native071` 起清空根节点的成对过滤，并去掉其他刚体指向根的过滤；树父子相邻接触由原生处理。可选的底盘—气簧安装座研究性排除，只写在具体碰撞形状上。当前入口逐条检查过滤关系；开启自碰撞时若根上仍有目标或任何目标指向根，直接拒绝启动。该行为与 [OpenUSD 对 FilteredPairsAPI 的定义](https://openusd.org/dev/api/usd_physics_page_front.html) 一致。

修正后的完成结果仍依赖碰撞表示：

| 原生试验 | 碰撞设置 | 结果 |
|---|---|---|
| `native072` | CAD 凸分解；单腿内部及底盘气簧座研究掩码 | 正趴端点 001：4.521 s；严格右侧躺端点 012：8.983 s；两者通过观察末尾行为门。左侧躺 004 超速失败 |
| `native073` | 重网格化 SDF 512；无额外研究掩码 | 正趴 001：3.411 s。左侧躺 004、右侧躺 012 均未到达目标，故障 5 |
| `native074` | 修正过滤的 CAD 凸分解，六条分步收腿/旋转程序 | 左侧躺 004 的六条对称程序均失败 |

`review072` 对正趴及右侧躺两条原生全刚体轨迹以 50 ms 间隔筛查完整 25 s，共 1000 帧。七帧需复核，其中一次腿—云台三角表面接触的有限 FCL 深度约 0.190 mm、采样分离平移约 0.190 mm；其余标记来自 CAD 表面低于地面约 3.5–3.9 mm。它们不能与旧的厘米级穿越混为一谈，也不是实车变形或材料允许量证明。`review073`、后续批次及其逐项结果另存，持续保留源快照和负结果。

四轴除了原生速度，还记录去圈后的每步角增量除以物理步长。简单接触正控中，位置求解器可能校正约 0.15 rad 而上报速度接近零，因此只查看原生速度会漏掉数值位置跳动；两项都必须满足上限。该有限步长量不是硬件瞬时速度。

当前新增可配置的九列翻正程序、上下腿映射、先保持各自轨道角收腿、切换角速度/倾角/连续等待，以及接稳轮速包络；传统组件、原生 ABI 调用同一实现。`build008` 的七包构建和两个 CTest 通过，`offline005` 的 2048 组补偿对照最大误差仍为 `2.3093e-14 N·m`。构建证据和模型运动证据分别保留。

新增 `cad-gimbal-guards`：仅为两段云台生成 30 个支撑平面构成的外包凸体，默认向外扩 1 mm，其他部位仍采用 CAD 凸分解。质量、惯量、闭链、摩擦、气簧与执行器均不改变。`native086` 的 cooked 凸体查询包含全部原始 CAD 顶点，最小平面余量分别约 0.590/0.854 mm；验证脚本为 `scripts/verify_recovery_guard_coverage.py`。它是有意保守的实验碰撞包络，可能阻止真实空腔允许的路线，失败不能作为机械不可能证明。

这一包络下，严格右侧躺 012 在 `native086` 约 8.914 s 完成，实际速度峰约 2.802 rad/s；严格左侧躺 004 的旧单侧快速路线在 `native085` 约 15.359 s 完成，但同样加入 1 mm 包络后 `native087` 未到位失败。两者不能拼成统一可靠策略。包络复测起点只在初始放置时上移 2 mm 留出余量，完整记录在 `prepare006/starts.json`；控制过程中没有根状态或关节状态重置。

新核心提供 `flip_handoff_orbit_limit_deg` 与 `flip_balance_when_upright`：前者要求两腿都转到车身下方的指定角度范围才离开翻正，后者允许翻正后段使用原有有界轮平衡。默认分别为 0/false，原有流程保持；原生对照已出现通过与失败两种结果，不能假定启用轮平衡总会改善。

`native076` 的正趴 ff/rr/fr/rf 在 30 s 控制窗口内均通过行为门，首次完成分别为 4.565/4.572/4.086/3.950 s；四轴原生速度最大 2.9334 rad/s。`review076` 对每条 610 帧、50 ms 间隔的完整窗 CAD 抽查均无新增标记。其左侧躺 rr 虽在 14.327 s 完成，却有 91 帧腿—云台接触标记，不能保留为已筛查通过路线。新核心 `native084` 重放正趴四组合，复现了相同的四个首次完成时间；逐刚体回放比对另存。上述仍是有限观察窗与有限采样，不是全姿态、连续几何或实车资格。

## 当前推荐的研究入口

先从原始 CAD 放置并筛查随机云台 yaw，再允许全失能被动沉降；生成初态不是静态平衡证明：

```bash
PYTHONPATH=/tmp/fallen_pose_analysis_deps \
  /home/noir/miniconda3/envs/isaacsim6/bin/python \
  scripts/prepare_traditional_recovery_starts.py \
  --output /tmp/recovery-starts-new --seed 261031
source scripts/resolve_isaac_python.sh
wheeled_prepare_native_loader
/home/noir/miniconda3/envs/isaacsim6/bin/python \
  scripts/simulate_traditional_self_righting.py \
  --output /tmp/recovery-native-new \
  --starts /tmp/recovery-starts-new/starts.json \
  --variants reports/traditional_self_righting_20261031/programs/sdf_baseline001.json \
  --self-collision --collision-geometry cad-sdf --sdf-resolution 512 \
  --settle .5 --seconds 30
```

`prepare002/starts.json` 含正趴、倒扣、左侧、右侧各四种腿组合，共 16 个 CAD 放置初态，拒绝并重抽了六个发生新增表面相交的 yaw 候选。因此 yaw 分布经过初态筛查条件化，不是无条件随机覆盖；0.5 s 沉降也不能声称已证明静态平衡。原始 CAD 三角筛查不作实体包含性证明。

研究程序 JSON 中 `flip_program` 每行依次为两腿膝角、两腿轨道角（度）、四轴巡航速度、加速度、到位后保持秒数、物理左/右轮速度；`relative_to_side` 决定前四列是下腿/上腿还是物理左/右腿，轮序始终固定。`fold_first` 决定是否先在各自实测轨道角上收腿。YAML 对应字段使用展平的九列数组。`--stop-when-all-faulted` 只用于搜索时提前结束全部故障的批次，回执保存实际观察时长；不会把缩短的观察窗算作成功。

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

自起序列完成：Capture/Complete 阶段连续 0.5 s 同时满足倾角 <8°、基座高度 305±30 mm、角速度模 <0.5 rad/s、水平速度模 <0.12 m/s、两轮名义表面离地绝对值 ≤3 mm、腿轴速度绝对值 <0.6 rad/s、无故障；此前实际四轴速度及去圈后轴角增量/物理步长均 ≤3.140001 rad/s、六闭链最大误差 ≤3 mm。

`recovery_sequence_completed` 只表示上述首个连续窗口。`passed` 还要求整个控制观察窗无故障/超速/闭链拒绝，并在观察末尾再次有连续 0.5 s 稳定窗口。后者不能解释成整个 20/30 秒从未失稳，更不能代替长期轮接地、静站漂移等验收项。CAD 自相交、真实碰撞动力学、硬件映射及完整传统交接另行判定。

单位为轴 rad/s、位置 m、角度 deg、力 N、时间 s。接触力 ≤2 N 只作诊断，不作为轮离地判据。峰值接触力不能直接换算机械损伤或材料安全界限。起身和翻正段不计作完成后的站立接地验收。

## 早期调试证据（不作为修正过滤后的行为通过）

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

旧阶段汇总见 `reports/traditional_self_righting_20261031/summary002.json`（38 次原生尝试、两个 CAD 回放筛查）；后续版本在 `summary*.json` 独立保存。初版 `summary.json` 和每轮原件保留，不覆盖失败数据或以最佳片段覆盖失败姿态。002 额外核对每轮源文件快照是否存在并与回执哈希一致；早期仅保留哈希而没有快照的调试不能称为完整源码可复现。

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

严格 90° 侧躺时，当前结构的腿/轮与地面仍有约 61–64 mm 间隙。`side_support001.json` 在每腿 4176 构型和云台 2016 构型的有限几何网格中，即使将云台极值作为乐观包络，COM 仍未越过当前底盘支撑面，距离最近边界约 5.65/10.98 mm。该结果不是动态自起不可能的数学证明；它说明仅靠缓慢改变腿角不能假定会自然翻正。这个早期静态网格不排除动态路线。后续修正自接触后的试验已出现个别严格侧躺完成窗口，尚未形成全组合可用的轨迹。

源资产的保守凸包存在机构内部伪接触；原生自碰撞关闭时又不能防止腿穿越自由云台。下一轮必须先修复并验证内部自接触碰撞表示，剔除相交初态，在此基础上继续仅使用四个腿电机和轮电机设计动作。倒扣需要按云台和两腿构型调整翻正后收腿/支撑顺序；持续站稳及完整传统控制交接也尚未完成。当前配置保持关闭，`hardware_deployment_ready=false`。
