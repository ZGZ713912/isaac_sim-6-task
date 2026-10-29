# pilot007最终结果：普通静止站立奖励候选仍未学会起立

更新于 `2026-10-27T15:09:14.171113+00:00`。本轮128/128次更新及固定冷测均已自然完成，状态`completed_unqualified`。
**21场标准趴地冷测起立0/21、联合站立基线0/21、10cm原地筛查0/21。**
模型已封存，learner、同项目只读复核和GPU上下文均已退出；没有续训或启动下一轮。

## 实际行为

| 原始测量 | 结果 |
|---|---:|
| 实际起立完成事件 | 0 |
| 21场终止 | 全部在约0.5m位移处，8.53–9.115s |
| 10s未起立超时 | 0；均在此前已终止，不能记为全部超时 |
| 全过程最大XY偏移 | 50.027cm |
| 原始车体高度峰值 | 270.11mm，约0.405s |
| 1s后高度中位数 | 200.05mm |
| 1s后倾角中位数 | 14.015° |
| 1s后平移速度中位数 | 5.908cm/s |
| 最后有效帧平均膝夹角 | 43.331° |
| 双轮速度峰值，200Hz快照 | 9.716rad/s |
| 腿速峰值，原始1kHz流 | 38.637rad/s |
| 闭链间隙原始峰值 | 0.08262mm |

21场的首次0.5m越界与终止处于同一个采样步；这是时序对应，不宣称没有同时出现其他保护条件。
模型在收腿后长期停在约200mm/14°，随后缓慢移动到训练范围边界，未形成305mm水平静站。
没有实际完成事件，因此不能对30s站立、站立接地或起立后抖动授予资格；
不从后缀重选“最好窗口”，不将起立过程的底盘接触纳入站立接地分母，
也不把≤2N直接判作明显离地。单次连续≤2ms失载宽容规则保持，但本轮无站立窗口可应用。
21场是同姿态的名义复制，不是21个独立随机种子。

## 奖励调整实际改变了什么

按用户纠正，真实103°失能趴地reset及42°/±0.6rad固定可训练初始bias未变。
P0直接调用`upright_locomotion.reward_terms`，使用零前后速度、零yaw及305mm命令，
逐项复用普通静站的10项高度、水平、速度/yaw跟踪与动作/腿速/力矩成本。
持续收腿、轮心对准、部分抬升奖励和8/s硬稳定瞬时加分已取消；路径势函数gain从3降至0.1。
保留宽高度误差平方成本4/s和真正完成前时间成本1/s；最高4/s连续联合静站奖励同时需要
高度、水平、低速度/角速度、双轮同时支撑且无车体支撑。
四腿与双轮六个输出始终可训练，没有固定腿或电机脚本。

TensorBoard128个更新点的稳定比例、起立完成事件均为0。
联合站稳奖励的rollout平均实际支付峰值仅约2.32e-7/步，末轮约5.19e-10/步，
远低于理想完全支撑安静状态0.02/步；普通20mm高度高斯在大部分低位状态也近乎为零。
第11–30与109–128更新窗口：主动高度均值211.44→205.88mm，倾角18.16→15.16°，
平移RMS6.89→4.42cm/s。它减少了运动和倾角，但没有学会抬到目标高度。
这些窗口是本轮描述，不是配对因果实验，也不能把“初始姿态”或某个系数认定为唯一根因。

原生接线检查和实际reward付款已通过，数值/优化器完整性也通过；
**这轮只改奖励的候选没有解决站立目标。** 下一项诊断建议是独立305mm直立出生的静站基线，
用全新actor、critic和空Adam验证基础静站，再设计趴地到站立的样本课程。
该方案尚未启动，不迁移本轮或旧模型权重。

![pilot007 TensorBoard曲线](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/pilot007_final_training.png)

图中均为本轮TensorBoard event的rollout统计，主动高度已排除失能段；
平均XY偏移不能替代冷测的全过程真实峰值。周期尖峰包含episode边界影响，不能直接当作动作成功。

## 冻结来源、消费和完整性

- 训练源：`b48a5139e9e0ffdf37cc214b36de7fe80930c1d3`，275个Python源匹配冻结提交。
- 源归档SHA-256：`55a67b0a298534a042b996e48ad9442569f4a1e3af83e633b8d720095419b160`。
- 最终模型SHA-256：`b6025a4c47e5a00533cc4252773ad2de591a24bc0056415b50e64c0c1ec6282a`。
- 256环境、rollout256、seed261005；从零actor、critic、空Adam，无旧模型读取、teacher或donor。
- 首轮12次Adam、9个actor与8个critic参数张量变化，六轴输出均实际更新。
- 最终17个Adam参数状态均为1536步；完整模型、RNG、冻结源、归档、配置、进程/GPU退出等16项检查通过。
- 9模块85项选定CPU检查通过，1项缺旧model fixture的metadata检查明确排除，负记录保留；
  新源18项原生接线检查通过、0终止、0PPO。以上工程检查均不授予行为资格。
- v2 requested强度0.25训练，腿P/D和气弹簧系数固定1，IMU晃动关闭；最终独立惯量±6%、
  完整范围、单组/组合域、原低摩擦和独立种子的鲁棒性仍未验收。
- 当前工作树HEAD还含只读复核器修复`47c74f12`；它在275个训练源之外，未热修改本轮训练源。
- 各代实际更新4+16+64+166+128+53+128＝**559**，不退款、不续训；不拼接不同代际最好分项。

| 代际 | 完整更新 | 实际结果 |
|---|---:|---|
| pilot001 | 4 | 正常停止，未续训 |
| pilot002 | 16 | 冷测0/21起立 |
| pilot003 | 64 | 冷测0/21起立，仅收腿 |
| pilot004 | 166 | 正常停止，冷测0/21起立 |
| pilot005 | 128 | 自然完成，0/21起立；瞬时约300mm后回落 |
| pilot006 | 53 | 正常封存，0/21起立；1s后约208mm/17.35°，最大XY约5.15cm |
| pilot007 | 128 | 自然完成，0/21起立；约200mm/14°，随后移动约50cm终止 |

## 长期13项验收

仍按原`docs/WHEEL_LEG_ACCEPTANCE.md`及`contracts/wheel_leg_acceptance_requirements_v2.json`，
本代无任何新增完整能力资格，未测不能算通过。

| 能力 | 状态 | 本代证据与边界 |
|---|---|---|
| 初始多姿态自起 | 未通过验收 | 标准趴地0/21；其他倒地/腿不对称分布未测 |
| 运动中真翻倒并恢复命令 | 未通过验收 | 未测 |
| 低/中/高静止站立 | 未通过验收 | 无实际起立完成事件，后续30s站立无可用窗口；三高度独立直立测试未测 |
| 低速正反直线 | 未通过验收 | 未测 |
| 高速正反直线 | 未通过验收 | 未测 |
| 双向100rpm车体自旋，至少80rpm | 未通过验收 | 实际车体转速与双向保持未测 |
| 前后配合左右转弯四象限 | 未通过验收 | 未测 |
| 静止变高 | 未通过验收 | 未测 |
| 运动中变高 | 未通过验收 | 未测 |
| 自旋中变高 | 未通过验收 | 未测 |
| 高速正反急刹 | 未通过验收 | 未测 |
| 平顺及腿轮协调 | 未通过验收 | 保留起立段原始诊断；完整动态谱和站立平顺未测 |
| 单组/组合域、噪声、种子鲁棒性 | 未通过验收 | v2强度0.25训练；名义冷测失败，完整强度/独立分组/低摩擦/独立种子未测 |

本轮没有实车动作、实车资格或集成ONNX部署资格。

## 可复核文件

- [最终行为摘要](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/pilot007_final_behavior_summary001.json)
- [完整性检查](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/pilot007_completion_integrity001.json)
- [TensorBoard最终统计](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/pilot007_status_tb001.json)
- [曲线及event来源回执](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/pilot007_final_training.provenance.json)
- [固定冷测判定](/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/standard_prone_20261027/pilot007_quiet_standing_dr025_u128_e256/evaluation/result.json)
- [原始行为与1kHz接触/腿流](/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/standard_prone_20261027/pilot007_quiet_standing_dr025_u128_e256/evaluation/native_trace.npz)
- [主回执](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/upright_curriculum_20261027/receipt.json)

本轮训练和验收已结束，heartbeat监控已删除；保留全部模型、日志、event和负结果。
