# 闭链误差仅记录，2026-11-03

用户要求取消闭链保护后从零重训。当前 v4 使用显式的
`wheel-leg-closure-monitor-v2 / record_only` 合同，5 mm 是诊断参考值。
每个物理步继续测量六组闭链端点的几何距离，每个策略周期保留峰值。
超限不再触发 episode 终止、重置、终止罚分或课程冷测拒绝。
训练未改用相对编码器目标；奖励与前一版 standing-v4 完全相同。

TensorBoard 保留 `UnifiedGetup/closure_gap_max_m`，新增平均值、超参考值比例、
主动区间超限数、闭链终止数和 record-only 标志。只读诊断保留原始 `gap` 和
`closure_gap_exceeded`；`closure_gap` 表示真正由闭链导致的终止，当前应为 false。
冷测同时报告原始峰值、参考值和超限情况，不把大误差伪装成小误差。

其他物理失败、冲击、边界、超时及站立/自起行为判据继续生效。256 环境、
seed107、10000 更新、保存间隔250、77标准正向自起/179静站起步不变。
策略100 Hz、物理步长0.005 s、decimation2、逐物理步PD200 Hz、PhysX64/32、
40–110°物理膝限位和腿电机40 N·m上限不变。旧 v1 guard 合同继续执行原保护；
旧 canonical 合同已逐字节保存为 before_closure_record_only 文件。

原生探针包含显式10 mm测量注入，检查真实终止/奖励路径，随后恢复测量函数。
注入不改机器人状态，属于接线验证；自然物理误差另行记录。CPU及原生验证不证明
已学会自起，也不授予机械精度或实车资格。新运行从全新actor、critic、
normalizers和空Adam开始，不读取旧策略作为初始化。
