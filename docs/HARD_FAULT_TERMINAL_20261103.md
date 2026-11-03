# 已有硬故障终止帧结算

新合同 `contracts/wheel_leg_prone_motion_v4_terminal_settlement_v1.json` 在 physical-interface 合同上只增加显式 `terminal_transition`。未启用的历史合同沿用原结算。

实际 acdd92ee 运行在第887–911次更新间出现多个极端 critic loss；第900次 value loss 为5.56796e29，同一事件有极大角速度、接触力及负奖励。1032次完整更新正常封存。原逻辑对已经触发机械或冲击终止的状态，仍先计算平方速度/冲击等密集代价，再叠加固定失败罚；这使损坏的终止后状态污染GAE和critic。巨幅物理失稳的历史lane和精确物理起因尚未查清。固定750和1000策略的各24秒、256环境Gaussian回放未复现该历史尖峰，不能冒称已原生复现巨幅失稳。

修复让已有 `mechanical`、`impact_bad` 或原有分组冲击故障终止帧仅支付原本角色的失败罚（自起40、其他3）。这些合法的终止学习样本仍进入PPO，bootstrap仍关闭。不使用闭链误差作为输入，不增加终止条件，不改变reset时间、物理、动作、正常帧standing-v4密集奖励、权重、普通refall/超时/位移失败结算。这里确实改变了硬故障终止帧的结算，故使用独立合同和全新learner，不能称同合同resume。

TensorBoard新增 `ProneMotion/hard_fault_terminal_events`（每策略步事件数，RSL在64步rollout上取均值）和 `ProneMotion/hard_fault_unsettled_reward_abs_peak`（各策略步原始终止罚绝对峰的rollout均值）。原始物理峰、密集奖励项和闭链诊断继续保留；诊断中的原始故障值不代表实际送入PPO的reward。诊断另存 `reward_before_terminal_settlement` 与 `hard_fault_terminal`，用于逐lane确认。

此修复仅针对已证实的终止帧学习污染；并未证明自起、静站、反向跟踪或物理稳定性已修好。此前750策略的收腿与支撑进度仍不能代替完整自起验收。新运行仍需四朝向、20秒主动自起期限及完成后的原保持段。物理200Hz、策略100Hz、PD200Hz、64/32、40–110度、40Nm、闭链record_only保持。

第一次原生启动被旧奖励核算断言拦截，0次PPO，名义/随机化均72/73；原断言仍要求硬故障帧支付全部密集项。已按显式新合同补充独立核算：先逐lane验证原始奖励保持，再核对已有故障掩码和实际终止，最后核对原有角色失败罚结算。保留该负结果，不绕过启动检查。
