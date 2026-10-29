# 本机当前策略的只读任务梯度诊断

## 动机与边界

pilot019/020从同一个2945状态各实际PPO128更新后，原八场景均0/8。
020关闭值裁剪降低了训练critic loss，但前进仍只有+.01344m/s，三档高度仍约.31m。
command_probe005原顺序复现了代表轨迹，命令列准确、名义样本动作/请求力矩未限幅；
同输入命令响应非零但弱。因此不能再按同一“继续适应”理由扩训。

本次只检验共享actor的不同命令样本是否在**旧策略首个score梯度**上相互抵消，
以及不同命令的优势均值对有限批梯度有多大贡献。不把梯度cosine、优势均值差异
或离线bucket中心化当唯一根因、真实Adam更新、改法有效或长训准入。

不改变任何训练合同、人口、课程、奖励、探索、物理、控制、出生或阈值。
不调用PPO.update或optimizer.step，不写checkpoint，不推进Adam/课程/消费时钟。
全部实际消费仍4417/20000，剩15583；继承1344、谱系上限18656、
transition_ceiling3271557120保持。即使换回旧控制模型，也不能退款其他臂。

## 原生记录与中性核对

共同诊断起点为完成的pilot020-3073及其保存CPU/CUDA/环境RNG，恢复后创建新episode，
不声称模拟器中途轨迹重放。模块和合同读取020冻结源
`runs/local_scratch_sources/20261018T064949Z-38101e30/source`，逐项核对原startup101来源。

串行两个独立进程，各512环境、3×384原生物理/策略帧。baseline用020冻结的原
`diagnose_local_credit.py --event-attribution --birth-attribution`。
候选使用新可选`--policy-credit`，仅在完整rollout边界复制原生storage已有的
policy输入、action、Gaussian mean/std及log-prob；不增加rollout中的actor forward或sample。
记录脚本自身源码和SHA并复制到输出，不修改020冻结目录。

两进程的全部initial/RNG/首Gaussian、全部原有rollout数组和critic tail必须逐位相同，
actor/critic/17Adam及原时钟各保持。只认证已记录数组；不扩称全部隐藏模拟器状态。
失败完整保留，配对不通过时不得据新增记录作机制结论。

## 纯CPU分析

从同一checkpoint构造原生MLP/Gaussian，精确load权重。与记录的CUDA mean/std/log-prob
核对数值误差；所有梯度仅在CPU分析副本通过autograd.grad求得，参数、.grad、RNG和
optimizer不改变。CPU针对性检查验证其与安装版本原生ratio=1的裁剪surrogate梯度
一致，包含可学习std，不额外采样。

按记录的当前命令分成互斥穷尽bucket：低/中/高静站、前/后行、正/反yaw及其余。
conditioning ownership严格沿原生mask，所有原生学习样本保留。使用原生整rollout
归一化advantage，将每个bucket优势拆成bucket内中心化项和均值项，验证两者梯度
之和重建原始score梯度；实际训练数据和PPO计算完全不变。

逐rollout报告样本量、优势均值/正数比例、残差、梯度范数、任务cosine和归属。
三段连续rollout来自一个checkpoint/RNG，不是独立训练种子。此处未执行mini-batch
排列、ratio裁剪后的epochs、entropy、梯度裁剪或Adam，所以不能预测真实更新方向。
仅发现负cosine也不足以删除任务、降低奖励或采用bucket中心化。

指标写入pilot020原RSL event的`LocalDiagnostics/PolicyCredit/rolloutN`，step3073。
配对/分析产物和失败另存机器回执。只有结合该证据与新轨迹/控制/实付，才能提出
下一项有界、明确命名和独立设计/预算认证的调整；本计划不授权自动追加PPO。
