# 平地策略200Hz时间合同

2026-10-20用户要求后续新版本使用200Hz策略频率。训练及评测默认计划已改为
`contracts/flat_base_finetune_200hz_v1.json`，旧50Hz计划仍供已保存模型复现。
这里的200Hz是每5ms计算一次动作，PPO的一轮仍由完整rollout和优化器更新组成。

| 项目 | 原计划 | 新默认 |
| --- | --- | --- |
| 策略步长 | 20ms / 50Hz | 5ms / 200Hz |
| 物理和反馈PD | 1ms / 1000Hz | 1ms / 1000Hz |
| 每动作物理步数 | 20 | 5 |
| 每环境rollout步数 | 96 | 384 |
| rollout物理时长 | 1.92s | 1.92s |
| γ | 0.99 | 0.9974905699336811 |
| λ | 0.95 | 0.9872585449014338 |

折扣和GAE按物理时间指数换算；奖励密度仍只乘一次policy_dt，事件成本不按采样次数
重复支付。动作一阶/二阶差分保留20ms参考间隔，在200Hz分别乘4/16。
命令加速度、升降速度、回合和启停时长仍以秒计。新训练每轮有4倍策略样本，
不能把相同轮数称为相同样本量或承诺训练更快。

观测35→动作6和历史长度1保留；previous_action现在来自5ms前。
原std和Adam完整恢复，但探索采样相关性随频率改变，原50Hz策略的闭环行为
必须重新测量，不能从权重相同推断行为相同。电机增益、力矩边界、轮响应和
地面0.2–0.5/轮子0.1的独立材质配置均保留。

新命令合同绑定原1338封存计划，再换算到200Hz。原计划引用的旧RMCS YAML路径
现已不存在，因此不把当前外部部署工作区当成训练来源，也不声称其硬件已切换频率。
新训练入口会验证准确的父checkpoint、计划、源码、完整状态恢复和时间迁移回执。

先生成只读准备回执，不启动训练或仿真：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python \
  /home/noir/Documents/workspace/isaac_wheeled_rl_train/scripts/prepare_flat_policy_200hz.py \
  --output /home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/flat_policy200_preparation_next
```

`policy200_migration001`因旧RMCS路径失败，保留负证据；002首次通过静态准备。
修复升频重复叠加历史critic预热后，当前准备为
`reports/flat_base_speed_20261020/policy200_migration003/`。
`preparation.json`只证明静态准备，不能证明原生状态恢复、容量或策略达标。

72项定向CPU测试通过（`cpu_policy200_001.xml`），覆盖独立worker/design预检、
折扣/GAE/rollout时间、差分成本、200Hz升降参考和错误时钟拒绝。
`policy200_evaluator_preflight002`也通过200Hz策略/1000Hz物理预检。
原生短回放与完整状态/容量结果将单独追加，不以静态预检替代。

`policy200_native001`原生回放已完成，128环境、每策略64回放、8场景、两车高、
地面μ0.2/0.5、两副本、12s，耗时373.97s。实读2400个策略步/12000个物理步，
每步1次动作解码、5次新反馈PD和5次反馈刷新，policy_dt=.005、physics_dt=.001。
原1338权重/后退限定参考完整52/64、50/64，全部判据均11/32，未授行为资格。
两者均为旧权重的零PPO回放；report中的generic trained标签不表示新训练。

0.42m/μ0.5最后5s，原权重前进+2.977、后退+0.355、旋转+4.027/−3.979；
限定参考前进+2.977、后退−2.493、旋转+4.027/−3.981。高位后退仍欠约0.51m/s，
低摩擦高速启动仍有早终止；12s也未覆盖完整正反启停周期。和旧50Hz报告的种子、
布局、时长不同，不能把差异唯一归因于频率，不能授长训或硬件资格。

兼容性扩展检查60项通过、1项失败（`cpu_policy200_compatibility001.xml`）：
失败是旧通用GUI fixture依赖的`models/v6_recovery_candidate_38000_20261008/policy.onnx`
已不在本机，保留缺失证据，不改旧fixture或替换成新模型掩盖。实际现用原1338
的`legacy_gui_preflight001`静态模型/合同检查通过；这不等于新GUI渲染验收。

## 完整状态恢复与训练容量

`policy200_capacity001`完成4096×384及12次critic minibatch反向检查，但暴露
旧迁移配方中的两次额外critic预热被重复叠加，actor当时未启用。没有优化器
写入，原权重和Adam未变；它不能证明actor反向容量。训练入口已修正：
仅升频迁移保留checkpoint已有的待预热数，不再重用历史架构迁移的额外预热。

`policy200_capacity002`使用003准备回执真实复核通过：原1338/actor1156、
待预热0、actor/critic/Adam及CPU/CUDA/环境RNG恢复六项全真；125源文件与
归档及当前源码SHA逐一相符。4096环境×384策略步＝1,572,864策略样本，
实读384次动作更新、1920次新反馈PD物理子步。12次minibatch反向均完成，
actor梯度非零、loss有限，优化器12次写入全部禁用；actor/critic/Adam前后精确相同。
采样156.97s、反向2.04s、整次含启动226.28s，Torch峰值分配3.74GB；
这是容量检查，不是收敛或稳态PPO提速证明。

核验回执为`policy200_capacity002/verification.json`。
200Hz迁移、短回放、两次容量检查累计新增PPO优化器更新0；本次高速请求消费
仍18optimizer＝6critic+12actor，最新账本`consumption004.json`，不退款旧失败。
没有用未通过的候选替换默认GUI，也没有启动长训或改动外部部署频率。

训练/评测默认计划已为200Hz；模拟预览及部署导出从合同读取频率，并拒绝
模型、评测或YAML频率错配。旧两个50Hz准备脚本仅可用
`--reproduce-frozen-50hz`显式复现历史实验。导出/时间合同的26项检查通过
（`cpu_policy200_export001.xml`，与此前72项有重叠，不相加）。
