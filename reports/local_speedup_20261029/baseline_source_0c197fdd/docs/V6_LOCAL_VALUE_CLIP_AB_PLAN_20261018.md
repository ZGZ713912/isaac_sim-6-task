# 本机200Hz价值函数裁剪受限对照

## 问题与假设

pilot018的2945模型原八场景32回合全跑满仍0/8。前进+.5实际+.07458m/s，
反向-.5实际-.04980m/s，三档高度.30830/.31082/.31507m，305mm静站yaw MAE
.31302rad/s。原初始/2944完整候选payload精确复现eval017，原种子、时长、
warmup及判据保持；不能以数值、存活、平滑或局部进展当正常。

原生单次PPO20个Adam小批真实推进：平均ratio越界13.543%、实际surrogate裁剪
分支7.021%、KL.00960，未见优化停摆。critic裁剪平坦样本平均13.120%；原值
MSE114.06278、当前值MSE113.44741，原生裁剪值损失113.97116。actor/critic
梯度分别裁剪，原系数4下critic梯度大不能单独解释actor学习失败。

这些是恢复RNG后新episode的首个384帧rollout，不是稳态窗口或唯一根因证明。
假设是长回报窗口下原值裁剪可能限制critic拟合，继而影响策略信用。本次只检验
关闭原生critic值裁剪是否改善实际跟踪；不保证它会改善，更不据此改失败成本200、
重置std、提高学习率或宣称bug已定位。当前分项GAE多数整环境bootstrap区间跨0，
也不支持仅凭噪声点估计修改探索。

## 精确起点及顺序

两臂均从pilot018已完成的2945完整学习状态开始：checkpoint
3cbd7aabf95d500d6ffa089728b28ad1a92c8e812be9d503d9c6bdf152ae2b0e，
Adam58900，182648832 transitions，继承消费1216，全部总消费4161。

1. pilot019控制臂：原v8逐字节合同，完整resume追加128至3073，原生
   use_clipped_value_loss=True，默认不开一次性PPO观测开关。
2. 控制正常完成退出后，原八场景比较初始、2945、控制3073，保留全部失败和
   traces。旧候选完整payload必须复现eval018。若实际达到原行为与机械门，
   可按原门接续，不必再消耗处理臂预算。
3. 若控制仍失败，实现精确命名v8→本机v9，处理臂从**相同pilot018-2945**开始，
   不能从控制3073开始。仅新增显式use_clipped_value_loss=False及声明的
   plan/design/budget来源变化；完整actor/critic/17Adam/std/CPU-CUDA-envRNG/
   课程保持，无新critic预热。处理最多128至3073。

原生Adam、GAE、采样、logprob、actor裁剪.2、value_loss_coef4、梯度裁剪1、
fixedLR1e-4、384帧/lambda1/gamma.9974905699336811/entropy.0005不变。
200Hz物理/策略、20ms动作差分参考、solver64/32、3mm、PD160/2.5、全部控制/
力矩/响应/DR/出生/普通倾角推扰/profiles概率/人口/reward/课程/评测保持。
v6命令仍上限.5m/s、1rad/s，不当完整高速任务。

## 处理臂工程准入

处理臂当前**尚未实现或启动**，控制结果也尚不存在；不能把计划当实际迁移通过。
控制期间可实现候选和针对性CPU正负例，不能改活learner或冻结源。

- 独立target design精确推导v9；旧v1–v8校验和绑定证据保持，不接受任意布尔
  覆盖、未知起点、改reward/物理/课程/消费。旧合同缺该字段继续使用原默认True。
- 严格认证pilot018 checkpoint/合同/seal/completion/结果、本计划与原生优化/
  分项信用报告SHA，控制真实完成/评测和不可变合并消费回执SHA。
- 针对性CPU检查完整Adam精确load、原生clipped/unclipped值损失及布尔类型，
  第一更新共用原advantage时actor/采样RNG保持、仅critic随后梯度可不同；不
  重新实现PPO优化器。所有负例和失败保留。
- 两个独立原生0PPO进程从同父保存RNG新episode，source/target各3×384帧。
  核验完整初态、首Gaussian及已记录动力学/命令/reward/mask数组逐位相同，
  actor/critic/17Adam/时钟保持，禁止update/step。只授工程试验，不授行为资格。
- 每个实际PPO从已提交冻结源码启动，核验独立initial_state逐值等父及实际
  首PPO推进；首处理2946应Adam58920、182845440 transitions、无新warmup。
  不重置std、不把CPU或0PPO等价当实际训练效果。

## 消费及曝光

控制128完成：谱系3073、Adam61460、207814656 transitions，总消费4289，
剩15711。控制消费128不能退款。若执行处理，命名合并回执继承消费收紧为
1344=原1216+控制128，处理谱系上限18656，总上限仍20000；从2945追加128
后处理谱系3073、Adam61460、207814656 transitions，总消费4417、剩15583。
处理transition_ceiling为3271557120，保留历史96帧曝光，不把旧2688更新伪作
384帧。部分完成/失败也保留全部实际消费，不能假造128完成回执。

若最终选回控制，处理消费也必须由新认证账本携带，不能用旧v8的1216账本
退掉另一臂。所有支线checkpoint/Adam/RNG/原始结果及冻结目录保留。

## 比较和退出

每段正常结束且真实退出后用该段冻结evaluate_local_chassis.py运行原八场景，
seed290929、四副本、10/16秒、1秒warmup、原阈值不变，保存全部失败/traces。
body pose每4帧可能post-reset，与active/ticks分开。未跨warmup的零MAE不是通过。
控制与处理为同起点的各128更新，不把PPO之后的不同轨迹伪称逐帧配对或跨种子
保证。等384帧曝光下比较原event稳定tag的六轴std/实付成本/回合秒数/value loss，
按结束回合与环境秒双归一机械率；history累计snapshot做差，不能相加。

关闭裁剪使平坦比例变0是实现定义，不是学习改善证据；必须检查原生回报/价值
误差及实际站立、前后速度、高度、yaw、机械表现。若无实质行为改善保留负结果，
下一轮须依据新证据选择受限调整，不能无限延长本假设。只有实际基础站立和
前后跟踪正常且机械无系统异常后才完整状态接续剩余预算长训。

只做本机仿真，不启动实体机器人/Kaiser，不运行本机TensorBoard或动6006/6007。
新指标进入原RSL event。保留worktree/冻结源，commit/push指定upstream并FF主区，
保留用户暂存.vscode/settings.json。
