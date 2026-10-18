# 同2560学习状态小熵对照的精确接续

已有V6_LOCAL_ENTROPY_AB_PLAN_20261018.md保持。控制pilot014已完成128，原八场景仅1/8；全部证据冻结在pilot014.json，无长训准入。处理v7从pilot013的同一2560完整状态分叉，entropy0.0005，无std重设、新critic预热或时钟重置，Gaussian仍原生采样且std可学习。奖励、200Hz物理/策略、solver64/32、3mm门、PD160/2.5、v5出生100ms渐入及20帧conditioning mask/动作历史、DR、人口和v6命令上限保持。

contracts/v6_local_scratch_200hz_v7.json由独立local_entropy_revision.REVISION认证：精确源计划/模型/合同/seal/completion、pilot013结果、动作信用结果、AB原计划与决策、控制结果及合并消费回执。target design仍从独立v18 expectations精确推导，仅声明entropy和plan/design/budget变化；原v1–v6校验未放宽。

第一次需--resume-local-entropy-revision，最多128，从已提交冻结源码启动；同v7合同后续普通resume。预算继承1088+控制128=1216，谱系上限18784；父2560学习/消费时钟保留，开始总3776，结束2688总3904，余16096。共同父2560不重复计费；选回控制或换谱系均须携带后续全部消费，不能复用旧1088预算退款。

83项针对性CPU及512入口静态前置通过，失败路径与纠正fixture日志保留。两个独立0PPO原生进程：10初始数组、62类全部288帧/512环境采样数组逐位相同，有效熵0/.0005，actor/critic/17Adam不变，优化器update/step禁止。这个对照仅证明更新前条件相同。

启动后核验独立initial_state逐值等父actor/critic/17Adam、学习std保留、全部RNG恢复、课程2560实际采样、无新critic预热。首seal2561/Adam51220，最终2688/Adam53760、132120576 transitions。恢复RNG后创建新episode，不称模拟器中途重放。活learner只读，不热改/重启/signal/制造快照。

完成退出后原八场景/种子290929/4副本/10或16秒/1秒warmup/原阈值，保留初始和共同父2560的原序列，旧候选完整复现；对比控制2688和处理2688，保存全部失败/traces。RSL原event匹配末100窗六轴std/实付成本/回合秒数，机械按结束回合与环境秒双归一。单次GPU两臂只支持有限假设；数值有限、std增加、reward或存活均不能单独授长训。
