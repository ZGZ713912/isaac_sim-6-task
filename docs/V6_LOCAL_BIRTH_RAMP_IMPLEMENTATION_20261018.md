# 本机 v5 出生接管实现

本机 v4-1024 固定评测仍0/8；长训门保持关闭。依据已冻结的
`V6_LOCAL_BIRTH_RAMP_PLAN_20261017.md`，本次只修订非名义出生的接管机制。
Kaiser及旧v1合同不采纳本机v5。

## 实现

`nonstandard_start` v2把选中出生的200ms被动零力矩保持改为立即100ms最终
六轴力矩渐入，保持原PD、驱动响应、速度力矩包络、气弹簧与solver64/32。
首物理帧增益0.05，20帧达到解析值1（float32实际0.99999994，与诊断逐位相同）。
名义出生、原profile/概率、capture1.5s/caps、课程origin、reward和entropy0不变。

`conditioning`独立承担学习所有权，整段20帧排除actor/critic；第20帧的返回状态
已可bootstrap，第21帧才重新成为学习样本。`holding`仍只表示实际零力矩保持；
v2不把它置true，也不沿用`held_interval`清空动作历史。旧v1回归同时验证。
新指标写入原RSL-RL event的`/nonstandard_start/`命名空间，单位为秒或无量纲比例。

## 精确全状态接续

命名入口为`--resume-local-birth-revision`。仅认证pilot008最终1024更新模型，
checkpoint/原合同/seal/completion/原报告/诊断/计划均绑定SHA；目标worker、独立design
与budget从冻结v5 plan精确重建。仅允许出生设置及相应本机plan/design/budget绑定不同。
拒绝任意父模型、从零初始化、移动课程origin、预算重置和其他合同漂移。

此前失败谱系1088与本谱系1024全部保留；本谱系上限18912，总上限20000。
第一段最多追加128，累计1152、全实验2240；没有新critic预热。后续同合同接续仍使用
原完整恢复路径。首PPO新增一次只读seal，供独立检查实际actor/critic/Adam推进；
原周期checkpoint不改，不中断活进程制造快照。

原生接续必须从提交冻结源启动，并核验actor、critic、17个Adam状态、学习过的std、
CPU/CUDA/环境RNG与1024课程时钟。恢复后创建新episode，不宣称模拟器中途重放。

## 验证与后续

原生0-PPO probe014与旧诊断probe011的完整初始q/dq/刚体位姿、策略输入、profile、
Gaussian首帧以及全部400帧27D状态和闭链gap逐位一致（包含NaN结束标记）。
两者2秒闭链0、存活484/512、峰1.382845mm；原生每帧学习mask正确，holding始终false，
20帧动作历史保留，末帧返回状态bootstrap有效。5秒独立种子确认及最终测试状态见
`reports/local_training_20261017/birth_v2_implementation001.json`。

首轮CPU运行145项通过，另1项旧集成fixture缺少`_pc_feedback_transport=None`；
生产env未改，补齐fixture后回归。后续一次扩大集合因`/tmp`用户quota失败4项，
失败日志与本轮生成的fixture均保留并移到home；改用home basetemp后183项通过。
最后新增计划计费负例的最终集合结果单独记入上述回执，不相加计数。

候选机械改善不等于策略正常：独立5秒候选仍有57次非轮接触倒地。
先追加128受限短训，退出后同原8场景/种子/时长/判据比较初始、v4-1024与新模型；
保存全部失败和traces，使用原event比较实际成本/分轴std/回合秒数，机械保护同时
按结束回合与环境秒归一。基础站立、前后跟踪及机械暴露实际正常后才可准入长训。
