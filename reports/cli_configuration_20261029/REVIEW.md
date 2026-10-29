# dev分支与可配置训练/play入口

统一分支已改名dev，旧dev前缀三分支保留为archive/dev。CLI源码已本地提交，未推送。常用参数保存在resolved计划和run_options以及PT中，不修改物理模型或验收门限；训练仍全新actor/critic/空Adam。

97项相关测试通过；真实CLI分别检查5000次、128环境、seed103、随机化off/on。独立80环境双域原生接线各30/30通过，无checkpoint读取和PPO；冻结342份源/输入退出无差异，进程/GPU退出。单车play使用全新CPU随机actor夹具，只完成5个100Hz策略步/50个1ms物理步，未恢复优化器，不能作为站稳或起立证据。所有13项长期能力均未由此授予通过。

当前训练环境20..256，保留原生隔离上限；预算至少20。默认10000次、64步采样、每250次保存，一次learn调用。--max_iterations改变时课程节点同比缩放，30秒站立和其他行为门限不变。--domain_randomization支持curriculum/on/off，--dr_strength支持0..1；--checkpoint仅play可用。完整命令见源码docs/WHEEL_LEG_UNIFIED_20261029.md。

机器可读回执receipt.json记录来源、各域原生结果及进程清理。
