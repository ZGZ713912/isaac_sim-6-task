# 当前从零策略的 RMUC 回放

这是 fresh35D、6 动作、200 Hz 策略/1000 Hz 物理的独立回放入口。原参考仓库的 `play_rmuc.sh wheel-leg` 只接受旧 V4 230D，不能直接加载这里的权重。

截至 2026-10-27T05:53:28.978821+00:00，以下已核验 checkpoint 完成 497 次更新，SHA256 `624590079e01f233b5a302c7491537cfaa2a2922f3b23a8bfb049aa5bf0f1407`。这只是训练中的候选。

```bash
/home/noir/.codex/worktrees/wheel-leg-domain-v2/isaac_wheeled_rl_train/play_rmuc.sh \
  --checkpoint /home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/fresh_unified_20261027/pilot018_velocity_quartic_physical_speed_mixed1536/model_496.pt
```

默认单车、名义模型、0.305m 站立出生；W/S 前后，A/D 转向，Q/E 高度，空格停止，P 暂停，R 重置。GUI 可选择低/中/高站立与六种倒地初态。原生保护或 30s 时限触发后暂停，R 重置。追加 `--scene inverted` 从翻倒开始；`--profile requested` 只改变单车抽样，不能作为全随机化评测。输出自动写入本 worktree 的 reports/interactive_play。

RMUC 使用原安装资产的真实启用三角碰撞几何：376 源网格、19356 三角形合并到原生任务的一张静态接触网格，保留任务地面材质；对应地图引用只负责显示，移除其碰撞 API，避免重复碰撞。为匹配当前原生站立/恢复的水平地面参考，全场相对选定出生点作刚体坐标对齐，主地面原始约 1.5 度倾角被对齐。原地图文件未改。当前 native 的高度/离地诊断仍参考出生平面，不能拿这次越障画面当正式地形或接地率验收。

校验为 CPU 合同预检、冻结 274 文件身份、严格 actor 加载及 4×35→4×6 有限推理、USD 几何写入/回读、唯一静态碰撞 mesh 和唯一 PhysicsScene。按用户要求没有启动 SimulationApp/GUI，原生运动为 0，GUI 可运行性尚未经实际启动验证。回执：`reports/robust_policy_repair_20261024/fresh018_rmuc_play_preparation007.json`。训练与 1536 更新后的既有串行全能力评测队列继续运行。
