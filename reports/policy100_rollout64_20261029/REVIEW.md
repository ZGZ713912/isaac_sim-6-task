# 策略100Hz、rollout64步交付

源码：`/home/noir/.codex/worktrees/prone-motion-100hz/isaac_wheeled_rl_train`，分支`codex/prone-motion-100hz`，提交`a4d2e1fda7e398c607f625cf7f596bc6df452ef7`。324个源/输入文件逐字节匹配提交；归档SHA256 `8a1e089562f463f66c334888cb406f18e8938666c1cdb787cb7a2801d8d9bae4`。

默认v2为100Hz策略、64步rollout；物理与反馈PD仍1kHz，每策略步10物理步。每环境每更新仍覆盖0.64秒/640物理步，256环境的策略样本为16384。训练连续10000次更新，每250次完整更新保存PT。策略、critic与Adam从零；禁止旧checkpoint/teacher/donor。

腿PD P160/D2.5、15.62独立左右轮模型与5.5/3.5ms有效轮I/O延迟、实测对称气簧物理力和编码器补偿均保留。原200Hz控制及全部辨识证据文件未改写；新控制仅派生身份和策略时序。gamma=0.99498743710662、lambda=0.9746794344808964保留秒尺度，动作变化代价按5ms参考归一化，事件奖励不变。

随机域额外0或1策略周期观测年龄在100Hz下为0或10ms，白噪声按10ms采样；实测轮I/O延迟保持。配置、startup、任务合同和TensorBoard延迟毫秒指标均显式记录。验收按实际策略周期计算完整30秒，并用10个1ms样本定位完成事件之后的接地起点：连续2ms仅诊断、3ms保留，再摔倒仍失败。

300项测试通过（85项任务/时序/运行时回归，215项受影响控制/域/模型），包括纯张量CUDA图等价。CPU配置预检通过。新版本名义/完整v2随机域各30项原生接线检查全部通过；实际100Hz/1kHz/10子步、35D/83D、奖励实付、实测物理弹簧、编码器补偿、轮延迟/限幅、允许起立触底和域切换新观测均通过。2个上下文、0PPO、0优化器尝试、0checkpoint读取；进程/GPU均已退出，源改动为零。未做100Hz行为验收或正式提速测量。

测试早期合成趴地fixture漏掉起立后运动段的失败日志保留，补齐测试输入后通过，生产门限未改。所有原始日志和机器可读回执保留在本目录。

本机run004原先200Hz训练已观察到正常封存于61次完整更新，model_stopped.pt存在；本轮未向它发送停止信号。原目录321个冻结文件核验匹配f4996e0d，进程/GPU已退出，无重启、无旧模型读取。100Hz改动隔离到新工作树。服务器未连接、修改或停训。

启动新训练（输出目录必须不存在）：

```bash
bash /home/noir/.codex/worktrees/prone-motion-100hz/isaac_wheeled_rl_train/scripts/train_prone_motion.sh \
  --output /home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/prone_motion_20261029/run100hz001
```

入口将再次运行独立两域0PPO检查，通过后才创建全新的训练策略；本次交付没有开始PPO。13项长期能力均未对100Hz新策略验收：R01, R02, S01, M01, M02, M03, M04, H01, H02, H03, B01, Q01, D01，全部qualified=false。
