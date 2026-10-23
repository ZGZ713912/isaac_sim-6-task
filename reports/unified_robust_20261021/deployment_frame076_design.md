# 新观测 ABI 的采样设计草案

状态：只读设计；尚未修改 RMCS，不能作为部署准入。

本轮重读当前 RMCS 的 `wheel_leg_controller.cpp`、`observation.cpp` 和 `rmcs_rl/src/rl_bridge.cpp`。`evaluate_policy_()`每个执行器tick接收新推理序号，命令参考按策略除频更新；`publish_model_feedback_()`当前每个执行器tick发布最新q/dq/IMU。`RlBridge`在自己的steady-clock策略时钟拼接观测并提交线程推理。因此只加两个200Hz坐标输出，会把它们与另一个采样时刻的传感器数据拼接。既有068独立C++数值回放不能证明这个集成时序。

拟实现一个显式启用的新profile，由 `WheelLegController` 在200Hz同一次采样中生成命令、gyro/gravity、连续模型关节角、速度、上一实际raw动作和站立位移，再保持这些输出至下一帧。状态、标定、恢复入口、PD及输出继续归controller；`RlBridge`仅承担观测路由与线程推理。

为避免两个独立策略时钟错相，拟给bridge增加可选的外部观测帧序号路径。有该路径时只对新帧提交推理，默认路径继续使用旧timeline。reset清除帧已读标志及历史，enable前帧序号0表示尚未就绪。重复帧不重复推理；回退/跳变的判定、提交失败重试及线程繁忙时的边界必须用故障注入测试明确。模型消费仍保持现有每执行器tick接收新有效动作的行为，不能误写成只在除频tick消费动作。

新增观测的两维是米乘20，使用068冻结107几何、v3生命周期、150ms滤波和50ms捕获等待。严重翻倒、整体reset、重新进入RL及标定改变都清除旧参考。完整RL入口在DM准备握手后直接进入策略控制，首帧动作历史来自测得成对共同角；六电机失能时零力矩，力矩/闭链/新鲜度/真实故障保护保持。旧转腿脚本、body45deg准入和接管blend不能混入该profile。

上线前还需：独立YAML35D布局及模型metadata绑定、200Hz策略/1000Hz PD、raw腿clip16、正负合同拒绝用例、真实调用链的coherent-frame测试和受影响包构建。构建助手已找到为RMCS `.script/build-rmcs`，在现有devcontainer内用该助手执行；本轮没有执行构建，也没有CAN/实车动作。
