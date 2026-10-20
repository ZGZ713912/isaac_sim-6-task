# 平地续训候选的 GUI 键盘回放

当前默认模型为`models/flat_static_grip1338_20261020/policy.onnx`，对应累计1338次成功优化、1156次actor更新。
地面摩擦0.2–0.5均匀随机，轮子0.1，策略50 Hz、物理/PD 1000 Hz。
基础固定回放8/8通过；完整速度/高度范围仍只有9/22，不能当作实车部署验收。

```bash
bash /home/noir/Documents/workspace/isaac_wheeled_rl_train/scripts/play_flat_base.sh \
  --onnx /home/noir/Documents/workspace/isaac_wheeled_rl_train/models/flat_static_grip1338_20261020/policy.onnx \
  --contract /home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/flat_base_20261019/pilot029_static_grip1338/config/contract.json
```

入口使用本机`isaacsim6`解释器、CPU物理和ONNX Runtime推理，单车GUI持续运行到关闭窗口。
GUI默认只使用4个CPU逻辑核，OpenMP/MKL线程也限制为4；`FLAT_PLAY_CPU_CORES=1..8`可显式调整。
控制台打印模型绝对路径、SHA-256及累计成功优化次数；窗口标题显示`flat_static_grip1338_20261020`。
该ONNX的SHA-256为`9cc82fd9355458305d372cd5b0299c9f35ae0d58ce81f05a94bdd7f66ee9f874`。
当前选择1338，因为1348虽保持基础8/8，最高高度回放新增三项失败检查。
先点击3D视窗取得键盘焦点：W/S前后，A/D转向，Q/E或T/G调高度，空格停止，P暂停，R复位。
默认键盘速度上限0.5 m/s、转向1 rad/s；左侧面板可调整速度、高度和跟随相机。
此平地模型入口不提供跳跃请求。

每次启动生成独立`reports/chassis_suite_flat_<UTC时间>/`，运行指标写入该目录的TensorBoard event。
模型、源checkpoint、训练方案、源合同及单环境投影都在启动前核验；原训练和模型文件保持。

本机原GUI启动文件依赖未提供的`isaacsim.util.debug_draw`。
入口在本次报告目录生成`flat_gui.kit`，省略此未使用的扩展并恢复原扩展目录。
力/速度叠加使用USD曲线，无需该绘图扩展；安装版启动文件保持原字节。
当前本机Fabric显示接口存在版本不一致警告，GUI改用USD场景委托和PhysX→USD变换同步；
仿真、观测、动作、材质、控制频率和模型权重保持。训练专用的短分组回合在交互合同中清空。
`gui_experience.json`保存来源和生成文件的SHA。首次启动可能从官方registry下载GUI组件。
`--gui-seconds 25`用于限定原生GUI启动检查，正常键控查看时不添加此参数。

已完成`reports/flat_base_play1338_20261020/cpu002/`的模型/合同/35→6推理CPU预检及19项已有键控/命令时钟检查。
首个CPU预检因旧的独立evaluation.seed断言未随交互种子投影而拒绝，修复入口投影后通过，失败输出保留。
原生`gui001`首次下载组件及编译着色器时CPU占用较高，后续未完成可视化初始化；
`gui002`仍在可视化初始化失败，`gui003`定位到训练短分组回合与交互场景冲突。
`gui004`完成单车推理和键盘订阅，但Fabric画面仍为空，不能称GUI验证通过。
修正显示同步后的`gui005`实际完成15秒窗口测试，状态`completed`，键盘订阅有效，
12条采样推进到约2.4秒仿真，高度约0.304 m；`viewport.png`已实际显示机器人与地面。
当前4核CPU物理回放慢于实时，不能据此声称50 Hz实时性能。
所有失败日志和截图保留，限定测试进程在结束后退出，未启动新的训练。
