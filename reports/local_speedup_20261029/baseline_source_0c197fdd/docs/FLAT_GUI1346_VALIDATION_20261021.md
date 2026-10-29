# 1346号模型：Isaac Sim GUI、键盘链路和仿真1倍速视频

本次按用户要求使用200Hz的1346号模型，模型SHA为
`a065c4688ff2dc86cde929125cd494af9460a4b307d1ea32f3bf9f0f36786dcf`。
物理/反馈仍为1000Hz，35观测、6动作，没有构建优化器或新增训练。
模型实际文件名为`flat_directional200_1346_preview_20261020.onnx`；播放入口现支持这一描述性名称和旧包中的`policy.onnx`，两者都必须与认证SHA一致；存在不同内容的同名候选会拒绝启动。

手动GUI命令：

```bash
bash /home/noir/Documents/workspace/isaac_wheeled_rl_train/scripts/play_flat_speed_preview.sh \
  --package /home/noir/Documents/workspace/isaac_wheeled_rl_train/models/flat_directional200_1346_preview_20261020
```

W/S前后、A/D旋转、Q/E或T/G升降、SPACE停车、P暂停、R重置。点击视口获取键盘焦点。键盘请求经过原合同1.5m/s²和4rad/s²斜率限制，不能把本GUI测试与突然改变目标的原生评测混为同一条件。

录制复用同一个`PolicyView.on_key`、`KeyboardCommand.advance`、ONNX推理、FixedCaseEnv和原物理闭链。自动测试通过脚本调用真实键盘回调的KEY_PRESS/KEY_RELEASE事件完成，报告明确标为脚本键盘回调，没有宣称人工或操作系统按键注入。原生键盘订阅同时处于启用状态。仅在不同用例之间执行原生reset，不回放/覆盖物理状态来制作动作视频；终止时先保存真实失败画面，随后才开始下一用例。

录制入口：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python -B \
  /home/noir/Documents/workspace/isaac_wheeled_rl_train/scripts/record_flat_gui_validation.py \
  --package /home/noir/Documents/workspace/isaac_wheeled_rl_train/models/flat_directional200_1346_preview_20261020 \
  --output /home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/NEW_UNIQUE_GUI_RECORDING \
  --device cpu
```

默认15项：原13个平地动作，包括站立、前后0.5m/s、双向yaw1、前进3/后退2.5m/s、双向yaw4、正反顺序平移/旋转启停；另加230mm和420mm车身高度。稳态高速各8秒，启停每段4秒。高度用例从原生305mm出生姿态发出高度请求，明确包含高度过渡，不冒称在目标高度的预置稳态出生。地面材质按原.2–.5采样，本次实际.44122854，轮.1、平均组合，有USD及PhysX回读。完整13动作×两高度×两地面摩擦的另一次原生结果单独记录在`reports/flat_gui1346_validation_20261021/native1346_reference001.json`，其中1346为73/104跑满，不能把新GUI的单一材质轨迹当作该全域验收。

2秒前进试录`smoke_cpu004`已成功：400策略步、2000物理反馈步、50帧/25FPS、视频精确2秒，GUI、键盘订阅和USD关节同步启用；双膝实际角行程5.53°/6.32°。实测仿真/墙钟时间约.113，因此GUI现场会慢于真实时间。录像每8个策略步取一帧，以25FPS播放，保持仿真1倍速；没有改变物理步长或通过丢物理步制造提速。失败处额外冻结1秒并在视频和帧清单显式标注，该冻结时间不计入仿真时长。

本机捕获像素使用[NVIDIA官方viewport buffer接口](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.utility/latest/USAGE_PYTHON.html)。必须等待像素回调完成，不能将“捕获已调度”的future当作PNG已经写完。早期失败试录均保留，正式录像使用修复后的冻结source003；入口文件名变动和异步文件捕获失败没有成为有效视频或新的训练消费。

15项时钟、终止保留、缓冲区复制、模型文件SHA和既有键盘时序测试通过。图表只从现有TensorBoard event导出，没有启动本机TensorBoard服务。

用户明确表示数据足够并要求停止play后，已核对PID1466118/start36970562身份并发送SIGINT；play及其编码子进程均确认退出，没有追加原生用例。Isaac Sim退出时没有执行Python最终收尾，原始`recording.json`/`report.json`仍保留最后写入状态，未改写成“全部完成”。停机身份和用户请求保存在`stop_requested001.json`，离线封装使用`analyze_flat_gui_recording.py --stopped-by-user`明确处理这一边界。

最终视频为`reports/flat_gui1346_validation_20261021/full_gui001/1346_gui_validation_1x.mp4`，2166帧、25FPS、86.64秒，SHA为`40515022b27286cb953575b860cc4c9bff9bf4a94da95efbcc43d32170c3c4a1`。12段完成记录含10段跑满、2段非轮触地失败，另有1段负向yaw启停在5.125秒被用户停止，最后两个高度项目未运行。总记录仿真84.58秒，视频包含两处显式标注的1秒失败冻结及不足一帧的取整边界。退出前最后3帧已经成功编码，但帧清单缓冲未刷新；2163帧有原生帧清单核对，额外3帧在回执中单独披露，其画面字幕5.04/5.08/5.12秒已人工检查与策略轨迹一致，不能冒称所有帧都有独立清单证明。

前进停车在5.725秒因nonwheel_contact终止，俯仰峰19.45°、末速度1.69m/s；反向顺序启停在13.73秒发生同类失败。即使使用原有指令斜率也没有完成这些停车测试。高速稳态末1秒均值前进3.096m/s、后退−2.427m/s，正反yaw约+4.001/−4.010rad/s。现场仿真/墙钟比约.084–.113，正常速度视频不能作为实时GUI性能已修复的证据。所有已录片段通过ffprobe和完整解码检查，分段结果、event导出曲线与来源SHA见最终`verification.json`及`index.html`。当前记录不授予实车资格，也不替换GUI默认模型或作为5000轮训练准入。
