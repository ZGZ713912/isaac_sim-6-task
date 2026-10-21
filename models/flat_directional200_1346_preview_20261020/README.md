# 200Hz冻结短训候选：仅仿真预览

本包是原1338父策略新增2次critic-only和6次actor更新的结果：学习计数1346、
actor1162。源ONNX导出前的训练次数未知。

策略200Hz、物理/反馈PD1000Hz、35维观测、6维动作、历史长度1。
已对8192个实际候选观察帧验证；单帧ONNX动作与原生对照记录最大误差0，
PyTorch/ONNX最大误差3.34e-6，详见`verification.json`。

该候选完整对照264/384回合跑满，11/96条件通过全部判据；高位/地面0.5后退约
−2.49m/s，尚未达到−3m/s。部分正反启停会横向越界。本包未通过部署验收，
停车航向保持的独立诊断不包含在本ONNX中。

使用本包查看GUI和键盘控制：

```bash
bash /home/noir/Documents/workspace/isaac_wheeled_rl_train/scripts/play_flat_speed_preview.sh \
  --package /home/noir/Documents/workspace/isaac_wheeled_rl_train/models/flat_directional200_1346_preview_20261020
```

点击viewport取得键盘焦点。W/S默认请求前后最高±3.0m/s，A/D默认请求原地
正反旋转最高±4.0rad/s；GUI的`Keyboard speed limits`两个滑条可降低上限。
实际策略指令仍遵循1.5m/s²、4.0rad/s²斜率，从静止分别约2s、1s达到上限；
行进中同时转向还受横向加速度约束。松开按键请求停车，SPACE清除运动按键。

原默认`play_flat_base.sh`仍指定旧1338模型。本次仅完成新包的CPU合同/图/推理验证，
没有启动新的GUI窗口或实车电机；实际渲染与交互结果需另行检查。
