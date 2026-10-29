# 气簧左右对称假设与无重力电机力矩验证

按用户最新指示，右腿暂时沿用左腿轴向力曲线。这是对称假设，不是新增右腿实测辨识。没有修改左腿拟合值或原始 39 个原生工况记录。

本次新增 7 个双腿同时受力的 Isaac Sim 6 原生工况，全部达到静态条件。两边均在各自气簧滑动关节施加同一 F(s)，由各自闭链几何传递到 hip/knee；正轴向推力不人为翻号。左右两侧电机坐标方向相反，力矩镜像差绝对值最大 0.000589 N·m，抽样闭链间隙最大 0.0231 mm。维持前一报告相同的无重力、固定基座、无地面接触、无摩擦、测量夹具 armature、PD 与限幅；使用固定位置参考预偏置，无在线气簧力矩前馈，运行中不覆写被动关节位置。

| β 参考 [deg] | 左 hip [N·m] | 左 knee [N·m] | 右 hip [N·m] | 右 knee [N·m] |
|---:|---:|---:|---:|---:|
| 56 | 18.878 | -18.878 | -18.878 | 18.878 |
| 60 | 18.912 | -18.912 | -18.912 | 18.912 |
| 75 | 18.282 | -18.282 | -18.282 | 18.282 |
| 90 | 15.840 | -15.840 | -15.840 | 15.840 |
| 98 | 13.292 | -13.292 | -13.291 | 13.291 |
| 100 | 5.609 | -5.609 | -5.608 | 5.608 |
| 101 | 3.709 | -3.709 | -3.709 | 3.709 |

新 P/P 左腿留出 610 点的平均弹性负载预测误差：hip RMS **0.929 N·m**，knee RMS **3.475 N·m**，合并 **2.544 N·m**。原目录模型分别为 4.459/6.154 N·m。可以认为平均负载已经接近本次实测，但不能称逐时刻完全一致；knee 剩余方向摩擦误差更明显。之前 0.426 N·m 是差动模型另加方向项后的结果，该项没有作为气簧力启用。

适用左/右候选角域仍为 56–101°；右腿有效性来自用户的对称假设。模型依旧标记为有效安装负载，未唯一识别纯气体压力。101° 浮点端点按前一探针规则最多容许 20 μm endpoint hold，公共组件严格拒绝超域。

独立导出：`/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/gas_spring_identified_symmetric_20261028/model.json`。默认训练入口仍加载资产目录下 `fit_10mpa.json`，本次没有替换生产默认模型或启动 PPO。加载双腿候选示例：

```python
from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve
left = IdentifiedGasSpringCurve.from_json("model/gas_spring_identified_symmetric_20261028/model.json")
right = left.for_side("right", assume_symmetry=True)
force_left_n = left.force(compression_left_m)
force_right_n = right.force(compression_right_m)
```

19 项相关测试通过，包括右侧使用须显式记录对称假设、轴向力符号、势能与闭链镜像映射。CAD 轴的微小左右差异低于约 10 ppm，不要求几何浮点数逐位相等。第一次数值检查误设了过紧容差、初次探针错用了旧右膝拼写，失败证据均保留；修正后结果见 `tests.log`、`native_symmetric.json`、`validation_receipt.json`。

复现（使用 Isaac Sim 环境解释器）：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/probe_identified_gas_spring.py --data reports/gas_spring_real2sim_20261027 --mode curves --models effective_monotone_table --paired --betas 56 60 75 90 98 100 101 --output reports/gas_spring_real2sim_20261027/symmetric_20261028/native_symmetric.json
python -m pytest -q tests/test_identified_gas_spring.py tests/test_gas_spring.py
python scripts/report_symmetric_gas_spring.py --data reports/gas_spring_real2sim_20261027 --out reports/gas_spring_real2sim_20261027/symmetric_20261028 --export-dir model/gas_spring_identified_symmetric_20261028
```

原始左腿拟合、来源和完整对照图见 [上一报告](../README.md)。
