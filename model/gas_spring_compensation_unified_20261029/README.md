# 统一训练的气簧源绑定

此包引用`dev/self_righting_policy@7359170b`原实测拟合模型和原零重力静态组件报告，绑定合并后的控制代码。`model/gas_spring_compensation_20261028/`原交付内容保持原样。

- 力模型仍为16个实测拟合节点及dev端点外推；左右共用轴向负载模型、按各侧几何换算电机补偿。
- `evidence.json`明确列出旧证据来源、新代码哈希及证据范围；`component_reference/traces.npz`是原trace的逐字节副本，SHA-256保持。
- `design.json`和`mechanism.json`只派生源绑定。当前100Hz长训练计划随后显式装入100Hz控制时序和左右15.62轮辨识配置。
- 原组件证据不等于合并后原生运动、策略、鲁棒性或实车验收。本次测试和运行边界见`docs/WHEEL_LEG_UNIFIED_20261029.md`。

实际腿控制为未限幅PD加一次气簧补偿、±40Nm限幅、相对软限位投影；恢复包装不重复叠加V5补偿。物理气簧施力保留，失能后电机力矩仍为零。
