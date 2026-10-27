# 新轮减速比 15.62（2026-10-27）

用户本次要求覆盖 2026-10-26 的“只改文档”决定：新训练使用 **15.62:1**，与 RMCS 双轮驱动一致。

`scripts/train_scratch_standing.py` 的默认入口现为：

- `contracts/scratch_standing_base_mechanism_v2.json`
- `contracts/scratch_standing_base_design_v2.json`
- `contracts/flat_base_200hz_control_v2.json`

三份文件独立冻结并绑定 SHA-256。`scripts/freeze_wheel_reduction.py` 可从保留的 v1 源文件重建；旧 v1、旧模型资产 manifest、旧录包及历史评测不覆盖。资产 manifest 内的 15.8 是旧资产来源记录；当前有效传动比由 v2 control 明确覆盖为 15.62，并在 task contract、静态预检和原生启动回读中单独记录。校验拒绝将新合同替换成 15.8 或原装减速箱比值。

轮端控制仍按 rad/s、N·m 计算，P=0.6、软件限矩 ±4.5 N·m、200 Hz 策略及 1 kHz 物理/PD 保持。轮端量不能再乘或除一次 15.62；只有转子端与轮端之间的换算使用该减速比。

## 录包换算

`scripts/identify_wheel_response.py` 的 `--gear-ratio` 目标口径默认为 **15.62**。脚本内固定的三份 2026-09-28 录包仍默认 `--recorded-gear-ratio 15.8`；输入由新驱动录制的数据时须显式指定 `--recorded-gear-ratio 15.62`。复现旧 15.8 计算须同时指定 `--gear-ratio 15.8 --recorded-gear-ratio 15.8`。

先用录包自己的减速比核对原始 RPM 与原 API 速度，再将轮列的全部相关通道换算一次。设录包减速比为 n₀、目标为 n：

| 量 | 乘数 |
|---|---|
| 轮角、轮速、轮速目标 | n₀/n |
| 指令/反馈力矩、历史软件限矩 | n/n₀ |
| 历史速度 P 增益 | (n/n₀)² |

正反方向及瞬时机械功率保持，四腿通道不变。新训练软件限矩仍是新轮端的 ±4.5 N·m；上述历史限矩换算只用于回放旧录包。结果明确保存录包/目标两种减速比和比例。

这类坐标换算不能证明更换后的惯量、摩擦、高速力矩或热包络。新训练暂保留既有轮端响应参数作为历史先验，`new_wheel_response_identified=false`；没有把旧 15.8 验证转成新轮验收。新轮响应须以新录包重新辨识和验证。本次不启动训练或机器人。
