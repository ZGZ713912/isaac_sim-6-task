# 轮腿统一分支与默认训练配置

当前统一开发分支为 `dev`（原名 `codex/wheel-leg-unified`），整合以下来源：

| 来源 | 冻结提交 | 采用内容 |
|---|---|---|
| `dev/self_righting_policy` | `2a5530c66a16f40cc53ed47c371baa9a20bdae3c` | 实测拟合气簧模型、补偿符号、端点外推及相对软限位投影 |
| `codex/prone-motion-long` | `f4996e0dc8aec96b6944c85a7bef6bbd2b6097a8` | 标准趴地与站立混采、运动/变高课程、10000次连续训练、每250次PT、左右15.62轮辨识、CUDA张量回放 |
| `codex/prone-motion-100hz` | `a4d2e1fda7e398c607f625cf7f596bc6df452ef7` | 100Hz策略、64步rollout、按秒保持折扣和动作变化代价 |

气簧实现与参数取自`7359170b`；合并期间dev新提交`2a5530c6`包含原暂存的模型/报告归档，已一起保留。100Hz提交已经包含long历史。合并保留三个来源的祖先关系。为使用确切的 `dev` 名称，旧 `dev/self_righting_policy`、`dev/completed-speed-resume-20261010`、`dev/wheel-leg-domain-v2` 改名为各自的 `archive/dev/...`，提交和工作目录保留。后续不新建 `codex/` 前缀分支。

## 默认入口

默认计划为`contracts/wheel_leg_prone_motion_v3.json`。在统一分支目录启动，输出目录必须不存在：

```bash
cd /home/noir/.codex/worktrees/wheel-leg-unified/isaac_wheeled_rl_train
bash scripts/train_prone_motion.sh \
  --task WheelLeg-ProneMotion-v3 \
  --num_envs 256 \
  --max_iterations 10000 \
  --num_steps_per_env 64 \
  --save_interval 250 \
  --seed 103 \
  --run_name train_10000_seed103 \
  --domain_randomization curriculum \
  --device cuda:0 \
  --headless
```

常用参数支持下划线和连字符两种拼写；`--help` 查看全部选项。`--run_name` 自动创建 `runs/prone_motion/<名字>`；也可用 `--output <新目录>` 指定绝对路径。已有训练目录和同名 `.native_probe` 均拒绝覆盖，换一个名称即可。训练禁止 `--checkpoint`，所有新运行仍从零初始化。

`--max_iterations`、`--num_envs`、`--seed`、`--save_interval`、`--num_steps_per_env`、`--learning_rate` 无需修改脚本。预算至少20次、训练环境20..256个；256为现有隔离物理环境的上限，没有因开放CLI而放宽。默认10000次时课程沿用原节点；例如5000次时，基础/全速/全自旋/全高度节点为500/3000/3500/2500次，完整随机域从4500次开始。时钟、模型、气簧补偿、验收标准不随预算改变；轮数更少不代表已学会。

域随机化通过 `--domain_randomization curriculum|on|off` 选择：默认 `curriculum` 渐进分组，`on` 从第一步使用组合 requested 域，`off` 全程名义域。`--dr_strength 0..1` 缩放指定v2范围，默认1。关闭或减弱训练随机化不授予完整随机域资格。启动时的独立零PPO接线检查仍验证名义与完整随机域；它不会进入训练优化器。

每次运行保存原合同 `curriculum_plan.input.json`、解析后的 `curriculum_plan.resolved.json` 和命令行 `run_options.json`；PT内也带解析计划及运行参数，便于按实际配置重放。`--headless` 默认开启；`--no-headless` 显示训练窗口。默认解释器保持本机 Isaac 环境，服务器用 `WHEELED_RL_PYTHON` 指定。

训练后选任意检查点播放，无需修改脚本：

```bash
bash scripts/play_prone_motion.sh \
  --task WheelLeg-ProneMotion-v3 \
  --checkpoint "$PWD/runs/prone_motion/train_10000_seed103/model_00250.pt" \
  --num_envs 1 \
  --scene prone \
  --domain_randomization off \
  --device cuda:0
```

play 默认打开GUI。`--scene` 支持 prone、stand、forward、reverse、spin_left、spin_right、arc_fl/fr/rl/rr、height；`--vx` 指定前后目标m/s（±2内），`--yaw_rate` 指定yaw rad/s（±100rpm内），`--height .23..42` 指定目标高度米，`--birth_height .23|.305|.42` 指定站姿出生高度。运动使用训练同一平滑升速/减速包络；prone场景由实际起立完成事件启动完整30秒静站等待，不是脚本控电机。单次默认70秒，原生结束事件提前结束预览；`--seconds` 调整总时长、`--loop` 重复场景并记录重置次数（不指定seconds时持续到关闭窗口或Ctrl-C）、`--headless --no-real_time` 无窗口快速预览。`--real_time` 仅控制GUI节奏，不改物理时钟。需要静站时使用 `--scene stand --height .305`。

play 仅恢复actor，使用原始计划和姿态库，验证PT版本、计划一致性、SHA及35→6结构，不创建PPO或恢复Adam。使用当前源码构建预览，结果写入独立 `runs/prone_motion_play/`；不是冻结训练源逐字节复现，也不替代固定病例的完整行为验收。复制模型到别处时可用 `--run <原运行目录>` 指定配套的agent配置、计划和姿态库。

CPU配置预检使用同一入口加`--preflight-only`，原生零PPO接线检查加`--native-probe`。普通启动先串行验证名义域和完整v2随机域，再创建新的actor、critic与空Adam。已有轮腿learner时入口拒绝并行启动；不要为切换分支停止或续训现有运行。

策略100Hz；物理、腿单环PD和轮控制1kHz；腿P=160、D=2.5，轮P=0.5。256环境、64步rollout，每次采样覆盖0.64秒、640个物理步；求解器64/32。连续调用一次`learn(10000)`，不插入中途冷测，每250次完整更新保存PT。原生终端进度、耗时和ETA保留。

标准趴地起立、完成后完整30秒静站、静站复习、前后±2m/s、双向100rpm自旋、四方向转弯及230/305/420mm高度课程沿用long计划。域随机化节点和范围保持：6000次前名义域，随后分组及组合渐入，9000次起完整v2范围。腿P/D、气簧系数固定1，独立惯量最终±6%，无IMU晃动。具体角色、幅度与随机化开关见[训练说明](PRONE_MOTION_TRAINING_20261028.md)。

显式`--task WheelLeg-ProneMotion-v1`选择历史200Hz/128步方案；v2选择历史100Hz/64步与long气簧适配层。`--plan`可指定仓库内与task版本对应的原始合同。历史009复现显式绑定旧mechanism，不自动采用新气簧补偿。

## 气簧与力矩路径

按用户确认，采用dev现有的16个实测拟合节点、插值和端点趋势。原文件`model/gas_spring_compensation_20261028/model.json`字节保持，SHA-256为`098fd7d5fbd876cd57d0b6e01e239eb16f6789690a5eadf7820ec20d974919a3`；没有生成新的解析拟合公式。左右共用轴向负载模型，分别用各自几何和符号；右侧仍为对称假设。

物理气簧按滑动副压缩量施加轴向力。补偿用PD同一份采样/延迟后的髋膝编码器，在目标绕圈处理之前保存反馈，通过dev几何表得到压缩量和长度雅可比。实测段插值保持；低压缩端为正值指数延伸，高压缩端为线性端点斜率延伸，超几何范围的物理力保持端值，覆盖40–107°软限位。

腿电机执行顺序为：未限幅PD请求加一次`−J_lengthᵀF`补偿，总和限制±40Nm，再执行dev相对软限位向外力矩投影。随后保留执行器强度、响应、总力矩保护与失能六电机置零；物理气簧继续生效。恢复环境包装层拥有这一补偿调用，底层V5在该路径显式跳过自身补偿。没有冻结腿、增加串级或叠加旧静态负载初始化。

CUDA执行图分别捕获同一模型的编码器几何/补偿与物理力纯张量计算，使用独立回放入口和独立输出存储；不捕获物理仿真、随机历史或优化器。模型参数保持dev原值，性能仍需在实际机器单独测量。

## 证据与验证边界

新源绑定包为`model/gas_spring_compensation_unified_20261029/`，明确引用dev原组件报告，保留原交付包。原24000步零重力组件结果只支持原组件模型，不能作为合并后起立、静站或运动的行为通过。

`tests/test_unified_training.py`使用冻结dev实现生成的281组几何/软限位/饱和样本，检查原力曲线、补偿、生产恢复控制包装和CUDA回放；同时核对100Hz/64步、单一补偿、旧计划保留和随机域配置。固定参考不含策略或旧权重。

本次不启动PPO或读取checkpoint。原100Hz训练独立收到SIGINT后在131次完整更新封存，源码核对无差异；确认进程与GPU退出后才运行本次独立双域零PPO接线探针。入口并行保护保留。集成报告位于主工作区`reports/branch_unification_20261029/`，原生检查结果以其中实际保存的报告为准。未运行合并后行为验收，全部13项长期能力均未由本次整合授予通过；继续按既有完整30秒、约10cm起立、原始接触及双向实际转速标准验收。
