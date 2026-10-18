# 本机值裁剪同起点对照实现

## 当前边界

原控制臂 pilot019 继续从冻结 `8a34f287a9288814cf301494ae7582ab34b3f149` 运行。工作树新增候选不会改变该进程。原周期 3000 封存已只读验证：9 actor、8 critic、17 Adam 的一、二阶矩全部推进且有限，Adam 60000、时钟 3000、193462272 transitions，91 份冻结源哈希和 CPU/CUDA/环境 RNG 封存通过。它只证明学习状态连续，不证明行为达标。见 `reports/local_training_20261017/pilot019_checkpoint3000.json`。

控制完成并按原八场景实测仍失败时，才生成命名本机 v9 合同。生成器要求实际 control completion、final seal、评测完整复现旧候选和不可变合并消费回执；证据未齐时拒绝生成，当前没有占位 SHA 或可启动的 v9 计划。若控制通过基础站立、前后跟踪和机械检查，则跳过候选。

## 唯一算法变化及恢复

`local_value_revision.py` 派生精确 v8→v9，算法只允许原生 `use_clipped_value_loss=True→False`。旧 v1–v8 缺该字段时保持原生默认 True；显式覆盖仅接受命名 v9 的布尔 False，拒绝数字 0 和其他值。CLI 新入口 `--resume-local-value-revision` 与其他迁移互斥，首次最多 128 更新，且只能从精确 pilot018-2945 的完整学习状态起步。

保持 value coefficient 4、actor clip .2、grad clip 1、固定 LR 1e-4、384 帧、lambda 1、gamma .9974905699336811、entropy .0005，以及物理、控制、reward、DR、出生和课程。actor/critic/17 Adam/学习 std/全部 RNG/课程时钟完整恢复，无额外 critic 预热。原生更新方法不重写。

控制 128 更新必须并入继承消费：1216→1344，谱系上限 18784→18656，总上限仍 20000，transition ceiling 3271557120。处理从共同父 2945 接续到 3073；双臂都完成后总消费 4417、剩 15583。不能从控制终点或通过旧账本退款。

## 已完成和待完成验证

84 项针对性 CPU 检查通过，包括原生已学 Adam 精确 load、20 次小批更新、mask 压紧、两类优势归一化、actor/采样/RNG 保持、原生 clipped-flat 梯度 0 与 unclipped 正确梯度、旧 v8 与观察器回归、缺控制回执拒绝。合成旧值偏移 fixture 只用来覆盖裁剪平坦分支，不是实际 rollout 结论。日志 `~/.local/state/isaac_wheeled_rl_train/local-200hz-20261018/value_cpu001.log`。

仍须实际控制完成/评测、预算合并、独立 target design 的正负例、512 静态前置、同父保存 RNG 的两个独立零 PPO 原生进程核对，再从已提交冻结源码启动处理。零 PPO 必须禁止 update/step，核对已记录完整初态、Gaussian、动力学、命令、reward、conditioning mask 和学习状态保持。通过这些工程检查也不授予行为或长训资格。
