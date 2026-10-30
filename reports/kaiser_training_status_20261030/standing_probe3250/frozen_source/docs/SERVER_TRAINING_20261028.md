# 本次标准趴地长训练的服务器环境

此文用于`train_prone_motion.sh`，替代历史`server_setup.md`对本入口的安装建议。
统一v3默认采用dev实测拟合气簧及一次编码器力矩补偿，来源见[统一说明](WHEEL_LEG_UNIFIED_20261029.md)。
新版默认策略100Hz、rollout64步；物理与腿PD保持1kHz。从零初始化、连续10000次更新、每250次保存一次PT；详见PRONE_MOTION_TRAINING_20261028.md中的时间尺度与域随机化单位说明。
环境修复与CPU预检不授予策略行为或GPU仿真资格，也不自动开始训练。

## Kaiser WSL 启动路径（2026-10-29）

Kaiser使用`$HOME/robot-rl-sim60/env/bin/python`，与下文的容器镜像分开。
在该WSL安装中，`libcuda.so.1`能由系统加载，但PhysX按名称加载的`libcuda.so`
需要`/usr/lib/wsl/lib`搜索路径。缺少此路径时，Torch仍能看到4090，PhysX却回退CPU，
最终在内部Warp内核处报`ProxyArray`参数错误。不能通过改成CPU训练、修改数组类或跳过预检解决。
共用的train/play启动器会在真实`libcuda.so`存在时加入WSL目录，并保留调用者已有路径。

Kaiser还缺少MaterialX依赖的`libXt.so.6`。没有sudo时可从系统已配置的Ubuntu仓库提取到用户目录：

```bash
mkdir -p "$HOME/.local/share/wheeled-rl/runtime-libs/packages"
cd "$HOME/.local/share/wheeled-rl/runtime-libs/packages"
apt-get download libxt6 libsm6 libice6
for package in ./*.deb; do
  dpkg-deb --extract "$package" "$HOME/.local/share/wheeled-rl/runtime-libs"
done
```

启动器自动加入该用户目录下的`usr/lib/x86_64-linux-gnu`，不改系统库或驱动。
其他提取位置可用`WHEELED_RL_RUNTIME_LIB_DIR`指定；原始下载包及SHA回执应保留。
真实GPU物理缓冲的设备读回与完整原生预检都要通过，不能仅凭`torch.cuda.is_available()`判断可训练。

## 已有镜像与依赖修复

服务器预装Sim `6.2.0-rc.14`、Isaac Lab源码`3.0.0`、Torch `2.11.0+cu130`、
torchvision `0.26.0+cu130`、Warp `1.17.0`、RSL-RL `5.5.1`。
Isaac Lab 3.0官方配套版本见[发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)。
服务器Sim仍是RC，不能以版本配套代替该二进制的原生验证。

原始`/opt/isaaclab-venv`有4项依赖错误。已保留修复前清单，使用以下相容版本：

```bash
/opt/isaaclab-venv/bin/python -m pip install --only-binary=:all: --no-deps \
  numpy==2.3.5 packaging==23.2 imageio-ffmpeg==0.6.0 \
  PyJWT==2.15.1 usd-validation-nvidia==1.23.0
```

NumPy同时满足cmeel-boost的`>=2.3,<2.4`与Numba的`<2.5`；
packaging满足ovphysx的`>=20,<24`；ffmpeg补齐robomimic声明的依赖。
加载完整Kit启动环境后，还补齐预捆绑msal的PyJWT依赖和两个simready扩展的USD校验依赖。
不升级或替换Torch、RSL、Warp及模拟器，不修改无卡模式的平台限制。

## 使用正确的Python运行时

### Kaiser

Kaiser 的项目目录为 `/home/kaiser/workspace/isaac_sim-6-task`，已有 Isaac 环境为
`/home/kaiser/robot-rl-sim60/env/bin/python`。train/play 现在自动检查该用户目录、
已激活环境和 PATH，无需修改脚本或使用本机 `/home/noir/...` 路径。
显式选择可使用 `--python /home/kaiser/robot-rl-sim60/env/bin/python`；其优先级高于
`WHEELED_RL_PYTHON`。两者都未设置才自动选择，自动候选需能定位四个训练模块。

```bash
cd /home/kaiser/workspace/isaac_sim-6-task
bash scripts/train_prone_motion.sh --help
bash scripts/train_prone_motion.sh \
  --task WheelLeg-ProneMotion-v3 \
  --num_envs 256 --max_iterations 10000 --num_steps_per_env 64 \
  --save_interval 250 --seed 103 --run_name train_10000_seed103 \
  --domain_randomization curriculum --device cuda:0 --headless
```

训练仍在独立双域原生探针通过后才启动，解释器检测不代替原生或行为验证。
如已有同名运行，保留旧目录并改 `--run_name`；不覆盖或续训。

### 预装 `/isaac-sim` 的容器镜像

不能直接执行镜像中的venv Python进行训练：镜像的`LD_LIBRARY_PATH`使其加载系统
Python 3.12.3动态库，却使用Isaac附带的3.12.14标准库，导致`ctypes`报
`undefined symbol: _PyErr_SetLocaleString`。
仅加载NVIDIA的环境脚本也不够：Kit的预捆绑路径会覆盖venv中修复的NumPy和packaging。

`scripts/server_isaac_python.sh`加载现有Isaac环境，优先使用匹配的libpython与venv包，
保留既有预加载项，最后`exec`真实Python以保持停止信号的进程归属。
未指定解释器且检测到该镜像时会选用此启动器。默认Sim目录`/isaac-sim`、Python`/opt/isaaclab-venv/bin/python`；
可用`WHEELED_ISAACSIM_ROOT`和`WHEELED_ISAAC_PYTHON`显式覆盖。

以下命令均在服务器的项目根目录执行：

```bash
chmod +x scripts/server_isaac_python.sh
export WHEELED_RL_PYTHON="$PWD/scripts/server_isaac_python.sh"
"$WHEELED_RL_PYTHON" -m pip check
bash scripts/train_prone_motion.sh --output /tmp/prone_motion_cpu_check001 --preflight-only
```

CPU预检输出目录必须是新的，不启动Kit、创建PPO或读取旧权重。
无卡模式不运行原生仿真；CPU/内存配额按挂卡后的实际分配另行核对。

## 挂卡后的训练命令

在服务器项目根目录执行，选择不存在的输出目录：

```bash
export WHEELED_RL_PYTHON="$PWD/scripts/server_isaac_python.sh"
bash scripts/train_prone_motion.sh --output "$PWD/runs/prone_motion/run_unified001"
```

先将 `upstream/dev/wheel-leg-unified` 的完整源码同步至服务器新目录，保持现有运行的冻结源与进程不动。仅在旧训练已结束并封存后使用新输出目录启动。

脚本先运行名义域与完整域的零PPO原生探针并检查保存的结果，全部通过后才开始
同一次`learn(10000)`。未通过不能仅凭返回码手动跳过。不要传本机旧009停止参数，
不要加载中断运行、旧teacher或donor。训练中无冷测，结束后再按相同病例比较周期PT。
服务器软件准备阶段不运行此命令。

## Lab 3 场景路径兼容性

GPU 首次探针的实际异常保存在 `run001.native_probe/progress.json` 的 `error` 字段：
`Unexpected number of real loop constraints`。它不是上方 Protobuf 或 AppLauncher 警告引起的退出。
原路径 `/World/envs/env_.*/Robot` 在服务器的 Lab 3 路径选择器中还匹配环境的地面、材质等子路径；
80 个探针环境各生成 6 台机器人，实际闭链球铰 2880 个，预期 480 个。
改用 `/World/envs/env_[0-9]+/Robot` 只匹配编号环境；闭链数量与求解器拓扑检查继续保留。
探针验证失败时也会显示保存的原始异常，避免只看到泛化的拒绝 PPO 信息。

闭链修复后还发现 Torch 的 `libtorch_cuda_linalg.so` 以名称延迟加载，但环境未包含
venv 的 `torch/lib`。单独执行 CUDA `eigvalsh` 复现，补充搜索路径后通过；
服务器 Python 启动器现自动加入该目录，继续使用原 Torch/CUDA 版本。

保留失败目录，修复后的启动必须使用新的目录。可以先单独执行两个域的零 PPO 检查：

```bash
WHEELED_RL_PYTHON="$PWD/scripts/server_isaac_python.sh" \
  bash scripts/train_prone_motion.sh --native-probe --output "$PWD/runs/prone_motion/gpu_probe002"
```

这只验证原生建模、控制、接触和域切换接线，不是自起、站立或运动能力验收。
