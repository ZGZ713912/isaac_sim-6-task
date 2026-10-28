# 本次标准趴地长训练的服务器环境

此文用于`train_prone_motion.sh`，替代历史`server_setup.md`对本入口的安装建议。
保持从零初始化、连续10000次更新、每250次保存一次PT，以及原课程、控制、奖励和域随机化合同。
环境修复与CPU预检不授予策略行为或GPU仿真资格，也不自动开始训练。

## 已有镜像与依赖修复

服务器预装Sim `6.2.0-rc.14`、Isaac Lab源码`3.0.0`、Torch `2.11.0+cu130`、
torchvision `0.26.0+cu130`、Warp `1.17.0`、RSL-RL `5.5.1`。
Isaac Lab 3.0官方配套版本见[发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)。
服务器Sim仍是RC，不能以版本配套代替该二进制的原生验证。

原始`/opt/isaaclab-venv`有4项依赖错误。已保留修复前清单，使用以下相容版本：

```bash
/opt/isaaclab-venv/bin/python -m pip install --only-binary=:all: --no-deps \
  numpy==2.3.5 packaging==23.2 imageio-ffmpeg==0.6.0
```

NumPy同时满足cmeel-boost的`>=2.3,<2.4`与Numba的`<2.5`；
packaging满足ovphysx的`>=20,<24`；ffmpeg补齐robomimic声明的依赖。
不升级或替换Torch、RSL、Warp及模拟器，不修改无卡模式的平台限制。

## 使用正确的Python运行时

不能直接执行镜像中的venv Python进行训练：镜像的`LD_LIBRARY_PATH`使其加载系统
Python 3.12.3动态库，却使用Isaac附带的3.12.14标准库，导致`ctypes`报
`undefined symbol: _PyErr_SetLocaleString`。
仅加载NVIDIA的环境脚本也不够：Kit的预捆绑路径会覆盖venv中修复的NumPy和packaging。

`scripts/server_isaac_python.sh`加载现有Isaac环境，优先使用匹配的libpython与venv包，
保留既有预加载项，最后`exec`真实Python以保持停止信号的进程归属。
默认Sim目录`/isaac-sim`、Python`/opt/isaaclab-venv/bin/python`；
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
bash scripts/train_prone_motion.sh --output "$PWD/runs/prone_motion/run001"
```

脚本先运行名义域与完整域的零PPO原生探针并检查保存的结果，全部通过后才开始
同一次`learn(10000)`。未通过不能仅凭返回码手动跳过。不要传本机旧009停止参数，
不要加载中断运行、旧teacher或donor。训练中无冷测，结束后再按相同病例比较周期PT。
服务器软件准备阶段不运行此命令。
