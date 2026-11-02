# 用 TUI 暂停、继续训练

`trainctl` 是原进程内的协作暂停。暂停期间不推进环境、采样、PPO、Adam、随机数或课程；继续时使用原 PID、原显存和原内存中的 rollout/仿真状态。不需要 Ctrl+C、`--resume` 或新输出目录。暂停不释放显存。

## 已接入的入口

`scripts/train_prone_motion.sh` 的正式训练自动注册到 trainctl，原来的启动参数保持不变。直接运行 `scripts/train_prone_motion.py` 也会在进入 learner 时自动注册。配置预检和独立 0-PPO 原生探针照常执行，不把预检当训练。

其他 RSL-RL 训练入口（包括独立工作树）在**下次启动时**使用：

```bash
trainctl run --name my-training -- /path/to/isaac/python scripts/train_example.py [原训练参数]
```

解释器可以是项目的 `scripts/server_isaac_python.sh` 启动器；支持 Python 的 `-u`、`-B`、`-O` 参数。目标必须是 `.py` 文件。控制器不会将自身工作树的 `src` 注入另一个项目。当前仅支持单进程 RSL-RL；分布式训练明确拒绝。

尚未接入控制器的旧进程不能凭空获得此功能。TUI 只列出已注册进程，既不向旧进程发送 SIGSTOP/SIGCONT，也不自动重启它们。

## 控制

在训练所在机器的另一个终端执行：

```bash
trainctl
```

- `↑/↓` 或 `j/k`：选择训练。
- `p`：请求暂停；`r`：继续；空格：切换。
- `q`：断开 TUI，训练保持原来的运行或暂停状态。

界面显示 `pause_requested` 时，训练仍在完成当前操作。主线程抵达下一次采样或 PPO 更新边界，且 CUDA 在途工作完成后才显示 `paused`。不会在 Adam 更新中途冻结。原生场景重建、评测或阻塞的 GPU 调用可能延迟响应；等待期间不能称已暂停。继续未获主线程确认时显示 `resuming`。

Kaiser 上训练时，本机直接打开远端 TUI：

```bash
trainctl --host kaiser
# 或
ssh -t kaiser '~/.local/bin/trainctl wheel-leg'
```

Kaiser 的 `trainctl` 已有另一项目的控制台，因此保留其原命令，轮腿入口为 `trainctl wheel-leg`（或 `wheel-leg-trainctl`）。本机 `trainctl --host kaiser` 会自动选择这一入口。

命令行也能控制：

```bash
trainctl list
trainctl status <ID>
trainctl pause <ID> --wait 30
trainctl resume <ID>
trainctl --host kaiser list
```

ID 可使用唯一前缀；也可用唯一名称或 PID。重名会拒绝操作。`pause`/`resume` 默认等待确认 10 秒，超时返回 2 并输出实际状态；请求仍然保留。`resume` 可以取消尚未确认的暂停。TUI 或 SSH 断开不会改变暂停请求。训练本身应继续使用原有 tmux/远端 supervisor 管理生命周期，trainctl 不接管进程启动后的终端生命周期。

暂停期间，原有 SIGINT/SIGTERM 仍交给训练脚本处理；控制器唤醒主线程，让既有正常停止/封存路径继续执行。TUI 没有停止或重启按键。

## 安装与数据

控制端不依赖 torch/Isaac，只需要 Linux Python 3.10+ 和标准库 curses：

```bash
python3 scripts/install_trainctl.py
~/.local/bin/trainctl list
```

安装器将完整控制器复制到 `~/.local/share/wheel-leg-trainctl/trainctl.py`，启动器位于 `~/.local/bin/trainctl`，因此清理工作树后命令仍可使用。升级后，已运行训练继续执行各自加载的旧控制器；新启动才使用新版本。也可在已安装本项目的 Python 环境中使用 pyproject 提供的同名入口。

如机器已有上述 transformer-rl 控制台，安装使用 `python3 scripts/install_trainctl.py --share-existing`。原启动器按完整 SHA-256 备份，所有原命令继续转交原启动器，只有新增的 `wheel-leg` 子命令进入轮腿控制台。其他未知同名程序仍拒绝覆盖。

每个训练通过仅当前用户可访问的 Unix socket 通信，校验 UID、启动 ticks、boot ID 和随机 job ID，避免 PID 复用误操作。记录默认在 `~/.local/state/trainctl/`，含控制器路径与 SHA-256；可用 `TRAINCTL_STATE_DIR`/`TRAINCTL_RUNTIME_DIR` 隔离测试。客户端不会通过 PID 发信号。

已有 RSL-RL TensorBoard event 增加 `TrainControl/pause_count`、`TrainControl/paused_seconds`、`TrainControl/active_seconds`，step 是已完成的更新。单位分别为次数、秒、秒。原 RSL 的 wall time/FPS 仍可能包含暂停时间；分析性能时结合这些 tag 排除人工暂停。本功能不启动 TensorBoard 服务。

新训练源码清单纳入控制模块。老检查点的 `--resume` 认证仅允许增加这些与暂停有关的文件，物理、观测、奖励和课程兼容要求不变。
