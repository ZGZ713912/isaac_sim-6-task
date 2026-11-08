"""A dependency-free terminal dashboard for managed packed-policy training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
import unicodedata

from .session_router import active_alias, open_session
from .training_control import read_json, verify_pin


LABELS = {"ready": "待启动", "starting": "启动中", "initializing": "初始化环境", "collecting": "采样中",
          "optimizing": "PPO 更新", "running": "训练中", "checkpointing": "保存暂停点", "paused": "已无损暂停",
          "stopping": "保存并退出中", "stopped": "已保存退出", "completed": "训练完成",
          "closing": "已保存 · 等待环境关闭", "ownership_pending": "会话锁仍被持有 · 等待确认",
          "evaluating": "控制评估中", "training": "训练中", "incomplete": "队列已结束 · 保留未完成项",
          "waiting_resources": "等待原任务资源准入",
          "failed": "训练失败", "interrupted": "中断 · 需要检查"}


def _text(value):
    return "".join(char for char in str(value) if char.isprintable() and char != "\x1b")


def _width(value):
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
               for char in value if not unicodedata.combining(char))


def _fit(value, width):
    value = _text(value)
    result = ""
    for char in value:
        if _width(result + char) > width:
            return result[:-1] + "…" if result else ""
        result += char
    return result


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _display_state(session, state=None):
    """只补充冻结配置与公开状态；不读取模型、指标日志或修改控制请求。"""
    state = dict(session.status() if state is None else state)
    if not state.get("structured"):
        return state
    recipe_path = session.root / "recipe.json"
    pin = next((item for item in session.spec["pins"]
                if item["path"] == str(recipe_path)), None)
    if pin is not None:
        recipe = read_json(verify_pin(pin))
        state.update(rollout_steps=recipe["rollout_steps"],
                     policy_dt_s=recipe["collection"]["policy_dt_s"],
                     ppo_epochs=recipe["ppo"]["epochs"],
                     ppo_minibatches=recipe["ppo"]["minibatches"])
    if state.get("checkpoint_update") is None and state.get("latest_checkpoint"):
        match = re.fullmatch(r"update_(\d+)\.pt", Path(state["latest_checkpoint"]).name)
        if match:
            state["checkpoint_update"] = int(match[1])
    return state


def render(state, *, width=84, color=False, message="", level="info", extra=None, overlay=None):
    """Render from the bounded worker projection, without parsing training logs."""
    width = max(44, min(width, 108))
    inner = width - 4
    if overlay is not None:
        return _overlay_frame(overlay, width=width, color=color)
    phase = state["phase"]
    pending = state.get("request_sequence", -1) > state.get("ack_sequence", -1)
    label = LABELS.get(phase, phase)
    if pending and state.get("active"):
        intent = {"pause": "等待更新边界暂停", "resume": "等待继续确认", "stop": "等待保存退出"}
        label += " · " + intent.get(state.get("requested"), "等待确认")
    update, target = state.get("update", 0), state["target_update"]
    budget_update = state.get("budget_update", update)
    ratio = min(1., max(0., budget_update / target))
    bar_width = max(10, inner - 23)
    filled = int(bar_width * ratio)
    bar = "━" * filled + "─" * (bar_width - filled)
    lines = ["TRAINING / 训练控制", "", f"{state['name']}   ·   {label}",
             f"{state['architecture']}    seed {state['seed']}    {state['device']}", "",
             f"{bar}  {ratio:6.1%}", f"更新  {update:,} / {target:,}"]
    if state.get("retired_consumption"):
        lines[6] = f"预算  {budget_update:,} / {target:,} · 保留模型更新  {update:,}"
        lines.append(f"历史丢失更新仍计费  {state['retired_consumption']['ppo_updates']:,}")
    if state.get('structured'):
        layout = f"{state['num_envs']} 环境"
        if state.get("rollout_steps") is not None:
            layout += f" × {state['rollout_steps']} 步"
        layout += f" · {state['history_length']} 帧含当前"
        period = state.get("policy_dt_s")
        if isinstance(period, (int, float)) and math.isfinite(period) and period > 0:
            layout += f" · 策略 {1 / period:g}Hz"
            lines.append(layout)
            span_ms = (state['history_length'] - 1) * period * 1000
            lines.append(f"历史跨度 {span_ms:g} ms · 评估由人手动发起")
        else:
            lines.append(layout)
        if state.get("ppo_epochs") is not None and state.get("ppo_minibatches") is not None:
            epochs, minibatches = state['ppo_epochs'], state['ppo_minibatches']
            lines.append(f"PPO {epochs} epochs × {minibatches} minibatches = {epochs * minibatches} 步 / 更新")
        position = state.get("solver_position_iterations")
        velocity = state.get("solver_velocity_iterations")
        if position is not None and velocity is not None:
            lines.append(f"闭链迭代 {position} / {velocity} · 按冻结配置保存")
        if state.get("curriculum_stage") is not None:
            lines.append(f"课程阶段 {state['curriculum_stage']} · {_text(state.get('curriculum_status', ''))}")
    if state.get("queue_counts"):
        counts = state["queue_counts"]
        lines[6] = f"{'评估步' if state.get('stage') == 'evaluating' else '当前更新'}  {update:,} / {target:,}"
        lines += [f"完整格 {counts['completed']}/{counts['total']} · 保留未完成 {counts['retained_incomplete']} · 未开始 {counts['not_started']}",
                  str(state.get("active_cell") or "待调度")]
        if state.get("evaluation_seed") is not None:
            lines.append(f"开发评估 seed {state['evaluation_seed']} · 四 seed × 50 场景")
    transitions = state.get("total_consumption", {}).get("returned_transitions", state.get("cumulative_transitions"))
    if transitions is not None:
        lines.append(f"累计采样  {transitions:,} transitions")
    lines.append("活动时间  " + _duration(state.get("session_active_elapsed_s", 0))
                 + "    暂停  " + _duration(state.get("paused_elapsed_s", 0)))
    if state.get("reward_mean") is not None:
        lines.append(f"Reward  {state['reward_mean']:.5g}    LR  {state.get('learning_rate', 0):.3g}")
    if phase == "waiting_resources" and state.get("resources"):
        resources = state["resources"]
        lines += [f"可用 RAM  {resources['MemAvailable'] / 1024**3:.2f} / 所需 {resources['minimum_ram_available_bytes'] / 1024**3:.1f} GiB",
                  f"可用 VRAM {resources['gpu_free_mib'] / 1024:.2f} / 所需 {resources['minimum_gpu_free_mib'] / 1024:.1f} GiB",
                  ("达到原任务资源门槛后自动继续。" if state.get('resources_resume_automatic', True)
                   else "尚未创建训练 attempt；资源恢复后使用 start / resume。")]
    lines += ["", f"最近保存  update {state['checkpoint_update']}" if state.get("checkpoint_update") is not None else "最近保存  尚无检查点"]
    if phase == "paused":
        lines.append("状态保留在原进程内；暂停期间仍占用内存和显存。按 f 保存退出即可腾卡。")
    elif phase == "stopped":
        lines.append("学习状态已保存；下次启动会重置物理回合与历史窗口。")
    elif phase == "completed":
        lines.append("训练预算已完成；控制性能仍需独立评估。")
    if state.get("boot_changed"):
        lines.append("主机 boot 已改变；研究队列需要重新审核接入。" if state.get("queue_counts")
                     else "主机 boot 已改变；跨 boot 启动需显式 --accept-new-boot。")
    if state.get("error"):
        lines.append("错误  " + _text(state["error"]))
    if extra:
        lines += ["", *extra]
    lines += ["", _KEYS, "离开面板 / SSH 断线不会停止训练。", "", str(state["root"])]
    message_row = -1
    if message:
        lines += ["", message]
        message_row = len(lines) - 1
    accent = "\x1b[38;5;81m" if color else ""
    reset = "\x1b[0m" if color else ""
    output = [accent + "╭" + "─" * (width - 2) + "╮" + reset]
    for index, line in enumerate(lines):
        line = _fit(line, inner)
        content = line + " " * (inner - _width(line))
        if color and index in (0, 2, 5):
            content = "\x1b[1;38;5;81m" + content + reset
        if color and index == message_row:
            content = _LEVEL_COLOR.get(level, "") + content + reset
        output.append(accent + "│" + reset + " " + content + " " + accent + "│" + reset)
    output.append(accent + "╰" + "─" * (width - 2) + "╯" + reset)
    return "\n".join(output)


def _overlay_frame(overlay, *, width, color):
    title, rows = overlay
    inner = width - 4
    lines = [str(title), ""] + [str(row) for row in rows] + ["", "任意键返回面板。"]
    accent = "\x1b[38;5;81m" if color else ""
    reset = "\x1b[0m" if color else ""
    output = [accent + "╭" + "─" * (width - 2) + "╮" + reset]
    for index, line in enumerate(lines):
        line = _fit(line, inner)
        content = line + " " * (inner - _width(line))
        if color and index == 0:
            content = "\x1b[1;38;5;81m" + content + reset
        output.append(accent + "│" + reset + " " + content + " " + accent + "│" + reset)
    output.append(accent + "╰" + "─" * (width - 2) + "╯" + reset)
    return "\n".join(output)


@contextmanager
def _terminal():
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


_KEYS = "p 暂停   r 继续   s 保存退出   f 释放显存   l 日志   ? 帮助   q 离开"
_LEVEL_COLOR = {"info": "", "ok": "\x1b[32m", "warn": "\x1b[33m", "error": "\x1b[31m"}


def _gpu_memory():
    """读一次显存占用；WSL 下由 Windows 驱动回答，任何失败都静默返回 None。"""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2., check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not rows:
        return None
    try:
        used, total = (int(value) for value in rows[0].split(",")[:2])
    except ValueError:
        return None
    return {"used_mib": used, "total_mib": total}


def _checkpoint_files(state):
    """校验暂停点能不能被 resume：载荷非空、元数据可解析、pin 未变。"""
    path = state.get("latest_checkpoint") or state.get("checkpoint")
    if not path:
        raise ValueError("会话还没有可用的保存点")
    payload = Path(path)
    if not payload.is_file() or payload.stat().st_size <= 0:
        raise ValueError(f"保存点载荷缺失或为空：{payload}")
    meta = Path(str(payload) + ".json")
    try:
        record = read_json(meta)
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        raise ValueError(f"保存点元数据不可解析：{meta}")
    pins = record.get("checkpoint_pins")
    for pin in pins if isinstance(pins, list) else ():
        if isinstance(pin, dict) and str(pin.get("path", "")) == str(payload):
            verify_pin(pin)
            break
    return payload, meta


def _help_lines():
    return [
        "p  暂停   等 update 边界，保存后冻结原进程（仍占用显存）",
        "r  继续   同进程解冻；进程已退出时恢复学习状态",
        "s  保存退出   等当前 update 完成，保存后结束进程（需确认）",
        "f  释放显存   暂停 → 校验保存点 → 结束进程 → 报告释放量（需确认）",
        "l  本会话 worker 日志尾部",
        "?  本页",
        "q  离开面板；Ctrl+C 同效。离开面板不会停止训练",
        "",
        "面板每 0.5 s 只读一次控制状态；不读模型、不读指标日志。",
        "暂停/继续/退出都由 worker 在 update 边界确认后生效，面板不猜。",
        "显存只有进程退出才归还；暂停不等于腾卡。",
    ]


def _log_lines(session):
    try:
        path = session.status().get("log")
    except (OSError, ValueError) as error:
        return [_text(error)]
    if not path:
        return ["尚未启动，无 worker 日志。"]
    try:
        return _tail(path, 40).splitlines() or ["（日志为空）"]
    except OSError as error:
        return [_text(error)]


class Release:
    """释放显存：暂停 → 校验保存点 → 保存退出 → 报告释放量。

    只组合既有控制请求（pause / stop），不改协议。任何一步失败都停在原地：
    不发强信号、不删文件、进程照旧。
    """

    PAUSE_TIMEOUT = 300.
    STOP_TIMEOUT = 300.

    def __init__(self, session, *, probe=_gpu_memory, clock=time.monotonic):
        self.session = session
        self.probe = probe
        self.clock = clock
        self.stage = "prepare"
        self.marker = clock()
        self.before = probe()

    def _elapsed(self):
        return _duration(self.clock() - self.marker)

    def step(self, state):
        """推进一格；返回 (still_running, message, level)。"""
        if self.stage == "prepare":
            if not state.get("active"):
                self.stage = "done"
                return False, "会话没有占用进程，无需释放显存。", "info"
            if state.get("phase") in ("paused", "stopped", "completed"):
                self.stage = "verify"
                return True, "已在暂停状态，直接校验保存点…", "info"
            self.session.command("pause")
            self.stage, self.marker = "pausing", self.clock()
            return True, "已提交暂停，等待 update 边界保存…", "info"
        if self.stage == "pausing":
            if state.get("phase") == "paused":
                self.stage = "verify"
                return True, "暂停已确认，校验保存点…", "info"
            if self.clock() - self.marker > self.PAUSE_TIMEOUT:
                self.stage = "aborted"
                return False, f"等暂停超时（{self._elapsed()}）；进程未受影响。", "error"
            return True, f"等待边界保存 {self._elapsed()}…", "info"
        if self.stage == "verify":
            try:
                payload, _ = _checkpoint_files(state)
            except ValueError as error:
                self.stage = "aborted"
                return False, f"保存点校验失败：{error}；未结束进程。", "error"
            self.session.command("stop")
            self.stage, self.marker = "stopping", self.clock()
            return True, f"{payload.name} 校验通过，正在保存并退出…", "info"
        if self.stage == "stopping":
            if not state.get("active") or state.get("phase") in ("stopped", "completed"):
                self.stage = "done"
                return False, self.report(state), "ok"
            if self.clock() - self.marker > self.STOP_TIMEOUT:
                self.stage = "aborted"
                return False, f"等进程退出超时（{self._elapsed()}）；按 l 看日志。", "error"
            return True, f"等待进程退出 {self._elapsed()}…", "info"
        return False, "", "info"

    def report(self, state):
        after = self.probe()
        parts = []
        if self.before and after:
            released = self.before["used_mib"] - after["used_mib"]
            parts.append(f"显存 {self.before['used_mib']:,} → {after['used_mib']:,} MiB，释放 {released:,} MiB")
        elif after:
            parts.append(f"当前显存 {after['used_mib']:,} MiB")
        parts.append(f"已保存到 update {state.get('checkpoint_update')}")
        parts.append(f"恢复：trainctl resume {state.get('name')} --wait 60（冷启动 1–2 分钟）")
        return "；".join(parts) + "。"


def _run_release(session, seconds):
    """命令行版：阻塞推进同一个状态机，超时不发强信号。"""
    machine = Release(session)
    deadline = time.monotonic() + seconds
    running, message, level = machine.step(session.status())
    while running and time.monotonic() < deadline:
        time.sleep(.5)
        running, message, level = machine.step(session.status())
    if running:
        message, level = "等待超时；进程仍在原处，未发送强信号。", "error"
    state = dict(session.status())
    state["release"] = {"stage": machine.stage, "message": message, "level": level}
    return state, message, level


def watch(session, *, plain=False):
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM") == "dumb":
        print(render(_display_state(session), color=False))
        return
    color = not plain and "NO_COLOR" not in os.environ
    message, level = "按 ? 查看键位；离开面板不会停止训练。", "info"
    confirm = None
    release = None
    overlay = None
    memory, memory_at = None, 0.
    live, live_at = None, 0.
    with _terminal():
        while True:
            columns, rows = shutil.get_terminal_size((84, 30))
            state = _display_state(session)
            now = time.monotonic()
            if now - memory_at > 5.:
                memory, memory_at = _gpu_memory(), now
            if confirm and now > confirm["deadline"]:
                confirm, message, level = None, "确认已超时，操作取消。", "warn"
            if release is not None:
                running, text, level = release.step(state)
                message = text
                if not running:
                    release = None
            if not state.get("active") and now - live_at > 5.:
                live, live_at = active_alias(), now
            extra = []
            if memory:
                spare = (memory["total_mib"] - memory["used_mib"]) / 1024
                extra.append(f"显存 {memory['used_mib']:,} / {memory['total_mib']:,} MiB · 剩余 {spare:.1f} GiB")
            if not state.get("active") and live:
                extra.append(f"该会话未在运行 · 当前活跃 {live}（trainctl watch {live}）")
            if confirm:
                extra.append(f"确认：再按 {confirm['key']} 执行，其它键取消（{confirm['deadline'] - now:.0f}s）")
            screen = render(state, width=columns - 1, color=color, message=message, level=level,
                            extra=extra, overlay=overlay)
            if rows < len(screen.splitlines()):
                screen = "\n".join(screen.splitlines()[:max(1, rows - 1)])
            sys.stdout.write("\x1b[H\x1b[2J" + screen)
            sys.stdout.flush()
            readable, _, _ = select.select([sys.stdin], [], [], .5)
            if not readable:
                continue
            key = sys.stdin.read(1)
            if overlay is not None:
                overlay = None
                continue
            if key in ("q", "\x04"):
                return
            lowered = key.lower()
            if key == "?":
                overlay, confirm = ("帮助 / 键位", _help_lines()), None
                continue
            if lowered == "l":
                overlay, confirm = ("worker 日志", _log_lines(session)), None
                continue
            if lowered in ("s", "f"):
                if confirm and confirm["key"] == lowered:
                    confirm = None
                    if lowered == "f":
                        release = Release(session)
                        message, level = "开始释放：暂停 → 校验保存点 → 退出进程…", "info"
                    else:
                        try:
                            session.command("stop")
                            message, level = "已提交保存退出；worker 在 update 边界确认。", "info"
                        except (OSError, ValueError) as error:
                            message, level = _text(error), "error"
                    continue
                confirm = {"key": lowered, "deadline": now + 6.}
                message = ("保存退出会结束进程，并把显存还给驱动。" if lowered == "s"
                           else "释放显存 = 暂停 + 校验保存点 + 结束进程；之后要 resume 冷启动。")
                level = "warn"
                continue
            if confirm:
                confirm = None
            if lowered == "r" and not state["active"]:
                try:
                    session.start()
                    message, level = "已提交恢复；worker 会从保存点重启。", "info"
                except (OSError, ValueError) as error:
                    message, level = _text(error), "error"
                continue
            operation = {"p": "pause", "r": "resume"}.get(lowered)
            if operation:
                try:
                    session.command(operation)
                    message, level = "请求已提交；面板显示 worker 确认后的真实状态。", "info"
                except (OSError, ValueError) as error:
                    message, level = _text(error), "error"

def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def _seconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("必须是有限非负秒数")
    return result


def _tail(path, lines):
    with Path(path).open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 65536))
        raw = stream.read(65536)
    values = raw.decode("utf-8", errors="replace").splitlines()
    if size > 65536:
        values = values[1:]
    return "\n".join(_text(line) for line in values[-lines:])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainctl", description="训练控制台 · 更新边界暂停 / 同进程继续 / 保存退出",
        epilog="暂停保留完整内存状态；保存退出后的恢复保留学习状态并重置回合。旧冻结 campaign 不会被自动接管。")
    parser.add_argument("--plain", action="store_true", help="禁用颜色；也遵循 NO_COLOR")
    commands = parser.add_subparsers(dest="operation")
    create = commands.add_parser("init", help="冻结配置与学习源码；不启动训练")
    create.add_argument("session", type=Path)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--env-factory", required=True)
    create.add_argument("--python", default=sys.executable, help="已配置好运行环境的训练 Python")
    create.add_argument("--python-path", type=Path, action="append", default=[], help="环境工厂的额外模块路径")
    create.add_argument("--updates", type=_positive, required=True)
    create.add_argument("--rollout-steps", type=_positive, default=48)
    create.add_argument("--checkpoint-interval", type=_positive, default=40)
    create.add_argument("--max-seconds", type=_seconds, default=3600.)
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--device", default="cpu")
    create.add_argument("--name")
    create.add_argument("--tensorboard", action="store_true")
    adopt = commands.add_parser("adopt", help="继承已核验研究记录并登记队列；不启动训练")
    adopt.add_argument("name")
    adopt.add_argument("--plan", type=Path, required=True)
    adopt.add_argument("--root", type=Path, required=True)
    for name, help_text in (("start", "启动新会话或从正常退出点继续"), ("resume", "同进程继续；已退出时恢复学习状态"),
                            ("pause", "等待更新边界，保存后暂停原进程"), ("stop", "等待当前更新完成，保存学习状态并退出"),
                            ("release", "校验保存点后保存退出，释放进程占用的显存"),
                            ("status", "读取控制状态，不读取训练原始日志"), ("watch", "交互面板；q / Ctrl+C 仅离开面板"),
                            ("logs", "显示本会话 worker 最近日志"), ("jobs", "显示研究队列的全部单元")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("session", type=Path)
        command.add_argument("--json", action="store_true", help="机器可读状态")
        if name in ("start", "resume"):
            command.add_argument("--accept-new-boot", action="store_true", help="明确授权本次跨 boot 启动")
        if name == "start":
            command.add_argument("--paused", action="store_true", help="初始化后先保存并暂停，不进行 PPO 更新")
        if name in ("pause", "resume", "stop", "release"):
            command.add_argument("--wait", type=_seconds, default=300. if name == "release" else 30.,
                                 help="最多等待确认的秒数；超时不发送强制信号")
        if name == "logs":
            command.add_argument("--lines", type=_positive, default=30)
    listing = commands.add_parser("list", help="列出目录中的受控会话")
    listing.add_argument("directory", type=Path, nargs="?", default=Path("runs"))
    listing.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.operation is None:
        parser.print_help()
        return 0
    message, level = "", "info"
    try:
        if args.operation == "init":
            from ..research.session import TrainingSession

            session = TrainingSession.create(args.session, args.config, env_factory=args.env_factory,
                updates=args.updates, python=args.python, device=args.device, seed=args.seed,
                rollout_steps=args.rollout_steps, checkpoint_interval=args.checkpoint_interval,
                max_seconds=args.max_seconds, python_paths=args.python_path, name=args.name, tensorboard=args.tensorboard)
            state = session.status()
        elif args.operation == "adopt":
            from ..research.training_queue import TrainingQueue

            state = TrainingQueue.adopt(args.root, args.plan, name=args.name).status()
        elif args.operation == "list":
            states = []
            if args.directory.is_dir():
                for child in sorted(args.directory.iterdir()):
                    if child.is_dir() and any((child / item).is_file() for item in ("session.json", "queue.json", "structured.json")):
                        states.append(_display_state(open_session(child)))
            if args.json:
                print(json.dumps(states, ensure_ascii=False, indent=2))
            else:
                print("TRAINING / 受控会话\n")
                for item in states:
                    print(f"{_text(item['name']):24}  {LABELS.get(item['phase'], item['phase'])}  "
                          f"{item.get('update', 0)}/{item['target_update']}  {_text(item['root'])}")
                if not states:
                    print("暂无会话。使用 trainctl init 创建配置快照。")
            return 0
        else:
            session = open_session(args.session)
            if getattr(session, "resolved_alias", None):
                message = f"别名 {args.session} → 当前活跃会话 {session.resolved_alias}"
            if args.operation == "jobs":
                from ..research.training_queue import TrainingQueue

                if not isinstance(session, TrainingQueue):
                    raise ValueError("jobs requires a managed research queue")
                rows = session.jobs()
                if args.json:
                    print(json.dumps(rows, ensure_ascii=False, indent=2))
                else:
                    for row in rows:
                        print(f"{row['cell']:48} {row['status']}")
                return 0
            if args.operation == "watch":
                watch(session, plain=args.plain)
                return 0
            if args.operation == "logs":
                path = session.status().get("log")
                print(_tail(path, args.lines) if path else "尚未启动，无 worker 日志。")
                return 0
            if args.operation == "start":
                state = session.start(paused=args.paused, accept_new_boot=args.accept_new_boot)
            elif args.operation == "resume" and not session.status()["active"]:
                state = session.start(accept_new_boot=args.accept_new_boot)
            elif args.operation == "release":
                state, message, level = _run_release(session, args.wait)
            elif args.operation in ("pause", "resume", "stop"):
                state = session.command(args.operation, wait_seconds=args.wait)
            else:
                state = session.status()
        state = _display_state(session, state)
        if args.operation == "release":
            message, level = state["release"]["message"], state["release"]["level"]
        if getattr(args, "json", False):
            print(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            color = sys.stdout.isatty() and not args.plain and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
            print(render(state, width=shutil.get_terminal_size((84, 30)).columns - 1, color=color,
                         message=message, level=level))
        if args.operation in ("pause", "resume", "stop", "release") and state.get("active"):
            return 2 if args.operation in ("stop", "release") or state.get("ack_sequence", -1) < state.get("request_sequence", 0) else 0
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError) as error:
        print(f"trainctl: {_text(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
