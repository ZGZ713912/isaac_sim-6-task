"""按冻结会话标记路由，不在导入时加载网络或仿真。"""
from pathlib import Path

from .training_control import read_json

DYNAMIC_PATH = Path.home() / ".config/transformer-rl/dynamic.json"
REGISTRY_PATH = Path.home() / ".config/transformer-rl/managed.json"


def _session_at(root):
    if (root / 'structured.json').is_file():
        from ..formal.runtime.session import StructuredTrainingSession
        return StructuredTrainingSession(root)
    from ..research.session import TrainingSession
    from ..research.training_queue import TrainingQueue
    return TrainingQueue(root) if (root / "queue.json").is_file() else TrainingSession(root)


def _read_mapping(path):
    value = read_json(path, {})
    return value if isinstance(value, dict) else {}


def _registry():
    return _read_mapping(REGISTRY_PATH)


def dynamic_rules():
    """别名 → 规则；目前只支持 ``{"formal": "active"}`` 这类跟随当前活跃会话。"""
    return _read_mapping(DYNAMIC_PATH)


def ranked_active(registry=None):
    """按最近更新时间倒序的活跃会话 [(alias, session)]。

    只读 status 投影，不发送控制请求、不读模型与指标日志。
    """
    registry = _registry() if registry is None else registry
    ranked = []
    for name, value in registry.items():
        if not isinstance(value, str):
            continue
        try:
            session = _session_at(Path(value))
            state = session.status()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not state.get("active"):
            continue
        ranked.append((str(state.get("updated_at") or ""), int(state.get("update") or 0), name, session))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(name, session) for _, _, name, session in ranked]


def active_alias(registry=None):
    """当前有活进程的别名；一个都没有时返回 None。"""
    ranked = ranked_active(registry)
    return ranked[0][0] if ranked else None


def open_session(value):
    root = Path(value).expanduser()
    if not root.exists():
        name = str(value)
        if dynamic_rules().get(name) == "active":
            ranked = ranked_active()
            if not ranked:
                raise ValueError(f"{name}: 当前没有正在运行的受控会话；"
                                 f"用 trainctl list <目录> 查看，或直接传会话路径")
            alias, session = ranked[0]
            try:
                session.resolved_alias = alias
            except (AttributeError, TypeError):
                pass
            return session
        registry = _registry()
        if name not in registry:
            raise ValueError(f"unknown managed session or queue: {name}")
        root = Path(registry[name])
    return _session_at(root)