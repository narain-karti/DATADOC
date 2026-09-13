"""Plugin registry: auto-discovery for built-ins + entry-points.

Built-ins are the deterministic core. External packages can register via:
    [project.entry-points."datadoc.plugins"]
    my_plugin = "my_package.my_plugin:MyPlugin"
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datadoc.plugins.base import BasePlugin


def _builtin_plugins() -> list["BasePlugin"]:
    from datadoc.plugins.duplicates import DuplicateRemoverPlugin
    from datadoc.plugins.missing_values import MissingValuePlugin
    from datadoc.plugins.outliers import OutlierPlugin
    from datadoc.plugins.datetime_feat import DatetimePlugin
    from datadoc.plugins.encoders import CategoricalEncoderPlugin
    from datadoc.plugins.rare import RareCategoryPlugin
    from datadoc.plugins.scaling import ScalingPlugin

    return [
        DuplicateRemoverPlugin(),
        MissingValuePlugin(),
        OutlierPlugin(),
        DatetimePlugin(),
        CategoricalEncoderPlugin(),
        RareCategoryPlugin(),
        ScalingPlugin(),
    ]


BUILTIN_PLUGIN_NAMES = [
    "DuplicateRemoverPlugin",
    "MissingValuePlugin",
    "OutlierPlugin",
    "DatetimePlugin",
    "CategoricalEncoderPlugin",
    "RareCategoryPlugin",
    "ScalingPlugin",
]


def _entry_point_plugins() -> list["BasePlugin"]:
    found: list["BasePlugin"] = []
    try:
        eps = entry_points(group="datadoc.plugins")
    except TypeError:
        # Python 3.10 compat: entry_points() returns dict
        try:
            eps = entry_points().get("datadoc.plugins", [])  # type: ignore
        except Exception:
            return []
    except Exception:
        return []
    for ep in eps:
        try:
            cls = ep.load()
            inst = cls() if isinstance(cls, type) else cls
            # validate interface duck-typing
            if hasattr(inst, "name") and hasattr(inst, "analyze") and hasattr(inst, "apply"):
                found.append(inst)
        except Exception:
            continue
    return found


def list_plugins() -> list["BasePlugin"]:
    """All registered plugins sorted by priority (lower runs first)."""
    plugs = _builtin_plugins() + _entry_point_plugins()
    # de-dupe by name, builtin wins
    seen: dict[str, "BasePlugin"] = {}
    for p in plugs:
        try:
            name = p.name
        except Exception:
            continue
        if name not in seen:
            seen[name] = p
    return sorted(seen.values(), key=lambda p: p.priority)


def get_plugin(name: str) -> "BasePlugin | None":
    for p in list_plugins():
        if p.name == name or p.name.lower() == name.lower():
            return p
    return None
