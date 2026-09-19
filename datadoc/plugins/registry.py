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
    from datadoc.plugins.target_encoder import TargetEncoderPlugin
    from datadoc.plugins.rare import RareCategoryPlugin
    from datadoc.plugins.polynomial import PolynomialFeaturesPlugin
    from datadoc.plugins.scaling import ScalingPlugin

    return [
        DuplicateRemoverPlugin(),
        MissingValuePlugin(),
        OutlierPlugin(),
        DatetimePlugin(),
        CategoricalEncoderPlugin(),
        TargetEncoderPlugin(),
        RareCategoryPlugin(),
        PolynomialFeaturesPlugin(),
        ScalingPlugin(),
    ]


BUILTIN_PLUGIN_NAMES = [
    "DuplicateRemoverPlugin",
    "MissingValuePlugin",
    "OutlierPlugin",
    "DatetimePlugin",
    "CategoricalEncoderPlugin",
    "TargetEncoderPlugin",
    "RareCategoryPlugin",
    "PolynomialFeaturesPlugin",
    "ScalingPlugin",
]


_CACHED_ENTRY_POINTS: list["BasePlugin"] | None = None


def _entry_point_plugins() -> list["BasePlugin"]:
    global _CACHED_ENTRY_POINTS
    if _CACHED_ENTRY_POINTS is not None:
        return _CACHED_ENTRY_POINTS

    found: list["BasePlugin"] = []
    try:
        eps = entry_points(group="datadoc.plugins")
    except TypeError:
        # Python 3.10 compat: entry_points() returns dict
        try:
            eps = entry_points().get("datadoc.plugins", [])  # type: ignore
        except Exception:
            eps = []
    except Exception:
        eps = []

    for ep in eps:
        try:
            cls = ep.load()
            inst = cls() if isinstance(cls, type) else cls
            # validate interface duck-typing
            if hasattr(inst, "name") and hasattr(inst, "analyze") and (hasattr(inst, "apply") or hasattr(inst, "transform")):
                found.append(inst)
        except Exception:
            continue
    _CACHED_ENTRY_POINTS = found
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


def resolve_plugins(plugin_names: list[str]) -> list["BasePlugin"]:
    """Resolve and order a list of plugin names or alias keywords (e.g. 'all', 'builtin')."""
    if not plugin_names:
        return []
    all_plugins = list_plugins()
    name_map = {p.name.lower(): p for p in all_plugins}

    resolved: dict[str, "BasePlugin"] = {}
    for spec in plugin_names:
        if not spec:
            continue
        spec_clean = spec.strip().lower()
        if spec_clean == "all":
            for p in all_plugins:
                resolved[p.name] = p
        elif spec_clean == "builtin":
            for p in _builtin_plugins():
                resolved[p.name] = p
        elif spec_clean in name_map:
            p = name_map[spec_clean]
            resolved[p.name] = p
        else:
            # Check prefix / suffix match (e.g. "missing" matching "MissingValuePlugin")
            matches = [
                p for p in all_plugins
                if spec_clean in p.name.lower()
            ]
            if len(matches) == 1:
                resolved[matches[0].name] = matches[0]

    return sorted(resolved.values(), key=lambda p: p.priority)
