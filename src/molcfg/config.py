"""Core Config container with attribute/path access, freeze, snapshot, and change notification."""

from __future__ import annotations

import copy
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from molcfg.errors import ConfigError, FrozenConfigError

# ---------------------------------------------------------------------------
# TOML serialization helpers
# ---------------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    """Format a Python scalar as a TOML value string."""
    # bool must come before int — isinstance(True, int) is True
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Cannot serialize {type(value).__name__!r} to TOML")


def _collect_toml(data: dict[str, Any], prefix: str, out: list[str]) -> None:
    """Recursively append TOML lines from *data* into *out*.

    Scalars are emitted first; nested dicts become ``[section]`` headers.
    """
    deferred: list[tuple[str, dict[str, Any]]] = []
    for key, val in data.items():
        if val is None:
            continue
        if isinstance(val, dict):
            deferred.append((key, val))
        else:
            out.append(f"{key} = {_toml_value(val)}")
    for key, sub in deferred:
        path = f"{prefix}.{key}" if prefix else key
        out.append("")
        out.append(f"[{path}]")
        _collect_toml(sub, path, out)


def _dict_to_toml(data: dict[str, Any]) -> str:
    """Convert a plain dict to a TOML-formatted string."""
    out: list[str] = []
    _collect_toml(data, "", out)
    # Strip any leading blank lines produced by the first subsection
    while out and out[0] == "":
        out.pop(0)
    result = "\n".join(out)
    return result + "\n" if result else ""


class Config:
    """Nested configuration container with attribute and path-style access.

    Supports freezing, snapshot/rollback, change callbacks, and serialization.
    """

    __slots__ = ("_data", "_frozen", "_snapshots", "_callbacks", "_metadata", "_prefix")

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        metadata: dict[str, dict[str, Any]] | None = None,
        prefix: str = "",
    ) -> None:
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_snapshots", [])
        object.__setattr__(self, "_callbacks", [])
        object.__setattr__(self, "_metadata", {} if metadata is None else metadata)
        object.__setattr__(self, "_prefix", prefix)
        if data is not None:
            for key, value in data.items():
                self._data[key] = self._wrap(key, value)

    # -- wrapping / unwrapping --

    def _wrap(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return Config(
                value,
                metadata=self._metadata,
                prefix=self._join_path(key),
            )
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value.to_dict()
        return value

    # -- attribute access --

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_"):
            try:
                data = object.__getattribute__(self, "_data")
            except AttributeError:
                data = None
            if data is not None and name in data:
                return data[name]
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"Config has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if self._frozen:
            raise FrozenConfigError(f"Cannot set {name!r} on a frozen Config")
        old = self._data.get(name)
        self._data[name] = self._wrap(name, value)
        self._set_metadata(name, value)
        self._notify(name, value, old)

    def __delattr__(self, name: str) -> None:
        if self._frozen:
            raise FrozenConfigError(f"Cannot delete {name!r} on a frozen Config")
        try:
            del self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute {name!r}") from None
        self._delete_metadata(name)

    # -- item / path access --

    def __getitem__(self, path: str) -> Any:
        parts = path.split(".")
        current: Any = self
        for part in parts:
            if isinstance(current, Config):
                try:
                    current = current._data[part]
                except KeyError:
                    raise KeyError(path) from None
            else:
                raise KeyError(path)
        return current

    def __setitem__(self, path: str, value: Any) -> None:
        if self._frozen:
            raise FrozenConfigError(f"Cannot set {path!r} on a frozen Config")
        parts = path.split(".")
        current = self
        for part in parts[:-1]:
            nxt = current._data.get(part)
            if not isinstance(nxt, Config):
                nxt = Config(
                    metadata=self._metadata,
                    prefix=current._join_path(part),
                )
                current._data[part] = nxt
            current = nxt
        old = current._data.get(parts[-1])
        current._data[parts[-1]] = current._wrap(parts[-1], value)
        current._set_metadata(parts[-1], value)
        self._notify(path, value, old)

    def __delitem__(self, path: str) -> None:
        if self._frozen:
            raise FrozenConfigError(f"Cannot delete {path!r} on a frozen Config")
        parts = path.split(".")
        current = self
        for part in parts[:-1]:
            nxt = current._data.get(part)
            if not isinstance(nxt, Config):
                raise KeyError(path)
            current = nxt
        try:
            del current._data[parts[-1]]
        except KeyError:
            raise KeyError(path) from None
        current._delete_metadata(parts[-1])

    def __contains__(self, path: str) -> bool:
        try:
            self[path]
            return True
        except KeyError:
            return False

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        return NotImplemented

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __deepcopy__(self, memo: dict[int, Any]):
        cls = type(self)
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        object.__setattr__(copied, "_data", copy.deepcopy(self._data, memo))
        object.__setattr__(copied, "_frozen", self._frozen)
        object.__setattr__(copied, "_snapshots", copy.deepcopy(self._snapshots, memo))
        object.__setattr__(copied, "_callbacks", copy.deepcopy(self._callbacks, memo))
        object.__setattr__(copied, "_metadata", copy.deepcopy(self._metadata, memo))
        object.__setattr__(copied, "_prefix", self._prefix)
        return copied

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self[path]
        except KeyError:
            return default

    # -- freeze / unfreeze --

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)
        for v in self._data.values():
            if isinstance(v, Config):
                v.freeze()

    def unfreeze(self) -> None:
        object.__setattr__(self, "_frozen", False)
        for v in self._data.values():
            if isinstance(v, Config):
                v.unfreeze()

    @property
    def frozen(self) -> bool:
        return self._frozen

    # -- snapshot / rollback --

    def snapshot(self) -> None:
        self._snapshots.append(copy.deepcopy((self._data, self._metadata)))

    def rollback(self) -> None:
        if not self._snapshots:
            raise ConfigError("No snapshots to rollback to")
        data, metadata = self._snapshots.pop()
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_metadata", metadata)

    # -- change notification --

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        self._callbacks.append(callback)

    def _notify(self, key: str, new_value: Any, old_value: Any) -> None:
        for cb in self._callbacks:
            cb(key, new_value, old_value)

    # -- class-level loaders --

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        """Load a JSON file and return a :class:`Config`.

        This is a convenience shortcut — source tracking is not recorded.
        For layered loading with provenance, use
        :class:`~molcfg.ConfigLoader` with a
        :class:`~molcfg.JsonFileSource` instead.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("JSON file must contain a top-level object")
        return cls(data)

    @classmethod
    def load_toml(cls, path: str | Path) -> "Config":
        """Load a TOML file and return a :class:`Config`.

        This is a convenience shortcut — source tracking is not recorded.
        For layered loading with provenance, use
        :class:`~molcfg.ConfigLoader` with a
        :class:`~molcfg.TomlFileSource` instead.
        """
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    # -- serialization --

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in self._data.items():
            result[k] = self._unwrap(v)
        return result

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_toml(self) -> str:
        """Serialize the config to a TOML-formatted string."""
        return _dict_to_toml(self.to_dict())

    def save_json(self, path: str | Path, **kwargs: Any) -> None:
        """Write the config as JSON to *path*."""
        Path(path).write_text(self.to_json(**kwargs), encoding="utf-8")

    def save_toml(self, path: str | Path) -> None:
        """Write the config as TOML to *path*."""
        Path(path).write_text(self.to_toml(), encoding="utf-8")

    def meta(self, path: str = "") -> dict[str, Any] | None:
        absolute_path = self._full_path(path)
        entry = self._metadata.get(absolute_path)
        if entry is None:
            return None
        result = copy.deepcopy(entry)
        history = result.get("history")
        if isinstance(history, list):
            result["history"] = tuple(history)
        return result

    def metadata(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._metadata)

    def _join_path(self, path: str) -> str:
        if not self._prefix:
            return path
        if not path:
            return self._prefix
        return f"{self._prefix}.{path}"

    def _full_path(self, path: str) -> str:
        return self._join_path(path)

    def _set_metadata(self, path: str, value: Any, source: str = "runtime") -> None:
        absolute_path = self._full_path(path)
        self._delete_metadata(path)
        for metadata_path in self._iter_metadata_paths(absolute_path, value):
            entry = self._metadata.setdefault(metadata_path, {"history": []})
            history = entry.setdefault("history", [])
            if not history or history[-1] != source:
                history.append(source)
            entry["source"] = source

    def _delete_metadata(self, path: str) -> None:
        absolute_path = self._full_path(path)
        for metadata_path in list(self._metadata):
            if metadata_path == absolute_path or metadata_path.startswith(f"{absolute_path}."):
                del self._metadata[metadata_path]

    def _iter_metadata_paths(self, path: str, value: Any):
        yield path
        if isinstance(value, Config):
            value = value.to_dict()
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_path = f"{path}.{key}"
                yield from self._iter_metadata_paths(nested_path, nested_value)
