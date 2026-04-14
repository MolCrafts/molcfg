"""Configuration sources: load config data from various backends."""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from abc import ABC, abstractmethod
from typing import Any

import yaml

_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(
    r"""
    [+-]?
    (?:
        (?:\d+\.\d*|\.\d+|\d+)
        (?:[eE][+-]?\d+)?
    )
    """,
    re.VERBOSE,
)


class Source(ABC):
    """Abstract base class for configuration sources."""

    @property
    def name(self) -> str:
        return getattr(self, "_name", self.__class__.__name__)

    @abstractmethod
    def load(self) -> dict[str, Any]:
        ...


class DictSource(Source):
    """Source that wraps a plain Python dict."""

    def __init__(self, data: dict[str, Any], name: str | None = None) -> None:
        self._data = data
        self._name = name or self.__class__.__name__

    def load(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


class JsonFileSource(Source):
    """Source that loads configuration from a JSON file."""

    def __init__(self, path: str | os.PathLike[str], name: str | None = None) -> None:
        self._path = path
        self._name = name or self.__class__.__name__

    def load(self) -> dict[str, Any]:
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"JSON root must be an object, got {type(data).__name__}")
        return data


class TomlFileSource(Source):
    """Source that loads configuration from a TOML file (read-only via tomllib)."""

    def __init__(self, path: str | os.PathLike[str], name: str | None = None) -> None:
        self._path = path
        self._name = name or self.__class__.__name__

    def load(self) -> dict[str, Any]:
        with open(self._path, "rb") as f:
            return tomllib.load(f)


class YamlFileSource(Source):
    """Source that loads configuration from a YAML file."""

    def __init__(self, path: str | os.PathLike[str], name: str | None = None) -> None:
        self._path = path
        self._name = name or self.__class__.__name__

    def load(self) -> dict[str, Any]:
        with open(self._path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise TypeError(f"YAML root must be a mapping, got {type(data).__name__}")
        return data


class EnvSource(Source):
    """Source that loads configuration from environment variables.

    Variables matching ``{prefix}{separator}...`` are split by the separator
    and nested into a dict hierarchy.  The prefix itself is stripped.

    Example::

        MYAPP_DB_HOST=localhost  →  {"db": {"host": "localhost"}}
    """

    def __init__(
        self,
        prefix: str = "",
        separator: str = "_",
        environ: dict[str, str] | None = None,
        *,
        coerce: bool = True,
        name: str | None = None,
    ) -> None:
        self._prefix = prefix.upper()
        self._separator = separator
        self._environ = environ if environ is not None else os.environ
        self._coerce = coerce
        self._name = name or self.__class__.__name__

    def load(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        full_prefix = f"{self._prefix}{self._separator}" if self._prefix else ""
        for key, value in self._environ.items():
            if full_prefix and not key.startswith(full_prefix):
                continue
            stripped = key[len(full_prefix):] if full_prefix else key
            parts = [p.lower() for p in stripped.split(self._separator)]
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = _coerce_value(value) if self._coerce else value
        return result


class CliSource(Source):
    """Source that parses CLI-style arguments into a nested dict.

    Supports ``--key=value`` and ``--key value`` forms.
    Dot-separated keys become nested dicts.

    Example::

        ["--db.host=localhost", "--db.port", "5432"]
        →  {"db": {"host": "localhost", "port": "5432"}}
    """

    def __init__(
        self,
        args: list[str],
        *,
        coerce: bool = True,
        name: str | None = None,
    ) -> None:
        self._args = args
        self._coerce = coerce
        self._name = name or self.__class__.__name__

    def load(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        i = 0
        while i < len(self._args):
            arg = self._args[i]
            if not arg.startswith("--"):
                i += 1
                continue
            arg = arg[2:]  # strip --
            if "=" in arg:
                key, value = arg.split("=", 1)
            else:
                key = arg
                next_index = i + 1
                if next_index < len(self._args) and not self._args[next_index].startswith("--"):
                    i = next_index
                    value = self._args[i]
                else:
                    value = ""
            parsed = _coerce_value(value) if self._coerce else value
            self._set_nested(result, key, parsed)
            i += 1
        return result

    @staticmethod
    def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        current = d
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


def _coerce_value(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if _INT_RE.fullmatch(stripped):
        return int(stripped)
    if _FLOAT_RE.fullmatch(stripped) and any(ch in stripped.lower() for ch in ".e"):
        return float(stripped)
    if stripped.startswith("[") or stripped.startswith("{") or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value
