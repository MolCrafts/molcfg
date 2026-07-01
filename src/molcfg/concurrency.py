"""Thread-safe config wrapper, file locking, and interpolation engine."""

from __future__ import annotations

import os
import re
import sys
import threading
from collections.abc import Callable
from typing import Any

from molcfg.config import Config
from molcfg.errors import CircularReferenceError

# File locking is platform-specific: fcntl is POSIX-only, Windows uses msvcrt.
# Pick the primitive at import so a missing module surfaces at load, not mid-lock.
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# -- Thread-safe wrapper --


class ThreadSafeConfig:
    """Thread-safe wrapper around a ``Config`` instance.

    All reads and writes are serialized through an ``RLock``.
    """

    __slots__ = ("_config", "_lock")

    def __init__(
        self,
        config: Config,
        lock: threading.RLock | None = None,
    ) -> None:
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_lock", lock if lock is not None else threading.RLock())

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, Config):
            return ThreadSafeConfig(value, self._lock)
        return value

    def __getattr__(self, name: str) -> Any:
        with self._lock:
            return self._wrap(getattr(self._config, name))

    def __setattr__(self, name: str, value: Any) -> None:
        with self._lock:
            setattr(self._config, name, value)

    def __delattr__(self, name: str) -> None:
        with self._lock:
            delattr(self._config, name)

    def __getitem__(self, path: str) -> Any:
        with self._lock:
            return self._wrap(self._config[path])

    def __setitem__(self, path: str, value: Any) -> None:
        with self._lock:
            self._config[path] = value

    def __delitem__(self, path: str) -> None:
        with self._lock:
            del self._config[path]

    def __contains__(self, path: str) -> bool:
        with self._lock:
            return path in self._config

    def __len__(self) -> int:
        with self._lock:
            return len(self._config)

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._config))

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._config.keys())

    def values(self) -> tuple[Any, ...]:
        with self._lock:
            return tuple(self._wrap(value) for value in self._config.values())

    def items(self) -> tuple[tuple[str, Any], ...]:
        with self._lock:
            return tuple((key, self._wrap(value)) for key, value in self._config.items())

    def get(self, path: str, default: Any = None) -> Any:
        with self._lock:
            if path in self._config:
                return self._wrap(self._config[path])
            return default

    def freeze(self) -> None:
        with self._lock:
            self._config.freeze()

    def unfreeze(self) -> None:
        with self._lock:
            self._config.unfreeze()

    def snapshot(self) -> None:
        with self._lock:
            self._config.snapshot()

    def rollback(self) -> None:
        with self._lock:
            self._config.rollback()

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        with self._lock:
            self._config.on_change(callback)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return self._config.to_dict()

    def to_json(self, **kwargs: Any) -> str:
        with self._lock:
            return self._config.to_json(**kwargs)

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._config.frozen


# -- File lock --


class FileLock:
    """Cross-platform exclusive file lock.

    Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows.

    Usage::

        with FileLock("/tmp/myapp.lock"):
            # exclusive access
            ...
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR)
        if sys.platform == "win32":
            # msvcrt locks a byte range from the current position; lock the first
            # byte. LK_LOCK blocks (retrying ~10s) then raises OSError.
            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(self._fd, fcntl.LOCK_EX)

    def release(self) -> None:
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# -- Interpolation engine --

_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def interpolate(
    data: dict[str, Any],
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve ``${...}`` placeholders in string values.

    Supports:
    - ``${path.to.key}`` — reference another config key
    - ``${env:VAR_NAME}`` — reference an environment variable

    Raises ``CircularReferenceError`` on circular references.
    """
    if environ is None:
        environ = dict(os.environ)
    resolved: dict[str, str] = {}
    return _interpolate_dict(data, data, environ, resolved, set())


def _interpolate_dict(
    node: dict[str, Any],
    root: dict[str, Any],
    environ: dict[str, str],
    resolved: dict[str, str],
    resolving: set[str],
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in node.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result[key] = _interpolate_dict(value, root, environ, resolved, resolving, full_key)
        elif isinstance(value, str) and "${" in value:
            result[key] = _resolve_string(value, full_key, root, environ, resolved, resolving)
        else:
            result[key] = value
    return result


def _resolve_string(
    template: str,
    current_key: str,
    root: dict[str, Any],
    environ: dict[str, str],
    resolved: dict[str, str],
    resolving: set[str],
) -> str:
    if current_key in resolved:
        return resolved[current_key]
    if current_key in resolving:
        raise CircularReferenceError(f"Circular reference detected at {current_key!r}")
    resolving.add(current_key)

    def replacer(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ref.startswith("env:"):
            var_name = ref[4:]
            return environ.get(var_name, "")
        # Config self-reference
        ref_value = _get_nested(root, ref)
        if ref_value is None:
            return match.group(0)  # leave unresolved
        if isinstance(ref_value, str) and "${" in ref_value:
            return _resolve_string(ref_value, ref, root, environ, resolved, resolving)
        return str(ref_value)

    result = _INTERP_RE.sub(replacer, template)
    resolving.discard(current_key)
    resolved[current_key] = result
    return result


def _get_nested(data: dict[str, Any], dotted_path: str) -> Any | None:
    parts = dotted_path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return None
        else:
            return None
    return current
