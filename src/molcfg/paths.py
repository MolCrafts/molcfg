"""Standard project paths for the molcrafts ecosystem.

Provides :func:`project_config_dir`, which resolves and idempotently creates
``~/.molcrafts/<name>/config/`` so downstream tools (e.g. ``molq`` writing a
SQLite database) share a stable, user-level configuration directory.

The module is deliberately self-contained: it has no internal molcfg
dependencies and uses only :mod:`pathlib` and :mod:`os` from the standard
library.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

_DEFAULT_BASE = "~/.molcrafts"
_HOME_ENV_VAR = "MOLCRAFTS_HOME"
_CONFIG_SUBDIR = "config"
_INVALID_NAME_CHARS = ("/", os.sep, "\\")
_RESERVED_NAMES = frozenset({"", ".", ".."})


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise ValueError(f"name must be a str, got {type(name).__name__}")
    if name in _RESERVED_NAMES:
        raise ValueError(f"name {name!r} is reserved or empty")
    if any(sep in name for sep in _INVALID_NAME_CHARS):
        raise ValueError(f"name {name!r} must be a single path segment (no '/', '\\', or os.sep)")


def _expand(path: str, environ: Mapping[str, str]) -> Path:
    """Expand a leading ``~`` against the injected mapping's ``HOME``.

    Mirrors ``Path.expanduser`` but reads from *environ* so a caller that
    injects an environment is fully isolated from :data:`os.environ`.
    """
    if path == "~" or path.startswith("~/"):
        home = environ.get("HOME")
        if home:
            tail = path[1:].lstrip("/")
            return Path(home) / tail if tail else Path(home)
    return Path(path)


def _resolve_base(environ: Mapping[str, str]) -> Path:
    raw = environ.get(_HOME_ENV_VAR)
    if raw is not None and raw.strip():
        return _expand(raw, environ)
    home = environ.get("HOME")
    if home is not None:
        return Path(home) / ".molcrafts"
    return Path(_DEFAULT_BASE).expanduser()


def project_config_dir(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return ``~/.molcrafts/<name>/config/``, creating it if missing.

    Args:
        name: Project identifier (e.g. ``"molq"``). Must be a single,
            non-empty path segment without ``/``, ``\\``, or :data:`os.sep`,
            and not equal to ``.`` or ``..``.
        environ: Optional environment mapping. Defaults to :data:`os.environ`.
            If ``MOLCRAFTS_HOME`` is set to a non-empty value (after
            ``str.strip``), it overrides ``~/.molcrafts`` as the base.
            Empty or whitespace-only values fall back to the default base.

    Returns:
        The resolved absolute path to ``<base>/<name>/config/``. The
        directory (and any missing parents) is created with
        ``mkdir(parents=True, exist_ok=True)``.

    Raises:
        ValueError: If ``name`` is empty, ``.``, ``..``, or contains a
            path separator. No filesystem changes are made in that case.

    Example:
        >>> path = project_config_dir("molq")  # doctest: +SKIP
        >>> path.is_dir()                       # doctest: +SKIP
        True
    """
    _validate_name(name)
    env = environ if environ is not None else os.environ
    base = _resolve_base(env)
    path = base / name / _CONFIG_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path
