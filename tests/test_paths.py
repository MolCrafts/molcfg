"""Tests for project_config_dir (resolves and creates ~/.molcrafts/<name>/config/)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import molcfg
from molcfg import project_config_dir


def test_default_base_creates_config_dir(tmp_path: Path) -> None:
    """ac-001: default base resolves to ~/.molcrafts/<name>/config/ and is created."""
    result = project_config_dir("molq", environ={"HOME": str(tmp_path)})

    assert result == tmp_path / ".molcrafts" / "molq" / "config"
    assert result.is_dir()


def test_molcrafts_home_overrides_base(tmp_path: Path) -> None:
    """ac-002: MOLCRAFTS_HOME overrides ~/.molcrafts as the base directory."""
    sandbox = tmp_path / "sandbox"
    result = project_config_dir(
        "molq",
        environ={"HOME": str(tmp_path), "MOLCRAFTS_HOME": str(sandbox)},
    )

    assert result == sandbox / "molq" / "config"
    assert result.is_dir()


def test_repeated_calls_are_idempotent(tmp_path: Path) -> None:
    """ac-003: calling twice returns the same path and does not raise."""
    first = project_config_dir("molq", environ={"HOME": str(tmp_path)})
    second = project_config_dir("molq", environ={"HOME": str(tmp_path)})

    assert first == second
    assert second.is_dir()


@pytest.mark.parametrize("override", ["", "   ", "\t\n"])
def test_empty_or_whitespace_molcrafts_home_falls_back(tmp_path: Path, override: str) -> None:
    """ac-004: empty / whitespace MOLCRAFTS_HOME falls back to ~/.molcrafts."""
    result = project_config_dir(
        "molq",
        environ={"HOME": str(tmp_path), "MOLCRAFTS_HOME": override},
    )

    assert result == tmp_path / ".molcrafts" / "molq" / "config"
    assert result.is_dir()


def test_molcrafts_home_with_tilde_is_expanded(tmp_path: Path) -> None:
    """MOLCRAFTS_HOME=~/custom expands relative to HOME."""
    result = project_config_dir(
        "molq",
        environ={"HOME": str(tmp_path), "MOLCRAFTS_HOME": "~/custom"},
    )

    assert result == tmp_path / "custom" / "molq" / "config"
    assert result.is_dir()


@pytest.mark.parametrize(
    "bad_name",
    ["", ".", "..", "a/b", "a" + os.sep + "b", "with\\sep"],
)
def test_invalid_name_raises_value_error(tmp_path: Path, bad_name: str) -> None:
    """ac-005: invalid name raises ValueError without creating directories."""
    with pytest.raises(ValueError):
        project_config_dir(bad_name, environ={"HOME": str(tmp_path)})

    assert not (tmp_path / ".molcrafts").exists()


def test_default_environ_falls_back_to_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ac-007: omitting environ kwarg uses os.environ for MOLCRAFTS_HOME."""
    monkeypatch.setenv("MOLCRAFTS_HOME", str(tmp_path))
    result = project_config_dir("molq")

    assert result == tmp_path / "molq" / "config"
    assert result.is_dir()


def test_re_exported_from_package() -> None:
    """ac-006: project_config_dir is importable from molcfg and listed in __all__."""
    assert "project_config_dir" in molcfg.__all__
    assert molcfg.project_config_dir is project_config_dir
