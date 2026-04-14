# molcfg

[![CI](https://github.com/MolCrafts/molcfg/actions/workflows/ci.yml/badge.svg)](https://github.com/MolCrafts/molcfg/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/molcrafts-molcfg.svg)](https://pypi.org/project/molcrafts-molcfg/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-16A34A.svg)](./LICENSE)

Configuration library for Python — layered sources, profiles, validation, and source tracking.

## Quick Start

```python
from molcfg import CliSource, ConfigLoader, DictSource, EnvSource

cfg = ConfigLoader([
    DictSource({"db": {"host": "localhost", "port": 5432}}, name="defaults"),
    EnvSource(prefix="APP", name="env"),
    CliSource(["--db.port=6432"], name="cli"),
]).load()

assert cfg["db.port"] == 6432
assert cfg.meta("db.port") == {"source": "cli", "history": ("defaults", "cli")}
```

```bash
pip install molcrafts-molcfg
```

## Features

- Layered loading from dicts, TOML/JSON files, environment variables, and CLI arguments
- `DEEP_MERGE`, `OVERRIDE`, and `APPEND` strategies — all return isolated copies
- Recursive schema validation with defaults, strict mode, and built-in constraints
- Source tracking via `Config.meta()` for every value
- Attribute and dotted-path access, freeze, snapshot, and rollback
- Thread-safe wrapper and POSIX file lock
- `${path.to.key}` and `${env:VAR}` interpolation
- No runtime dependencies

---

Built with love by [MolCrafts](https://github.com/MolCrafts)
