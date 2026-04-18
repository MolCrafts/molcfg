# molcfg

Python configuration library for services, CLIs, and internal tooling that need predictable loading, merging, and validation.

[Get started](getting-started.md){ .md-button .md-button--primary }
[API reference](api.md){ .md-button }

## What it gives you

- Layered loading from dicts, JSON/TOML/YAML files, environment variables, and CLI arguments
- Deep merge, override, and append strategies — all return isolated copies
- Recursive schema validation with defaults, strict unknown-field checks, and built-in constraints
- `Registry` + `Build` for resolving config strings like `"silu"` into Python classes or instances
- Source tracking via `Config.meta()` so every value can be explained
- Attribute and dotted-path access, freeze, snapshot, and rollback on the same `Config` object
- Thread-safe wrapper and POSIX file lock for concurrent access patterns
- `${path.to.key}` and `${env:VAR}` interpolation with circular-reference detection

## Quick example

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

## Documentation map

- [Getting Started](getting-started.md) — installation, first config, layered loading
- [Sources](sources.md) — file, environment, and CLI source details
- [Validation](validation.md) — schemas, defaults, strict mode, constraints
- [Merge](merge.md) — merge strategies, `ConfigLoader`, `ProfileLoader`
- [Registry](registry.md) — string tags → classes/instances for config-driven factories
- [Concurrency](concurrency.md) — thread-safe wrapper, file locks, interpolation
- [API Reference](api.md) — complete exported surface
