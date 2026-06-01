<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molcfg
</h1>

<p><strong>Layered configuration for Python — predictable loading, merging, validation, and source tracking.</strong></p>

<p>
  <a href="https://github.com/MolCrafts/molcfg/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molcfg/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/molcrafts-molcfg/"><img src="https://img.shields.io/pypi/v/molcrafts-molcfg?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/molcrafts-molcfg/"><img src="https://img.shields.io/pypi/pyversions/molcrafts-molcfg?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff"></a>
</p>

<p>
  <a href="https://github.com/MolCrafts/molcfg/tree/master/docs"><b>Documentation</b></a> &nbsp;&middot;&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;&middot;&nbsp;
  <a href="#molcrafts-ecosystem"><b>Ecosystem</b></a>
</p>

</div>

molcfg loads configuration from dicts, files, environment variables, and CLI
arguments into a single layered `Config`, remembering where every value came
from. It adds schema validation, string-tag construction, interpolation, and
thread-safe access on top.

## Capabilities

| Module | Capability |
|--------|------------|
| `config`      | `Config` container — attribute and dotted-path access, freeze, snapshot/rollback, change notification, JSON/TOML/YAML round-trip |
| `source`      | Config sources — `DictSource`, `JsonFileSource`, `TomlFileSource`, `YamlFileSource`, `EnvSource`, `CliSource` behind a common `Source` interface |
| `merge`       | `OVERRIDE` / `APPEND` / `DEEP_MERGE` strategies, `ConfigLoader` for layered loading, and `ProfileLoader` for profile-based config |
| `validation`  | Type-and-constraint validation — `Range`, `OneOf`, `Pattern`, `Length` descriptors, `Build`, and the `validate` entry point |
| `registry`    | `Registry` — maps string keys (e.g. `"silu"`) to factories, resolving short and long forms into classes or instances |
| `concurrency` | `ThreadSafeConfig` wrapper, POSIX `FileLock`, and the `${path}` / `${env:VAR}` `interpolate` engine |
| `paths`       | `project_config_dir` — resolves and creates the shared `~/.molcrafts/<name>/config/` directory |
| `errors`      | Exception hierarchy — `ConfigError`, `FrozenConfigError`, `CircularReferenceError`, `ValidationError` |

## Install

```bash
pip install molcrafts-molcfg
```

Requires Python 3.12+. The only runtime dependency is `pyyaml`.

## Quick start

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

## Documentation

Full documentation lives in [`docs/`](https://github.com/MolCrafts/molcfg/tree/master/docs):

- [Getting started](https://github.com/MolCrafts/molcfg/blob/master/docs/getting-started.md)
- [Sources](https://github.com/MolCrafts/molcfg/blob/master/docs/sources.md)
- [Validation](https://github.com/MolCrafts/molcfg/blob/master/docs/validation.md)
- [Merge strategies](https://github.com/MolCrafts/molcfg/blob/master/docs/merge.md)
- [Registry](https://github.com/MolCrafts/molcfg/blob/master/docs/registry.md)
- [Concurrency](https://github.com/MolCrafts/molcfg/blob/master/docs/concurrency.md)
- [API reference](https://github.com/MolCrafts/molcfg/blob/master/docs/api.md)

## MolCrafts ecosystem

| Project | Role |
|---------|------|
| [molpy](https://github.com/MolCrafts/molpy)     | Python toolkit — the shared molecular data model & workflow layer |
| [molrs](https://github.com/MolCrafts/molrs)     | Rust core — molecular data structures & compute kernels (native + WASM) |
| [molpack](https://github.com/MolCrafts/molpack) | Packmol-grade molecular packing (Rust + Python) |
| [molvis](https://github.com/MolCrafts/molvis)   | WebGL molecular visualization & editing |
| [molexp](https://github.com/MolCrafts/molexp)   | Workflow & experiment-management platform |
| [molnex](https://github.com/MolCrafts/molnex)   | Molecular machine-learning framework |
| [molq](https://github.com/MolCrafts/molq)       | Unified job queue — local / SLURM / PBS / LSF |
| **molcfg**                                      | Layered configuration library — this repo |
| [mollog](https://github.com/MolCrafts/mollog)   | Structured logging, stdlib-compatible |
| [molhub](https://github.com/MolCrafts/molhub)   | Molecular dataset hub |
| [molmcp](https://github.com/MolCrafts/molmcp)   | MCP server for the ecosystem |
| [molrec](https://github.com/MolCrafts/molrec)   | Atomistic record specification |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause — see [LICENSE](LICENSE).

<hr>

<div align="center">
<sub>Crafted with 💚 by <a href="https://github.com/MolCrafts">MolCrafts</a></sub>
</div>
