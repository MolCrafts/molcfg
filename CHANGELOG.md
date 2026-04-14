# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-04-13

### Added

- `YamlFileSource` for loading configuration from YAML files.

### Changed

- PyYAML is now a required runtime dependency (previously molcfg had no runtime dependencies).
- Dropped the "zero-dependency" framing from the package description and docs.

## [1.0.0] - 2026-04-11

### Added

- Zensical-based documentation site under `docs/` with usage, validation, source loading, concurrency, and release pages.
- `Config.meta()` and loader-level source tracking for merged configuration values.
- Automatic type coercion for `EnvSource` and `CliSource`, with `coerce=False` escape hatch.
- Recursive schema validation for nested schemas and `list[Schema]`.
- `validate(..., allow_extra=False, apply_defaults=True)` for stricter and more useful validation flows.
- Release engineering files: `README.md`, `CONTRIBUTING.md`, `RELEASING.md`, `LICENSE`, `zensical.toml`.

### Changed

- Promoted package version from `0.1.0` to `1.0.0`.
- Changed the PyPI distribution name to `molcrafts-molcfg` while keeping the Python import path as `molcfg`.
- Expanded project metadata in `pyproject.toml` for packaging, documentation, and release publishing.
- Strengthened `ThreadSafeConfig` so nested reads stay inside the same lock boundary.
- Fixed merge and source-loading aliasing so callers receive isolated nested data.

### Fixed

- `CliSource` no longer consumes the next CLI flag as a value for a flag without an explicit value.
- `Config` attribute access now correctly returns config values even when keys collide with method names like `items` or `keys`.
- Snapshot and rollback semantics now preserve metadata alongside configuration data.

## [0.1.0] - 2026-02-07

### Added

- Initial release of `molcfg` with config container, sources, merge strategies, validation helpers, interpolation, and concurrency utilities.
