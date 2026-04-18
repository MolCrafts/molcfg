# Release Notes

## 1.3.0

Release date: 2026-04-18

### Added

- **`Registry[T]`** — tag-to-factory container. `registry.build("silu")`
  returns an instance; `registry.get("silu")` returns the registered class
  itself (for APIs that take `type[T]`). Decorator registration via
  `@registry("key")`.
- **Long-form specs with kwargs** — `registry.build({"type": "leaky_relu",
  "negative_slope": 0.1})` constructs `LeakyReLU(negative_slope=0.1)`.
- **`Build(registry)` marker** for `typing.Annotated` — declare
  `activation: Annotated[nn.Module, Build(activations)] = "silu"` in a
  schema and `validate()` resolves the config string into an instance
  before type-checking. Works the same across JSON/TOML/YAML sources, on
  explicit values and defaults alike.
- `docs/registry.md` guide covering short vs. long form, build vs. get,
  and `validate()` integration.

### Changed

- `validate()` now calls `get_type_hints(..., include_extras=True)` so
  `Annotated` metadata is visible to the new `Build` marker. No impact
  on existing schemas.

### Breaking changes

None.

## 1.0.0

Release date: 2026-04-12

First stable release.

### What's included

- `Config` with attribute access, dotted-path access, freeze, snapshot, rollback, and change callbacks
- `ConfigLoader` and `ProfileLoader` with source metadata tracking via `Config.meta()`
- `DictSource`, `JsonFileSource`, `TomlFileSource`, `EnvSource`, `CliSource`
- Automatic scalar and JSON-like coercion for `EnvSource` and `CliSource` (disable with `coerce=False`)
- `merge()` with `DEEP_MERGE`, `OVERRIDE`, and `APPEND` strategies — all paths return isolated copies
- Recursive schema validation with defaults, strict mode, and built-in constraints (`Range`, `Length`, `Pattern`, `OneOf`)
- `ThreadSafeConfig` with shared lock support; `FileLock` for cross-process coordination
- `interpolate()` with `${path.to.key}` and `${env:VAR}` resolution and circular-reference detection

### Breaking changes

None — this is the initial stable release.
