# API Reference

## Config

```python
class Config(data: dict | None = None, *, metadata=None, prefix="")
```

Nested configuration container. Nested dicts are automatically wrapped into child `Config` instances.

### Access

| Operation | Example |
|-----------|---------|
| Attribute read | `cfg.db.host` |
| Dotted-path read | `cfg["db.host"]` |
| Attribute write | `cfg.db.host = "x"` |
| Dotted-path write | `cfg["db.host"] = "x"` |
| Containment | `"db.host" in cfg` |
| Delete | `del cfg.db.host` |

### Methods

- `get(path, default=None)` — safe dotted-path read
- `keys()`, `values()`, `items()` — top-level iteration
- `to_dict()` — recursive plain-dict export
- `to_json(**kwargs)` — JSON string; passes kwargs to `json.dumps`
- `freeze()` — recursively freeze; raises `FrozenConfigError` on write
- `unfreeze()` — recursively unfreeze
- `snapshot()` — push current state onto the snapshot stack
- `rollback()` — pop and restore the last snapshot
- `on_change(callback)` — register `callback(path, new_value, old_value)`
- `meta(path)` — return `{"source": str, "history": tuple}` for a dotted path
- `metadata()` — return the full metadata dict

### Properties

- `frozen` — `True` if the config is currently frozen

---

## Sources

All sources inherit from `Source` and expose a single method:

- `load() -> dict[str, Any]`

### DictSource

```python
DictSource(data: dict, *, name: str = "")
```

Wraps an in-memory dict.

### JsonFileSource

```python
JsonFileSource(path: str | Path, *, name: str = "")
```

### TomlFileSource

```python
TomlFileSource(path: str | Path, *, name: str = "")
```

### EnvSource

```python
EnvSource(prefix: str = "", *, coerce: bool = True, environ: dict | None = None, name: str = "")
```

Reads environment variables and maps them to nested keys by splitting on `_`. `prefix` is stripped and not included in the output key.

### CliSource

```python
CliSource(args: list[str], *, coerce: bool = True, name: str = "")
```

Parses `--key=value` and `--key value` arguments. Dotted keys map to nested dicts.

---

## Merge

### merge()

```python
merge(base: dict, override: dict, strategy: MergeStrategy = DEEP_MERGE) -> dict
```

Returns a new dict. Neither input is mutated.

### MergeStrategy

```python
class MergeStrategy(enum.Enum):
    DEEP_MERGE = "deep_merge"
    OVERRIDE   = "override"
    APPEND     = "append"
```

### ConfigLoader

```python
ConfigLoader(sources: list[Source], strategy: MergeStrategy = DEEP_MERGE)
```

- `load() -> Config` — merge all sources in order, attach metadata

### ProfileLoader

```python
ProfileLoader(base_sources: list[Source], profiles: dict[str, Source], strategy: MergeStrategy = DEEP_MERGE)
```

- `load(profile: str | None = None) -> Config` — load base sources and optionally apply a named profile overlay

---

## Validation

### validate()

```python
validate(
    data: dict,
    schema: type,
    *,
    apply_defaults: bool = False,
    allow_extra: bool = True,
) -> dict
```

Raises `ValidationError` on type mismatch, missing required fields, or constraint failure.

### Constraints

```python
Range(min: float | None = None, max: float | None = None)
Length(min: int | None = None, max: int | None = None)
Pattern(pattern: str)
OneOf(*choices)
```

Attach via `__constraints__ = {"field": [constraint, ...]}` on the schema class.

### Build

```python
Build(registry: Registry)
```

`typing.Annotated` metadata marker. When `validate()` encounters
`field: Annotated[T, Build(reg)]`, the raw value is passed through
`reg.build(...)` before the field's type is checked. Works on default
values too.

---

## Registry

```python
Registry(name: str, *, discriminator: str = "type")
```

String-tag → factory container. Generic: `Registry[T]("name")`.

### Registration

```python
reg.register("silu", nn.SiLU)     # direct
@reg("leaky_relu")                # decorator
class MyLeakyReLU(nn.Module): ...
```

Keys are lowercased. Duplicate registration raises `ValueError`.

### Access

- `reg.build(spec) -> T | None` — resolve into an instance. Accepts
  `str` (short form), `dict` with a `type` key + kwargs (long form),
  `None` (pass-through), or an existing instance (idempotent).
- `reg.get(name) -> type[T] | callable | None` — return the registered
  factory/class without instantiating. `None` pass-through. For APIs
  that take `type[T]` and construct later.
- `reg.keys() -> list[str]` — sorted list of registered keys.
- `"key" in reg` — containment check (case-insensitive).

See the [Registry guide](registry.md) for the full tutorial.

---

## Concurrency

### ThreadSafeConfig

```python
ThreadSafeConfig(config: Config, lock: threading.RLock | None = None)
```

Wraps all reads and writes in the provided lock (or a new `RLock`). Exposes the same interface as `Config`.

### FileLock

```python
FileLock(path: str | Path)
```

Context manager. Acquires `fcntl.LOCK_EX` on entry, releases on exit.

- `acquire()` / `release()` for manual management

### interpolate()

```python
interpolate(data: dict, environ: dict | None = None) -> dict
```

Resolves `${path.to.key}` and `${env:VAR}` placeholders. Raises `CircularReferenceError` on circular references.

---

## Errors

All errors inherit from `ConfigError`.

| Exception | Raised when |
|-----------|-------------|
| `ConfigError` | Base class |
| `FrozenConfigError` | Writing to a frozen `Config` |
| `ValidationError` | Schema validation fails |
| `CircularReferenceError` | Interpolation detects a cycle |
