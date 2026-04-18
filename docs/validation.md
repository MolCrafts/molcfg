# Validation

`validate()` checks a plain dict against a class-based schema and returns a validated copy.

## Basic usage

```python
from molcfg import validate

class DbSchema:
    host: str
    port: int

result = validate({"host": "localhost", "port": 5432}, DbSchema)
assert result == {"host": "localhost", "port": 5432}
```

Annotations drive type checking. `validate()` raises `ValidationError` on mismatch.

## Defaults

Annotate fields with default values and pass `apply_defaults=True`:

```python
class DbSchema:
    host: str = "localhost"
    port: int = 5432

result = validate({}, DbSchema, apply_defaults=True)
assert result == {"host": "localhost", "port": 5432}
```

## Strict mode

Reject keys not declared in the schema:

```python
validate(
    {"host": "localhost", "unknown_key": True},
    DbSchema,
    allow_extra=False,
)
# raises ValidationError: unexpected field 'unknown_key'
```

## Nested schemas

Annotate a field with another class to validate recursively:

```python
class DbSchema:
    host: str
    port: int = 5432

class AppSchema:
    db: DbSchema
    debug: bool = False

result = validate({"db": {"host": "localhost"}}, AppSchema, apply_defaults=True)
assert result == {"db": {"host": "localhost", "port": 5432}, "debug": False}
```

`list[Schema]` and `dict[K, V]` are also supported.

## Optional fields

Use `field: Type | None = None`:

```python
class Schema:
    name: str
    alias: str | None = None
```

## Supported types

- Primitives: `str`, `int`, `float`, `bool`, `None`
- `Literal["a", "b"]`
- `list[T]`, `dict[K, V]`
- Nested class schemas
- `T | None` for optional fields

## Constraints

Attach constraints to schema fields via `__constraints__`:

```python
from molcfg import Length, OneOf, Pattern, Range

class ServerSchema:
    host: str
    port: int
    env: str
    token: str
    __constraints__ = {
        "port": [Range(1, 65535)],
        "env": [OneOf("dev", "staging", "prod")],
        "token": [Length(min=32)],
        "host": [Pattern(r"^[\w\.\-]+$")],
    }
```

`validate()` checks constraints after type validation. `ValidationError` includes the field name and constraint that failed.

## Resolving string tags into objects

For fields whose config value is a string tag (e.g. `activation = "silu"`)
but whose Python value should be a class or instance, use `Build` with a
`Registry`. See the [Registry guide](registry.md) for details.

```python
from typing import Annotated
from molcfg import Build, Registry, validate

activations = Registry("activation")
activations.register("silu", SiLU)

class Schema:
    activation: Annotated[SiLU, Build(activations)] = "silu"

validate({"activation": "silu"}, Schema, apply_defaults=True)
# {"activation": <SiLU instance>}
```
