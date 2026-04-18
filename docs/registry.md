# Registry

`Registry` turns config strings into Python objects. It solves the common
"tagged factory" problem: a config says `activation = "silu"`, and the code
wants an `nn.SiLU()` instance.

## Quick start

```python
import torch.nn as nn
from molcfg import Registry

activations = Registry[nn.Module]("activation")
activations.register("silu", nn.SiLU)
activations.register("gelu", nn.GELU)
activations.register("leaky_relu", nn.LeakyReLU)

activations.build("silu")
# -> nn.SiLU()

activations.build({"type": "leaky_relu", "negative_slope": 0.1})
# -> nn.LeakyReLU(negative_slope=0.1)

activations.build(None)
# -> None
```

Two equivalent TOML forms:

```toml
# short
activation = "gelu"

# long (with kwargs)
[activation]
type = "leaky_relu"
negative_slope = 0.1
```

## Registration

Three equivalent forms:

```python
# direct
activations.register("silu", nn.SiLU)

# decorator on register
@activations.register("my_act")
class MyAct(nn.Module): ...

# decorator via __call__
@activations("my_act2")
class MyAct2(nn.Module): ...
```

Keys are lowercased; duplicate registration raises `ValueError`.

## Building vs. looking up

| Call | Returns | Use when |
|---|---|---|
| `registry.build(spec)` | an **instance** | the config value should become the object you use |
| `registry.get(name)` | the **factory/class** | the API you pass to wants `type[T]` and constructs later (e.g. `Allegro(latent_activation=nn.SiLU)`) |

```python
activations.get("silu")    # -> nn.SiLU  (the class)
activations.get(None)      # -> None
activations.build("silu")  # -> nn.SiLU()  (an instance)
```

## Building

`registry.build(spec)` accepts:

| Input | Behavior |
|---|---|
| `None` | returns `None` (so callers can keep `Optional` fields) |
| `str` | `factory()` with no kwargs |
| `dict` with `type` key | `factory(**{k: v for k, v in dict.items() if k != "type"})` |
| anything else | returned unchanged (idempotent) |

The discriminator key defaults to `"type"` and can be changed:

```python
optimizers = Registry("optimizer", discriminator="name")
optimizers.register("adam", Adam)
optimizers.build({"name": "adam", "lr": 1e-3})
```

Unknown key — the error lists the candidates:

```
ValueError: registry='activation': 'silux' not in ['gelu', 'leaky_relu', 'silu']
```

## Integration with `validate()`

Declare the field with `Annotated[T, Build(registry)]`. During validation
the raw value is passed through `registry.build(...)` before type-checking.

```python
from typing import Annotated
from molcfg import Build, Registry, validate
import torch.nn as nn

activations = Registry[nn.Module]("activation")
activations.register("silu", nn.SiLU)
activations.register("gelu", nn.GELU)
activations.register("leaky_relu", nn.LeakyReLU)

class ModelSchema:
    hidden_dim: int = 128
    activation: Annotated[nn.Module, Build(activations)] = "silu"
    gate: Annotated[nn.Module | None, Build(activations)] = None

result = validate(
    {"hidden_dim": 256, "activation": {"type": "leaky_relu", "negative_slope": 0.1}},
    ModelSchema,
    apply_defaults=True,
)
assert isinstance(result["activation"], nn.LeakyReLU)
assert result["gate"] is None
```

Defaults are resolved too: `activation = "silu"` as a default becomes an
`nn.SiLU()` instance in the validated output.

## Standalone use

`Registry` has no dependency on `Config` or `validate`. Use it directly
from any loader:

```python
cfg = Config.load_toml("config.toml")
model = build_model(
    hidden_dim=cfg["model.hidden_dim"],
    activation=activations.build(cfg["model.activation"]),
)
```
