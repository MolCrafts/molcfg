"""String-tag → factory registry for config-driven instance construction.

A ``Registry`` maps short string keys (e.g. ``"silu"``) to factories
(classes or callables) that produce instances.  It supports both a short
form (``"silu"`` → zero-arg build) and a long form
(``{"type": "leaky_relu", "negative_slope": 0.1}`` → kwargs build).

The registry is deliberately self-contained: it does not know about
``Config`` or ``validate`` and can be used standalone.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")

Factory = type | Callable[..., Any]


def _normalize_key(key: str) -> str:
    return str(key).lower()


class Registry(Generic[T]):
    """Maps string keys to factories that produce instances of ``T``.

    Parameters
    ----------
    name:
        Human-readable label, included in error messages.
    discriminator:
        Key used to pick a factory when the spec is a dict.  Defaults to
        ``"type"``.
    """

    def __init__(self, name: str, *, discriminator: str = "type") -> None:
        self.name = name
        self.discriminator = discriminator
        self._factories: dict[str, Factory] = {}

    def register(self, key: str, factory: Factory | None = None) -> Any:
        """Register ``factory`` under ``key``.

        Can be used as a direct call ``reg.register("silu", nn.SiLU)`` or as
        a decorator ``@reg.register("silu")``.
        """
        if factory is None:

            def deco(f: Factory) -> Factory:
                self._register(key, f)
                return f

            return deco
        self._register(key, factory)
        return factory

    def __call__(self, key: str) -> Callable[[Factory], Factory]:
        """Decorator form: ``@reg("silu")``."""

        def deco(f: Factory) -> Factory:
            self._register(key, f)
            return f

        return deco

    def build(self, spec: Any) -> T | None:
        """Resolve ``spec`` into an instance.

        Accepted forms:

        - ``None`` → ``None`` (caller's Optional handling)
        - ``str`` → ``factory()`` with no kwargs
        - ``dict`` with discriminator key → ``factory(**kwargs)``
        - any other value → returned unchanged (idempotent; assumes already
          an instance)
        """
        if spec is None:
            return None
        if isinstance(spec, str):
            return self._instantiate(spec, {})
        if isinstance(spec, dict):
            if self.discriminator not in spec:
                raise ValueError(
                    f"registry={self.name!r}: dict spec missing "
                    f"{self.discriminator!r} key; got keys {sorted(spec)}"
                )
            kwargs = {k: v for k, v in spec.items() if k != self.discriminator}
            return self._instantiate(spec[self.discriminator], kwargs)
        return spec

    def get(self, key: str | None) -> Factory | None:
        """Return the registered factory for ``key`` without instantiating.

        For the common "config says ``'silu'``, API wants the class
        ``nn.SiLU`` to construct later" case.  ``None`` passes through so
        optional fields (``latent_activation = null``) stay null.
        """
        if key is None:
            return None
        try:
            return self._factories[_normalize_key(key)]
        except KeyError as exc:
            raise ValueError(f"registry={self.name!r}: {key!r} not in {self.keys()}") from exc

    def keys(self) -> list[str]:
        """Sorted list of registered keys."""
        return sorted(self._factories)

    def __contains__(self, key: str) -> bool:
        return _normalize_key(key) in self._factories

    def _register(self, key: str, factory: Factory) -> None:
        normalized = _normalize_key(key)
        if normalized in self._factories:
            raise ValueError(f"registry={self.name!r}: key {normalized!r} already registered")
        self._factories[normalized] = factory

    def _instantiate(self, key: str, kwargs: dict[str, Any]) -> T:
        try:
            factory = self._factories[_normalize_key(key)]
        except KeyError as exc:
            raise ValueError(f"registry={self.name!r}: {key!r} not in {self.keys()}") from exc
        return factory(**kwargs)
