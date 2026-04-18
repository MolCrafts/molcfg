"""Type and constraint validation for config values."""

from __future__ import annotations

import copy
import re
import types
import typing
from typing import Annotated, Any, Literal, get_type_hints

from molcfg.errors import ValidationError
from molcfg.registry import Registry

# -- Constraint descriptors --


class Range:
    """Constrain a numeric value to [min_val, max_val]."""

    def __init__(self, min_val: int | float, max_val: int | float) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def check(self, value: Any, path: str) -> str | None:
        if not isinstance(value, (int, float)):
            return f"{path}: expected numeric, got {type(value).__name__}"
        if value < self.min_val or value > self.max_val:
            return f"{path}: {value} not in range [{self.min_val}, {self.max_val}]"
        return None


class OneOf:
    """Constrain a value to a fixed set of options."""

    def __init__(self, *values: Any) -> None:
        self.values = values

    def check(self, value: Any, path: str) -> str | None:
        if value not in self.values:
            return f"{path}: {value!r} not one of {self.values!r}"
        return None


class Pattern:
    """Constrain a string value to match a regex pattern."""

    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern)

    def check(self, value: Any, path: str) -> str | None:
        if not isinstance(value, str):
            return f"{path}: expected str for pattern check, got {type(value).__name__}"
        if not self.pattern.search(value):
            return f"{path}: {value!r} does not match pattern {self.pattern.pattern!r}"
        return None


class Length:
    """Constrain the length of a sized value."""

    def __init__(self, min_len: int = 0, max_len: int | None = None) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def check(self, value: Any, path: str) -> str | None:
        try:
            length = len(value)
        except TypeError:
            return f"{path}: value has no length"
        if length < self.min_len:
            return f"{path}: length {length} < minimum {self.min_len}"
        if self.max_len is not None and length > self.max_len:
            return f"{path}: length {length} > maximum {self.max_len}"
        return None


type Constraint = Range | OneOf | Pattern | Length


# -- Build marker (registry-driven deserialization) --


class Build:
    """``Annotated`` metadata that resolves a field through a ``Registry``.

    Example:
        ``activation: Annotated[nn.Module, Build(activations)] = "silu"``

    When ``validate()`` encounters this field, the raw value (string, dict,
    or already-built instance) is passed through ``registry.build(...)``
    before the usual type check runs.
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry


# -- Type checking engine --


def _strip_annotated(tp: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return ``(base, metadata)`` for ``Annotated[X, *meta]``; pass-through otherwise."""
    if typing.get_origin(tp) is Annotated:
        args = typing.get_args(tp)
        return args[0], args[1:]
    return tp, ()


def _type_matches(value: Any, expected: Any) -> bool:
    """Check if *value* matches a non-schema type annotation *expected*."""
    origin = typing.get_origin(expected)

    # Plain types (int, str, etc.)
    if origin is None:
        if expected is type(None):
            return value is None
        if isinstance(expected, type):
            return isinstance(value, expected)
        return True

    # X | Y  (UnionType from PEP 604)
    if origin is types.UnionType:
        return any(_type_matches(value, arg) for arg in typing.get_args(expected))

    # typing.Union (legacy, but get_type_hints may still produce it)
    if origin is typing.Union:
        return any(_type_matches(value, arg) for arg in typing.get_args(expected))

    # Literal
    if origin is Literal:
        return value in typing.get_args(expected)

    # list[X]
    if origin is list:
        if not isinstance(value, list):
            return False
        args = typing.get_args(expected)
        if args:
            return all(_type_matches(item, args[0]) for item in value)
        return True

    # dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            return False
        args = typing.get_args(expected)
        if args and len(args) == 2:
            return all(
                _type_matches(k, args[0]) and _type_matches(v, args[1]) for k, v in value.items()
            )
        return True

    return isinstance(value, origin) if isinstance(origin, type) else True


def _validate_value(
    value: Any,
    expected: Any,
    path: str,
    *,
    allow_extra: bool,
    apply_defaults: bool,
) -> tuple[Any, list[str]]:
    expected, metadata = _strip_annotated(expected)
    build_marker = next((m for m in metadata if isinstance(m, Build)), None)
    if build_marker is not None:
        try:
            value = build_marker.registry.build(value)
        except ValueError as exc:
            return value, [f"{path}: {exc}"]
        # A Build marker yields an instance (or None), not a dict — skip the
        # nested-schema recursion path and type-check directly.
        if value is None and _type_is_optional(expected):
            return value, []
        if _type_matches(value, expected):
            return value, []
        return value, [f"{path}: expected {expected}, got {type(value).__name__}"]

    if value is None and _type_is_optional(expected):
        return value, []

    schema_type = _schema_type(expected)
    if schema_type is not None:
        if not isinstance(value, dict):
            return value, [f"{path}: expected {expected}, got {type(value).__name__}"]
        try:
            normalized = validate(
                value,
                schema_type,
                path,
                allow_extra=allow_extra,
                apply_defaults=apply_defaults,
            )
        except ValidationError as exc:
            return value, exc.errors
        return normalized, []

    origin = typing.get_origin(expected)
    if origin is list:
        if not isinstance(value, list):
            return value, [f"{path}: expected {expected}, got {type(value).__name__}"]
        args = typing.get_args(expected)
        if not args:
            return value, []
        errors: list[str] = []
        normalized_items: list[Any] = []
        for index, item in enumerate(value):
            normalized_item, item_errors = _validate_value(
                item,
                args[0],
                f"{path}[{index}]",
                allow_extra=allow_extra,
                apply_defaults=apply_defaults,
            )
            errors.extend(item_errors)
            normalized_items.append(normalized_item)
        return (normalized_items if apply_defaults else value), errors

    if origin is dict:
        if not isinstance(value, dict):
            return value, [f"{path}: expected {expected}, got {type(value).__name__}"]
        args = typing.get_args(expected)
        if not args or len(args) != 2:
            return value, []
        key_type, value_type = args
        errors: list[str] = []
        normalized_items: dict[Any, Any] = {}
        for key, item in value.items():
            _, key_errors = _validate_value(
                key,
                key_type,
                f"{path}.<key>",
                allow_extra=allow_extra,
                apply_defaults=apply_defaults,
            )
            errors.extend(key_errors)
            item_path = f"{path}.{key}" if isinstance(key, str) else f"{path}[{key!r}]"
            normalized_item, item_errors = _validate_value(
                item,
                value_type,
                item_path,
                allow_extra=allow_extra,
                apply_defaults=apply_defaults,
            )
            errors.extend(item_errors)
            normalized_items[key] = normalized_item
        return (normalized_items if apply_defaults else value), errors

    if _type_matches(value, expected):
        return value, []
    return value, [f"{path}: expected {expected}, got {type(value).__name__}"]


# -- Schema validation --


def validate(
    data: dict[str, Any],
    schema: type,
    prefix: str = "",
    *,
    allow_extra: bool = True,
    apply_defaults: bool = False,
) -> dict[str, Any]:
    """Validate *data* against a class-based *schema*.

    The schema class should have type-annotated fields.  Constraints are
    specified via a class-level ``__constraints__`` dict mapping field names
    to lists of constraint objects.

    Returns the validated data dict.  Raises ``ValidationError`` on failure.
    """
    hints = get_type_hints(schema, include_extras=True)
    constraints: dict[str, list[Constraint]] = getattr(schema, "__constraints__", {})
    errors: list[str] = []
    validated: dict[str, Any] = {}

    for field, expected_type in hints.items():
        path = f"{prefix}{field}" if not prefix else f"{prefix}.{field}"
        if field not in data:
            # Check if optional (X | None)
            if _type_is_optional(expected_type):
                continue
            if hasattr(schema, field):
                if apply_defaults:
                    default_value = copy.deepcopy(getattr(schema, field))
                    normalized_default, default_errors = _validate_value(
                        default_value,
                        expected_type,
                        path,
                        allow_extra=allow_extra,
                        apply_defaults=apply_defaults,
                    )
                    if default_errors:
                        errors.extend(default_errors)
                    else:
                        validated[field] = normalized_default
                continue  # has default
            errors.append(f"{path}: missing required field")
            continue

        value = data[field]

        normalized_value, type_errors = _validate_value(
            value,
            expected_type,
            path,
            allow_extra=allow_extra,
            apply_defaults=apply_defaults,
        )
        if type_errors:
            errors.extend(type_errors)
            continue

        for constraint in constraints.get(field, []):
            err = constraint.check(normalized_value, path)
            if err:
                errors.append(err)
        validated[field] = normalized_value

    if not allow_extra:
        for field in sorted(set(data) - set(hints)):
            path = f"{prefix}{field}" if not prefix else f"{prefix}.{field}"
            errors.append(f"{path}: unexpected field")

    if errors:
        raise ValidationError(errors)

    if apply_defaults:
        result = copy.deepcopy(data) if allow_extra else {}
        result.update(validated)
        return result

    return data


def _type_is_optional(tp: Any) -> bool:
    """Return True if tp allows None (e.g. int | None)."""
    tp, _ = _strip_annotated(tp)
    origin = typing.get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        return type(None) in typing.get_args(tp)
    return False


def _schema_type(tp: Any) -> type | None:
    if isinstance(tp, type):
        try:
            hints = get_type_hints(tp)
        except Exception:
            return None
        if hints:
            return tp
        return None

    origin = typing.get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        args = [arg for arg in typing.get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return _schema_type(args[0])
    return None
