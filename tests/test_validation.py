"""Tests for type checking, constraints, and schema validation."""

from typing import Literal

import pytest

from molcfg import Length, OneOf, Pattern, Range, validate
from molcfg.errors import ValidationError


class TestTypeValidation:
    def test_valid_int(self):
        class Schema:
            port: int

        validate({"port": 5432}, Schema)

    def test_invalid_int(self):
        class Schema:
            port: int

        with pytest.raises(ValidationError):
            validate({"port": "abc"}, Schema)

    def test_float_accepts_int(self):
        class Schema:
            cutoff: float

        validate({"cutoff": 8}, Schema)

    def test_float_rejects_bool(self):
        class Schema:
            cutoff: float

        with pytest.raises(ValidationError):
            validate({"cutoff": True}, Schema)

    def test_optional_field_missing(self):
        class Schema:
            name: str | None

        validate({}, Schema)  # should not raise

    def test_optional_field_present(self):
        class Schema:
            name: str | None

        validate({"name": "hello"}, Schema)

    def test_literal(self):
        class Schema:
            mode: Literal["fast", "slow"]

        validate({"mode": "fast"}, Schema)
        with pytest.raises(ValidationError):
            validate({"mode": "invalid"}, Schema)

    def test_list_type(self):
        class Schema:
            tags: list[str]

        validate({"tags": ["a", "b"]}, Schema)
        with pytest.raises(ValidationError):
            validate({"tags": [1, 2]}, Schema)

    def test_dict_type(self):
        class Schema:
            meta: dict[str, int]

        validate({"meta": {"a": 1}}, Schema)
        with pytest.raises(ValidationError):
            validate({"meta": {"a": "x"}}, Schema)

    def test_nested_schema(self):
        class DbSchema:
            host: str
            port: int

        class AppSchema:
            db: DbSchema

        validate({"db": {"host": "localhost", "port": 5432}}, AppSchema)

    def test_nested_schema_reports_nested_field_error(self):
        class DbSchema:
            host: str

        class AppSchema:
            db: DbSchema

        with pytest.raises(ValidationError, match="db.host"):
            validate({"db": {"host": 123}}, AppSchema)

    def test_list_of_nested_schema(self):
        class ServiceSchema:
            name: str

        class AppSchema:
            services: list[ServiceSchema]

        validate({"services": [{"name": "api"}, {"name": "worker"}]}, AppSchema)

    def test_missing_required_field(self):
        class Schema:
            name: str

        with pytest.raises(ValidationError, match="missing"):
            validate({}, Schema)

    def test_field_with_default(self):
        class Schema:
            name: str = "default"

        validate({}, Schema)  # should not raise

    def test_apply_defaults_returns_completed_data(self):
        class Schema:
            host: str = "localhost"
            port: int

        result = validate({"port": 5432}, Schema, apply_defaults=True)
        assert result == {"host": "localhost", "port": 5432}

    def test_allow_extra_false_rejects_unknown_fields(self):
        class Schema:
            host: str

        with pytest.raises(ValidationError, match="unexpected field"):
            validate({"host": "localhost", "debug": True}, Schema, allow_extra=False)

    def test_nested_defaults_are_applied(self):
        class DbSchema:
            host: str = "localhost"
            port: int = 5432

        class AppSchema:
            db: DbSchema

        result = validate({"db": {}}, AppSchema, apply_defaults=True)
        assert result == {"db": {"host": "localhost", "port": 5432}}


class TestConstraints:
    def test_range_pass(self):
        class Schema:
            port: int
            __constraints__ = {"port": [Range(1, 65535)]}

        validate({"port": 8080}, Schema)

    def test_range_fail(self):
        class Schema:
            port: int
            __constraints__ = {"port": [Range(1, 65535)]}

        with pytest.raises(ValidationError, match="range"):
            validate({"port": 70000}, Schema)

    def test_range_skips_none_on_an_optional_field(self):
        class Schema:
            cutoff: float | None
            __constraints__ = {"cutoff": [Range(0, 10)]}

        validate({"cutoff": None}, Schema)

    def test_oneof_pass(self):
        class Schema:
            level: str
            __constraints__ = {"level": [OneOf("debug", "info", "warning", "error")]}

        validate({"level": "info"}, Schema)

    def test_oneof_fail(self):
        class Schema:
            level: str
            __constraints__ = {"level": [OneOf("debug", "info")]}

        with pytest.raises(ValidationError, match="not one of"):
            validate({"level": "trace"}, Schema)

    def test_pattern_pass(self):
        class Schema:
            email: str
            __constraints__ = {"email": [Pattern(r"@")]}

        validate({"email": "a@b.com"}, Schema)

    def test_pattern_fail(self):
        class Schema:
            email: str
            __constraints__ = {"email": [Pattern(r"^[a-z]+@")]}

        with pytest.raises(ValidationError, match="pattern"):
            validate({"email": "INVALID"}, Schema)

    def test_length_pass(self):
        class Schema:
            name: str
            __constraints__ = {"name": [Length(1, 50)]}

        validate({"name": "hello"}, Schema)

    def test_length_fail(self):
        class Schema:
            name: str
            __constraints__ = {"name": [Length(5, 10)]}

        with pytest.raises(ValidationError, match="length"):
            validate({"name": "hi"}, Schema)

    def test_multiple_errors(self):
        class Schema:
            port: int
            host: str
            __constraints__ = {"port": [Range(1, 100)]}

        with pytest.raises(ValidationError) as exc_info:
            validate({"port": 200, "host": 123}, Schema)
        assert len(exc_info.value.errors) == 2
