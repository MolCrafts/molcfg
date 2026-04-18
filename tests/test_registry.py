"""Tests for Registry and Build (config-string → instance resolution)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest

from molcfg import Build, Config, Registry, validate
from molcfg.errors import ValidationError


@dataclass
class Adam:
    lr: float = 0.001


@dataclass
class SiLU:
    pass


@dataclass
class GELU:
    pass


@dataclass
class LeakyReLU:
    negative_slope: float = 0.01


@pytest.fixture
def activations() -> Registry:
    reg = Registry("activation")
    reg.register("silu", SiLU)
    reg.register("gelu", GELU)
    reg.register("leaky_relu", LeakyReLU)
    return reg


class TestRegistryBuild:
    def test_short_form_str(self, activations: Registry) -> None:
        assert isinstance(activations.build("silu"), SiLU)

    def test_short_form_case_insensitive(self, activations: Registry) -> None:
        assert isinstance(activations.build("SiLU"), SiLU)

    def test_long_form_dict_with_kwargs(self, activations: Registry) -> None:
        result = activations.build({"type": "leaky_relu", "negative_slope": 0.2})
        assert isinstance(result, LeakyReLU)
        assert result.negative_slope == 0.2

    def test_none_passthrough(self, activations: Registry) -> None:
        assert activations.build(None) is None

    def test_instance_passthrough(self, activations: Registry) -> None:
        already = SiLU()
        assert activations.build(already) is already

    def test_unknown_key_error(self, activations: Registry) -> None:
        with pytest.raises(ValueError, match="'silux' not in"):
            activations.build("silux")

    def test_unknown_key_error_lists_candidates(self, activations: Registry) -> None:
        with pytest.raises(ValueError, match=r"gelu.*leaky_relu.*silu"):
            activations.build("silux")

    def test_dict_missing_discriminator(self, activations: Registry) -> None:
        with pytest.raises(ValueError, match="missing 'type'"):
            activations.build({"negative_slope": 0.1})

    def test_custom_discriminator(self) -> None:
        reg: Registry = Registry("opt", discriminator="name")
        reg.register("adam", Adam)
        result = reg.build({"name": "adam", "lr": 0.9})
        assert isinstance(result, Adam)
        assert result.lr == 0.9


class TestRegistryGet:
    """``.get(name)`` returns the factory/class without instantiation."""

    def test_returns_factory_class(self, activations: Registry) -> None:
        assert activations.get("silu") is SiLU

    def test_case_insensitive(self, activations: Registry) -> None:
        assert activations.get("SiLU") is SiLU

    def test_none_passthrough(self, activations: Registry) -> None:
        assert activations.get(None) is None

    def test_unknown_key_error(self, activations: Registry) -> None:
        with pytest.raises(ValueError, match="'silux' not in"):
            activations.get("silux")


class TestRegistryRegistration:
    def test_decorator_via_call(self) -> None:
        reg: Registry = Registry("layer")

        @reg("mlp")
        @dataclass
        class MLP:
            dim: int = 32

        instance = reg.build({"type": "mlp", "dim": 64})
        assert isinstance(instance, MLP)
        assert instance.dim == 64

    def test_decorator_via_register(self) -> None:
        reg: Registry = Registry("layer")

        @reg.register("conv")
        @dataclass
        class Conv:
            pass

        assert isinstance(reg.build("conv"), Conv)

    def test_duplicate_registration_raises(self) -> None:
        reg: Registry = Registry("dup")
        reg.register("x", SiLU)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("x", GELU)

    def test_contains(self, activations: Registry) -> None:
        assert "silu" in activations
        assert "SiLU" in activations
        assert "missing" not in activations

    def test_keys_sorted(self, activations: Registry) -> None:
        assert activations.keys() == ["gelu", "leaky_relu", "silu"]


class TestValidateIntegration:
    def test_annotated_build_resolves_string(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[SiLU, Build(reg)]

        result = validate({"activation": "silu"}, Schema, apply_defaults=True)
        assert isinstance(result["activation"], SiLU)

    def test_annotated_build_resolves_dict(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("leaky_relu", LeakyReLU)

        class Schema:
            activation: Annotated[LeakyReLU, Build(reg)]

        result = validate(
            {"activation": {"type": "leaky_relu", "negative_slope": 0.3}},
            Schema,
            apply_defaults=True,
        )
        assert result["activation"].negative_slope == 0.3

    def test_annotated_build_default_is_resolved(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[SiLU, Build(reg)] = "silu"

        result = validate({}, Schema, apply_defaults=True)
        assert isinstance(result["activation"], SiLU)

    def test_annotated_build_optional_none(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[SiLU | None, Build(reg)] = None

        result = validate({"activation": None}, Schema, apply_defaults=True)
        assert result["activation"] is None

    def test_annotated_build_optional_present(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[SiLU | None, Build(reg)] = None

        result = validate({"activation": "silu"}, Schema, apply_defaults=True)
        assert isinstance(result["activation"], SiLU)

    def test_annotated_build_unknown_key_fails_validation(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[SiLU, Build(reg)]

        with pytest.raises(ValidationError, match="not in"):
            validate({"activation": "relu"}, Schema)

    def test_annotated_build_type_mismatch_after_resolve(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[GELU, Build(reg)]

        with pytest.raises(ValidationError, match="expected"):
            validate({"activation": "silu"}, Schema)

    def test_idempotent_instance_passes_through(self) -> None:
        reg: Registry = Registry("activation")
        reg.register("silu", SiLU)

        class Schema:
            activation: Annotated[SiLU, Build(reg)]

        already = SiLU()
        result = validate({"activation": already}, Schema, apply_defaults=True)
        assert result["activation"] is already


# Reusable registry and schema for file-format roundtrips.
_FORMAT_REG: Registry = Registry("activation")
_FORMAT_REG.register("silu", SiLU)
_FORMAT_REG.register("gelu", GELU)
_FORMAT_REG.register("leaky_relu", LeakyReLU)


class ModelSchema:
    hidden_dim: int = 128
    activation: Annotated[object, Build(_FORMAT_REG)] = "silu"


class TestFileFormatRoundtrips:
    """Short & long forms work identically across JSON / TOML / YAML."""

    def test_json_short_form(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text('{"hidden_dim": 64, "activation": "gelu"}')
        cfg = Config.load_json(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], GELU)
        assert result["hidden_dim"] == 64

    def test_json_long_form(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text('{"activation": {"type": "leaky_relu", "negative_slope": 0.25}}')
        cfg = Config.load_json(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], LeakyReLU)
        assert result["activation"].negative_slope == 0.25

    def test_toml_short_form(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text('hidden_dim = 32\nactivation = "silu"\n')
        cfg = Config.load_toml(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], SiLU)

    def test_toml_long_form(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text('[activation]\ntype = "leaky_relu"\nnegative_slope = 0.15\n')
        cfg = Config.load_toml(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], LeakyReLU)
        assert result["activation"].negative_slope == 0.15

    def test_yaml_short_form(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("hidden_dim: 16\nactivation: gelu\n")
        cfg = Config.load_yaml(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], GELU)
        assert result["hidden_dim"] == 16

    def test_yaml_long_form(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("activation:\n  type: leaky_relu\n  negative_slope: 0.05\n")
        cfg = Config.load_yaml(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], LeakyReLU)
        assert result["activation"].negative_slope == 0.05

    def test_yaml_default_applied(self, tmp_path: Path) -> None:
        """When the field is absent in YAML, default 'silu' resolves too."""
        path = tmp_path / "c.yaml"
        path.write_text("hidden_dim: 8\n")
        cfg = Config.load_yaml(path)
        result = validate(cfg.to_dict(), ModelSchema, apply_defaults=True)
        assert isinstance(result["activation"], SiLU)
