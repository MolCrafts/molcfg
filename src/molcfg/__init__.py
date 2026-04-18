"""molcfg — configuration library for the molcrafts ecosystem."""

from molcfg.concurrency import FileLock, ThreadSafeConfig, interpolate
from molcfg.config import Config
from molcfg.errors import (
    CircularReferenceError,
    ConfigError,
    FrozenConfigError,
    ValidationError,
)
from molcfg.merge import ConfigLoader, MergeStrategy, ProfileLoader, merge
from molcfg.registry import Registry
from molcfg.source import (
    CliSource,
    DictSource,
    EnvSource,
    JsonFileSource,
    Source,
    TomlFileSource,
    YamlFileSource,
)
from molcfg.validation import Build, Length, OneOf, Pattern, Range, validate

__all__ = [
    # Config
    "Config",
    # Errors
    "ConfigError",
    "CircularReferenceError",
    "FrozenConfigError",
    "ValidationError",
    # Sources
    "Source",
    "CliSource",
    "DictSource",
    "EnvSource",
    "JsonFileSource",
    "TomlFileSource",
    "YamlFileSource",
    # Merge
    "ConfigLoader",
    "MergeStrategy",
    "ProfileLoader",
    "merge",
    # Validation
    "Build",
    "Length",
    "OneOf",
    "Pattern",
    "Range",
    "validate",
    # Registry
    "Registry",
    # Concurrency
    "FileLock",
    "ThreadSafeConfig",
    "interpolate",
]
