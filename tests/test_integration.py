"""Integration test: full pipeline load → merge → validate → freeze → read."""

import json

import pytest

from molcfg import (
    Config,
    ConfigLoader,
    DictSource,
    JsonFileSource,
    Range,
    interpolate,
    validate,
)
from molcfg.errors import FrozenConfigError


class TestFullPipeline:
    def test_load_merge_validate_freeze_read(self, tmp_path):
        # 1. Write a JSON config file
        json_path = tmp_path / "app.json"
        json_path.write_text(
            json.dumps(
                {
                    "db": {"host": "file-host", "port": 9999},
                }
            )
        )

        # 2. Define sources: defaults → file override
        defaults = DictSource(
            {
                "db": {"host": "localhost", "port": 5432, "name": "mydb"},
                "app": {"debug": True},
            }
        )
        file_src = JsonFileSource(json_path)

        # 3. Load & merge
        loader = ConfigLoader([defaults, file_src])
        cfg = loader.load()

        assert cfg["db.host"] == "file-host"
        assert cfg["db.port"] == 9999
        assert cfg["db.name"] == "mydb"  # from defaults
        assert cfg["app.debug"] is True

        # 4. Validate
        class DbSchema:
            host: str
            port: int
            name: str
            __constraints__ = {"port": [Range(1, 65535)]}

        validate(cfg["db"].to_dict(), DbSchema)

        # 5. Freeze
        cfg.freeze()
        with pytest.raises(FrozenConfigError):
            cfg["app.debug"] = False

        # 6. Read via attribute and path
        assert cfg.db.host == "file-host"
        assert cfg["db.name"] == "mydb"

    def test_interpolation_in_pipeline(self):
        raw = {
            "base_dir": "/opt/app",
            "log_dir": "${base_dir}/logs",
            "data_dir": "${base_dir}/data",
            "env_user": "${env:TEST_USER}",
        }
        resolved = interpolate(raw, environ={"TEST_USER": "alice"})
        cfg = Config(resolved)
        assert cfg.log_dir == "/opt/app/logs"
        assert cfg.data_dir == "/opt/app/data"
        assert cfg.env_user == "alice"

    def test_validation_defaults_and_metadata_in_pipeline(self):
        defaults = DictSource(
            {"db": {"host": "localhost"}},
            name="defaults",
        )
        cli = DictSource(
            {"db": {"port": 5432}},
            name="cli",
        )
        cfg = ConfigLoader([defaults, cli]).load()

        class DbSchema:
            host: str
            port: int = 5432

        validated = validate(cfg["db"].to_dict(), DbSchema, apply_defaults=True)

        assert validated == {"host": "localhost", "port": 5432}
        assert cfg.meta("db.host") == {
            "source": "defaults",
            "history": ("defaults",),
        }
        assert cfg.meta("db.port") == {
            "source": "cli",
            "history": ("cli",),
        }
