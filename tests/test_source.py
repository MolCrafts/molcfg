"""Tests for configuration sources."""

import json

import pytest

from molcfg import (
    CliSource,
    DictSource,
    EnvSource,
    JsonFileSource,
    TomlFileSource,
    YamlFileSource,
)


class TestDictSource:
    def test_load(self):
        src = DictSource({"a": 1, "b": {"c": 2}})
        assert src.load() == {"a": 1, "b": {"c": 2}}

    def test_load_returns_copy(self):
        original = {"a": 1}
        src = DictSource(original)
        result = src.load()
        result["a"] = 999
        assert src.load()["a"] == 1

    def test_load_returns_deep_copy(self):
        original = {"a": {"b": 1}}
        src = DictSource(original)

        result = src.load()
        result["a"]["b"] = 999

        assert src.load()["a"]["b"] == 1


class TestJsonFileSource:
    def test_load(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"db": {"host": "localhost"}}))
        src = JsonFileSource(p)
        assert src.load() == {"db": {"host": "localhost"}}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            JsonFileSource("/nonexistent.json").load()

    def test_non_object_root(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(TypeError, match="object"):
            JsonFileSource(p).load()


class TestTomlFileSource:
    def test_load(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text('[db]\nhost = "localhost"\nport = 5432\n')
        src = TomlFileSource(p)
        data = src.load()
        assert data == {"db": {"host": "localhost", "port": 5432}}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            TomlFileSource("/nonexistent.toml").load()


class TestYamlFileSource:
    def test_load(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("db:\n  host: localhost\n  port: 5432\n")
        src = YamlFileSource(p)
        assert src.load() == {"db": {"host": "localhost", "port": 5432}}

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert YamlFileSource(p).load() == {}

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            YamlFileSource("/nonexistent.yaml").load()

    def test_non_mapping_root(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- 1\n- 2\n")
        with pytest.raises(TypeError, match="mapping"):
            YamlFileSource(p).load()


class TestEnvSource:
    def test_with_prefix(self):
        environ = {
            "MYAPP_DB_HOST": "localhost",
            "MYAPP_DB_PORT": "5432",
            "OTHER_VAR": "ignored",
        }
        src = EnvSource(prefix="MYAPP", separator="_", environ=environ)
        data = src.load()
        assert data == {"db": {"host": "localhost", "port": 5432}}

    def test_without_prefix(self):
        environ = {"DB_HOST": "localhost"}
        src = EnvSource(prefix="", separator="_", environ=environ)
        data = src.load()
        assert data == {"db": {"host": "localhost"}}

    def test_coerces_bool_and_json_values(self):
        environ = {
            "APP_DEBUG": "true",
            "APP_TAGS": '["a", "b"]',
        }
        src = EnvSource(prefix="APP", separator="_", environ=environ)
        data = src.load()
        assert data == {"debug": True, "tags": ["a", "b"]}

    def test_can_disable_coercion(self):
        environ = {"APP_PORT": "5432"}
        src = EnvSource(prefix="APP", separator="_", environ=environ, coerce=False)
        data = src.load()
        assert data == {"port": "5432"}


class TestCliSource:
    def test_key_equals_value(self):
        src = CliSource(["--db.host=localhost", "--db.port=5432"])
        data = src.load()
        assert data == {"db": {"host": "localhost", "port": 5432}}

    def test_key_space_value(self):
        src = CliSource(["--db.host", "localhost", "--db.port", "5432"])
        data = src.load()
        assert data == {"db": {"host": "localhost", "port": 5432}}

    def test_mixed(self):
        src = CliSource(["--db.host=localhost", "--db.port", "5432"])
        data = src.load()
        assert data == {"db": {"host": "localhost", "port": 5432}}

    def test_non_flag_args_ignored(self):
        src = CliSource(["positional", "--key=val"])
        data = src.load()
        assert data == {"key": "val"}

    def test_flag_without_value_does_not_consume_next_option(self):
        src = CliSource(["--flag", "--other=1"])
        data = src.load()
        assert data == {"flag": "", "other": 1}

    def test_coerces_bool_and_json_values(self):
        src = CliSource(["--debug=true", '--tags=["a","b"]'])
        data = src.load()
        assert data == {"debug": True, "tags": ["a", "b"]}

    def test_can_disable_coercion(self):
        src = CliSource(["--port=5432"], coerce=False)
        data = src.load()
        assert data == {"port": "5432"}
