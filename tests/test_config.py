"""Tests for Config core: access, freeze, snapshot, callbacks, serialization."""

import json

import pytest

from molcfg import Config
from molcfg.errors import ConfigError, FrozenConfigError


class TestAttributeAccess:
    def test_simple_attribute(self):
        cfg = Config({"name": "test"})
        assert cfg.name == "test"

    def test_nested_attribute(self):
        cfg = Config({"db": {"host": "localhost", "port": 5432}})
        assert cfg.db.host == "localhost"
        assert cfg.db.port == 5432

    def test_missing_attribute_raises(self):
        cfg = Config({"a": 1})
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent

    def test_set_attribute(self):
        cfg = Config({})
        cfg.name = "hello"
        assert cfg.name == "hello"

    def test_del_attribute(self):
        cfg = Config({"x": 1})
        del cfg.x
        with pytest.raises(AttributeError):
            _ = cfg.x

    def test_attribute_access_prefers_config_value_over_method_name(self):
        cfg = Config({"items": 1, "keys": 2})
        assert cfg.items == 1
        assert cfg.keys == 2


class TestPathAccess:
    def test_dotted_path(self):
        cfg = Config({"db": {"host": "localhost"}})
        assert cfg["db.host"] == "localhost"

    def test_single_key(self):
        cfg = Config({"key": "value"})
        assert cfg["key"] == "value"

    def test_missing_path_raises(self):
        cfg = Config({"a": 1})
        with pytest.raises(KeyError):
            _ = cfg["a.b.c"]

    def test_set_path(self):
        cfg = Config({})
        cfg["db.host"] = "localhost"
        assert cfg["db.host"] == "localhost"

    def test_del_path(self):
        cfg = Config({"db": {"host": "localhost", "port": 5432}})
        del cfg["db.host"]
        with pytest.raises(KeyError):
            _ = cfg["db.host"]

    def test_contains(self):
        cfg = Config({"a": {"b": 1}})
        assert "a.b" in cfg
        assert "a.c" not in cfg


class TestFreeze:
    def test_freeze_prevents_setattr(self):
        cfg = Config({"db": {"host": "localhost"}})
        cfg.freeze()
        with pytest.raises(FrozenConfigError):
            cfg.db.host = "other"

    def test_freeze_prevents_setitem(self):
        cfg = Config({"key": "val"})
        cfg.freeze()
        with pytest.raises(FrozenConfigError):
            cfg["key"] = "new"

    def test_freeze_prevents_delattr(self):
        cfg = Config({"key": "val"})
        cfg.freeze()
        with pytest.raises(FrozenConfigError):
            del cfg.key

    def test_unfreeze_allows_mutation(self):
        cfg = Config({"key": "val"})
        cfg.freeze()
        cfg.unfreeze()
        cfg.key = "new"
        assert cfg.key == "new"

    def test_frozen_property(self):
        cfg = Config({})
        assert not cfg.frozen
        cfg.freeze()
        assert cfg.frozen


class TestSnapshot:
    def test_snapshot_and_rollback(self):
        cfg = Config({"x": 1})
        cfg.snapshot()
        cfg.x = 2
        assert cfg.x == 2
        cfg.rollback()
        assert cfg.x == 1

    def test_multiple_snapshots_stack(self):
        cfg = Config({"x": 1})
        cfg.snapshot()
        cfg.x = 2
        cfg.snapshot()
        cfg.x = 3
        cfg.rollback()
        assert cfg.x == 2
        cfg.rollback()
        assert cfg.x == 1

    def test_rollback_empty_raises(self):
        cfg = Config({})
        with pytest.raises(ConfigError):
            cfg.rollback()


class TestChangeNotification:
    def test_callback_on_setattr(self):
        changes: list[tuple[str, object, object]] = []
        cfg = Config({"x": 1})
        cfg.on_change(lambda k, n, o: changes.append((k, n, o)))
        cfg.x = 2
        assert len(changes) == 1
        assert changes[0] == ("x", 2, 1)

    def test_callback_on_setitem(self):
        changes: list[tuple[str, object, object]] = []
        cfg = Config({})
        cfg.on_change(lambda k, n, o: changes.append((k, n, o)))
        cfg["a.b"] = 42
        assert len(changes) == 1
        assert changes[0][0] == "a.b"
        assert changes[0][1] == 42


class TestMetadata:
    def test_runtime_setitem_tracks_source_metadata(self):
        cfg = Config({})
        cfg["db.host"] = "localhost"
        assert cfg.meta("db.host") == {
            "source": "runtime",
            "history": ("runtime",),
        }

    def test_nested_assignment_updates_shared_metadata(self):
        cfg = Config({"db": {"host": "localhost"}})
        cfg.db.host = "prod-db"
        assert cfg.meta("db.host") == {
            "source": "runtime",
            "history": ("runtime",),
        }

    def test_delete_removes_metadata(self):
        cfg = Config({})
        cfg["db.host"] = "localhost"
        del cfg["db.host"]
        assert cfg.meta("db.host") is None

    def test_snapshot_and_rollback_restore_metadata(self):
        cfg = Config({})
        cfg["db.host"] = "localhost"
        cfg.snapshot()
        cfg["db.host"] = "prod-db"
        cfg.rollback()
        assert cfg["db.host"] == "localhost"
        assert cfg.meta("db.host") == {
            "source": "runtime",
            "history": ("runtime",),
        }


class TestSerialization:
    def test_to_dict(self):
        cfg = Config({"db": {"host": "localhost", "port": 5432}})
        d = cfg.to_dict()
        assert d == {"db": {"host": "localhost", "port": 5432}}
        assert isinstance(d["db"], dict)

    def test_to_json(self):
        cfg = Config({"name": "test", "count": 3})
        j = cfg.to_json(sort_keys=True)
        assert json.loads(j) == {"name": "test", "count": 3}

    def test_to_toml_scalars(self):
        cfg = Config({"name": "test", "count": 3, "flag": True})
        toml_str = cfg.to_toml()
        assert 'name = "test"' in toml_str
        assert "count = 3" in toml_str
        assert "flag = true" in toml_str

    def test_to_toml_nested(self):
        cfg = Config({"db": {"host": "localhost", "port": 5432}})
        toml_str = cfg.to_toml()
        assert "[db]" in toml_str
        assert 'host = "localhost"' in toml_str
        assert "port = 5432" in toml_str

    def test_to_toml_list(self):
        cfg = Config({"tags": ["a", "b", "c"]})
        toml_str = cfg.to_toml()
        assert 'tags = ["a", "b", "c"]' in toml_str

    def test_to_toml_skips_none(self):
        cfg = Config({"name": "test", "opt": None})
        toml_str = cfg.to_toml()
        assert "opt" not in toml_str

    def test_save_and_load_json_roundtrip(self, tmp_path):
        cfg = Config({"db": {"host": "localhost", "port": 5432}, "debug": False})
        path = tmp_path / "cfg.json"
        cfg.save_json(path)
        loaded = Config.load_json(path)
        assert loaded.to_dict() == cfg.to_dict()

    def test_save_and_load_toml_roundtrip(self, tmp_path):
        cfg = Config({"db": {"host": "localhost", "port": 5432}, "debug": False})
        path = tmp_path / "cfg.toml"
        cfg.save_toml(path)
        loaded = Config.load_toml(path)
        assert loaded.to_dict() == cfg.to_dict()

    def test_load_json_invalid_raises(self, tmp_path):
        from molcfg.errors import ConfigError
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")  # JSON array, not an object
        with pytest.raises(ConfigError):
            Config.load_json(path)

    def test_save_json_creates_file(self, tmp_path):
        cfg = Config({"x": 1})
        path = tmp_path / "out.json"
        cfg.save_json(path)
        assert path.exists()

    def test_save_toml_creates_file(self, tmp_path):
        cfg = Config({"x": 1})
        path = tmp_path / "out.toml"
        cfg.save_toml(path)
        assert path.exists()


class TestDunderMethods:
    def test_repr(self):
        cfg = Config({"a": 1})
        assert "Config" in repr(cfg)

    def test_eq(self):
        assert Config({"a": 1}) == Config({"a": 1})
        assert Config({"a": 1}) != Config({"a": 2})

    def test_len(self):
        assert len(Config({"a": 1, "b": 2})) == 2

    def test_iter(self):
        cfg = Config({"a": 1, "b": 2})
        assert set(cfg) == {"a", "b"}

    def test_get_default(self):
        cfg = Config({"a": 1})
        assert cfg.get("a") == 1
        assert cfg.get("b", 99) == 99
