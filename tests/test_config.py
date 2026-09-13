"""Tests for datadoc.toml loading, including the no-tomli fallback parser."""

import sys

import pytest

from datadoc.cli.app import (
    _load_toml_file,
    _parse_simple_toml,
    _pipeline_config,
)

INIT_STYLE = """\
# DATADOC config - committed for reproducibility
target = "churn"        # e.g. "churn"
task = "auto"      # auto | classification | regression
drop_identifiers = false
scaling = "standard"
clip_outliers = true  # enable me
categorical_threshold = 20
rare_category_min_frequency = 0.0  # e.g. 0.02 groups rare into __RARE__
identifier_columns = ["PassengerId", "Ticket"]  # forced IDs
ignored_columns = []  # nothing ignored
strict_schema = true
"""


def test_fallback_strips_inline_comments():
    cfg = _parse_simple_toml(INIT_STYLE)
    assert cfg["target"] == "churn"
    assert cfg["rare_category_min_frequency"] == 0.0
    assert isinstance(cfg["rare_category_min_frequency"], float)
    assert cfg["clip_outliers"] is True
    assert cfg["categorical_threshold"] == 20


def test_fallback_parses_lists_and_quotes():
    cfg = _parse_simple_toml(INIT_STYLE)
    assert cfg["identifier_columns"] == ["PassengerId", "Ticket"]
    assert cfg["ignored_columns"] == []


def test_fallback_hash_inside_quotes_survives():
    cfg = _parse_simple_toml('target = "a#b"\n')
    assert cfg["target"] == "a#b"


def test_pipeline_config_accepts_commented_toml(tmp_path, monkeypatch):
    p = tmp_path / "datadoc.toml"
    p.write_text(INIT_STYLE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Force the fallback path even where tomllib exists.
    monkeypatch.setitem(sys.modules, "tomllib", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("tomllib", "tomli"):
            raise ImportError("forced fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = _pipeline_config()
    assert cfg.target == "churn"
    assert cfg.rare_category_min_frequency == 0.0
    assert cfg.identifier_columns == ["PassengerId", "Ticket"]


def test_explicit_toml_file_with_lists(tmp_path):
    p = tmp_path / "custom.toml"
    p.write_text('[datadoc]\ntarget = "y"\nidentifier_columns = ["a"]\n', encoding="utf-8")
    loaded = _load_toml_file(p)
    assert loaded["identifier_columns"] == ["a"]
    cfg = _pipeline_config(config_file=str(p))
    assert cfg.identifier_columns == ["a"]


def test_explicit_flags_beat_config_file(tmp_path):
    p = tmp_path / "custom.toml"
    p.write_text("[datadoc]\ndrop_identifiers = false\n", encoding="utf-8")
    cfg = _pipeline_config(drop_identifiers=True, config_file=str(p))
    assert cfg.drop_identifiers is True
    cfg2 = _pipeline_config(config_file=str(p))
    assert cfg2.drop_identifiers is False


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
