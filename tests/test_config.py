from pathlib import Path

from harness.config import Config


def _config(**kw) -> Config:
    base = {
        "ollama_host": "http://h",
        "model": "m",
        "timeout_seconds": 300,
        "temperature": 0.2,
        "max_tokens": 2048,
        "max_tokens_ceiling": 8192,
        "max_fix_attempts": 5,
        "max_total_iterations": 40,
        "workspace_root": Path("workspace"),
    }
    base.update(kw)
    return Config(**base)


def test_with_overrides_replaces_only_the_values_given():
    config = _config().with_overrides(model="other", timeout_seconds=900)

    assert config.model == "other"
    assert config.timeout_seconds == 900
    assert config.ollama_host == "http://h"
    assert config.max_fix_attempts == 5


def test_with_overrides_ignores_none_and_zero():
    config = _config().with_overrides(host=None, max_fix_attempts=None, timeout_seconds=0)

    assert config.ollama_host == "http://h"
    assert config.max_fix_attempts == 5
    assert config.timeout_seconds == 300


def test_config_load_reads_the_default_yaml():
    config = Config.load()

    assert config.model
    assert config.max_fix_attempts >= 1
    assert config.timeout_seconds > 0
