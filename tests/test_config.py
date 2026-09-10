from pathlib import Path

from harness.cli import _parse_endpoints
from harness.config import Config, Endpoint


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


def test_resolved_endpoints_is_the_single_host_when_none_configured():
    config = _config(ollama_host="http://h", model="m", timeout_seconds=200)

    assert config.resolved_endpoints() == [Endpoint("http://h", "m", 200)]


def test_resolved_endpoints_uses_the_explicit_list_when_present():
    eps = (Endpoint("http://a", "m1", 100), Endpoint("http://b", "m2", 100))
    config = _config(endpoints=eps)

    assert config.resolved_endpoints() == list(eps)


def test_parse_endpoints_splits_host_and_model_and_defaults_the_model():
    config = _config(model="fallback", timeout_seconds=300)
    parsed = _parse_endpoints(
        ["http://a:11434,qwen2.5-coder:7b", "http://b:11434"], config=config
    )

    assert parsed == (
        Endpoint("http://a:11434", "qwen2.5-coder:7b", 300),
        Endpoint("http://b:11434", "fallback", 300),
    )
