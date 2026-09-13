import pytest

from harness.llm_client import OllamaClient, OllamaError


def test_generate_wraps_a_requests_exception_as_ollama_error(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        client.generate("prompt")


def test_generate_wraps_an_invalid_timeout_as_ollama_error(monkeypatch):
    # requests rejects timeout=0 outright with a ValueError raised before
    # any network I/O -- 0 does not mean "no timeout" the way it might for
    # some other client. This must surface as a clean OllamaError, the
    # same as any other call failure, not an unhandled crash.
    import requests

    client = OllamaClient("http://h", "m", timeout_seconds=0)

    def raises_for_zero_timeout(*a, **k):
        if k.get("timeout") == 0:
            raise ValueError(
                "Attempted to set connect timeout to 0, but the timeout cannot be "
                "set to a value less than or equal to 0."
            )
        raise AssertionError("unexpected timeout value")

    monkeypatch.setattr(requests, "post", raises_for_zero_timeout)

    with pytest.raises(OllamaError, match="Invalid call to Ollama"):
        client.generate("prompt")
