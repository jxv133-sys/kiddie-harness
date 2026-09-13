import json

import pytest

from harness.llm_client import OllamaClient, OllamaError


class _FakeStreamResponse:
    """Mimics the bits of a `requests.Response` used for a streamed call:
    `raise_for_status()` and `iter_lines()` yielding raw bytes, exactly
    like a real streaming response does."""

    def __init__(self, lines: list[str | Exception]):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line.encode()


class _NonStreamResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


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


def _stream_lines(*fragments: str, done_reason: str = "stop") -> list[str]:
    lines = [json.dumps({"response": f, "done": False}) for f in fragments]
    lines.append(json.dumps({"response": "", "done": True, "done_reason": done_reason}))
    return lines


def test_generate_without_on_chunk_does_not_stream(monkeypatch):
    import requests

    captured = {}

    def fake_post(url, *, json, timeout, **kw):
        captured.update(json=json, stream=kw.get("stream"))
        return _NonStreamResponse({"response": "hi", "done": True, "done_reason": "stop"})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    client.generate("prompt")

    assert captured["json"]["stream"] is False
    assert captured["stream"] is None  # requests' own stream=True kwarg never passed


def test_generate_with_on_chunk_streams_and_reports_cumulative_text(monkeypatch):
    import requests

    captured = {}

    def fake_post(url, *, json, timeout, stream=False, **kw):
        captured["payload"] = json
        captured["stream_kwarg"] = stream
        return _FakeStreamResponse(_stream_lines("def f", "():\n", "    pass\n"))

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)
    seen: list[str] = []

    result = client.generate("prompt", on_chunk=seen.append)

    assert captured["payload"]["stream"] is True
    assert captured["stream_kwarg"] is True
    # the final "done" line carries no new text but still triggers one
    # last on_chunk call, same cumulative text as the line before it
    assert seen == ["def f", "def f():\n", "def f():\n    pass\n", "def f():\n    pass\n"]
    assert result.text == "def f():\n    pass\n"
    assert result.done_reason == "stop"
    assert not result.truncated


def test_generate_streaming_reports_truncation_via_done_reason(monkeypatch):
    import requests

    def fake_post(*a, **kw):
        return _FakeStreamResponse(_stream_lines("partial", done_reason="length"))

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    result = client.generate("prompt", on_chunk=lambda t: None)

    assert result.truncated


def test_generate_streaming_skips_blank_lines(monkeypatch):
    import requests

    def fake_post(*a, **kw):
        return _FakeStreamResponse(["", *_stream_lines("hi"), ""])

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    result = client.generate("prompt", on_chunk=lambda t: None)

    assert result.text == "hi"


def test_generate_streaming_raises_on_a_non_json_line(monkeypatch):
    import requests

    def fake_post(*a, **kw):
        return _FakeStreamResponse(["not json at all"])

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    with pytest.raises(OllamaError, match="non-JSON stream line"):
        client.generate("prompt", on_chunk=lambda t: None)


def test_generate_streaming_raises_if_the_stream_ends_without_a_done_line(monkeypatch):
    import requests

    def fake_post(*a, **kw):
        return _FakeStreamResponse([json.dumps({"response": "hi", "done": False})])

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    with pytest.raises(OllamaError, match="ended without a final"):
        client.generate("prompt", on_chunk=lambda t: None)


def test_generate_streaming_wraps_a_connection_drop_mid_stream(monkeypatch):
    import requests

    def fake_post(*a, **kw):
        return _FakeStreamResponse(
            [json.dumps({"response": "hi", "done": False}), requests.ConnectionError("reset")]
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    with pytest.raises(OllamaError, match="Lost connection"):
        client.generate("prompt", on_chunk=lambda t: None)


def test_generate_streaming_swallows_a_broken_on_chunk_callback(monkeypatch):
    # A live-view display bug must never abort a real, in-progress
    # generation -- the model call is what matters, the callback is just
    # a spectator.
    import requests

    def fake_post(*a, **kw):
        return _FakeStreamResponse(_stream_lines("hi"))

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient("http://h", "m", timeout_seconds=5)

    def broken_on_chunk(text):
        raise RuntimeError("display bug")

    result = client.generate("prompt", on_chunk=broken_on_chunk)

    assert result.text == "hi"
