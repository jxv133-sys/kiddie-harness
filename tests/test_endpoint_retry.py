import threading

import pytest

from harness import orchestrator
from harness.llm_client import OllamaError
from harness.orchestrator import RunCancelled, _with_endpoint_retry


def test_succeeds_immediately_when_fn_does_not_raise():
    assert _with_endpoint_retry(lambda: "ok") == "ok"


def test_retries_and_recovers_from_a_transient_failure():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise OllamaError("connection reset")
        return "recovered"

    assert _with_endpoint_retry(fn) == "recovered"
    assert len(calls) == 2


def test_raises_the_last_error_once_attempts_are_exhausted():
    calls = []

    def fn():
        calls.append(1)
        raise OllamaError(f"attempt {len(calls)}")

    with pytest.raises(OllamaError, match="attempt 3"):
        _with_endpoint_retry(fn)
    assert len(calls) == 3


def test_attempts_parameter_overrides_the_default():
    calls = []

    def fn():
        calls.append(1)
        raise OllamaError("down")

    with pytest.raises(OllamaError):
        _with_endpoint_retry(fn, attempts=1)
    assert len(calls) == 1  # no retry at all with attempts=1


def test_cancellation_during_backoff_raises_run_cancelled(monkeypatch):
    # Long enough that the backoff poll loop genuinely has time to notice
    # cancel_event -- the autouse fast-retry fixture collapses this to
    # near-zero everywhere else, which would make this specific test race.
    monkeypatch.setattr(orchestrator, "_ENDPOINT_RETRY_BACKOFF_SECONDS", 0.3)
    cancel_event = threading.Event()
    calls = []

    def fn():
        calls.append(1)
        cancel_event.set()  # simulate Stop landing right after the failure
        raise OllamaError("down")

    with pytest.raises(RunCancelled):
        _with_endpoint_retry(fn, cancel_event=cancel_event)
    assert len(calls) == 1  # never reached a second attempt
