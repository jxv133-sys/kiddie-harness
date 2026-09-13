import pytest

from harness import orchestrator


@pytest.fixture(autouse=True)
def _fast_endpoint_retries(monkeypatch):
    """`orchestrator._with_endpoint_retry` backs off for real seconds
    between attempts in production -- harmless there, but it would turn
    any test that simulates a connection failure (FakeClient raising
    OllamaError) into a multi-second test for no reason. Attempts stay
    the same (so "retried N times" behaviour is still exercised); only
    the backoff collapses to effectively nothing.
    """
    monkeypatch.setattr(orchestrator, "_ENDPOINT_RETRY_BACKOFF_SECONDS", 0.001)
