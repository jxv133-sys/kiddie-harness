import json
from pathlib import Path

from harness.config import Endpoint
from harness.llm_client import OllamaClient
from harness.orchestrator import MultiFileLoop, partition_clients_by_role
from harness.session import Session

from .fakes import FakeClient, make_config


def _clients(*hosts: str) -> list[OllamaClient]:
    return [OllamaClient(h, "m", 60) for h in hosts]


def test_no_roles_tagged_reproduces_todays_behaviour():
    # Every endpoint at the "balanced" default (the only option before
    # roles existed): the first is both the plan/integration client and
    # a full worker, and nothing overrides critic.
    endpoints = [Endpoint("http://a", "m", 60), Endpoint("http://b", "m", 60)]
    clients = _clients("http://a", "http://b")

    primary, workers, critic_override = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is None


def test_smart_endpoint_becomes_primary_and_critic_and_is_excluded_from_workers():
    endpoints = [
        Endpoint("http://smart", "big", 60, role="smart"),
        Endpoint("http://q1", "small", 60, role="quick"),
        Endpoint("http://q2", "small", 60, role="quick"),
    ]
    clients = _clients("http://smart", "http://q1", "http://q2")

    primary, workers, critic_override = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert critic_override is clients[0]
    assert workers == clients[1:]  # the smart endpoint does no per-file grind


def test_only_quick_endpoints_falls_back_to_the_first_as_primary():
    endpoints = [Endpoint("http://q1", "m", 60, role="quick"), Endpoint("http://q2", "m", 60, role="quick")]
    clients = _clients("http://q1", "http://q2")

    primary, workers, critic_override = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is None  # no "smart" tag anywhere -- nothing to override with


def test_an_unrecognised_role_behaves_like_balanced():
    endpoints = [Endpoint("http://a", "m", 60, role="typo-not-a-real-role")]
    clients = _clients("http://a")

    primary, workers, critic_override = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is None


def test_a_lone_smart_endpoint_still_does_its_own_per_file_work():
    # workers must never end up empty just because the only endpoint
    # available happens to be tagged "smart".
    endpoints = [Endpoint("http://only", "m", 60, role="smart")]
    clients = _clients("http://only")

    primary, workers, critic_override = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is clients[0]


def test_critic_routes_to_the_smart_client_not_the_worker_that_built_the_file(tmp_path: Path):
    config = make_config(tmp_path, critic_enabled=True)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    worker = FakeClient(
        [plan, "- spec", "def thing():\n    return 1\n"], host="http://worker"
    )
    smart = FakeClient(['{"follows_spec": true, "issues": ""}'], host="http://smart")

    result = MultiFileLoop(
        worker, config, session, pool_clients=[worker], critic_client=smart
    ).run("goal")

    assert result.success
    assert len(smart.calls) == 1  # only the critic call
    assert len(worker.calls) == 3  # plan, spec, codegen -- never critic
