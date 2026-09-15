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

    primary, workers, critic_override, branch_pool = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is None
    assert branch_pool == clients  # both "balanced" -- both branch-eligible


def test_smart_endpoint_becomes_primary_and_critic_but_still_joins_the_workers():
    # A "smart" endpoint is preferred for plan/critic -- it isn't kept
    # out of the per-file grind for being capable; if anything that's a
    # reason to give it files too, not sideline it.
    endpoints = [
        Endpoint("http://smart", "big", 60, role="smart"),
        Endpoint("http://q1", "small", 60, role="quick"),
        Endpoint("http://q2", "small", 60, role="quick"),
    ]
    clients = _clients("http://smart", "http://q1", "http://q2")

    primary, workers, critic_override, branch_pool = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert critic_override is clients[0]
    assert workers == clients  # smart included alongside the quick ones
    assert branch_pool == [clients[0]]  # only the smart one -- both quicks excluded


def test_only_quick_endpoints_falls_back_to_the_first_as_primary():
    endpoints = [Endpoint("http://q1", "m", 60, role="quick"), Endpoint("http://q2", "m", 60, role="quick")]
    clients = _clients("http://q1", "http://q2")

    primary, workers, critic_override, branch_pool = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is None  # no "smart" tag anywhere -- nothing to override with
    assert branch_pool == []  # no endpoint is ever branch-eligible if all are "quick"


def test_an_unrecognised_role_behaves_like_balanced():
    endpoints = [Endpoint("http://a", "m", 60, role="typo-not-a-real-role")]
    clients = _clients("http://a")

    primary, workers, critic_override, branch_pool = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is None
    assert branch_pool == clients  # not "quick" -- still branch-eligible, like "balanced"


def test_a_lone_smart_endpoint_still_does_its_own_per_file_work():
    endpoints = [Endpoint("http://only", "m", 60, role="smart")]
    clients = _clients("http://only")

    primary, workers, critic_override, branch_pool = partition_clients_by_role(endpoints, clients)

    assert primary is clients[0]
    assert workers == clients
    assert critic_override is clients[0]
    assert branch_pool == clients


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


def test_the_smart_endpoint_can_still_build_files_alongside_a_quick_one(tmp_path: Path):
    # "why not have the math expert do math too" -- a smart endpoint
    # isn't excluded from the per-file dispatch pool. Two independent
    # files, two workers, each given exactly enough queued responses for
    # one file (spec + codegen): whichever specific file either one
    # claims, the totals below only add up if both workers actually did
    # real per-file work -- proof the smart endpoint wasn't sidelined to
    # plan/critic only.
    config = make_config(tmp_path, critic_enabled=False)
    session = Session.create(config.workspace_root)
    plan = json.dumps(
        {
            "files": [
                {"path": "a.py", "purpose": "x", "depends_on": []},
                {"path": "b.py", "purpose": "y", "depends_on": []},
            ]
        }
    )
    smart = FakeClient([plan, "- spec", "def thing():\n    return 1\n"], host="http://smart")
    quick = FakeClient(["- spec", "def thing():\n    return 2\n"], host="http://quick")

    result = MultiFileLoop(smart, config, session, pool_clients=[smart, quick]).run("goal")

    assert result.success
    assert len(smart.calls) == 3  # plan, plus one file's spec + codegen
    assert len(quick.calls) == 2  # the other file's spec + codegen
