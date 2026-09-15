import json

from harness.llm_client import OllamaError
from harness.steps.super_review import Issue, confirm_issue, find_issues

from .fakes import FakeClient

_FILES = [("main.py", "print('hi')\n"), ("helper.py", "def add(a, b):\n    return a + b\n")]


def test_find_issues_parses_a_list_of_issues():
    client = FakeClient(
        [
            json.dumps(
                {
                    "issues": [
                        {"file": "main.py", "description": "never imports helper.py"},
                    ]
                }
            )
        ]
    )

    issues = find_issues(client, "a script", _FILES, temperature=0.2, max_tokens=512)

    assert issues == [Issue(file="main.py", description="never imports helper.py")]


def test_find_issues_parses_an_empty_list():
    client = FakeClient([json.dumps({"issues": []})])

    issues = find_issues(client, "a script", _FILES, temperature=0.2, max_tokens=512)

    assert issues == []


def test_find_issues_fails_open_on_an_unreachable_host():
    client = FakeClient([OllamaError("connection reset")])

    issues = find_issues(client, "a script", _FILES, temperature=0.2, max_tokens=512)

    # An unreliable opinion must never invent a finding -- a call that
    # can't even complete means "found nothing", not "something's wrong".
    assert issues == []


def test_find_issues_fails_open_on_unparseable_json():
    client = FakeClient(["not json at all"])

    issues = find_issues(client, "a script", _FILES, temperature=0.2, max_tokens=512)

    assert issues == []


def test_find_issues_skips_a_malformed_entry_without_crashing():
    client = FakeClient(
        [
            json.dumps(
                {
                    "issues": [
                        {"file": "main.py", "description": "real issue"},
                        {"file": "main.py"},  # missing "description" -- dropped, not fatal
                    ]
                }
            )
        ]
    )

    issues = find_issues(client, "a script", _FILES, temperature=0.2, max_tokens=512)

    assert issues == [Issue(file="main.py", description="real issue")]


def test_confirm_issue_parses_a_true_verdict():
    client = FakeClient([json.dumps({"confirmed": True})])
    issue = Issue(file="main.py", description="never imports helper.py")

    confirmed = confirm_issue(client, _FILES, issue, temperature=0.2, max_tokens=512)

    assert confirmed is True


def test_confirm_issue_parses_a_false_verdict():
    client = FakeClient([json.dumps({"confirmed": False})])
    issue = Issue(file="main.py", description="never imports helper.py")

    confirmed = confirm_issue(client, _FILES, issue, temperature=0.2, max_tokens=512)

    assert confirmed is False


def test_confirm_issue_fails_open_toward_not_confirmed_on_an_unreachable_host():
    client = FakeClient([OllamaError("connection reset")])
    issue = Issue(file="main.py", description="never imports helper.py")

    confirmed = confirm_issue(client, _FILES, issue, temperature=0.2, max_tokens=512)

    assert confirmed is False


def test_confirm_issue_fails_open_toward_not_confirmed_on_unparseable_json():
    client = FakeClient(["not json at all"])
    issue = Issue(file="main.py", description="never imports helper.py")

    confirmed = confirm_issue(client, _FILES, issue, temperature=0.2, max_tokens=512)

    assert confirmed is False
