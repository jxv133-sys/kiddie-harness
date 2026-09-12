import json

from harness.llm_client import OllamaError
from harness.steps.critic import critique_file

from .fakes import FakeClient


def test_critique_file_parses_a_passing_verdict():
    client = FakeClient([json.dumps({"follows_spec": True, "issues": ""})])

    result = critique_file(
        client, "- print hi", "print('hi')\n", "main.py", temperature=0.2, max_tokens=512
    )

    assert result.follows_spec
    assert result.issues == ""


def test_critique_file_parses_a_failing_verdict_with_issues():
    client = FakeClient(
        [json.dumps({"follows_spec": False, "issues": "never prints the goodbye message"})]
    )

    result = critique_file(
        client, "- print hi\n- print bye", "print('hi')\n", "main.py",
        temperature=0.2, max_tokens=512,
    )

    assert not result.follows_spec
    assert result.issues == "never prints the goodbye message"


def test_critique_file_fails_open_on_an_unreachable_host():
    client = FakeClient([OllamaError("connection reset")])

    result = critique_file(client, "- x", "x = 1\n", "a.py", temperature=0.2, max_tokens=512)

    # An unreliable opinion must never be the thing that sinks a file that
    # every real check already passed -- a critic call that can't even
    # complete is treated as "no objection", not "no good".
    assert result.follows_spec


def test_critique_file_fails_open_on_unparseable_json():
    client = FakeClient(["not json at all"])

    result = critique_file(client, "- x", "x = 1\n", "a.py", temperature=0.2, max_tokens=512)

    assert result.follows_spec


def test_critique_file_fails_open_when_the_required_field_is_missing():
    client = FakeClient([json.dumps({"issues": "something"})])  # no follows_spec key

    result = critique_file(client, "- x", "x = 1\n", "a.py", temperature=0.2, max_tokens=512)

    assert result.follows_spec
