import json

import pytest

from harness.llm_client import OllamaError
from harness.steps.plan import FileTask, PlanError, _parse_free_form, plan_files

from .fakes import FakeClient


def test_parse_free_form_reads_a_numbered_markdown_list():
    text = (
        "### Files Needed:\n\n"
        "1. **prime.py**\n   - **Purpose:** check whether a number is prime\n"
        "2. **primes.py**\n   - Purpose: return the first n primes\n"
    )

    data = _parse_free_form(text)

    assert [f["path"] for f in data["files"]] == ["prime.py", "primes.py"]
    assert "prime" in data["files"][0]["purpose"].lower()


def test_parse_free_form_reads_json_inside_a_fence_after_reasoning():
    text = (
        "<think>\nI need one module.\n</think>\n"
        '```json\n{"files": [{"path": "core.py", "purpose": "logic"}]}\n```'
    )

    assert _parse_free_form(text) == {"files": [{"path": "core.py", "purpose": "logic"}]}


def test_plan_files_parses_schema_constrained_json():
    payload = json.dumps(
        {
            "files": [
                {"path": "cli.py", "purpose": "Entry point"},
                {"path": "core.py", "purpose": "Core logic"},
            ]
        }
    )
    client = FakeClient([payload])

    tasks = plan_files(client, "build a todo app", temperature=0.2, max_tokens=512)

    assert tasks == [
        FileTask(path="cli.py", purpose="Entry point"),
        FileTask(path="core.py", purpose="Core logic", depends_on=("cli.py",)),
    ]


def test_plan_files_raises_on_invalid_json():
    client = FakeClient(["not json"])

    with pytest.raises(PlanError):
        plan_files(client, "build a todo app", temperature=0.2, max_tokens=512, max_attempts=1)


def test_plan_files_raises_on_empty_file_list():
    client = FakeClient([json.dumps({"files": []})])

    with pytest.raises(PlanError):
        plan_files(client, "build a todo app", temperature=0.2, max_tokens=512, max_attempts=1)


def test_plan_files_retries_when_the_first_plan_is_empty():
    client = FakeClient(
        [
            json.dumps({"files": []}),  # degenerate grammar fill
            json.dumps({"files": [{"path": "core.py", "purpose": "logic"}]}),  # retry
        ]
    )

    tasks = plan_files(client, "x", temperature=0.2, max_tokens=512)

    assert tasks == [FileTask(path="core.py", purpose="logic")]
    assert len(client.calls) == 2
    assert client.temperature_calls[1] > client.temperature_calls[0]
    assert "empty list" in client.calls[1]


def test_plan_files_does_not_retry_an_unreachable_host():
    # A dead connection isn't fixed by a blunter prompt or a hotter
    # temperature -- it should propagate immediately, not burn through
    # the same timeout two or three more times before giving up.
    client = FakeClient([OllamaError("Read timed out"), "should never be requested"])

    with pytest.raises(OllamaError):
        plan_files(client, "x", temperature=0.2, max_tokens=512, max_attempts=3)

    assert len(client.calls) == 1


def test_plan_files_raises_after_exhausting_retries():
    client = FakeClient([json.dumps({"files": []})] * 3)

    with pytest.raises(PlanError):
        plan_files(client, "x", temperature=0.2, max_tokens=512, max_attempts=3)


def test_plan_files_recovers_via_a_markdown_list_on_the_final_attempt():
    client = FakeClient(
        [
            json.dumps({"files": []}),  # schema attempt 0
            json.dumps({"files": []}),  # schema attempt 1
            "### Files\n1. **prime.py** - checks primality\n2. **primes.py** - lists primes\n",
        ]
    )

    tasks = plan_files(client, "primes", temperature=0.2, max_tokens=512, max_attempts=3)

    assert [t.path for t in tasks] == ["prime.py", "primes.py"]
    # the last call was unconstrained
    assert client.temperature_calls[-1] > client.temperature_calls[0]


def test_plan_files_drops_non_python_entries():
    payload = json.dumps(
        {
            "files": [
                {"path": "requirements.txt", "purpose": "deps"},
                {"path": "core.py", "purpose": "logic"},
                {"path": "README.md", "purpose": "docs"},
            ]
        }
    )
    client = FakeClient([payload])

    tasks = plan_files(client, "x", temperature=0.2, max_tokens=512)

    assert tasks == [FileTask(path="core.py", purpose="logic")]


def test_plan_files_raises_when_no_python_files_remain():
    payload = json.dumps({"files": [{"path": "setup.cfg", "purpose": "config"}]})
    client = FakeClient([payload])

    with pytest.raises(PlanError):
        plan_files(client, "x", temperature=0.2, max_tokens=512, max_attempts=1)


def test_plan_files_flattens_subdirectory_paths_to_bare_filenames():
    payload = json.dumps(
        {
            "files": [
                {"path": "pkg/core.py", "purpose": "logic"},
                {"path": "app/main.py", "purpose": "entry"},
            ]
        }
    )
    client = FakeClient([payload])

    tasks = plan_files(client, "x", temperature=0.2, max_tokens=512)

    assert tasks == [
        FileTask(path="core.py", purpose="logic"),
        FileTask(path="main.py", purpose="entry", depends_on=("core.py",)),
    ]


def test_plan_files_default_depends_on_is_every_earlier_file():
    payload = json.dumps(
        {
            "files": [
                {"path": "a.py", "purpose": "leaf"},
                {"path": "b.py", "purpose": "uses a"},
                {"path": "c.py", "purpose": "uses a and b"},
            ]
        }
    )
    tasks = plan_files(FakeClient([payload]), "x", temperature=0.2, max_tokens=512)

    assert tasks[0].depends_on == ()
    assert tasks[1].depends_on == ("a.py",)
    assert tasks[2].depends_on == ("a.py", "b.py")


def test_plan_files_uses_an_explicit_depends_on_when_given():
    payload = json.dumps(
        {
            "files": [
                {"path": "config.py", "purpose": "constants"},
                {"path": "core.py", "purpose": "logic"},
                {"path": "main.py", "purpose": "entry", "depends_on": ["core.py"]},
            ]
        }
    )
    tasks = plan_files(FakeClient([payload]), "x", temperature=0.2, max_tokens=512)

    assert tasks[2].depends_on == ("core.py",)  # not config.py


def test_plan_files_drops_forward_and_unknown_depends_on_entries():
    payload = json.dumps(
        {
            "files": [
                {"path": "a.py", "purpose": "leaf", "depends_on": ["b.py", "nope.py"]},
                {"path": "b.py", "purpose": "x", "depends_on": ["a.py"]},
            ]
        }
    )
    tasks = plan_files(FakeClient([payload]), "x", temperature=0.2, max_tokens=512)

    # a.py: forward dep b.py and unknown nope.py both dropped -> fall back to () (no earlier files)
    assert tasks[0].depends_on == ()
    assert tasks[1].depends_on == ("a.py",)


def test_plan_files_deduplicates_repeated_paths_keeping_the_first():
    payload = json.dumps(
        {
            "files": [
                {"path": "main.py", "purpose": "entry point"},
                {"path": "helper.py", "purpose": "helpers"},
                {"path": "main.py", "purpose": "entry point again"},
            ]
        }
    )
    client = FakeClient([payload])

    tasks = plan_files(client, "x", temperature=0.2, max_tokens=512)

    assert tasks == [
        FileTask(path="main.py", purpose="entry point"),
        FileTask(path="helper.py", purpose="helpers", depends_on=("main.py",)),
    ]
