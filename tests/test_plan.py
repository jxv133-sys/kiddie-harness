import json

import pytest

from harness.steps.plan import FileTask, PlanError, plan_files

from .fakes import FakeClient


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
        FileTask(path="core.py", purpose="Core logic"),
    ]


def test_plan_files_raises_on_invalid_json():
    client = FakeClient(["not json"])

    with pytest.raises(PlanError):
        plan_files(client, "build a todo app", temperature=0.2, max_tokens=512, max_attempts=1)


def test_plan_files_raises_on_empty_file_list():
    client = FakeClient([json.dumps({"files": []})])

    with pytest.raises(PlanError):
        plan_files(client, "build a todo app", temperature=0.2, max_tokens=512, max_attempts=1)


def test_plan_files_retries_without_schema_when_the_first_plan_is_empty():
    client = FakeClient(
        [
            json.dumps({"files": []}),  # schema-constrained attempt: degenerate
            json.dumps({"files": [{"path": "core.py", "purpose": "logic"}]}),  # retry
        ]
    )

    tasks = plan_files(client, "x", temperature=0.2, max_tokens=512)

    assert tasks == [FileTask(path="core.py", purpose="logic")]
    assert len(client.calls) == 2
    assert client.temperature_calls[1] > client.temperature_calls[0]


def test_plan_files_extracts_json_from_a_reasoning_wrapped_retry():
    client = FakeClient(
        [
            json.dumps({"files": []}),
            (
                "<think>\nThe user needs one module.\n</think>\n"
                '```json\n{"files": [{"path": "core.py", "purpose": "logic"}]}\n```'
            ),
        ]
    )

    tasks = plan_files(client, "x", temperature=0.2, max_tokens=512)

    assert tasks == [FileTask(path="core.py", purpose="logic")]


def test_plan_files_raises_after_exhausting_retries():
    client = FakeClient([json.dumps({"files": []})] * 3)

    with pytest.raises(PlanError):
        plan_files(client, "x", temperature=0.2, max_tokens=512, max_attempts=3)


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
        FileTask(path="main.py", purpose="entry"),
    ]


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
        FileTask(path="helper.py", purpose="helpers"),
    ]
