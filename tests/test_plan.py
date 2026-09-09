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
        plan_files(client, "build a todo app", temperature=0.2, max_tokens=512)


def test_plan_files_raises_on_empty_file_list():
    client = FakeClient([json.dumps({"files": []})])

    with pytest.raises(PlanError):
        plan_files(client, "build a todo app", temperature=0.2, max_tokens=512)
