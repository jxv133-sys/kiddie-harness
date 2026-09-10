from harness.steps.plan import FileTask
from harness.steps.spec import write_spec

from .fakes import FakeClient


def test_write_spec_returns_trimmed_response_text():
    client = FakeClient(["  - does a thing\n- does another thing  \n"])
    task = FileTask(path="core.py", purpose="Core logic")

    spec_text = write_spec(client, "build a todo app", task, temperature=0.2, max_tokens=256)

    assert spec_text == "- does a thing\n- does another thing"
    assert len(client.calls) == 1


def test_write_spec_strips_a_reasoning_models_think_block():
    client = FakeClient(
        ["<think>\nThe user wants bullets. Let me think about core.py.\n</think>\n- does a thing"]
    )
    task = FileTask(path="core.py", purpose="Core logic")

    spec_text = write_spec(client, "build a todo app", task, temperature=0.2, max_tokens=256)

    assert spec_text == "- does a thing"
