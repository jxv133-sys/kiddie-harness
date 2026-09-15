from harness.steps.plan import FileTask
from harness.steps.spec import looks_like_a_spec, write_spec

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


def test_looks_like_a_spec_accepts_real_bullet_points():
    assert looks_like_a_spec("- does a thing\n- does another thing")
    assert looks_like_a_spec("some preamble\n- a bullet buried after it")


def test_looks_like_a_spec_rejects_disclaimer_prose():
    # A real, observed failure mode: a small model responding with
    # commentary about the task instead of an actual spec.
    assert not looks_like_a_spec(
        "Sure, here's a specification for the file based on your request."
    )
    assert not looks_like_a_spec("")
