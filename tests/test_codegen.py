from harness.steps.codegen import fix_file, generate_file

from .fakes import FakeClient


def test_generate_file_uses_python_rules_for_a_py_path():
    client = FakeClient(["x = 1\n"])

    generate_file(client, "goal", path="core.py", temperature=0.2, max_tokens=512)

    prompt = client.calls[0]
    assert "You are a Python code generator" in prompt
    assert 'if __name__ == "__main__"' in prompt
    assert "goal" in prompt


def test_generate_file_uses_html_rules_for_an_html_path():
    client = FakeClient(["<p>hi</p>"])

    generate_file(client, "goal", path="index.html", temperature=0.2, max_tokens=512)

    prompt = client.calls[0]
    assert "You are a HTML code generator" in prompt
    assert "well-formed HTML document" in prompt
    assert 'if __name__ == "__main__"' not in prompt


def test_generate_file_uses_css_rules_for_a_css_path():
    client = FakeClient(["body {}"])

    generate_file(client, "goal", path="style.css", temperature=0.2, max_tokens=512)

    prompt = client.calls[0]
    assert "You are a CSS code generator" in prompt
    assert "declaration blocks" in prompt


def test_generate_file_uses_javascript_rules_for_a_js_path():
    client = FakeClient(["function f() {}"])

    generate_file(client, "goal", path="app.js", temperature=0.2, max_tokens=512)

    prompt = client.calls[0]
    assert "You are a JavaScript code generator" in prompt
    assert "no Node-only APIs" in prompt


def test_generate_file_uses_batch_rules_for_a_bat_path():
    client = FakeClient(["@echo off"])

    generate_file(client, "goal", path="setup.bat", temperature=0.2, max_tokens=512)

    prompt = client.calls[0]
    assert "You are a Batch code generator" in prompt
    assert "@echo off" in prompt
    assert 'if __name__ == "__main__"' not in prompt


def test_generate_file_uses_batch_rules_for_a_cmd_path():
    client = FakeClient(["@echo off"])

    generate_file(client, "goal", path="setup.cmd", temperature=0.2, max_tokens=512)

    assert "You are a Batch code generator" in client.calls[0]


def test_generate_file_uses_powershell_rules_for_a_ps1_path():
    client = FakeClient(["Write-Host 'hi'"])

    generate_file(client, "goal", path="deploy.ps1", temperature=0.2, max_tokens=512)

    prompt = client.calls[0]
    assert "You are a PowerShell code generator" in prompt
    assert "Verb-Noun" in prompt


def test_generate_file_falls_back_to_python_for_an_unrecognized_extension():
    client = FakeClient(["x = 1\n"])

    generate_file(client, "goal", path="notes.txt", temperature=0.2, max_tokens=512)

    assert "You are a Python code generator" in client.calls[0]


def test_fix_file_names_the_right_language_in_its_prompt():
    client = FakeClient(["body { color: red; }"])

    fix_file(
        client,
        code="body { color: red;",
        error="unclosed block",
        stage="compile",
        path="style.css",
        temperature=0.2,
        max_tokens=512,
    )

    prompt = client.calls[0]
    assert "You are a CSS code fixer" in prompt
    assert "unclosed block" in prompt


def test_generated_code_strips_fences_regardless_of_language():
    client = FakeClient(["```html\n<p>hi</p>\n```"])

    result = generate_file(client, "goal", path="index.html", temperature=0.2, max_tokens=512)

    assert result.code == "<p>hi</p>"
