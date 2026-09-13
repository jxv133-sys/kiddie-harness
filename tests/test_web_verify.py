from pathlib import Path

from harness.steps.verify import css_check, html_check, js_check, verify_generated_file


def test_html_check_passes_on_well_formed_markup(tmp_path: Path):
    f = tmp_path / "index.html"
    f.write_text("<html><head><title>x</title></head><body><p>hi</p></body></html>")
    result = html_check(f)
    assert result.success
    assert result.stage == "compile"


def test_html_check_fails_on_an_unclosed_tag(tmp_path: Path):
    f = tmp_path / "index.html"
    f.write_text("<html><body><div><p>hi</p></body></html>")
    result = html_check(f)
    assert not result.success
    assert "div" in result.output


def test_html_check_fails_on_a_stray_closing_tag(tmp_path: Path):
    f = tmp_path / "index.html"
    f.write_text("<html><body></p></body></html>")
    result = html_check(f)
    assert not result.success
    assert "</p>" in result.output


def test_html_check_ignores_void_elements(tmp_path: Path):
    f = tmp_path / "index.html"
    f.write_text('<html><body><img src="x.png"><br><input type="text"></body></html>')
    result = html_check(f)
    assert result.success


def test_html_check_ignores_self_closing_tags(tmp_path: Path):
    f = tmp_path / "index.html"
    f.write_text("<html><body><div /></body></html>")
    result = html_check(f)
    assert result.success


def test_html_check_handles_mismatched_nesting(tmp_path: Path):
    # <b> is closed before <i>, which was opened after it -- a real,
    # common small-model mistake.
    f = tmp_path / "index.html"
    f.write_text("<p><b>bold <i>and italic</b></i></p>")
    result = html_check(f)
    assert not result.success


def test_css_check_passes_on_balanced_rules(tmp_path: Path):
    f = tmp_path / "style.css"
    f.write_text("body { color: red; }\n.card { padding: 4px; }\n")
    result = css_check(f)
    assert result.success
    assert result.stage == "compile"


def test_css_check_fails_on_an_unclosed_block(tmp_path: Path):
    f = tmp_path / "style.css"
    f.write_text("body { color: red;\n.card { padding: 4px; }\n")
    result = css_check(f)
    assert not result.success


def test_css_check_fails_on_an_unterminated_string(tmp_path: Path):
    f = tmp_path / "style.css"
    f.write_text('body::before { content: "unterminated; }\n')
    result = css_check(f)
    assert not result.success


def test_css_check_ignores_comments(tmp_path: Path):
    f = tmp_path / "style.css"
    f.write_text("/* a { b: } unbalanced inside a comment */\nbody { color: red; }\n")
    result = css_check(f)
    assert result.success


def test_js_check_passes_on_balanced_code(tmp_path: Path):
    f = tmp_path / "app.js"
    f.write_text("function f(x) {\n  return [x, {a: 1}];\n}\n")
    result = js_check(f)
    assert result.success
    assert result.stage == "compile"


def test_js_check_fails_on_an_unclosed_brace(tmp_path: Path):
    f = tmp_path / "app.js"
    f.write_text("function f(x) {\n  return x;\n")
    result = js_check(f)
    assert not result.success


def test_js_check_fails_on_an_unterminated_string(tmp_path: Path):
    f = tmp_path / "app.js"
    f.write_text("const s = 'unterminated;\n")
    result = js_check(f)
    assert not result.success


def test_js_check_ignores_line_and_block_comments(tmp_path: Path):
    f = tmp_path / "app.js"
    f.write_text("// unbalanced { here\n/* also ( unbalanced */\nfunction f() {}\n")
    result = js_check(f)
    assert result.success


def test_verify_generated_file_routes_by_extension(tmp_path: Path):
    html = tmp_path / "index.html"
    html.write_text("<p>ok</p>")
    css = tmp_path / "style.css"
    css.write_text("body { color: red; }")
    js = tmp_path / "app.js"
    js.write_text("function f() {}")
    py = tmp_path / "main.py"
    py.write_text("def broken(:\n")

    assert verify_generated_file(html).success
    assert verify_generated_file(css).success
    assert verify_generated_file(js).success
    # falls through to the Python pipeline for everything else
    py_result = verify_generated_file(py)
    assert not py_result.success
    assert py_result.stage == "compile"
