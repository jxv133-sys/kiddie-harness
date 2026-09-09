from harness.postprocess import strip_code_fences


def test_strips_fenced_block_with_language_tag():
    text = "```python\nprint('hi')\n```"
    assert strip_code_fences(text) == "print('hi')"


def test_strips_fenced_block_without_language_tag():
    text = "```\nprint('hi')\n```"
    assert strip_code_fences(text) == "print('hi')"


def test_leaves_unfenced_code_untouched():
    text = "print('hi')"
    assert strip_code_fences(text) == "print('hi')"


def test_trims_surrounding_whitespace():
    text = "\n\n  print('hi')  \n\n"
    assert strip_code_fences(text) == "print('hi')"


def test_does_not_strip_when_prose_surrounds_fence():
    text = "Here is the code:\n```python\nprint('hi')\n```"
    result = strip_code_fences(text)
    assert "print('hi')" in result
