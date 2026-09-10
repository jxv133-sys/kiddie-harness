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


def test_extracts_fenced_block_when_prose_surrounds_it():
    text = "Here is the code:\n```python\nprint('hi')\n```"
    assert strip_code_fences(text) == "print('hi')"


def test_extracts_fenced_block_after_prose_and_trailing_note():
    text = "Sure, here you go:\n```python\nprint('hi')\n```\nLet me know if that works."
    assert strip_code_fences(text) == "print('hi')"


def test_takes_last_fenced_block_when_several_present():
    text = (
        "First attempt:\n```python\nprint('draft')\n```\n"
        "Actually this is better:\n```python\nprint('final')\n```"
    )
    assert strip_code_fences(text) == "print('final')"


def test_drops_orphan_think_close_tag_then_extracts_code():
    # Reasoning models (e.g. deepseek-style) emit chain-of-thought that ends
    # with a bare "</think>" (the opening tag consumed by Ollama's template),
    # followed by the real answer in a fenced block.
    text = (
        "We need to print the first 20 Fibonacci numbers. I'll write a loop.\n"
        "</think>\n\n"
        "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
        "        print(a)\n        a, b = b, a + b\n\n\nfib(20)\n```"
    )
    assert strip_code_fences(text) == (
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
        "        print(a)\n        a, b = b, a + b\n\n\nfib(20)"
    )


def test_drops_paired_think_block_then_returns_bare_code():
    text = "<think>\nplan: just print it\n</think>\nprint('hi')"
    assert strip_code_fences(text) == "print('hi')"


def test_think_close_tag_with_no_fence_returns_remaining_text():
    text = "reasoning about the fix\n</think>\nprint('fixed')"
    assert strip_code_fences(text) == "print('fixed')"
