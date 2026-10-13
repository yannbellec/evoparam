Typing examples (spec REQ-301): one file per public entry point, checked by mypy in
evoparam/tests/test_typing.py. Files named wrong_*.py must fail the check. They are not
collected by pytest (no test_ prefix) and are never run.
