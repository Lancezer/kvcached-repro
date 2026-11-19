"""Attach cleanup context without replacing the original exception."""
import json
import sys
from contextlib import contextmanager
from traceback import format_exception


@contextmanager
def cleanup_operation(operation, **target):
    try:
        yield
    except Exception as error:
        # Outer operations fill missing fields while retaining the innermost failure site.
        outer = {key: value for key, value in target.items() if value is not None}
        error.cleanup_context = dict(outer, operation=operation) | getattr(error, 'cleanup_context', {})
        raise


def cleanup_diagnostic(error, **target):
    context = {key: value for key, value in target.items() if value is not None}
    return dict(context | getattr(error, 'cleanup_context', {}),
                traceback=''.join(format_exception(type(error), error, error.__traceback__)))


def emit_cleanup_diagnostic(error, **target):
    # A failed diagnostic stream must not replace the cleanup exception.
    try:
        print(json.dumps({'cleanup_diagnostic': cleanup_diagnostic(error, **target)}),
              file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass
