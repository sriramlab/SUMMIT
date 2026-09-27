"""Shared CLI parsing and validation, without importing numerical runtimes."""
from __future__ import annotations

import argparse
import math


class ArgumentParser(argparse.ArgumentParser):
    """Require complete option names, including during early runtime setup."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def explicit_options(argv):
    """Return explicitly supplied long options before the end-of-options marker."""
    result = set()
    for token in argv:
        if token == "--":
            break
        if token.startswith("--"):
            result.add(token.split("=", 1)[0])
    return result


def positive_gib(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("memory must be a finite positive GiB value")
    return number


def gib_bytes(value):
    number = positive_gib(value) * 2**30
    if not math.isfinite(number) or number < 1:
        raise argparse.ArgumentTypeError("memory must be at least one byte and finite")
    return int(number)
