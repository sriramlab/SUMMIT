"""Shared CLI spelling and validation, without importing numerical runtimes."""
from __future__ import annotations

import argparse
from copy import copy
import math


class CanonicalHelpFormatter(argparse.HelpFormatter):
    """Show one spelling per option while accepting compatibility aliases."""

    def _format_action_invocation(self, action):
        if len(action.option_strings) > 1 and action.option_strings[0].startswith("--"):
            action = copy(action)
            action.option_strings = action.option_strings[:1]
        return super()._format_action_invocation(action)


def option_action(parser, option):
    action = parser._option_string_actions.get(option)
    if action is None and parser.allow_abbrev:
        matches = [a for name, a in parser._option_string_actions.items() if name.startswith(option)]
        if len(matches) == 1:
            action = matches[0]
    return action


def explicit_options(parser, argv):
    """Resolve aliases/accepted abbreviations for mode-specific validation."""
    result = set()
    for token in argv:
        if token == "--":
            break
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            action = option_action(parser, option)
            result.add(action.option_strings[0] if action is not None else option)
    return result


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", CanonicalHelpFormatter)
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args=None, namespace=None):
        import sys
        tokens = list(sys.argv[1:] if args is None else args)
        parsed, rest = super().parse_known_args(tokens, namespace)
        seen = {}
        for index, token in enumerate(tokens):
            if token == "--":
                break
            option, sep, raw = token.partition("=")
            action = option_action(self, option) if option.startswith("--") else None
            if action is None or len(action.option_strings) < 2 or action.nargs is not None:
                continue
            # These aliases take one scalar value. Argparse has already checked
            # arity and type, including --option=value and negative numbers.
            raw = raw if sep else tokens[index + 1]
            value = action.type(raw) if action.type else raw
            previous = seen.get(action.dest)
            if previous is not None and previous[0] != option and previous[1] != value:
                self.error(f"conflicting values for aliases {previous[0]} and {option}; use {action.option_strings[0]} once")
            seen[action.dest] = (option, value)
        return parsed, rest


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
