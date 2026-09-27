"""Select a command before its thread/NUMA bootstrap or numerical imports."""
from __future__ import annotations

import sys


def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens[:1] == ["pgs"]:
        from .prediction.cli import main as pgs_main
        return pgs_main(tokens[1:], prog="summit pgs")
    if tokens[:1] == ["reference"] and tokens[1:2] != ["zpass"]:
        from .ldscore.generalized_gxe_variant_cli import main as reference_main
        return reference_main(tokens[1:], prog="summit reference")
    # The legacy CLI reads argv during import to establish its runtime. Keep
    # that ordering, and retain it for the existing reference zpass command.
    original = sys.argv
    try:
        sys.argv = [original[0], *tokens]
        from .cli import main as legacy_main
        return legacy_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
