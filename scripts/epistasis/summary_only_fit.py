"""Validation launcher: run the public fit command with cohort readers disabled."""
from unittest.mock import patch
import sys
from summit.epistasis.cli import main

if __name__ == "__main__":
    with patch(
        "summit.prediction.genotype.source_from_spec",
        side_effect=AssertionError("genotypes unavailable"),
    ), patch(
        "summit.prediction.cli._aligned_table",
        side_effect=AssertionError("phenotypes unavailable"),
    ):
        main(sys.argv[1:])
