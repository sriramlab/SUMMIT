"""Compatibility launcher for installed supplied-target input preparation.

The implementation lives in summit.epistasis.inputs and is also available as
``summit epistasis make-inputs recipe.json --out inputs``.
"""
import argparse
from pathlib import Path
from summit.epistasis.inputs import build_trans_inputs as build


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--genome-build", default="GRCh37")
    p.add_argument("--target", required=True)
    background = p.add_mutually_exclusive_group(required=True)
    background.add_argument("--background-chromosome")
    background.add_argument("--interaction-variants", type=Path)
    p.add_argument("--training-samples", type=Path, required=True)
    p.add_argument("--confirmation-samples", type=Path, required=True)
    p.add_argument("--phenotypes", type=Path, required=True)
    p.add_argument("--phenotype-column", required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--covariates", type=Path, required=True)
    p.add_argument("--covariate-columns", default="")
    p.add_argument(
        "--structure-covariates", default="PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10"
    )
    p.add_argument("--local-window-bp", type=int, default=100000)
    p.add_argument("--minimum-cell-fraction", type=float, default=0.02)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
