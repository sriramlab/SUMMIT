#!/usr/bin/env python3
"""Convert biallelic PLINK 2 linear-regression results to SUMMIT inputs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cov-rank",
        type=int,
        required=True,
        help="Rank of the non-intercept GWAS covariates.",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new --out path.")
    if args.cov_rank < 0:
        parser.error("--cov-rank must be nonnegative.")

    table = pd.read_csv(args.glm, sep="\t", dtype=str)
    table.columns = table.columns.str.lstrip("#")
    required = {"ID", "A1", "REF", "ALT", "OBS_CT", "BETA", "SE", "TEST"}
    missing = required - set(table.columns)
    if missing:
        parser.error(f"Missing linear-regression columns: {', '.join(sorted(missing))}")
    table = table.loc[table["TEST"].eq("ADD")].copy()
    if table.empty:
        parser.error("No additive-variant tests found.")
    if table["ALT"].str.contains(",", regex=False).any():
        parser.error("Use biallelic variants.")
    if not (table["A1"].eq(table["REF"]) | table["A1"].eq(table["ALT"])).all():
        parser.error("An effect allele is absent from REF/ALT.")
    if table["ID"].duplicated().any():
        parser.error("Variant IDs must be unique.")

    beta = pd.to_numeric(table["BETA"], errors="coerce")
    se = pd.to_numeric(table["SE"], errors="coerce")
    result = pd.DataFrame(
        {
            "SNP": table["ID"],
            "A1": table["A1"],
            "A2": np.where(table["A1"].eq(table["REF"]), table["ALT"], table["REF"]),
            "N": pd.to_numeric(table["OBS_CT"], errors="coerce"),
            "BETA": beta,
            "SE": se,
            "Z": beta / se,
            "COV_RANK": args.cov_rank,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(result)} variants to {args.out}")


if __name__ == "__main__":
    main()
