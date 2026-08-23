#!/usr/bin/env python3
"""Create current-format toy inputs from the bundled example files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def _write_sumstats() -> None:
    src = HERE / "sim_50k_h2_0.25_p_0.01.sumstat"
    base = OUT / "sim_50k_h2_0.25_p_0.01.beta_se.sumstat"
    trait_a = OUT / "trait_a.sumstats"
    trait_b = OUT / "trait_b.sumstats"

    df = pd.read_csv(src, sep=r"\s+")
    if {"BETA", "SE"}.issubset(df.columns):
        out = df.loc[:, ["SNP", "A1", "A2", "N", "BETA", "SE"]].copy()
    elif "Z" in df.columns:
        n = pd.to_numeric(df["N"], errors="raise").to_numpy(dtype=np.float64)
        z = pd.to_numeric(df["Z"], errors="raise").to_numpy(dtype=np.float64)
        se = 1.0 / np.sqrt(n)
        beta = z * se
        out = df.loc[:, ["SNP", "A1", "A2", "N"]].copy()
        out["BETA"] = beta
        out["SE"] = se
    else:
        raise RuntimeError(f"{src} has neither BETA/SE nor Z columns.")

    out.to_csv(base, sep="\t", index=False, float_format="%.10g")
    out.to_csv(trait_a, sep="\t", index=False, float_format="%.10g")

    out.to_csv(trait_b, sep="\t", index=False, float_format="%.10g")


def _write_manifest() -> None:
    manifest = OUT / "rg_manifest.fixed1.tsv"
    trait_a = (OUT / "trait_a.sumstats").resolve()
    trait_b = (OUT / "trait_b.sumstats").resolve()
    rows = pd.DataFrame(
        [
            {
                "phen1": "trait_a",
                "phen2": "trait_b",
                "sumstats1": str(trait_a),
                "sumstats2": str(trait_b),
                "overlap_covariance": 1.0,
                "cov_rank1": 0,
                "cov_rank2": 0,
            }
        ]
    )
    rows.to_csv(manifest, sep="\t", index=False)


def _write_environment() -> None:
    env_path = OUT / "small.env"
    if env_path.exists():
        return

    fam = pd.read_csv(HERE / "small.fam", sep=r"\s+", header=None, usecols=[0, 1])
    fam.columns = ["FID", "IID"]
    idx = np.arange(fam.shape[0], dtype=np.float64)
    fam["ENV"] = np.sin(idx / 11.0)
    fam.to_csv(env_path, sep="\t", index=False, float_format="%.8g")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_sumstats()
    _write_manifest()
    _write_environment()
    print(f"Prepared example inputs under {OUT}")


if __name__ == "__main__":
    main()
