#!/usr/bin/env python3
"""Generate synthetic genotypes, covariates, and GWAS statistics for examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "synthetic"
SEED, SAMPLES, VARIANTS = 20260909, 512, 2048
FILES = ("small.bed", "small.bim", "small.fam", "small.cov", "small.env",
         "small.annot", "trait_a.sumstats", "trait_b.sumstats", "traits.tsv", "rg_manifest.tsv")


def generate(output: Path) -> None:
    output = output.resolve()
    description = {"kind": "summit.synthetic_examples", "version": 1,
                   "seed": SEED, "samples": SAMPLES, "variants": VARIANTS}
    marker = output / "synthetic.json"
    if output.exists():
        if (marker.is_file() and json.loads(marker.read_text()) == description
                and all((output / name).is_file() for name in FILES)):
            return
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    n, m = SAMPLES, VARIANTS
    # All values are generated from the seed; no external data are read.
    frequencies = rng.uniform(0.1, 0.5, m)
    calls = rng.binomial(2, frequencies, size=(n, m)).astype(np.float64)
    iid = np.array([f"SIM{i:06d}" for i in range(n)])
    snp = np.array([f"sim_variant_{j:06d}" for j in range(m)])
    chromosome = 1 + np.arange(m) // (m // 4)
    bp = 1000 * (1 + np.arange(m) % (m // 4))
    to_bed(str(output / "small.bed"), calls, properties={
        "fid": iid, "iid": iid, "sid": snp, "chromosome": chromosome,
        "bp_position": bp, "allele_1": ["A"] * m, "allele_2": ["C"] * m,
    }, count_A1=True)
    covariates = rng.standard_normal((n, 2))
    environment = rng.standard_normal(n)
    pd.DataFrame({"FID": iid, "IID": iid, "cov1": covariates[:, 0],
                  "cov2": covariates[:, 1]}).to_csv(output / "small.cov", sep="\t", index=False)
    pd.DataFrame({"FID": iid, "IID": iid, "environment": environment}).to_csv(
        output / "small.env", sep="\t", index=False)
    annotation = (np.arange(m) % 2).astype(int)
    pd.DataFrame({"CHR": chromosome, "BP": bp, "SNP": snp,
                  "bin1": 1 - annotation, "bin2": annotation}).to_csv(
        output / "small.annot", sep="\t", index=False)
    z = np.column_stack((np.ones(n), covariates))
    u = np.linalg.qr(z, mode="reduced")[0]
    g = calls - u @ (u.T @ calls)
    g /= np.std(g, axis=0, ddof=z.shape[1])
    effects = rng.standard_normal(m) / np.sqrt(m)
    y = np.sqrt(0.4) * (g @ effects) + np.sqrt(0.6) * rng.standard_normal(n)
    y -= u @ (u.T @ y)
    y /= np.std(y, ddof=z.shape[1])
    xx = np.sum(g * g, axis=0)
    xy = g.T @ y
    beta = xy / xx
    se = np.sqrt(np.maximum(y @ y - xy * beta, 0.0) / (n - z.shape[1] - 1) / xx)
    summary = pd.DataFrame({"SNP": snp, "A1": "A", "A2": "C", "N": n,
                            "BETA": beta, "SE": se, "COV_RANK": 2})
    # Identical traits make the supplied-overlap example's value of 1 explicit.
    for name in ("trait_a", "trait_b"):
        summary.to_csv(output / f"{name}.sumstats", sep="\t", index=False)
    pd.DataFrame({"FID": iid, "IID": iid, "trait_a": y, "trait_b": y}).to_csv(
        output / "traits.tsv", sep="\t", index=False)
    pd.DataFrame([{"phen1": "trait_a", "phen2": "trait_b",
                   "sumstats1": str(output / "trait_a.sumstats"),
                   "sumstats2": str(output / "trait_b.sumstats"),
                   "overlap_covariance": 1.0, "cov_rank1": 2,
                   "cov_rank2": 2}]).to_csv(output / "rg_manifest.tsv", sep="\t", index=False)
    marker.write_text(json.dumps(description, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.out)
    print(f"Synthetic examples ready: {args.out.resolve()}")


if __name__ == "__main__":
    main()
