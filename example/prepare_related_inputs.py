#!/usr/bin/env python3
"""Generate a synthetic cohort with sibling pairs for the related-individuals example.

1,200 people: 400 sibling pairs and 400 unrelated individuals, 2,048 SNPs on four chromosomes, two covariates and two
traits. Siblings share half of their parents' haplotypes, and a shared family effect makes their phenotypes more
alike than their SNPs explain. Writes BED genotypes, covariates, phenotypes, the sibling pair list and OLS summary
statistics with the covariate rank, all from a fixed seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "related"
SEED, PAIRS, SINGLES, VARIANTS = 20261025, 400, 400, 2048
FILES = ("cohort.bed", "cohort.bim", "cohort.fam", "cohort.cov", "traits.tsv", "pairs.tsv",
         "trait_a.tsv", "trait_b.tsv", "trait_a.sumstats", "trait_b.sumstats")


def ols_sumstats(calls, y, covariates, snp):
    """PLINK-style marginal OLS with the covariates: BETA, SE and COV_RANK per SNP."""
    n = calls.shape[0]
    q = np.linalg.qr(np.column_stack((np.ones(n), covariates)), mode="reduced")[0]
    g = calls - q @ (q.T @ calls)
    r = y - q @ (q.T @ y)
    xx = np.sum(g * g, axis=0)
    beta = (g.T @ r) / xx
    df = n - q.shape[1] - 1
    se = np.sqrt(np.maximum(r @ r - beta * beta * xx, 0.0) / df / xx)
    return pd.DataFrame({"SNP": snp, "A1": "A", "A2": "C", "N": n, "BETA": beta, "SE": se,
                         "COV_RANK": covariates.shape[1]})


def generate(output: Path) -> None:
    output = output.resolve()
    description = {"kind": "summit.related_example", "version": 1, "seed": SEED,
                   "pairs": PAIRS, "singles": SINGLES, "variants": VARIANTS}
    marker = output / "related.json"
    if output.exists():
        if (marker.is_file() and json.loads(marker.read_text()) == description
                and all((output / name).is_file() for name in FILES)):
            return
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    m = VARIANTS
    freq = rng.uniform(0.1, 0.5, m)
    hap = lambda k: (rng.random((k, m)) < freq).astype(np.int8)
    # Each sibling draws one of the two haplotypes of each parent.
    mother, father = hap(2 * PAIRS).reshape(PAIRS, 2, m), hap(2 * PAIRS).reshape(PAIRS, 2, m)
    sibs = []
    for _ in range(2):
        pick_m, pick_f = rng.integers(0, 2, (PAIRS, 1, m)), rng.integers(0, 2, (PAIRS, 1, m))
        sibs.append(np.take_along_axis(mother, pick_m, 1)[:, 0] + np.take_along_axis(father, pick_f, 1)[:, 0])
    singles = hap(SINGLES) + hap(SINGLES)
    calls = np.vstack([np.stack(sibs, 1).reshape(2 * PAIRS, m), singles]).astype(np.float64)
    n = calls.shape[0]
    iid = np.array([f"SIM{i:06d}" for i in range(n)])
    snp = np.array([f"sim_variant_{j:06d}" for j in range(m)])
    chromosome = 1 + np.arange(m) // (m // 4)
    bp = 1000 * (1 + np.arange(m) % (m // 4))
    to_bed(str(output / "cohort.bed"), calls, properties={
        "fid": iid, "iid": iid, "sid": snp, "chromosome": chromosome,
        "bp_position": bp, "allele_1": ["A"] * m, "allele_2": ["C"] * m,
    }, count_A1=True)
    covariates = rng.standard_normal((n, 2))
    pd.DataFrame({"FID": iid, "IID": iid, "cov1": covariates[:, 0], "cov2": covariates[:, 1]}).to_csv(
        output / "cohort.cov", sep="\t", index=False)
    pd.DataFrame({"IID1": iid[0:2 * PAIRS:2], "IID2": iid[1:2 * PAIRS:2], "RELATEDNESS": 0.5}).to_csv(
        output / "pairs.tsv", sep="\t", index=False)
    # Two traits with h2 0.3 and genetic correlation 0.5; siblings also share a family effect (variance 0.3).
    std = (calls - calls.mean(0)) / calls.std(0, ddof=1)
    b1 = rng.standard_normal(m) / np.sqrt(m)
    b2 = 0.5 * b1 + np.sqrt(0.75) * rng.standard_normal(m) / np.sqrt(m)
    family = np.concatenate([np.repeat(rng.standard_normal(PAIRS), 2), rng.standard_normal(SINGLES)])
    traits = {}
    for name, b in (("trait_a", b1), ("trait_b", b2)):
        g = std @ b
        y = np.sqrt(0.3) * g / g.std() + np.sqrt(0.3) * family + np.sqrt(0.4) * rng.standard_normal(n)
        y += covariates @ np.array([0.2, -0.1])
        traits[name] = y
        pd.DataFrame({"FID": iid, "IID": iid, "y": y}).to_csv(output / f"{name}.tsv", sep="\t", index=False)
        ols_sumstats(calls, y, covariates, snp).to_csv(output / f"{name}.sumstats", sep="\t", index=False)
    pd.DataFrame({"FID": iid, "IID": iid, **traits}).to_csv(output / "traits.tsv", sep="\t", index=False)
    marker.write_text(json.dumps(description, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.out)
    print(f"Related-individuals example ready: {args.out.resolve()}")


if __name__ == "__main__":
    main()
