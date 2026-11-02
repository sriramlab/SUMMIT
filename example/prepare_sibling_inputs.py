#!/usr/bin/env python3
"""Generate synthetic sibships for the direct-effects example.

300 extended families, each with two sibships whose mothers are sisters (cousins), two or three siblings per sibship,
2,000 SNPs on two chromosomes and one trait. The trait has direct heritability 0.4. Siblings also share an environment
(variance 0.2) and an effect of their parents' genotypes (variance 0.1, a second polygenic score correlated 0.5 with
the direct one), which a population estimate absorbs and the direct estimate does not. Writes BED genotypes, the
sibship table, an annotation with two MAF classes, covariates and the trait, all from a fixed seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "siblings"
SEED, FAMILIES, VARIANTS = 20261102, 300, 2000
FILES = ("siblings.bed", "siblings.bim", "siblings.fam", "sibships.tsv", "siblings.annot", "siblings.cov", "traits.tsv")


def child(rng, mother, father):
    """One child's two haplotypes from each parent's pair of haplotypes (unlinked SNPs)."""
    m = mother.shape[1]
    return np.stack([mother[rng.integers(0, 2, m), np.arange(m)], father[rng.integers(0, 2, m), np.arange(m)]])


def generate(output: Path) -> None:
    output = output.resolve()
    description = {"kind": "summit.sibling_example", "version": 1, "seed": SEED, "families": FAMILIES}
    marker = output / "siblings.json"
    if output.exists():
        if (marker.is_file() and json.loads(marker.read_text()) == description
                and all((output / name).is_file() for name in FILES)):
            return
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    m = VARIANTS
    freq = rng.uniform(0.05, 0.5, m)
    founder = lambda: (rng.random((2, m)) < freq).astype(np.int8)
    haps, parents, rows = [], [], []
    for f in range(FAMILIES):
        grandmother, grandfather = founder(), founder()
        for s in range(2):                           # two sisters, each with an unrelated husband
            mother, father = child(rng, grandmother, grandfather), founder()
            for _ in range(rng.integers(2, 4)):
                haps.append(child(rng, mother, father))
                parents.append(mother.sum(0) + father.sum(0))
                rows.append((f"SIM{len(rows):06d}", f"{f}_{s}", f"{f}"))
    G = np.array([h.sum(0) for h in haps], dtype=np.float64)
    n = G.shape[0]
    iid = np.array([r[0] for r in rows])
    chromosome = 1 + np.arange(m) // (m // 2)
    bp = 1000 * (1 + np.arange(m) % (m // 2))
    snp = np.array([f"sim_variant_{j:06d}" for j in range(m)])
    to_bed(str(output / "siblings.bed"), G, properties={
        "fid": iid, "iid": iid, "sid": snp, "chromosome": chromosome, "bp_position": bp,
        "allele_1": ["A"] * m, "allele_2": ["C"] * m}, count_A1=True)
    sib = pd.DataFrame(rows, columns=["IID", "SIBSHIP", "FAMILY"])
    sib.to_csv(output / "sibships.tsv", sep="\t", index=False)
    maf = np.minimum(G.mean(0) / 2, 1 - G.mean(0) / 2)
    pd.DataFrame({"CHR": chromosome, "BP": bp, "SNP": snp, "CM": 0, "MAF_LOW": (maf < 0.2).astype(int),
                  "MAF_HIGH": (maf >= 0.2).astype(int)}).to_csv(output / "siblings.annot", sep="\t", index=False)
    age = rng.uniform(40, 70, n)
    pd.DataFrame({"FID": iid, "IID": iid, "age": age, "PC1": rng.standard_normal(n)}).to_csv(
        output / "siblings.cov", sep="\t", index=False)
    std = lambda X: (X - X.mean(0)) / X.std(0)
    b = rng.standard_normal(m)
    b2 = 0.5 * b + np.sqrt(0.75) * rng.standard_normal(m)
    direct = std(G) @ b
    parental = std(np.array(parents, dtype=float)) @ b2         # an effect of the parents' genotypes
    shared = rng.standard_normal(sib.SIBSHIP.nunique())[pd.factorize(sib.SIBSHIP)[0]]
    y = (np.sqrt(0.4) * direct / direct.std() + np.sqrt(0.1) * parental / parental.std() + np.sqrt(0.2) * shared
         + np.sqrt(0.3) * rng.standard_normal(n) + 0.02 * age)
    pd.DataFrame({"FID": iid, "IID": iid, "trait": y}).to_csv(output / "traits.tsv", sep="\t", index=False)
    marker.write_text(json.dumps(description, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.out)
    print(f"Sibling example ready: {args.out.resolve()}")


if __name__ == "__main__":
    main()
