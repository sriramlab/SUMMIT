#!/usr/bin/env python3
"""Generate a synthetic admixed cohort for the local-ancestry example.

2,000 people with two source populations, 22 short chromosomes of 150 SNPs each, local-ancestry calls in admix-kit
.lanc format and one trait. Each haplotype is a mosaic of ancestry tracts around the person's global ancestry, and
alleles follow the frequency of the tract's ancestry. The trait has SNP heritability 0.3 and a further 0.1 from
untagged variants whose frequencies differ between the populations, which only local ancestry tags.

  python prepare_local_ancestry_inputs.py               write the inputs under out/local_ancestry/
  python prepare_local_ancestry_inputs.py --sumstats    after `summit local-ancestry markers` and `merge`: the
                                                        covariates with THETA and OLS summary statistics of the
                                                        SNPs alone and of SNPs and local-ancestry markers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import open_bed, to_bed

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "local_ancestry"
SEED, N, CHROMOSOMES, SITES, STEP = 20261102, 2000, 22, 150, 25_000
SWITCH = 0.02            # probability that a tract ends between neighboring sites (mean tract 1.25 Mb)
UNTAGGED = 400           # untagged variants per genome
FILES = ("cohort.bed", "cohort.bim", "cohort.fam", "cohort.cov", "cohort.annot", "trait.tsv")


def ols_sumstats(calls, y, covariates, snp):
    """PLINK-style marginal OLS with the covariates: BETA, SE and COV_RANK per marker."""
    n = calls.shape[0]
    q = np.linalg.qr(np.column_stack((np.ones(n), covariates)), mode="reduced")[0]
    g = calls - q @ (q.T @ calls)
    r = y - q @ (q.T @ y)
    xx = np.sum(g * g, axis=0)
    beta = (g.T @ r) / xx
    se = np.sqrt(np.maximum(r @ r - beta * beta * xx, 0.0) / (n - q.shape[1] - 1) / xx)
    return pd.DataFrame({"SNP": snp, "A1": "A", "A2": "C", "N": n, "BETA": beta, "SE": se,
                         "COV_RANK": covariates.shape[1]})


def tracts(rng, theta, length):
    """Ancestry (1 = population 1) along `length` sites for one haplotype of each person."""
    anc = np.empty((theta.size, length), dtype=np.int8)
    anc[:, 0] = rng.random(theta.size) < theta
    for k in range(1, length):
        redraw = rng.random(theta.size) < SWITCH
        anc[:, k] = np.where(redraw, rng.random(theta.size) < theta, anc[:, k - 1])
    return anc


def write_lanc(path, anc0, anc1):
    """admix-kit .lanc: per person, segment ends (site index, exclusive) with the two haplotypes' ancestries."""
    lines = []
    for a, b in zip(anc0, anc1):
        change = np.flatnonzero((np.diff(a) != 0) | (np.diff(b) != 0)) + 1
        stops = np.r_[change, a.size]
        lines.append(" ".join(f"{s}:{a[s - 1]}{b[s - 1]}" for s in stops))
    path.write_text(f"{anc0.shape[1]} {anc0.shape[0]}\n" + "\n".join(lines) + "\n")


def generate(output: Path) -> None:
    output = output.resolve()
    description = {"kind": "summit.local_ancestry_example", "version": 1, "seed": SEED, "n": N}
    marker = output / "local_ancestry.json"
    if output.exists():
        if (marker.is_file() and json.loads(marker.read_text()) == description
                and all((output / name).is_file() for name in FILES)):
            return
        raise FileExistsError(f"Use a new output directory: {output}")
    (output / "lanc").mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    iid = np.array([f"SIM{i:06d}" for i in range(N)])
    theta = rng.beta(5, 2, N)
    calls, snp_ids, chrom, pos, untagged = [], [], [], [], []
    for c in range(1, CHROMOSOMES + 1):
        anc = [tracts(rng, theta, SITES), tracts(rng, theta, SITES)]
        # Balding-Nichols allele frequencies of the two populations (F = 0.15 each)
        p0 = rng.uniform(0.05, 0.95, SITES)
        f = 0.15
        p = np.vstack([rng.beta(p0 * (1 - f) / f, (1 - p0) * (1 - f) / f) for _ in range(2)])
        g = sum((rng.random((N, SITES)) < p[a, np.arange(SITES)]).astype(np.int8) for a in anc)
        calls.append(g)
        bp = STEP * (1 + np.arange(SITES))
        names = [f"sim_variant_{c}_{k}" for k in range(SITES)]
        snp_ids += names
        chrom += [c] * SITES
        pos += list(bp)
        # untagged variants at random sites, drawn from the local ancestry there
        k = rng.choice(SITES, UNTAGGED // CHROMOSOMES, replace=True)
        pu = np.vstack([rng.beta(0.5 * (1 - f) / f, 0.5 * (1 - f) / f, k.size) for _ in range(2)])
        untagged.append(sum((rng.random((N, k.size)) < pu[a[:, k], np.arange(k.size)]).astype(np.int8) for a in anc))
        pd.DataFrame({"#CHROM": c, "POS": bp, "ID": names, "REF": "C", "ALT": "A"}).to_csv(
            output / "lanc" / f"chr{c}.pvar", sep="\t", index=False)
        pd.DataFrame({"#IID": iid}).to_csv(output / "lanc" / f"chr{c}.psam", sep="\t", index=False)
        write_lanc(output / "lanc" / f"chr{c}.lanc", *anc)
    G = np.hstack(calls).astype(np.float64)
    m = G.shape[1]
    to_bed(str(output / "cohort.bed"), G, properties={
        "fid": iid, "iid": iid, "sid": np.array(snp_ids), "chromosome": np.array(chrom),
        "bp_position": np.array(pos), "allele_1": ["A"] * m, "allele_2": ["C"] * m}, count_A1=True)
    pc1 = theta + 0.05 * rng.standard_normal(N)        # a genotype PC that tracks global ancestry imperfectly
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": pc1}).to_csv(output / "cohort.cov", sep="\t", index=False)
    annot = pd.DataFrame({"CHR": chrom, "BP": pos, "SNP": snp_ids, "CM": 0, "SNP_ALL": 1})
    annot.to_csv(output / "cohort.annot", sep="\t", index=False)
    std = lambda X: (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    gs = std(G) @ rng.standard_normal(m)
    gu = std(np.hstack(untagged).astype(float)) @ rng.standard_normal(sum(u.shape[1] for u in untagged))
    y = (np.sqrt(0.3) * gs / gs.std() + np.sqrt(0.1) * gu / gu.std() + np.sqrt(0.6) * rng.standard_normal(N)
         + 0.5 * pc1)
    pd.DataFrame({"FID": iid, "IID": iid, "trait": y}).to_csv(output / "trait.tsv", sep="\t", index=False)
    marker.write_text(json.dumps(description, indent=2) + "\n")


def sumstats(output: Path) -> None:
    """Covariates with THETA, and summary statistics of the SNPs and of the merged fileset."""
    cov = pd.read_csv(output / "cohort.cov", sep="\t", dtype={"FID": str, "IID": str})
    theta = pd.read_csv(output / "markers.theta.tsv", sep="\t", dtype={"FID": str, "IID": str})
    cov = cov.merge(theta, on=["FID", "IID"])
    cov.to_csv(output / "cohort_theta.cov", sep="\t", index=False)
    y = pd.read_csv(output / "trait.tsv", sep="\t", dtype={"IID": str}).set_index("IID").loc[cov.IID, "trait"].to_numpy()
    covariates = cov[["PC1", "THETA"]].to_numpy()
    for name, prefix in (("snp", "cohort"), ("joint", "joint")):
        bed = open_bed(str(output / f"{prefix}.bed"), count_A1=True)
        ols_sumstats(bed.read(dtype="float64"), y, covariates, bed.sid).to_csv(
            output / f"trait.{name}.sumstats", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sumstats", action="store_true", help="write covariates with THETA and summary statistics")
    args = parser.parse_args()
    if args.sumstats:
        sumstats(args.out)
        print(f"Summary statistics written: {args.out.resolve()}")
    else:
        generate(args.out)
        print(f"Local-ancestry example ready: {args.out.resolve()}")


if __name__ == "__main__":
    main()
