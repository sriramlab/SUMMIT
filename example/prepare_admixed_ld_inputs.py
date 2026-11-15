#!/usr/bin/env python3
"""Generate source panels and an admixed cohort for the `summit admixed-ld` example.

Two source populations each have founder haplotypes. A sampled haplotype copies founders and switches founder at
`copy_rate` per cM, which gives LD within each population. An admixed haplotype also carries breakpoints since
admixture, at T/100 per cM for T generations. Each segment between breakpoints draws its ancestry (the first population
with probability theta_h) and starts a fresh copying path in that population. This is the model that `summit
admixed-ld` assumes: LD within a source only on unbroken stretches, local ancestry with covariance
theta_h (1 - theta_h) exp(-T d / 100), and global ancestry theta that varies between people. Genetic positions are
1 cM per Mb and are written to the CM column of the BIM files.

  python prepare_admixed_ld_inputs.py     write the inputs under out/admixed_ld/

The output holds panel_afr and panel_eur (1,000 people each), the cohort (3,000 people with mean global ancestry 0.75
from the first population, admixed 8 generations ago), cohort.cov (two genotype PCs), trait.sumstats (OLS summary
statistics of a trait with h2 0.3 adjusted for the PCs) and cohort.annot (two MAF classes). The second panel lists a
third of the alleles the other way round. No participant dataset is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 3
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "admixed_ld"
FILES = [f"{p}.{e}" for p in ("panel_afr", "panel_eur", "cohort") for e in ("bed", "bim", "fam")] + [
    "cohort.cov", "trait.sumstats", "cohort.annot"]


def _haplotypes(rng, founders, cm, n_hap, copy_rate, anc=None, break_rate=0.0):
    """Haplotypes (n_hap x M int8) and their local ancestry. founders: {"A": F x M, "E": F x M} for admixed haplotypes,
    or a single entry for a panel of that source. anc: per-haplotype probability of ancestry A (admixed only)."""
    keys = list(founders)
    M = cm.size
    out = np.empty((n_hap, M), dtype=np.int8)
    F = founders[keys[0]].shape[0]
    if anc is None:
        a = np.full(n_hap, keys[0] == "A")
    else:
        a = rng.random(n_hap) < anc
    f = rng.integers(0, F, n_hap)
    la = np.empty((n_hap, M), dtype=np.int8)
    for j in range(M):
        if j > 0:
            dcm = cm[j] - cm[j - 1]
            if dcm < 0:                                   # new chromosome: everything restarts
                brk = np.ones(n_hap, dtype=bool)
                sw = np.ones(n_hap, dtype=bool)
            else:
                brk = (rng.random(n_hap) < 1 - np.exp(-break_rate * dcm) if anc is not None
                       else np.zeros(n_hap, dtype=bool))
                sw = rng.random(n_hap) < 1 - np.exp(-copy_rate * dcm)
            if anc is not None:
                a = np.where(brk, rng.random(n_hap) < anc, a)
            f = np.where(sw | brk, rng.integers(0, F, n_hap), f)
        if anc is None:
            out[:, j] = founders[keys[0]][f, j]
        else:
            out[:, j] = np.where(a, founders["A"][f, j], founders["E"][f, j])
        la[:, j] = a
    return out, la


def simulate(seed=1, n_chr=2, snps_per_chr=600, chr_cm=60.0, n_panel=1200, n_cohort=3000, mean_theta=0.75,
             sd_theta=0.12, generations=8.0, n_founders=40, copy_rate=3.0):
    """Dosages (counting A1) of the two source panels and the admixed cohort, the cohort's global and local ancestry,
    and the variant table (CHR SNP CM BP A1 A2)."""
    rng = np.random.default_rng(seed)
    chrom = np.repeat(np.arange(1, n_chr + 1), snps_per_chr)
    cm = np.concatenate([np.sort(rng.uniform(0, chr_cm, snps_per_chr)) for _ in range(n_chr)])
    bp = np.rint(cm * 1e6).astype(np.int64) + 1                     # 1 cM per Mb
    M = cm.size
    pE = rng.uniform(0.05, 0.95, M)
    pA = np.clip(pE + rng.normal(0, 0.25, M), 0.03, 0.97)
    founders = {s: (rng.random((n_founders, M)) < p[None, :]).astype(np.int8) for s, p in (("A", pA), ("E", pE))}
    hA, _ = _haplotypes(rng, {"A": founders["A"]}, cm, 2 * n_panel, copy_rate)
    hE, _ = _haplotypes(rng, {"E": founders["E"]}, cm, 2 * n_panel, copy_rate)
    k = mean_theta * (1 - mean_theta) / sd_theta ** 2 - 1
    theta = rng.beta(mean_theta * k, (1 - mean_theta) * k, n_cohort)
    th_hap = np.clip(np.repeat(theta, 2) + rng.normal(0, 0.05, 2 * n_cohort), 0.0, 1.0)
    hC, la = _haplotypes(rng, founders, cm, 2 * n_cohort, copy_rate, anc=th_hap, break_rate=generations / 100.0)
    geno = lambda h: (h[0::2] + h[1::2]).astype(np.float64)
    variants = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"sim{j:05d}" for j in range(M)], "CM": cm, "BP": bp,
                             "A1": np.where(rng.random(M) < 0.5, "A", "C"), "A2": "G"})
    return dict(variants=variants, panel_A=geno(hA), panel_E=geno(hE), cohort=geno(hC), theta=theta,
                local_ancestry=(la[0::2] + la[1::2]).astype(np.float64), pA=pA, pE=pE, generations=generations)


def write_bed(prefix, G, variants, id_prefix):
    """PLINK 1 fileset of dosages G (samples x variants) counting A1. Returns the sample IDs."""
    from bed_reader import to_bed
    iid = np.array([f"{id_prefix}{i:06d}" for i in range(G.shape[0])])
    to_bed(str(prefix) + ".bed", G.astype(np.float32), count_A1=True,
           properties=dict(fid=iid, iid=iid, chromosome=variants.CHR.to_numpy(), sid=variants.SNP.to_numpy(),
                           cm_position=variants.CM.to_numpy(), bp_position=variants.BP.to_numpy(),
                           allele_1=variants.A1.to_numpy(), allele_2=variants.A2.to_numpy()))
    return iid


def principal_components(G, k=2):
    X = (G - G.mean(0)) / np.maximum(G.std(0), 1e-12)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * s[:k]


def gwas(G, y, C):
    """Marginal OLS of y on each SNP with the covariates C (an intercept is added): BETA and SE."""
    n = G.shape[0]
    Z = np.column_stack([np.ones(n), C])
    Q, _ = np.linalg.qr(Z)
    yr = y - Q @ (Q.T @ y)
    X = G - Q @ (Q.T @ G)
    sxx = np.einsum("ij,ij->j", X, X)
    with np.errstate(divide="ignore", invalid="ignore"):           # NaN for SNPs without variance
        beta = (X.T @ yr) / sxx
        resid = yr[:, None] - X * beta[None, :]
        df = n - Z.shape[1] - 1
        se = np.sqrt(np.einsum("ij,ij->j", resid, resid) / df / sxx)
    return beta, se


def generate(output: Path, h2: float = 0.3) -> None:
    output = output.resolve()
    params = dict(seed=SEED, n_chr=6, snps_per_chr=400, chr_cm=60.0, n_panel=1000, n_cohort=3000, generations=8.0)
    description = {"kind": "summit.admixed_ld_example", "version": 1, "h2": h2, **params}
    marker = output / "admixed_ld.json"
    if output.exists():
        if (marker.is_file() and json.loads(marker.read_text()) == description
                and all((output / name).is_file() for name in FILES)):
            return
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    S = simulate(**params)
    v = S["variants"]
    write_bed(output / "panel_afr", S["panel_A"], v, "SIMA")
    vE = v.copy()
    swap = np.arange(len(v)) % 3 == 0                                # the second panel lists some alleles the other way
    vE.loc[swap, ["A1", "A2"]] = vE.loc[swap, ["A2", "A1"]].to_numpy()
    GE = S["panel_E"].copy()
    GE[:, swap] = 2 - GE[:, swap]
    write_bed(output / "panel_eur", GE, vE, "SIME")
    G = S["cohort"]
    iid = write_bed(output / "cohort", G, v, "SIM")
    C = principal_components(G, 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(output / "cohort.cov", sep="\t",
                                                                                 index=False)
    rng = np.random.default_rng(SEED + 1)
    sd = G.std(0)
    poly = sd > 0
    g = ((G[:, poly] - G[:, poly].mean(0)) / sd[poly]) @ rng.normal(0.0, 1.0, int(poly.sum()))
    y = np.sqrt(h2) * g / g.std() + np.sqrt(1 - h2) * rng.standard_normal(G.shape[0]) + 0.3 * C[:, 0] / C[:, 0].std()
    beta, se = gwas(G, y, C)
    ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta, "SE": se,
                  "Z": beta / se, "COV_RANK": C.shape[1]})[ok].to_csv(output / "trait.sumstats", sep="\t",
                                                                     index=False, float_format="%.8g")
    p = G.mean(0) / 2
    common = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"CHR": v.CHR, "BP": v.BP, "SNP": v.SNP, "CM": v.CM, "maf_low": 1 - common,
                  "maf_high": common}).to_csv(output / "cohort.annot", sep="\t", index=False)
    marker.write_text(json.dumps(description, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.out)
    print(f"Admixed-LD example ready: {args.out.resolve()}")


if __name__ == "__main__":
    main()
