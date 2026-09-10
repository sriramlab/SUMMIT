# Input files

Use the same genome build, SNP naming, and annotation definitions throughout
an analysis. Input examples below use invented IDs.

## Genotypes

`--geno` accepts a `.bed` or `.pgen` filename, or a prefix with one complete trio:

| Format | Files |
|---|---|
| PLINK 1 | `.bed`, `.bim`, `.fam` |
| PLINK 2 | `.pgen`, `.pvar`, `.psam` |

PGEN input must contain biallelic diploid variants and a plain-text PVAR.
Stored fractional dosages are retained. Missing values are mean-imputed.
Convert BGEN or compressed PVAR input before running SUMMIT. If both genotype
formats share a prefix, specify the desired filename.

PGEN is supported for genome-wide, windowed, and one-environment G×E LD scores,
and PGS. The direct generalized reference executor currently reads BED.

## GWAS summary statistics

Supply a whitespace-delimited table with a header:

| Meaning | Accepted columns |
|---|---|
| SNP ID | `SNP`, `ID`, `snp`, `id` |
| Effect allele | `A1`, `ALT` |
| Other allele | `A2`, `REF` |
| Sample size | `N`, `OBS_CT` |
| Effect estimate | `BETA`, `beta` |
| Standard error | `SE`, `STDERR`, `se`, `stderr` |

```text
SNP A1 A2 N BETA SE
sim_variant_000001 A C 50000 0.01 0.005
```

For h² and rg, provide BETA and SE; Z alone is insufficient. These methods
reconstruct marginal linear-regression score statistics, so verify suitability
before using statistics from a different association model.

Optional `COV_RANK` or `P_EFF` gives the number of non-intercept GWAS covariates;
`--cov-rank` overrides it. The rg calculation uses this metadata. The current
h² calculation uses a covariate-rank value of zero, including when h² supplies
the denominator for rg. [Methods](Methods.md) explains the calculation.

## LD scores and annotations

LD-score files begin with `CHR SNP BP`, optionally `CM`, followed by LD-score
columns. An annotation file contains either:

- `CHR BP SNP`, optional `CM`, and named annotation columns; or
- a numeric matrix with one row per input SNP, optionally headed by column names.

Row order must match the genotype or LD-score SNP order. Omit `--annot` for a
single-component model. Disjoint bins and overlapping annotations have different
interpretations; see [Heritability and genetic correlation](Heritability-and-genetic-correlation.md).

For chromosome-split h²/rg files, `@` expands to chromosomes 1–22, for example
`reference/chr@.gw.ldscore.gz`.

## Covariates, environments, and phenotypes

These tables use `FID IID` followed by named data columns:

```text
FID IID cov1 cov2
SIM000001 SIM000001 0.2 -0.4
```

Keep IDs as strings. Covariates must be numeric; encode categorical fixed
effects before using the main `summit` command. The contextual Python and PGS
APIs can apply saved categorical coding recipes.

For one-environment G×E, an environment table has exactly one selected
environment column. Trait scoring requires finite selected phenotypes on the
reference sample set. For PGS, each trait can declare its own sample subset;
keep files also have a `FID IID` header.
