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
PCGC preparation, and PGS. The direct quantitative generalized reference
executor currently reads BED.

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

Use each SNP's observed sample size in `N`; do not replace varying values with
the file maximum. SUMMIT accounts for these differences in its null moments
and regression weights. For rg, see the sample-overlap assumptions in
[Methods](Methods.md).

Optional `COV_RANK` or `P_EFF` gives the number of non-intercept GWAS covariates;
`--cov-rank` overrides it. The rg calculation uses this metadata. The current
h² calculation uses a covariate-rank value of zero, including when h² supplies
the denominator for rg. [Methods](Methods.md) explains the calculation.

PCGC uses moments prepared from individual binary outcomes and genotypes.
The [PCGC guide](Binary-traits-and-PCGC.md) describes its sample table,
population scales, risks, and annotation format.

## Files used for inference

Reference and trait summaries are stored differently across workflows:

| Workflow | Reference input | Trait input |
|---|---|---|
| Additive h²/rg | LD scores, usually `.ldscore.gz` | Summary statistics with BETA and SE |
| One-environment quantitative G×E | `.gxe.ref.json` describing the reference files | `.gxe.gwas.tsv.gz`, `.gxe.gwis.tsv.gz`, and `.gxe.moments.json` per trait |
| Generalized quantitative G×E | `.generalized-gxe-variant-ldscore-v1.npz` | `.generalized-gxe-trait-summary-v1.npz` |
| Additive or generalized G×E PCGC | `.binary.ldscores.npz` | `.binary.sumstats.npz` |

PCGC preparation (`--make-binary-sumstats`) computes both sets of moments and
writes them separately. Fit with `--h2 study.binary.sumstats.npz` and
`--ldscores study.binary.ldscores.npz`. These trait summaries contain PCGC score
products with same-person terms removed, plus any requested trait-dependent
uncertainty summaries. NPZ stores the multidimensional arrays without flattening
them into BETA/SE tables.

The summary records its expected reference identity. SUMMIT checks preparation
metadata and array checksums before fitting; renaming or moving the files does
not change compatibility. Use the reference saved with the summary, including
the same realized random-vector estimate. Standard PCGC's reference weights
depend on the disease-risk model and analyzed sample.

Existing combined `.binary.npz` files are accepted through `--h2` alone.
`--binary-output-format combined` selects that format during preparation.
See [Generalized G×E PCGC](../pcgc_gxe.md#matching-reference-and-summary-files)
for the compatibility checks and [Multiple environments](Multiple-environments.md)
for the quantitative Python interface.

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
