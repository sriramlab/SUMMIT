# SUMMIT

**S**ummary-statistics-based **U**nified **M**ethod for **M**ultivariate **I**nference of **T**raits

SUMMIT estimates LD scores, SNP heritability, genetic covariance, and genetic
correlation from reference genotypes and GWAS summary statistics. The current
implementation is centered on per-SNP LD scores and SCORE-scale summary-statistic
moments.

## Main Features

- Genome-wide randomized, fixed-window, and GxE LD scores from PLINK 1
  BED/BIM/FAM hard calls or biallelic diploid PLINK 2 PGEN/PVAR/PSAM dosages.
- Covariate-adjusted and annotation-partitioned LD scores.
- Optional fixed-window LD scores with `--ld-wind-kb`.
- Optional GxE LD scores with `--env`; SUMMIT writes both additive-interaction
  cross-LD and interaction-interaction LD scores.
- Heritability estimation from `BETA`/`SE` summary statistics.
- Optional score-scale constrained LDSC-style IRWLS for h2 and bivariate
  genetic covariance/rg via `--weight-mode ldsc`; the default remains
  SUMMIT/HE.
- Genetic correlation estimation with either a fixed overlap intercept or a
  summary-estimated intercept.
- Batch genetic-correlation manifests, including a sparse fast path for
  fixed-intercept analyses.
- Integrated allele validation/alignment for rg (enabled by default), with
  strand-ambiguous SNPs dropped by default and `--no-align-alleles` as an
  explicit escape hatch for pre-harmonized inputs.

Trace-summary (`.tr/.MN`) input is intentionally not supported in the current
refactored h2/rg path; use per-SNP LD scores.

## Installation

### Requirements

- Linux or macOS
- Python 3.10 or newer
- Conda or Miniconda
- C++17 compiler
- BLAS/LAPACK and OpenMP support

The supplied environment uses OpenBLAS from conda-forge. If MKL is available
through `MKLROOT` or the active conda environment, CMake may use MKL instead.

```bash
git clone https://github.com/bronsonj98/SUMMIT.git
cd SUMMIT
conda env create -f environment.yml
conda activate summit
pip install -v .
```

Check the install:

```bash
summit --help
python -c "import summit, summit.gwldcore, summit.winldcore; print('SUMMIT OK')"
```

For source-tree development without installing, the compatibility launcher is:

```bash
python src/summit.py --help
```

## Input Formats

### Summary Statistics

SUMMIT expects whitespace-delimited summary statistics with these columns:

| Required column | Accepted names |
| --- | --- |
| SNP ID | `SNP`, `ID`, `snp`, `id` |
| effect allele | `A1`, `ALT` |
| other allele | `A2`, `REF` |
| sample size | `N`, `OBS_CT` |
| effect estimate | `BETA`, `beta` |
| standard error | `SE`, `STDERR`, `se`, `stderr` |

Optional `COV_RANK` or `P_EFF` records the number of non-intercept covariates
used in the GWAS, and `--cov-rank` overrides that metadata. The current h2 path
intentionally reconstructs its SCORE-scale moment with `cov_rank=0`; this is an
experimental convention even when covariate metadata are present. The rg path
uses the resolved covariate rank. Z-only summary statistics are no longer a
supported analysis input because SUMMIT reconstructs moments from `BETA`, `SE`,
and `N`.

### LD Scores

LD-score files are whitespace-delimited and must begin with:

```text
CHR SNP BP [CM] L2_or_annotation_columns...
```

Use `@` as a chromosome placeholder for split files, for example
`/path/to/chr@.gw.ldscore.gz`. SUMMIT resolves `@` to chromosomes 1 through 22.

### Annotations

If `--annot` is omitted, SUMMIT runs a single-component model. Otherwise,
annotations can be:

- LDSC-style full annotations with `CHR`, `BP`, `SNP`, optional `CM`, and one or
  more annotation columns.
- Thin annotation matrices with one row per BIM/LD-score SNP. If a header is
  present, column names are used as annotation names.

### Reference Genotypes

`--geno` accepts an explicit `.bed` or `.pgen` path, or a prefix with exactly
one complete genotype trio:

- PLINK 1 `.bed/.bim/.fam` input supports genome-wide, windowed, and GxE LD
  scores, including the existing hard-call-specific options.
- PLINK 2 `.pgen/.pvar/.psam` input supports randomized genome-wide,
  deterministic windowed, and GxE estimators for biallelic diploid variants.
  The PVAR must be plain text rather than `.pvar.zst`.

For PGEN input, SUMMIT bulk-decodes stored REF-allele dosages into one reusable
variant-block buffer, mean-imputes missing dosage values, and standardizes each
variant over the retained samples. Thus the target is LD in the decoded dosage
matrix. A hard-call PGEN estimates the same matrix quantity as BED; an imputed
dosage PGEN need not give exactly the same finite-sample LD scores as hard
calls.

The PGEN paths are CPU-only and require `--impute-method mean` (the default).
The randomized genome-wide path additionally requires `--ddof 1` and the dense
kernels; it does not support `--device cuda`, Mailman, HWE imputation,
`--correct-skew`, or `--write-kmoments`. Windowed PGEN uses bounded streamed
panels and supports `--win-panel-cols` and `--win-cache-mb`; GxE PGEN uses a
persistent streamed dosage reader. Multiallelic and non-diploid variants are
out of scope. If BED and PGEN trios share a prefix, pass the desired `.bed` or
`.pgen` filename explicitly.

See [PGEN LD-score estimation](docs/pgen_gwld.md) for the estimand,
normalization, implementation details, and validation design.

## Common Commands

All examples below assume SUMMIT is installed and available as `summit`.

### Genome-wide LD scores

```bash
summit \
  --geno ref_panel.bed \
  --annot mafld.annot.gz \
  --covar covariates.txt \
  --out outs/ref.mafld \
  --nvecs 1000 \
  --step_size 1000 \
  --num-threads 8
```

This writes `outs/ref.mafld.gw.ldscore.gz`, `outs/ref.mafld.gw.M`,
`outs/ref.mafld.gw.mc.tsv`, and `outs/ref.mafld.gw.log`. The MC file reports
the default annotation-level integrated random-probe noise diagnostic. Add
`--write-ld-mc-var` (or `--write-ld-mc-ci`) to write optional per-SNP MC
variances, standard errors, and approximate pointwise 95% conditional MC
intervals.
Add `--write-kmoments` for single-component LD scores when you want the
model-based `--rg-se-method kmoments` path.

See [Genome-wide LD-score Monte Carlo noise](docs/gwld_mc_noise.md) for the
estimand and sufficient-statistic calculation.

For a PGEN dosage panel, use the same command with an explicit PGEN path and
mean imputation:

```bash
summit \
  --geno ref_panel.pgen \
  --annot mafld.annot.gz \
  --covar covariates.txt \
  --out outs/ref.mafld.pgen \
  --nvecs 1000 \
  --step_size 1000 \
  --impute-method mean \
  --num-threads 8
```

### Windowed LD scores

```bash
summit \
  --geno ref_panel.bed \
  --annot mafld.annot.gz \
  --covar covariates.txt \
  --ld-wind-kb 20000 \
  --out outs/ref.mafld.20mb \
  --num-threads 8
```

This writes `*.win.ldscore.gz`, `*.win.M`, and `*.win.M_5_50`.

### GxE LD scores

```bash
summit \
  --geno ref_panel.bed \
  --env environment.txt \
  --covar covariates.txt \
  --annot mafld.annot.gz \
  --out outs/ref.mafld.env \
  --nvecs 1000 \
  --num-threads 8
```

The environment file must contain `FID`, `IID`, and exactly one environment
column. SUMMIT writes `*.gxe.ldscore.gz` and `*.gee.ldscore.gz`.

### Heritability

```bash
summit \
  --h2 trait.sumstats.gz \
  --ldscores ref.mafld.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --out outs/trait.mafld \
  --njack chr \
  --num-threads 4
```

`--h2` can also be a directory of summary-statistic files or a chromosome-split
path spec. Results are written to `<out>.results.tsv` and `<out>.log`.

To replace the default HE estimating instrument with score-scale constrained
LDSC-style IRWLS:

```bash
summit \
  --h2 trait.sumstats.gz \
  --ldscores ref.mafld.gw.ldscore.gz \
  --ldscores-w ref.regression.l2.ldscore.gz \
  --ldsc-m ref.mafld.gw.M \
  --annot mafld.annot.gz \
  --weight-mode ldsc \
  --out outs/trait.mafld.ldsc \
  --njack chr
```

The one-column `--ldscores-w` file is recommended, especially for overlapping
annotations. If it is omitted, SUMMIT uses the row sum of the primary LD-score
columns. LDSC mode reruns IRWLS inside every delete block and is currently
available for ordinary `--h2` and single-pair/regular-manifest `--rg`, but not
fast cached h2 or fast rg manifests. See
[Score-scale constrained LDSC weighting](docs/ldsc_weight_mode.md).

### Genetic Correlation With a Fixed Intercept

For non-overlapping studies, the fixed intercept is usually zero:

```bash
summit \
  --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --intercept-rg 0 \
  --ldscores ref.mafld.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --align-alleles \
  --out outs/trait1.trait2.fixed0 \
  --njack chr
```

For overlapping individual-level samples, use either:

- `--intercept-rg c`, where `c = y_overlap' y_overlap / sqrt(N1*N2)`, or
- `--pheno-rg pheno1.txt,pheno2.txt` with optional
  `--pheno-rg-cov cov1.txt,cov2.txt` so SUMMIT computes the overlap intercept.

Phenotype files must have a header, `FID IID`, and the phenotype in the last
column. Covariate files must have `FID IID` followed by covariates.

### Genetic Correlation With a Summary-Estimated Intercept

When `--intercept-rg` and `--pheno-rg` are omitted, SUMMIT estimates the
cross-trait intercept from summary statistics first, then plugs it into the
SCORE normal equations.

The intercept regression LD scores must be one-dimensional. If the primary
analysis uses partitioned LD scores, provide a separate scalar `--ldscores-reg`
file:

```bash
summit \
  --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --ldscores ref.mafld.gw.ldscore.gz \
  --ldscores-reg ref.total.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --align-alleles \
  --out outs/trait1.trait2.unconstrained \
  --njack chr
```

SUMMIT uses a scalar LD score for the unconstrained rg intercept fit. A 1D
`--ldscores-reg` file is preferred. If a multi-column regression LD file is
provided, SUMMIT collapses it to total LD by default; this is valid only when
the LD-score columns are non-overlapping, such as a disjoint MAF-LD partition.
For overlapping annotations, precompute or provide a genuine scalar regression
LD score instead.

### Batch rg Manifests

Build a manifest from phenotype, covariate, and sumstats sources:

```bash
summit \
  --make-rg-manifest outs/rg_manifest.tsv \
  --phen-dir phenotypes/ \
  --cov-dir covariates/ \
  --sum-dir sumstats_map.tsv \
  --pair-list pairs.tsv \
  --out outs/manifest_build
```

Run a manifest with fixed intercepts:

```bash
summit \
  --rg outs/rg_manifest.tsv \
  --rg-manifest-fast \
  --ldscores ref.mafld.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --out outs/rg_batch \
  --njack chr
```

The fast path requires finite `intercept_rg` values in the manifest and currently
supports jackknife SEs only. Omit `--rg-manifest-fast` when you need the regular
per-pair path or summary-estimated intercepts.

For many models that share annotation columns, such as baseline plus one focal
cell-type annotation, place every unique column in one union annotation/LD-score
pair and provide a model manifest:

```text
model  bins                                    aliases
ct_001 ["base","Coding","CT001_H3K4me1"]    ["base","Coding","focal"]
ct_002 ["base","Coding","CT002_H3K4me1"]    ["base","Coding","focal"]
```

Then run all models in one process:

```bash
summit \
  --rg outs/rg_manifest.tsv \
  --rg-manifest-fast \
  --rg-model-manifest models.tsv \
  --ldscores baseline_celltypes.union.gw.ldscore.gz \
  --annot baseline_celltypes.union.annot.gz \
  --out outs/rg_celltypes \
  --njack chr
```

`bins` and `aliases` accept JSON string lists or comma-separated names. Every
model must have the same ordered aliases so the combined output has a stable
schema; the output includes a `model` column. SUMMIT loads and aligns traits
once, computes union SNP sufficient statistics once, and then fits each selected
model independently. The selected columns and their order therefore define the
same estimator as separate invocations on each model. When consolidating
existing per-cell-type traces, verify that their shared baseline annotation and
LD-score columns are identical before retaining one shared copy.

## Example Scripts

The `example/` directory contains small, runnable scripts. The bundled summary
statistics are legacy Z-format, so `prepare_example_inputs.py` creates compatible
toy `BETA`/`SE` files under `example/out/`. The fixed-intercept rg example reuses
the same toy signal twice and therefore sets `--intercept-rg 1`; use
`--intercept-rg 0` for genuinely non-overlapping studies.

```bash
cd example
./estimate_gwldscore.sh
./estimate_partitioned_gwldscore.sh
./estimate_windowed_ldscore.sh
./estimate_gxe_ldscore.sh
./h2_ldscore.sh
./rg_fixed_intercept.sh
./rg_unconstrained.sh
./rg_manifest_fixed_intercept.sh
```

The scripts prefer the installed `summit` executable and fall back to
`python ../src/summit.py` when run from a source checkout.

## Selected Options

### Analysis modes

- `--geno`: compute genome-wide, windowed, or GxE LD scores from a PLINK 1
  BED/BIM/FAM or PLINK 2 PGEN/PVAR/PSAM path or prefix.
- `--h2`: estimate heritability for one file, a directory, or split-file spec.
- `--rg`: estimate rg for a comma-separated pair or a manifest TSV.
- `--make-rg-manifest`: build an rg manifest.

Exactly one of these modes must be specified.

### h2/rg options

- `--ldscores`: primary LD-score file or chromosome-split spec.
- `--ldscores-reg`: optional scalar LD-score file for unconstrained rg intercept
  regression.
- `--weight-mode`: `he` (default) or score-scale constrained `ldsc` IRWLS for
  h2 and bivariate genetic covariance/rg.
- `--ldscores-w`: optional scalar regression-SNP LD score used in LDSC weights.
- `--ldsc-m`: fixed reference annotation masses; split files with `@` are summed.
- `--ldsc-irwls-iters`, `--ldsc-irwls-tol`: LDSC update controls.
- `--annot`: annotation file or split-file spec; omitted means single component.
- `--max-chisq`: main chi-square filter; use `auto` for `max(80, 0.001*Nmax)`.
- `--intercept-chisq-thr`: chi-square filter used only for intercept regression.
- `--chisq-action`: `drop`, `clip`, `warn`, or `none`.
- `--intercept-rg`: fixed rg intercept on SUMMIT's SCORE scale.
- `--pheno-rg`, `--pheno-rg-cov`: compute fixed intercept from overlapping
  phenotype/covariate files.
- `--align-alleles`: validate and align the second summary-statistic file to
  the first (default).
- `--no-align-alleles`: explicitly skip allele validation/alignment for inputs
  already guaranteed to have identical orientation.
- `--keep-ambiguous`: keep strand-ambiguous SNPs during allele alignment.
- `--njack`: integer SNP blocks, `chr`, `chr:d`, `chr:d:R`, or `chr:d:R:seed`.
- `--rg-se-method`: `jackknife`, `delta`, `robust`, or `kmoments`.
- `--write-jack`: save jackknife replicate dumps.
- `--write-normeq`: save explicit SCORE normal-equation JSON for eligible rg runs.

### LD-score options

- `--nvecs`: random vectors for stochastic genome-wide LD scores.
- `--write-ld-mc-var` / `--write-ld-mc-ci`: additionally write per-SNP,
  per-annotation MC variances, SEs, and approximate pointwise 95% conditional
  MC intervals; the compact annotation-level diagnostic is written by default.
- `--skip-ld-mc`: disable the default MC diagnostic.
- `--step_size`: SNP block size for LD-score computation; for PGEN this also
  controls the reusable dosage decode buffer.
- `--win-panel-cols`, `--win-cache-mb`: decoded panel width and bounded
  prepared-panel cache for PGEN windowed LD. Automatic caching is capped and
  also respects `--target-mem` when supplied.
- `--covar`: covariate file with `FID IID` and covariate columns.
- `--env`: one-column environment file for GxE LD scores.
- `--ld-wind-kb`: compute fixed-window LD scores instead of randomized
  genome-wide LD scores.
- `--rand-samp`: random subset of samples; a ratio in `(0,1]` or an integer
  sample count.
- `--rand-dist`: `spherical`, `normal`/`gaussian`, or `rademacher`.
- `--dtype`: `float32` or `float64`.
- `--device`: `cpu` or `cuda[:index]` for supported genome-wide LD-score runs.
- `--num-threads`: cap BLAS/OpenMP thread pools.
- `--target-xz-mem`, `--target-mem`: memory budgets in GB.

## Output Files

- Genome-wide LD scores: `<out>.gw.ldscore.gz`, `<out>.gw.M`,
  `<out>.gw.mc.tsv`, `<out>.gw.log`, optionally `<out>.gw.mcvar.gz` and
  `<out>.gw.kmoments`.
- Windowed LD scores: `<out>.win.ldscore.gz`, `<out>.win.M`,
  `<out>.win.M_5_50`, `<out>.win.log`.
- GxE LD scores: `<out>.gxe.ldscore.gz`, `<out>.gee.ldscore.gz`,
  `<out>.gxe.log`.
- h2: `<out>.results.tsv`, `<out>.log`, optionally `<out>.<trait>.jack`.
- rg: `<out>.log`, optionally `<out>.rg.jack` and
  `<out>.rg.scoreeq.json`.
- rg manifest: `<out>/batch.log`, `<out>/manifest.results.tsv`, and per-pair
  logs/results where applicable.

## Citation

If you use SUM-RHE/SUMMIT, please cite:

```text
Jeong, M., Pazokitoroudi, A., Liu, Z., & Sankararaman, S. (2024).
Scalable summary statistics-based heritability estimation method with
individual genotype level accuracy. Genome Research, gr.279207.124.
https://doi.org/10.1101/gr.279207.124
```
