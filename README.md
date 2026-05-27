# SUMMIT

**S**ummary-statistics-based **U**nified **M**ethod for **M**ultivariate **I**nference of **T**raits

SUMMIT estimates LD scores, SNP heritability, genetic covariance, and genetic
correlation from reference genotypes and GWAS summary statistics. The current
implementation is centered on per-SNP LD scores and SCORE-scale summary-statistic
moments.

## Main Features

- Genome-wide randomized LD scores from PLINK bed/bim/fam reference genotypes.
- Covariate-adjusted and annotation-partitioned LD scores.
- Optional fixed-window LD scores with `--ld-wind-kb`.
- Optional GxE LD scores with `--env`; SUMMIT writes both additive-interaction
  cross-LD and interaction-interaction LD scores.
- Heritability estimation from `BETA`/`SE` summary statistics.
- Genetic correlation estimation with either a fixed overlap intercept or a
  summary-estimated intercept.
- Batch genetic-correlation manifests, including a sparse fast path for
  fixed-intercept analyses.
- Allele alignment for rg (`--align-alleles`) with strand-ambiguous SNPs
  dropped by default.

Trace-summary (`.tr/.MN`) input is intentionally not supported in the current
refactored h2/rg path; use per-SNP LD scores.

## Installation

### Requirements

- Linux or macOS
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

Optional `COV_RANK` or `P_EFF` gives the number of non-intercept covariates used
in the GWAS. If it is absent, SUMMIT uses `cov_rank=0`; an explicit `--cov-rank`
overrides the file. Z-only summary statistics are no longer a supported analysis
input because SUMMIT reconstructs exact SCORE-scale moments from `BETA`, `SE`,
`N`, and `cov_rank`.

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

This writes `outs/ref.mafld.gw.ldscore.gz` and `outs/ref.mafld.gw.log`.
Add `--write-kmoments` for single-component LD scores when you want the
model-based `--rg-se-method kmoments` path.

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

`--collapse-reg-ld` is kept only as a compatibility option. It sums a multi-column
regression LD file before the intercept fit and prints a warning. This is valid
only when the LD-score columns are non-overlapping, such as a disjoint MAF-LD
partition. For overlapping annotations, precompute or provide a genuine scalar
regression LD score instead.

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

- `--geno`: compute LD scores from PLINK bed/bim/fam input.
- `--h2`: estimate heritability for one file, a directory, or split-file spec.
- `--rg`: estimate rg for a comma-separated pair or a manifest TSV.
- `--make-rg-manifest`: build an rg manifest.

Exactly one of these modes must be specified.

### h2/rg options

- `--ldscores`: primary LD-score file or chromosome-split spec.
- `--ldscores-reg`: optional scalar LD-score file for unconstrained rg intercept
  regression.
- `--annot`: annotation file or split-file spec; omitted means single component.
- `--max-chisq`: main chi-square filter; use `auto` for `max(80, 0.001*Nmax)`.
- `--intercept-chisq-thr`: chi-square filter used only for intercept regression.
- `--chisq-action`: `drop`, `clip`, `warn`, or `none`.
- `--intercept-rg`: fixed rg intercept on SUMMIT's SCORE scale.
- `--pheno-rg`, `--pheno-rg-cov`: compute fixed intercept from overlapping
  phenotype/covariate files.
- `--align-alleles`: align the second summary-statistic file to the first.
- `--keep-ambiguous`: keep strand-ambiguous SNPs during allele alignment.
- `--njack`: integer SNP blocks, `chr`, `chr:d`, `chr:d:R`, or `chr:d:R:seed`.
- `--rg-se-method`: `jackknife`, `delta`, `robust`, or `kmoments`.
- `--write-jack`: save jackknife replicate dumps.
- `--write-normeq`: save explicit SCORE normal-equation JSON for eligible rg runs.

### LD-score options

- `--nvecs`: random vectors for stochastic genome-wide LD scores.
- `--step_size`: SNP block size for LD-score computation.
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

- Genome-wide LD scores: `<out>.gw.ldscore.gz`, `<out>.gw.log`, optionally
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
