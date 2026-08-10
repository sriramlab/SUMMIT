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
- Full one-environment GENIE-style G + GxE + NxE estimation from marginal
  score summaries and an in-sample XX/XW/WX/WW trace bundle.
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

### GxE, heterogeneous noise, and marginal GWIS scores

Build the phenotype-independent feature cache once for a fixed cohort,
environment, covariate design, SNP set, and annotation:

```bash
summit \
  --gxe-build-cache \
  --geno ref_panel --env environment.txt --covar covariates.txt \
  --annot mafld.annot.gz \
  --gxe-kernel-mode standardized --gxe-genotype-scale sample \
  --write-gxe-jackknife --njack 100 \
  --out outs/design
```

Randomized traces can then be divided across disjoint probe identities. The
recommended low-Monte-Carlo-noise layout uses eight B128 jobs covering B1024;
the probe-indexed RNG makes this equivalent to another tiling within declared
floating-point tolerance:

```bash
summit \
  --geno ref_panel --env environment.txt --covar covariates.txt \
  --annot mafld.annot.gz \
  --gxe-feature-cache outs/design.gxe.cache.npz \
  --gxe-reference-shard --gxe-probe-offset 0 \
  --nvecs 128 --seed 20260808 --rand-dist rademacher \
  --dtype float32 --step_size 500 \
  --gxe-kernel-mode standardized --gxe-genotype-scale sample \
  --write-gxe-jackknife --njack 100 \
  --out outs/shard.000
```

Merge only checked, disjoint shards, then score all selected traits in one
genotype pass:

```bash
summit \
  --gxe-merge-shards outs/shard.*.gxe.shard.json \
  --gxe-feature-cache outs/design.gxe.cache.npz \
  --out outs/reference.B1024

summit \
  --gxe-score-reference outs/reference.B1024.gxe.ref.json \
  --geno ref_panel --env environment.txt --covar covariates.txt \
  --gxe-pheno phenotypes.txt --gxe-pheno-cols trait1,trait2 \
  --out outs/scores
```

The environment file must contain `FID`, `IID`, and exactly one environment
column. Every selected phenotype must be finite on the cache's fixed cohort;
build a common-cohort wide input first. The environment is standardized, and
the intercept, environment, and supplied covariates are projected from the
phenotype and both genetic feature panels. The projected phenotype is rescaled
so `y'y = rank(P)`; the fitter verifies this contract rather than guessing a
score scale.

The merged reference contains all four directional trace-score panels (`gxx`,
`gxe`, `exg`, and `gee`), per-SNP projected norms/NxE diagonals, and exact
two-sided deletion intersections. Each trait triplet contains direct marginal
additive and interaction scores plus the indispensable scalar
`y' diag(E^2) y`.

Exact jackknife generation requires at least 100 probes by default. Lower
counts can strongly contaminate the delete-block SE through randomized
within-block trace noise; the diagnostic-only override is
`--allow-low-probe-gxe-jackknife`.

```bash
summit \
  --gxe-fit outs/reference.B100.gxe.ref.json \
  --gxe-gwas outs/scores.trait1.gxe.gwas.tsv.gz \
  --gwis outs/scores.trait1.gxe.gwis.tsv.gz \
  --gxe-moments outs/scores.trait1.gxe.moments.json \
  --out outs/trait1
```

For several traits sharing one reference, a strict batch manifest avoids
reloading and reaggregating the reference for every fit while retaining full
per-trait hash and SNP-axis validation:

```json
{
  "kind": "summit.gxe.fit_batch",
  "schema_version": 1,
  "reference": "reference.B100.gxe.ref.json",
  "traits": [
    {
      "name": "trait1",
      "moments": "scores.trait1.gxe.moments.json",
      "gwas": "scores.trait1.gxe.gwas.tsv.gz",
      "gwis": "scores.trait1.gxe.gwis.tsv.gz",
      "out": "fits/trait1"
    }
  ]
}
```

```bash
summit --gxe-fit-batch fit_batch.json --out outs/batch_fit
```

All trait result pairs are preflighted and published as one no-overwrite
transaction. Downstream consumers should wait for successful command/job
completion before reading a batch.

The default `--gxe-kernel-mode standardized` uses SUMMIT's covariate-adjusted
partial-correlation convention: form `P G` and `P[diag(E)G]`, then normalize
each valid projected column to squared norm `rank(P)`. This is the same
post-projection convention used by SUMMIT's additive LD/score machinery and is
invariant to nonsingular rescaling of a retained genotype column. The explicit
`--gxe-kernel-mode genie --gxe-genotype-scale hwe` sensitivity mode instead
keeps the natural norms of HWE-scaled projected columns to reproduce GENIE's
kernel definition. These are different random-effect estimands, so their
artifacts cannot be mixed; both choices are recorded and hash-bound.

Do not pass a conventional PLINK 2 `--glm interaction` `ADDxE` Z statistic as
`--gwis`: it is conditional on the SNP main effect. GENIE requires the marginal
cross-product of projected `G*E` with projected phenotype. SUMMIT's generated
files declare `SCORE_MODE=marginal_cross_product` and the loader rejects other
declared modes. Every score/reference artifact is SHA-256-bound. Phenotype
summaries are also bound to the exact feature cache, so the same scores can be
reused across B50/B100 trace checkpoints from that cache without another
genotype pass.

Generation and fitting refuse an existing output prefix unless
`--gxe-overwrite` is supplied explicitly; files are written atomically with
owner-only permissions. See [docs/gxe_genie.md](docs/gxe_genie.md) for the equations,
identifiability checks, file contract, and validation requirements.

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
- `--env`: one-column environment file for the GxE reference/score bundle.
- `--gxe-build-cache`: write `<out>.gxe.cache.npz` and stop.
- `--gxe-feature-cache`, `--gxe-reference-shard`, `--gxe-probe-offset`: reuse
  the exact feature definition and generate a disjoint, non-fit-able trace shard.
- `--gxe-merge-shards`: validate and merge shard manifests into a fit-able reference.
- `--gxe-score-reference`: produce marginal scores and NxE moments from a
  sealed reference; `--gxe-pheno-cols` selects columns from `--gxe-pheno`.
- `--write-gxe-jackknife`: write within-block intersections needed for
  two-sided GENIE kernel-deletion SEs; block layout comes from `--njack`.
- `--allow-low-probe-gxe-jackknife`: diagnostic override for fewer than 100
  probes; resulting SEs are not production-calibrated.
- `--ld-wind-kb`: compute fixed-window LD scores instead of randomized
  genome-wide LD scores.
- `--rand-samp`: random subset of samples; a ratio in `(0,1]` or an integer
  sample count.
- `--rand-dist`: `spherical`, `normal`/`gaussian`, or `rademacher`.
- `--dtype`: `float32` or `float64`.
- `--gxe-native-backend direct`: opt into the Linux C++ direct BED backend for
  phenotype-free, one-annotation, `float32` or `float64`, standardized/sample-scaled GxE
  references. It uses the same Philox probes and exact projected-feature
  algebra as the Python oracle and seals the loaded native binary and source
  snapshot in every artifact.
- `--use-mailman auto` (default): for ordinary additive LD scores, use the
  existing Mailman implementation only at `B<=10` and only with HWE
  imputation. The existing Mailman kernel is not used for GxE because it does
  not implement the required interaction-before-projection algebra.
- `--device`: `cpu` or `cuda[:index]` for supported genome-wide LD-score runs.
- `--num-threads`: cap BLAS/OpenMP thread pools.
- `--target-xz-mem`, `--target-mem`: sketch-panel budgets in GiB or `auto`
  (default). Auto uses the tightest observable memory limit and retains both a
  fixed reserve and a proportional margin; cluster workflows should keep an
  explicit budget when the scheduler allocation is not visible to the process.

## Output Files

- Genome-wide LD scores: `<out>.gw.ldscore.gz`, `<out>.gw.log`, optionally
  `<out>.gw.kmoments`.
- Windowed LD scores: `<out>.win.ldscore.gz`, `<out>.win.M`,
  `<out>.win.M_5_50`, `<out>.win.log`.
- GxE reference: `<out>.{gxx,gxe,exg,gee}.ldscore.gz`,
  `<out>.gxe.diag.tsv.gz`, `<out>.gxe.ref.json`, and optionally
  `<out>.gxe.jackknife.npz`.
- GxE cache/shard: `<out>.gxe.cache.npz` or directional panels plus
  `<out>.gxe.shard.identity.json` and `<out>.gxe.shard.json`.
- Batched GxE phenotype summaries: `<out>.<trait>.gxe.{gwas,gwis}.tsv.gz` and
  `<out>.<trait>.gxe.moments.json`.
- GxE fit: `<out>.gxe.results.tsv`, `<out>.gxe.fit.json`, `<out>.gxe.log`;
  batch fitting writes one result/JSON pair per manifest trait plus the global
  batch log prefix.
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
