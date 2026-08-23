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
- Full one-environment GENIE-style G + GxE + NxE estimation from marginal
  score summaries and an in-sample XX/XW/WX/WW trace bundle.
- Heritability estimation from `BETA`/`SE` summary statistics.
- Exact chromosome-jackknife batched h2 with an optional reusable binary cache.
- Genetic correlation estimation with either a supplied sample-overlap
  covariance or ancillary summary-only overlap-covariance estimation.
- Batch genetic-correlation manifests, including a sparse fast path for
  supplied-overlap analyses.
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

The supplied environment uses OpenBLAS 0.3.31 or newer; the direct GxE backend
rejects 0.3.30 because it contained a parallel-GEMM race. A portable build uses
the environment BLAS and keeps the independent integrity layer enabled.
Private OpenBLAS remains an explicit compatibility/testing option; it is not
selected automatically and it is not allowed to disable integrity.

The production GxE build uses a private pthread-BLIS archive and keeps the
independent integrity layer enabled. Repeated fixed-input tests isolated the
intermittent corruption to the high-thread BLIS/OpenMP execution path; the
same production-shaped product was bit-identical in 1,000 private
pthread-BLIS calls. OpenBLAS and OpenMP-BLIS remain compatibility and
diagnostic configurations rather than production GxE backends.
The pthread-BLIS path executes the requested probe width in one GEMM, so it
has no backend-specific width-32 subdivision. Native source and target output
scratch is capacity-based, retained across genotype blocks, and released at
the source-to-target phase boundary.

The accepted private pthread-BLIS build disables numerical checksum/recompute
by default after fixed-input backend qualification. It retains immutable
thread ownership, serialized vendor entry, exact input/output mapping and NUMA
contracts, finite/dimension checks, and zero repair/drop requirements. Shared
or OpenBLAS compatibility builds retain checksum protection; a checksum is not
used as a substitute for selecting a clean production backend.

The build records the selected archive SHA-256 and threading layer. Application
OpenMP performs decode and non-BLAS loops; BLIS owns a separate pthread team
whose affinity is inherited from the authenticated CPU set for each call.
This separation avoids the failing nested/shared OpenMP execution path while
preserving explicit placement. `OMP_WAIT_POLICY=PASSIVE` is recommended for
this two-pool design. Explicit user environment settings win. Private OpenBLAS
can still be requested with
`-DGXELDCORE_USE_PRIVATE_OPENBLAS=ON` and
`-DGXELDCORE_PRIVATE_OPENBLAS_ARCHIVE=/path/to/libopenblas.a`, but it remains a
guarded compatibility configuration. Set
`SUMMIT_GXE_VERIFY_FEATURE_MOMENTS=always` only for strict duplicate-product
stress testing. If MKL is available through `MKLROOT` or the active conda
environment, CMake may use MKL instead.

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

- Full annotation tables with `CHR`, `BP`, `SNP`, optional `CM`, and one or more
  annotation columns.
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

SUMMIT does not read BGEN directly. Convert BGEN input to a biallelic diploid
PGEN/PVAR/PSAM trio first.

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
An explicit `.pgen` can replace `.bed`; PGEN uses mean-imputed dosages.

### GxE, heterogeneous noise, and marginal GWIS scores

Generate one reusable phenotype-independent reference for a fixed cohort,
environment, covariate design, SNP set, and annotation. Probe panels are tiled
automatically within the requested memory budget:

```bash
summit \
  --geno ref_panel --env environment.txt --covar covariates.txt \
  --annot mafld.annot.gz \
  --gxe-kernel-mode standardized --gxe-genotype-scale sample \
  --nvecs 1024 --seed 20260808 --rand-dist rademacher \
  --target-mem 16 \
  --out outs/reference.B1024
```

Then score all selected traits in one genotype pass:

```bash
summit \
  --gxe-score-reference outs/reference.B1024.gxe.ref.json \
  --geno ref_panel --env environment.txt --covar covariates.txt \
  --gxe-pheno phenotypes.txt --gxe-pheno-cols trait1,trait2 \
  --out outs/scores
```

The environment file must contain `FID`, `IID`, and exactly one environment
column. Every selected phenotype must be finite on the reference's fixed cohort;
build a common-cohort wide input first. The environment is standardized, and
the intercept, environment, and supplied covariates are projected from the
phenotype and both genetic feature panels. The projected phenotype is rescaled
so `y'y = rank(P)`; the fitter verifies this contract rather than guessing a
score scale.

The reference contains all four directional trace-score panels (`gxx`,
`gxe`, `exg`, and `gee`), per-SNP projected norms/NxE diagonals, and
compact same-person kernel-product statistics when at least two probes are
used. It does not write probe sketches or deletion-jackknife state. Each trait
triplet contains direct marginal additive and interaction scores plus the
indispensable scalar `y' diag(E^2) y`.

The XW and WX matrices are transposes before squaring, but their per-SNP LD
scores are not duplicates: XW contains annotation-weighted squared row norms
of `X'W`, whereas WX contains its squared column norms. Their fully aggregated
normal-equation entries agree only after swapping the left/source annotation
indices. The native pipeline obtains both directions from one shared target
GEMM, so retaining both panels does not repeat the expensive matrix product.

To reuse that reference with a different trait-specific cohort, score one
trait at a time explicitly:

```bash
summit \
  --gxe-score-reference outs/reference.B1024.gxe.ref.json \
  --gxe-population-reference \
  --geno trait_cohort --env trait_environment.txt --covar trait_covariates.txt \
  --gxe-pheno trait.txt --gxe-pheno-col trait1 \
  --out outs/trait1.population
```

This path requires standardized kernels, the same ordered variants/alleles,
annotation and named environment/covariate convention, but permits a different
sample count and trait missingness. It writes exact study NxE and
genetic-by-NxE design moments alongside the marginal scores. Genetic
kernel-product traces are transferred by separate same-person and
different-person finite-cohort factors; this assumes the reference and study
sample the same joint genotype/environment population.

Several environments on an identical complete-case cohort can share every
streamed genotype read while retaining independent four-component models:

```bash
summit \
  --geno ref_panel --env environments.txt --covar covariates.txt \
  --gxe-env-cols age,sex,bmi,alcohol,smoking \
  --gxe-native-backend direct --gxe-parallel-environment-groups auto \
  --num-threads 64 --step_size 2000 \
  --nvecs 128 --seed 20260808 --rand-dist rademacher \
  --target-xz-mem 16 --gxe-total-memory-gib auto \
  --out outs/reference.multi
```

This writes `reference.multi.<environment>.gxe.ref.json` plus a batch manifest
at `reference.multi.gxe.multi.json`. The implementation holds only each
environment's current global source tile in RAM and never spills probe state to
disk. With `--gxe-native-backend direct --rand-dist rademacher`, one persistent
C++ context duplicates
the validated BED/BIM/FAM descriptors and owns BED decode/standardization,
NumPy-compatible Philox probe generation, the packed multi-environment feature
plan, randomized source construction, fixed-effect projection, paired target
products, normalization, and all four directional score reductions. Python
constructs and seals the dimensions/design/annotation plan, validates native
evidence, and transactionally publishes the final arrays; decoded genotype and
intermediate numerical panels never cross the Python/C++ boundary. The same
descriptor-owned path is used for a phenotype-free single `--env` direct
reference (as a one-environment plan). Feature-cache, reference-shard, and
phenotype-scoring workflows retain their purpose-specific paths.
The protected source is accumulated directly into the first half of its final
read-only `[S, e*S]` mapping. Sealing fills only the weighted half, eliminating
one full-panel copy and reducing the panel live peak from three panels to two.
Environments with different missingness are rejected; intersect their
sample rows explicitly or place them in separate batches. The resulting models
are independent per environment, not a cross-environment covariance model.
On a multi-socket host, `auto` may divide four or more direct-backend
environments between two process-isolated socket groups when at least 32 total
threads are requested. Each group has a private BLAS runtime and disjoint
physical cores; a canonical batch manifest is published only after both group
manifests and every referenced artifact pass hash validation. Use `1` for the
lowest aggregate memory footprint or `2` to require the isolated layout.

New GxE reference generation does not create block-local deletion-jackknife
metadata or `.gxe.jackknife.npz` artifacts. These optional outputs did not
change the LD-score point estimates and added a separate, probe-sensitive
uncertainty path. Existing sealed references that already contain jackknife
metadata remain readable for reproducibility, while new fits omit jackknife
standard errors unless uncertainty is supplied by a future explicit method.

```bash
summit \
  --gxe-fit outs/reference.B1024.gxe.ref.json \
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
  "reference": "reference.B1024.gxe.ref.json",
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
summaries are also bound to the exact reference manifest and design, so files
from different cohorts, environments, variants, or kernel conventions cannot
be mixed.

Generation and fitting refuse an existing output prefix unless
`--gxe-overwrite` is supplied explicitly; files are written atomically with
owner-only permissions. See [docs/gxe_genie.md](docs/gxe_genie.md) for the file
contract and validation requirements, and the compiled
[reference-estimator report](docs/reference_gxe_estimator.pdf) for the complete
derivation and GENIE covariate audit.

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

For large trait collections sharing one LD-score/annotation model,
`--h2-batch-fast` reuses exact chromosome-unit sufficient statistics and can
reuse cached score moments via `--h2-cache-dir`. This path is HE-only, requires
`--njack chr[:...]`, and rejects `--chisq-action clip`; see
[Fast batched h2](docs/fast_h2_batch.md).

Additional regression-weighting modes are described in the
[technical documentation](docs/ldsc_weight_mode.md).

### Genetic Correlation With a Supplied Sample-Overlap Covariance

For non-overlapping studies, the supplied sample-overlap covariance is zero:

```bash
summit \
  --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --overlap-covariance-rg 0 \
  --ldscores ref.mafld.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --align-alleles \
  --out outs/trait1.trait2.fixed0 \
  --njack chr
```

For overlapping individual-level samples, use either:

- `--overlap-covariance-rg c_ov`, where
  `c_ov = y_overlap' y_overlap / sqrt(N1*N2)`, or
- `--pheno-rg pheno1.txt,pheno2.txt` with optional
  `--pheno-rg-cov cov1.txt,cov2.txt` so SUMMIT computes the sample-overlap
  covariance.

Phenotype files must have a header, `FID IID`, and the phenotype in the last
column. Covariate files must have `FID IID` followed by covariates.

### Genetic Correlation With Summary-Only Overlap-Covariance Estimation

When `--overlap-covariance-rg` and `--pheno-rg` are omitted, SUMMIT estimates
the overlapping phenotype covariance from HE-scale summary moments first, then
passes it to the main covariance equation.

The LD scores used for summary-only overlap-covariance estimation must be
one-dimensional. If the primary analysis uses partitioned LD scores, provide a
separate scalar `--ldscores-reg` file:

```bash
summit \
  --rg trait1.sumstats.gz,trait2.sumstats.gz \
  --ldscores ref.mafld.gw.ldscore.gz \
  --ldscores-reg ref.total.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --align-alleles \
  --out outs/trait1.trait2.summary_overlap_covariance \
  --njack chr
```

SUMMIT uses a scalar LD score for the summary-only overlap-covariance fit. A 1D
`--ldscores-reg` file is preferred. If a multi-column regression LD file is
provided, SUMMIT collapses it to total LD by default; this is valid only when
the LD-score columns are non-overlapping, such as a disjoint MAF-LD partition.
For overlapping annotations, precompute or provide a genuine scalar regression
LD score instead.

### rg Allele Harmonization

Allele validation and alignment are enabled by default in single-pair,
regular-manifest, fast-manifest, and multi-model rg. Named `A1/A2` or `ALT/REF`
columns are required. Direct, swapped, strand-complement, and swapped-
complement orientations are handled vectorially, with trait-2 effects flipped
only when needed.

Harmonization currently accepts A/C/G/T alleles. Invalid labels, equal alleles,
incompatible pairs, and strand-ambiguous A/T or C/G SNPs are dropped by
default. `--keep-ambiguous` uses literal allele-label orientation and does not
infer strand from allele frequency; EAF-assisted resolution is not implemented.
Use `--no-align-alleles` only for inputs already guaranteed to have identical
orientation.

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

Run a manifest with supplied overlap covariances:

```bash
summit \
  --rg outs/rg_manifest.tsv \
  --rg-manifest-fast \
  --ldscores ref.mafld.gw.ldscore.gz \
  --annot mafld.annot.gz \
  --out outs/rg_batch \
  --njack chr
```

Regular manifests may mix finite supplied `overlap_covariance` values with
omitted values; omitted values use SUMMIT's summary-only overlap-covariance
estimation and matching delete refits. The fast path requires a finite supplied
`overlap_covariance` in every row, uses HE weighting and jackknife SEs only, and
does not support
`--adjust-delta` or normal-equation dumps. With chromosome jackknife its fixed-
unit sparse corrections reproduce the regular supplied-overlap estimator.
Integer block mode instead uses pre-drop blocks and therefore differs from the
regular post-filter block jackknife. Omit `--rg-manifest-fast` for alternative
weighting or summary-only overlap-covariance estimation. Add
`--rg-fast-no-pair-logs` for large batches to retain only `batch.log` and
`manifest.results.tsv`.

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
toy `BETA`/`SE` files under `example/out/`. The supplied-overlap rg example
reuses the same toy signal twice and therefore sets
`--overlap-covariance-rg 1`; use `--overlap-covariance-rg 0` for genuinely
non-overlapping studies.

```bash
cd example
./estimate_gwldscore.sh
./estimate_partitioned_gwldscore.sh
./estimate_windowed_ldscore.sh
./estimate_gxe_ldscore.sh
./h2_ldscore.sh
./rg_supplied_overlap_covariance.sh
./rg_unconstrained.sh
./rg_manifest_supplied_overlap_covariance.sh
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
- `--ldscores-reg`: optional scalar LD-score file for summary-only rg
  overlap-covariance estimation.
- `--annot`: annotation file or split-file spec; omitted means single component.
- `--max-chisq`: main chi-square filter; use `auto` for `max(80, 0.001*Nmax)`.
- `--overlap-covariance-chisq-thr`: chi-square filter used only for summary-only
  overlap-covariance estimation.
- `--chisq-action`: `drop`, `clip`, `warn`, or `none`.
- `--overlap-covariance-rg`: supplied sample-overlap covariance on SUMMIT's HE
  scale.
- `--pheno-rg`, `--pheno-rg-cov`: compute the supplied sample-overlap covariance
  from overlapping phenotype/covariate files.
- `--align-alleles`: validate and align the second summary-statistic file to
  the first (default).
- `--no-align-alleles`: explicitly skip allele validation/alignment for inputs
  already guaranteed to have identical orientation.
- `--keep-ambiguous`: keep strand-ambiguous SNPs during allele alignment.
- `--njack`: integer SNP blocks, `chr`, `chr:d`, `chr:d:R`, or `chr:d:R:seed`.
- `--rg-se-method`: `jackknife`, `delta`, `robust`, or `kmoments`.
- `--write-jack`: save jackknife replicate dumps.
- `--write-normeq`: save explicit SCORE normal-equation JSON for eligible rg runs.
- `--h2-batch-fast`, `--h2-cache-dir`: exact chromosome-jackknife HE batching
  and optional reusable trait-moment cache.
- `--rg-manifest-fast`, `--rg-model-manifest`: supplied-overlap HE batching and
  optional multi-model column selection.
- `--rg-fast-no-pair-logs`: omit fast-mode per-pair logs while retaining the
  batch log and combined result table.

### LD-score options

- `--nvecs`: random vectors for stochastic genome-wide LD scores.
- `--write-ld-mc-var` / `--write-ld-mc-ci`: additionally write per-SNP,
  per-annotation MC variances, SEs, and approximate pointwise 95% conditional
  MC intervals; the compact annotation-level diagnostic is written by default
  when `nvecs >= 2` unless skipped.
- `--skip-ld-mc`: disable the default MC diagnostic.
- `--step_size`: SNP block size for LD-score computation; for PGEN this also
  controls the reusable dosage decode buffer.
- `--win-panel-cols`, `--win-cache-mb`: decoded panel width and bounded
  prepared-panel cache for PGEN windowed LD. Automatic caching is capped and
  also respects `--target-mem` when supplied.
- `--covar`: covariate file with `FID IID` and covariate columns.
- `--env`: one-column environment file for the GxE reference/score bundle.
- `--gxe-score-reference`: produce marginal scores and NxE moments from a
  sealed reference; `--gxe-pheno-cols` selects columns from `--gxe-pheno`.
- `--gxe-population-reference`: explicitly score one `--gxe-pheno-col` in a
  different cohort using a standardized population reference.
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
- `--gxe-parallel-environment-groups auto|1|2`: for four or more direct
  references sharing one cohort, use one process or two socket-local isolated
  groups. `auto` selects two only when the allowed topology and total thread
  request support it.
- `--force_affinity_all false` (default): preserve scheduler or `taskset`
  placement. Set true only when intentionally discarding an inherited cpuset.
- `--use-mailman auto` (default): packed Mailman execution is eligible only at
  `B<=10`. The additive and direct GxE estimators share the native Mailman
  pre/post primitives; the GxE context supplies its own interaction/projection
  transforms around those primitives. Every `B>10` direct GxE job uses dense
  BLIS, with memory tiling when required—never Mailman and never a hard-coded
  width-32 backend split.
- `--device`: `cpu` or `cuda[:index]` for supported genome-wide LD-score runs.
- `--num-threads`: cap BLAS/OpenMP thread pools.
- `--target-xz-mem`, `--target-mem`: sketch-panel budgets in GiB or `auto`
  (default). Auto uses the tightest observable memory limit and retains both a
  fixed reserve and a proportional margin; cluster workflows should keep an
  explicit budget when the scheduler allocation is not visible to the process.
- `--gxe-total-memory-gib`: independent modeled complete-process peak in GiB
  or `auto` (default). The direct GxE planner jointly searches environment and
  probe tile counts, minimizes genotype passes, and rejects plans whose decoded
  blocks, native context/output state, integrity/vendor workspace, thread
  stacks, allocator slack, publication allowance, and 20% headroom exceed this
  bound. The selected plan and exact phase/component arithmetic are retained in
  reference, group, and combined manifests.

## Output Files

- Genome-wide LD scores: `<out>.gw.ldscore.gz`, `<out>.gw.M`, and
  `<out>.gw.log`; `<out>.gw.mc.tsv` is written by default when `nvecs >= 2`
  unless `--skip-ld-mc`; `<out>.gw.mcvar.gz` and `<out>.gw.kmoments` are
  optional.
- Windowed LD scores: `<out>.win.ldscore.gz`, `<out>.win.M`,
  `<out>.win.M_5_50`, `<out>.win.log`.
- GxE reference: `<out>.{gxx,gxe,exg,gee}.ldscore.gz`,
  `<out>.gxe.diag.tsv.gz`, and `<out>.gxe.ref.json`. Reference construction
  writes no deletion-jackknife or probe-state artifact.
- Batched GxE phenotype summaries: `<out>.<trait>.gxe.{gwas,gwis}.tsv.gz` and
  `<out>.<trait>.gxe.moments.json`.
- GxE fit: `<out>.gxe.results.tsv`, `<out>.gxe.fit.json`, `<out>.gxe.log`;
  batch fitting writes one result/JSON pair per manifest trait plus the global
  batch log prefix.
- h2: `<out>.results.tsv`, `<out>.log`, optionally `<out>.<trait>.jack`; the
  result table records the estimator mode.
- rg: `<out>.log`, optionally `<out>.rg.jack` and
  `<out>.rg.scoreeq.json`.
- rg manifest: `<out>/batch.log`, `<out>/manifest.results.tsv`, and per-pair
  logs/results where applicable; the combined table records the estimator mode.

## Citation

If you use SUM-RHE/SUMMIT, please cite:

```text
Jeong, M., Pazokitoroudi, A., Liu, Z., & Sankararaman, S. (2024).
Scalable summary statistics-based heritability estimation method with
individual genotype level accuracy. Genome Research, gr.279207.124.
https://doi.org/10.1101/gr.279207.124
```
