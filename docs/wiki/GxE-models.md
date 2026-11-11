# G×E models

For quantitative traits, the one-environment model estimates additive genetic
variance, interaction variance, residual variance, and environment-dependent
residual variance. It can reuse one reference calculation across traits.
SUMMIT-GxE estimates the joint model of many exposures; see
[SUMMIT-GxE: many exposures](#summit-gxe-many-exposures).
For binary outcomes, see [Generalized G×E PCGC](Generalized-GxE-PCGC.md).

## Inputs

Supply genotypes, a numeric environment table, covariates, and optional SNP
annotations. Environment and covariate tables start with `FID IID`. The
reference uses one fixed set of samples and variants.

## 1. Build the reference

```bash
summit --geno reference.bed --env environment.tsv --covar covariates.tsv \
  --nvecs 1024 --seed 1 --rand-dist rademacher --memory-gib 16 \
  --num-threads 8 --out results/reference
```

The default standardizes the environment and adjusts for the intercept,
environment, and covariates. Genetic and interaction feature columns are
normalized after that adjustment. `results/reference.gxe.ref.json` identifies
the saved reference files. The reference can be reused only with compatible
SNPs, annotations, and feature definitions.

### Native backend on Linux and macOS

The default one-environment backend uses NumPy. To select the optimized native
backend, add `--gxe-native-backend direct --rand-dist rademacher`. BED input is
supported on both Linux and macOS with the OpenBLAS build described in
[Installation](Installation.md).

For a small native example, run from the repository root:

```bash
python example/prepare_example_inputs.py
summit --geno example/out/synthetic/small.bed \
  --env example/out/synthetic/small.env \
  --annot example/out/synthetic/small.annot \
  --covar example/out/synthetic/small.cov \
  --out example/out/small.2bins.env.native \
  --nvecs 100 --block-size 256 --seed 1 --dtype float64 --num-threads 2 \
  --gxe-native-backend direct --rand-dist rademacher
```

Use a new output prefix if you have already run this command. The native path
requires Rademacher probes; set the same distribution in a NumPy run when
comparing results. The shell example `estimate_gxe_ldscore.sh` uses the default
probe distribution, so its estimates need not match this native example exactly.

## 2. Compute trait summaries

```bash
summit --gxe-score-reference results/reference.gxe.ref.json \
  --geno reference.bed --env environment.tsv --covar covariates.tsv \
  --gxe-pheno phenotypes.tsv --gxe-pheno-cols trait1,trait2 \
  --num-threads 8 --out results/scores
```

All selected traits must be finite on the reference sample set. SUMMIT adjusts
and normalizes the phenotypes and computes marginal additive and interaction
scores in one genotype pass. Each trait produces additive `.gxe.gwas.tsv.gz`
and interaction `.gxe.gwis.tsv.gz` tables, plus a `.gxe.moments.json` file
needed for fitting.

A conventional conditional interaction statistic, such as PLINK's `ADDxE`
coefficient, is not the marginal interaction score required here. Use SUMMIT's
trait summaries unless you have verified the generating formula independently.

## 3. Fit a trait

```bash
summit --gxe-fit results/reference.gxe.ref.json \
  --gxe-gwas results/scores.trait1.gxe.gwas.tsv.gz \
  --gwis results/scores.trait1.gxe.gwis.tsv.gz \
  --gxe-moments results/scores.trait1.gxe.moments.json \
  --njack 200 --out results/trait1
```

Results include `.gxe.results.tsv`, `.gxe.fit.json`, and `.gxe.log`.
Standard errors use SNP-block deletion of the completed reference and trait
summaries. Reference LD scores are held fixed during those deletions.

Use a new output prefix. `--gxe-overwrite` explicitly permits replacing an
existing result.

## Many traits or environments

`--gxe-fit-batch fit_batch.json` fits multiple traits against one reference.
The manifest identifies each trait's moments, GWAS, GWIS, and output prefix:

```json
{
  "kind": "summit.gxe.fit_batch",
  "schema_version": 1,
  "reference": "reference.gxe.ref.json",
  "traits": [{
    "name": "trait1",
    "moments": "scores.trait1.gxe.moments.json",
    "gwas": "scores.trait1.gxe.gwas.tsv.gz",
    "gwis": "scores.trait1.gxe.gwis.tsv.gz",
    "out": "fits/trait1"
  }]
}
```

`--gxe-env-cols exposure1,exposure2` builds independent one-environment models
while sharing genotype reads. The environments must have the same complete
sample set. For a joint model that includes covariance between environmental
responses, see [SUMMIT-GxE](#summit-gxe-many-exposures) and
[Multiple environments](Multiple-environments.md).

## A different study cohort

`--gxe-population-reference` allows scoring one selected trait in a different
cohort. Supply that cohort's genotypes, environment, covariates, and phenotype,
with `--gxe-pheno-col`. The model requires matching variant and feature
definitions and assumes compatible genotype–environment distributions between
reference and study. Cohort-specific residual moments are computed in the study.

Each cohort standardizes its own environment and normalizes its own features,
so the reference's kernel moments are moved to the study's people
([Methods](Methods.md#classic-one-environment-transfer)). The study's people
can differ from the reference in their environment distribution, for example
when missingness depends on age. The reference records its environment's power
sums, and scoring records the study's. The fit then forms the study's
same-person moments and corrects the different-person moments for the study's
environment composition.

A reference or score set made before these sums were recorded still fits. Its
fit keeps the earlier sample-size scaling and warns: a study that is not a
random sample of the reference with respect to the environment then
misestimates the G×E variance. Rebuild the reference and rescore the trait to
use the correction. The fit's `normal_equation_diagnostics` record
`same_person_transport` and `different_person_transport`.

The G×E variance refers to the study's own standardized environment. When the
study's environment mean differs from the reference's, the two cohorts' G×E
components are not the same quantity, because the model has no G–G×E
covariance term.

## Alternative scaling

`--gxe-kernel-mode raw_projected --gxe-genotype-scale hwe` uses the natural
projected column norms of HWE-scaled genotypes for comparison with GENIE's
kernel definition. It changes the variance-component interpretation. Keep
reference and trait scaling consistent and report the chosen mode.

[Methods](Methods.md) describes the feature definitions and reference transfer.

## SUMMIT-GxE: many exposures

SUMMIT-GxE is the multi-exposure gene-by-context model of Jeong and
Sankararaman, "Joint estimation of gene-by-context covariance across many
exposures in biobanks". For K exposures with standardized values
`c = (c_1, ..., c_K)` and basis `phi(c) = (1, c_1, ..., c_K)`, the effect of
SNP j in context c is `phi(c)' b_j`, with

```math
\mathrm{Cov}(b_j)=\Omega/M,\qquad
\Omega=\begin{pmatrix}\omega_{00}&\omega_{0E}^T\\ \omega_{0E}&\Omega_{EE}\end{pmatrix}.
```

`omega_00` is the additive genetic variance at the average context,
`omega_0E` holds the covariances between the additive and context-specific
coefficients, and `Omega_EE` is the covariance matrix of the context-specific
coefficients. The residual variance `phi(c)' Psi phi(c)` also depends on the
exposures. SUMMIT-GxE estimates Ω and Ψ jointly by the method of moments from
two inputs: the per-SNP interaction statistics of each trait, and a reference
of stochastic directional LD scores that depends only on the genotypes,
exposures and covariates. The reference is computed once for a set of
individuals and a basis, and reused for every trait measured in them. This is
the joint model of
[Multiple environments](Multiple-environments.md) with the basis above;
[Methods](Methods.md#generalized-per-snp-reference-ld-scores) gives the
equations.

The amplification, effect heterogeneity and G×E heritability of the paper are
functions of Ω:

```math
a_k=\frac{\omega_{0k}}{\omega_{00}},\qquad
S=\Omega_{EE}-\frac{\omega_{0E}\,\omega_{0E}^T}{\omega_{00}},\qquad
h^2_{G\times E}=\mathrm{tr}(\Sigma_c\Omega_{EE})
=\omega_{00}\,a^T\Sigma_c\,a+\mathrm{tr}(\Sigma_c S).
```

The amplification `a_k` is the proportional change of all genetic effects per
standard deviation of exposure k. The effect heterogeneity S is the covariance
of the context-specific coefficients that is uncorrelated with the additive
coefficients; [Methods](Methods.md#amplification-and-residual-response) writes
the same decomposition with `gamma` for a. The G×E heritability uses
`Sigma_c`, the covariance matrix of the standardized exposures in the
individuals of the model, and is usually reported relative to `omega_00`. The
genetic covariance between contexts c and c' is `phi(c)' Omega phi(c')`, which
gives genetic correlations and variance ratios between contexts. The fit
returns Ω; [Outputs and standard errors](#outputs-and-standard-errors) shows
how to compute these quantities and their standard errors.

### Running SUMMIT-GxE

SUMMIT-GxE runs through the Python API. On the command line,
`summit reference plan` sizes the reference step without reading genotypes,
and `summit reference inspect` validates a written reference; there is no
command-line fit. For nine exposures and four SNP classes:

```bash
summit reference plan --samples 300000 --variants 450000 --basis 10 \
  --annotations 4 --nvecs 128 --memory-gib 150 --genotype-format bed \
  --num-threads 8
```

The reference executor requires PLINK BED input and a native build described
in [Multiple environments](Multiple-environments.md#workflow). The helpers
below are in `scripts/generalized_gxe/workflow.py`, which the research runners
import; it is not part of the installed package. Prepare the inputs on one
ordered set of individuals:

- **Exposures.** Center each exposure and scale it to unit variance in the
  individuals of the model. Code binary exposures 0/1 and standardize them in
  the same way. `standardized` uses the sample standard deviation.
- **Covariates.** Supply the mean model as one design matrix with an intercept
  column. It should contain the exposures, their squares and pairwise
  products, the products of each exposure with the leading genotype PCs, and
  the other covariates. SUMMIT does not create these columns. Exposure terms
  omitted from the mean can be mistaken for G×E, and omitted exposure-by-PC
  terms for effect heterogeneity. `fixed_basis(design)` returns the
  orthonormal basis that both steps project out and drops collinear columns,
  such as the square of a binary exposure.
- **Residual variance.** `rank_reduced_symmetric_context_residual_basis`
  returns the products of basis columns used by the residual-variance model.
  It drops the square of a binary exposure, which the other terms span.
- **SNP classes and blocks.** One column of ones gives a single class of SNPs;
  see [SNP classes](#snp-classes). `balanced_inference_block_ids(M, 200)`
  assigns 200 contiguous jackknife blocks in BIM order whose sizes differ by
  at most one SNP; the fit requires such balanced blocks.

```python
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts/generalized_gxe")  # from the repository root
from workflow import (
    balanced_inference_block_ids, fixed_basis, read_plink_axes,
    rank_reduced_symmetric_context_residual_basis, run_reference,
    run_trait_batch, standardized,
)
from summit import gxeldcore
from summit.ldscore.generalized_gxe_fit_v1 import fit_generalized_gxe_variant_model_v1

axes = read_plink_axes("geno/cohort")  # cohort.bed, .bim and .fam
# exposures (N x K), design (N x F) and phenotypes (N x T) follow axes.sample_ids
contexts = np.column_stack([standardized(column) for column in exposures.T])
basis = np.column_stack([np.ones(axes.n), contexts])
basis_names = ("intercept", *exposure_names)
fixed = fixed_basis(design)
residual, residual_names, _ = rank_reduced_symmetric_context_residual_basis(
    basis, basis_names
)
annotations, annotation_names = np.ones((axes.m, 1)), ("all",)
blocks, labels = balanced_inference_block_ids(axes.m, 200)
shared = dict(
    axes=axes, basis=basis, fixed=fixed, annotations=annotations,
    annotation_names=annotation_names, inference_block_ids=blocks,
    inference_block_labels=labels, residual_names=residual_names,
)

# Reference: 128 random-sign variant probes, two genotype passes.
reference = run_reference(
    **shared, basis_names=basis_names, probes=128, seed=1, threads=8,
    memory_bytes=64 * 2**30, native_module=gxeldcore,
    output=Path("results/reference"),
)
# Interaction statistics of all traits in one genotype pass.
traits, _, _ = run_trait_batch(
    **shared, reference_run=reference, phenotypes=phenotypes,
    trait_names=trait_names, residual_basis=residual,
    output=Path("results/traits"),
)
fit = fit_generalized_gxe_variant_model_v1(
    reference.artifact, traits, trait_selector=trait_names[0]
)
omega = fit.raw_omegas.sum(axis=0)  # (K+1) x (K+1), summed over SNP classes
```

The two steps write `results/reference.generalized-gxe-variant-ldscore-v1.npz`
and `results/traits.generalized-gxe-trait-summary-v1.npz`. The trait step
projects the covariates out of each phenotype and scales it to squared norm
N − F. Phenotypes must be finite in every selected individual. To add traits
later, load the reference with `load_generalized_gxe_variant_reference_v1` and
pass `ReferenceRun(artifact, None, path)` from `workflow.py` as
`reference_run`. Saved files are fitted as shown in
[Multiple environments](Multiple-environments.md#reference-and-trait-files).

### Outputs and standard errors

| Output | Field |
|---|---|
| Ω of each SNP class | `fit.raw_omegas`, classes × (K+1) × (K+1) |
| Genome-wide Ω | `fit.raw_omegas.sum(axis=0)` |
| Residual-variance coefficients Ψ | `fit.raw_residual_coefficients`, in the order of `residual_names` |
| Coefficients and their covariance | `fit.raw_coefficients`, `fit.raw_covariance` |
| Standard errors | `fit.raw_standard_errors`; jackknife alone in `fit.raw_jackknife_standard_errors` |
| How the standard errors were formed | `fit.manifest["uncertainty"]` |

Standard errors use a delete-one jackknife over the reference's SNP blocks,
with the directional LD scores held fixed, plus the between-block signal
covariance that the jackknife misses
([Methods](Methods.md#between-block-signal-covariance)). Because the
phenotype contains the genetic values of every block, block contributions are
not independent, and the jackknife alone understates the variance of
amplification, increasingly so at higher heritability. The added term is the
default since commit 1f136ac and uses only aggregates saved with the reference
and trait summary. `standard_errors="jackknife"` keeps the jackknife alone. A
trait summary without context moments falls back to the jackknife with a
`CrossBlockSignalUnavailableWarning`.

Derived quantities are recomputed in each jackknife replicate, and the
between-block term is propagated by the delta method:

```python
from summit.context.crossblock_signal import derived_variance
from summit.context.oracle import coefficients_to_omegas

c = len(fit.component_index)
sigma_c = np.cov(contexts, rowvar=False)  # K x K, held fixed

def omega_of(theta):
    return coefficients_to_omegas(theta[:c], fit.component_index).sum(axis=0)

def amplification(k):
    return lambda theta: omega_of(theta)[0, k] / omega_of(theta)[0, 0]

def heterogeneity(k):  # diagonal entry S_kk
    return lambda theta: (
        omega_of(theta)[k, k] - omega_of(theta)[0, k] ** 2 / omega_of(theta)[0, 0]
    )

def h2_gxe_ratio(theta):  # G×E heritability relative to omega_00
    om = omega_of(theta)
    return np.trace(sigma_c @ om[1:, 1:]) / om[0, 0]

for name, f in [("a_1", amplification(1)), ("S_11", heterogeneity(1)),
                ("h2gxe_over_h2", h2_gxe_ratio)]:
    result = derived_variance(f, fit.raw_coefficients, fit.raw_loo_coefficients,
                              fit.raw_crossblock_covariance)
    print(name, result["estimate"], result["standard_error"])
```

`derived_variance` also returns the jackknife and between-block parts of the
variance separately.

### Single-exposure analyses

A single-exposure analysis of exposure k reduces the genetic basis to
`(1, c_k)` and keeps the same individuals, covariates and residual-variance
terms. It is a restricted fit of the joint system. The
[chromosome estimator](Chromosome-LD-and-annotation-prediction.md#fitting-with-standard-errors),
which computes directional LD scores between SNPs of the same chromosome as in
the paper, fits the joint and restricted models on one set of
jackknife systems with one shared between-block term.
`chromosome_paired_contrast` then gives the standard error of a
joint-minus-single difference:

```python
from summit.context.spec import ContextComponentIndex, ContextPairIndex
from summit.ldscore.generalized_gxe_chromosome import (
    chromosome_paired_contrast, fit_transferred_chromosome_models)

index = ContextComponentIndex(annotation_names, ContextPairIndex(len(basis_names)))
c = len(index)
total = c + len(residual_names)  # genetic coefficients, then residual

def single(k):
    genetic = [e.index for e in index.entries if {e.q, e.r} <= {0, k}]
    return np.array(genetic + list(range(c, total)))

# reference_chunks, trait_chunks, transport, rows and moments as on that page
fits = fit_transferred_chromosome_models(
    reference_chunks, trait_chunks, transport=transport, study_rows=rows,
    models={"joint": np.arange(total), "single_1": single(1)}, **moments)
a1 = lambda om: om.sum(axis=0)[0, 1] / om.sum(axis=0)[0, 0]
fits["joint"].derived(a1)
chromosome_paired_contrast(fits["joint"], a1, fits["single_1"], a1)
```

For a reference built with `run_reference`, which sums over all SNP pairs of
the genome, `restricted_genetic_fit` in `workflow.py` fits a restricted model
with one SNP class and jackknife standard errors only.

### Traits measured in subsets

The reference is computed in all individuals of the model. For a trait
measured in some of them, pass their FAM row indices `rows`, in increasing
order, as `row_selection` to `run_trait_batch`, with `basis[rows]`,
`residual[rows]`, `phenotypes[rows]` and `fixed_basis(design[rows])`. Keep the exposures
standardized in all individuals. The fit forms the trait's own same-person
terms and corrects the different-person terms for the trait's exposure
distribution
([transfer](Multiple-environments.md#transferring-the-reference-to-the-traits-people)).
With a reference written with `mode="composable"`, passing `study_rows=rows`
to the fit gives the exact same-person term; keep such references in
protected storage.

### SNP classes

Each annotation column gives a class of SNPs its own covariance, and the
genome-wide Ω is their sum. Amplification, effect heterogeneity and G×E
heritability are computed from the sum, as `omega_of` does above. The paper's
main analysis used four classes: MAF below or at least 0.05, each split at the
median LD score within the MAF class. With per-SNP `maf` and `ldscore` vectors
in BIM order, for example SUMMIT [LD scores](LD-scores.md) of the same
genotypes:

```python
common = maf >= 0.05
high_ld = np.zeros(maf.size, dtype=bool)
for group in (~common, common):
    high_ld[group] = ldscore[group] > np.median(ldscore[group])
annotations = np.column_stack(
    [g & h for g in (~common, common) for h in (~high_ld, high_ld)]
).astype(float)
annotation_names = ("maf_low_ld_low", "maf_low_ld_high",
                    "maf_common_ld_low", "maf_common_ld_high")
```

Pass these to `run_reference` and `run_trait_batch` in place of the single
column. With the chromosome estimator,
`combine_chromosome_annotations(chunk, np.ones((4, 1)), ("all",))` merges the
four classes into one without reading genotypes again.
