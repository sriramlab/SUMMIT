# Multiple environments

For a quantitative trait, a joint model estimates a covariance matrix of SNP
effects across a context basis. For example, `[1, exposure1, exposure2]` includes
baseline effects, each environmental response, and their covariances. Contexts
can be continuous, categorical, or a specified combination.

The [cross-trait extension](Cross-trait-analysis.md) estimates the corresponding
covariance between two traits, with separate sample masks and overlap handling.

For binary disease outcomes, use the [generalized PCGC path](Generalized-GxE-PCGC.md).
The PCGC command estimates the same types of effect covariance on a declared
liability scale, accounting for disease risks and case–control sampling.
It prepares its own reference and trait moments.

For quantitative analyses split across chromosomes, see
[Chromosome LD and annotation prediction](Chromosome-LD-and-annotation-prediction.md).

## Workflow

1. Fit the context coding on the reference cohort: continuous centers/scales,
   categorical levels, and any retained basis transformation.
2. Specify fixed effects, SNP annotations, and one genotype scaling rule.
3. Estimate generalized per-SNP reference LD scores.
4. Compute study trait summaries with the same variant and context definitions.
5. Fit genetic and residual components; use post-hoc SNP blocks for standard errors.

The reference estimator reads genotypes twice. It uses one genotype scale
across all context columns. The context-dependent features are
`P diag(phi_q) G`, where P removes fixed effects.

This estimator has a Python interface. The command below plans
resources and inspects results; it does not yet provide a general fit-spec CLI.
The native reference executor requires BED and supports protected OpenBLAS on
Linux and macOS, or private BLIS on Linux; see [Installation](Installation.md).

## Plan resources

```bash
summit reference plan \
  --samples 10000 --variants 100000 --basis 3 --annotations 2 --nvecs 128 \
  --memory-gib 8 --genotype-format bed --num-threads 4 \
  --block-size 1024 --rhs-tile-columns 36 --rhs-policy tiled
```

Planning reads no genotypes. More basis columns increase the number of genetic
components as `Q*(Q+1)/2`; this can increase computation substantially.

## Synthetic reference example

From the repository root:

```bash
python example/prepare_example_inputs.py
env BLIS_NUM_THREADS=4 OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 \
  python example/estimate_generalized_gxe_variant_ldscore.py \
  --output example/out/generalized --probes 16 --njack 20 --threads 4
summit reference inspect \
  example/out/generalized.generalized-gxe-variant-ldscore-v1.npz
```

The example source shows how to construct the executor and adapt its result.
`scripts/generalized_gxe/workflow.py` contains reference, trait, and fitting
helpers used by the research runners. Those runners require explicit local inputs.

## Reference and trait files

The quantitative Python workflow stores the two inputs separately:

- `.generalized-gxe-variant-ldscore-v1.npz` contains the reference LD-score moments.
- `.generalized-gxe-trait-summary-v1.npz` contains the corresponding trait moments.

Once both files have been prepared with matching definitions, fit a trait with:

```python
from summit.ldscore.generalized_gxe_reference_v1 import (
    load_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_trait_summary import (
    load_generalized_gxe_trait_summary,
)
from summit.ldscore.generalized_gxe_fit_v1 import fit_generalized_gxe_variant_model_v1

reference = load_generalized_gxe_variant_reference_v1(
    "results/reference.generalized-gxe-variant-ldscore-v1.npz"
)
trait = load_generalized_gxe_trait_summary(
    "results/trait.generalized-gxe-trait-summary-v1.npz"
)
fit = fit_generalized_gxe_variant_model_v1(reference, trait, trait_selector="trait1")
```

Use a trait name saved in the summary. The fitter checks compatibility and
uses the saved SNP-block definitions for its jackknife. A reference can be
reused across traits with compatible samples, scaling, contexts, and fixed
effects. [PCGC preparation](Generalized-GxE-PCGC.md) also saves separate reference and
trait files, using `.binary.ldscores.npz` and `.binary.sumstats.npz`. Its fitter
requires the reference identified by the trait summary.

## Reusing annotation columns

Default `summary` output stores the reference summaries needed for fitting.
Optional `composable` output also retains annotations, directional panels,
and sample-aligned component diagonals. **Keep composable files in protected
storage when the input is participant data.**

`compose_generalized_gxe_variant_references_v1` combines compatible annotation
columns without reading genotypes again. It requires the same ordered samples,
variants, context definitions, and scaling. Verify those inputs from the
original reference records; equal dimensions do not establish equivalence.

## Reading a joint fit

The fitted matrix Ω gives the genetic covariance between contexts x and z as
`x.T @ Ω @ z`; setting x=z gives genetic variance. Raw moment estimates may be
indefinite. An optional positive-semidefinite projection should be reported
separately from the raw estimate.

Eigenvalues depend on the reference context metric. Repeated eigenvalues
identify a subspace, so individual eigenvectors should not receive separate
biological interpretations. With overlapping annotations, report the combined
surface before interpreting conditional annotation coefficients.

[Methods](Methods.md) gives the equations. [Contextual Python API](Contextual-Python-API.md)
covers categorical summaries, transformations, and other research functions.
