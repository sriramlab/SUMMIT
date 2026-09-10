# Multiple environments

A joint model estimates a covariance matrix of SNP effects across a context
basis. For example, `[1, exposure1, exposure2]` includes baseline effects,
each environmental response, and their covariances. Contexts can be continuous,
categorical, or a specified combination.

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

This estimator has a Python interface. The separate command below plans
resources and inspects results; it does not yet provide a general fit-spec CLI.
The native reference executor currently requires BED and the Linux BLIS build
noted in [Installation](Installation.md).

## Plan resources

```bash
summit-generalized-gxe-variant-ldscore plan \
  --samples 10000 --variants 100000 --basis 3 --annotations 2 --probes 128 \
  --memory-bytes 8589934592 --genotype-format bed --threads 4 \
  --variant-block-width 1024 --rhs-tile-columns 36 --rhs-policy tiled
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
summit-generalized-gxe-variant-ldscore inspect \
  example/out/generalized.generalized-gxe-variant-ldscore-v1.npz
```

The example source shows how to construct the executor and adapt its result.
`scripts/generalized_gxe/workflow.py` contains reference, trait, and fitting
helpers used by the research runners. Those runners require explicit local inputs.

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
