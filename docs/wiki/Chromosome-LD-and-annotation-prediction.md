# Chromosome LD and annotation-dependent prediction

The chromosome estimator computes within-chromosome SNP-pair LD and combines
the moments in a single genome-wide fit. It estimates one covariance matrix
per annotation using the same participants, genotype scaling, context basis,
and fixed-effect span for reference and trait summaries. Population transfer
is not implemented for these summaries.

## Normalization and approximation

For annotation weights A[j,k], define M[k] as their sum over all chromosomes.
A chromosome contributes

```text
K[c,k,qr] = sum_{j in c} A[j,k] F[q,j] F[r,j]' / M[k]
```

with the symmetric second orientation included when q != r, and
`F[q] = (I - UU') diag(phi[q]) G`. Global annotation masses give each chromosome
its share of the genome-wide variance. An annotation absent on one chromosome
contributes zero there.

Joint estimation includes one shared set of residual coefficients. The
approximation sets cross-chromosome residual-projected kernel products to
zero. Local LD makes this plausible, but population structure, relatedness,
and genotype–context dependence can affect its accuracy. Assess agreement
with a genome-wide reference for the intended cohort.

## Preparing and combining chromosome summaries

`GeneralizedGxENativeBEDExecutor` can read a chromosome from a merged BED using
`variant_start` and `annotations.shape[0]`, or from a chromosome BED with
`variant_start=0`. It makes two passes over that interval. Check chromosome
membership and shared sample and variant identities before execution.

Supplying normalized residual `phenotypes` and a `residual_basis` computes
trait summaries in the same passes. Set the planner's `num_traits` and
`num_residual_components` accordingly. Sharing this reference across traits
requires identical sample and design definitions.

`reduce_chromosome_result` reduces the reference and trait moments.
`write_chromosome_moments` saves them with checksums and provenance to a new
file. `joint_chromosome_equations(..., expected_chromosomes=range(1, 23))`
checks for missing, duplicate, and incompatible chromosome contributions
before assembly. The residual basis must begin with the constant-one kernel.

`combine_chromosome_annotations(chunk, weights, names)` reuses these summaries
for `A_new = A @ weights`, including nonnegative overlapping combinations.
An all-one column combines disjoint bins covering the panel into an
unpartitioned model. The fit recomputes annotation masses from the combined
moments, allowing overall and partitioned estimates from the same genotype
passes.

SNP-block deletion removes selected target rows and updates retained annotation
masses while holding source reference sketches fixed. This is an approximate
jackknife; it excludes reference-probe variation and cross-chromosome LD.

## Annotation-dependent prediction

The public prediction API accepts `AnnotationDesign` and `AnnotationPrior`:

```python
from summit.prediction import AnnotationDesign, AnnotationPrior

design = AnnotationDesign(
    weights=annotations,              # M by K, nonnegative; overlap allowed
    names=annotation_names,
    variant_identity=trait.scale.variant_identity,
)
prior = AnnotationPrior(design, covariances)  # K by Q by Q, each PSD
candidate = prior.candidate(
    "annotated", residual_variances,
    {"source": "discovery-only covariance estimates"},
)
```

The SNP prior is `Lambda[j] = sum_k A[j,k] Omega[k] / M[k]`. Homogeneous and
annotated candidates can share genotype reads and run in the same batch.
Annotation designs must match the ordered training SNPs and alleles.

Native annotation fits using BLIS require `GXELDCORE_GEMM_INTEGRITY=ON`
and `GXELDCORE_GEMM_CHECKSUM=ON` when building. These numerical checks are
required by the prediction backend.

Planning, checkpoints, model saving, and scoring use the regular
[prediction API](PGS-API.md). The fitted weights use the supplied per-SNP
priors. Scoring saved weights requires no annotation input.

For the CLI, write a design with `write_annotation_design(path, design)` and
use a candidate with `operation: "annotation"`, `annotation_design` (a path
relative to the fit specification), `covariances` (K by Q by Q), and nonempty
`provenance`, alongside its `id`. The usual architecture/context and residual
specifications still apply. Candidate construction does not estimate the
annotation covariances or select hyperparameters; those must come from the
declared training/tuning procedure.

With overlap, Omega[k] is a conditional covariance contribution. The covariance
of a variant set S is `sum_{j in S} Lambda[j]`. Requiring PSD components is a
sufficient, interpretable restriction, but it excludes negative conditional
increments. In particular, an all-SNP-plus-coding PSD prior only adds coding
variance. Start with disjoint coding/noncoding bins if both enrichment and
depletion should be possible. Larger overlapping models need shrinkage,
identifiability checks, and adequate information per component.
