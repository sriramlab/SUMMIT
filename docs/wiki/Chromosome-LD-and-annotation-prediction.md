# Chromosome LD and annotation-dependent prediction

The chromosome estimator computes within-chromosome SNP-pair LD and combines
the moments in a single genome-wide fit. It estimates one covariance matrix
per annotation using the same participants, genotype scaling, context basis,
and fixed-effect span for reference and trait summaries. A shared reference
can be transferred to one trait's own moments with
`transferred_chromosome_equations` (see below).

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

## Transferring a shared reference to a trait

`transferred_chromosome_equations` keeps the trait's own score, trace and
residual moments and transfers the reference genetic Gram to the trait's
people ([Methods](Methods.md#reference-transfer-and-fitting)):
`T_S = D_S + s (T_R - D_R) + kron(L, F_S - s F_R)`.

For a reference root with per-person diagonals, one call does both parts:

```python
from summit.ldscore.generalized_gxe_chromosome import (
    prepare_chromosome_transport, transferred_chromosome_equations)

transport = prepare_chromosome_transport(reference_chunks, diagonals, phi)  # once per root
terms = transport.for_study(rows)                                           # once per trait
equations = transferred_chromosome_equations(
    reference_chunks, trait_chunks, **terms, deleted_blocks=deleted, **moments)
```

`diagonals` maps each chromosome to its `C x N_R` component diagonal (the
`global_kernel_diagonal` of `reference_aux.npz`, normalized by the genome-wide
annotation masses). `phi` is the reference basis and `rows` are the trait's
people as reference sample indices. `moments` are the reference and trait
residual moments, `residual_names` and `expected_chromosomes`.
`own_row_transported_equations` wraps the last two steps for a single system.
The genome-wide D_R and D_S use the summed diagonals, so they include
cross-chromosome products.

The pieces can also be passed separately:

- Same-person part: `study_same_person` with its `same_person_transport` label;
  or `reference_kernel_diagonal` with `study_rows` for the exact own-row term;
  or `reference_fourth_moments` and `study_fourth_moments` for the summary form.
- Different-person part: each chromosome's raw intercept same-person entries,
  either `reference_same_person_intercepts` (a mapping built with
  `chromosome_same_person_intercept`) or the chunks' `same_person_intercept`,
  plus `reference_second_moments`, `study_second_moments` and the fourth
  moments. `reduce_chromosome_result` stores `same_person_intercept` (K x K, a
  sum over people), and `combine_chromosome_annotations` carries it.
  `MaskedTraitBatch` stores each trait's `context_second_moments` and
  `context_fourth_moments` in its `common` record.

Without these inputs a part keeps its legacy scaling, which is biased when
missingness depends on the contexts, and warns when the sizes differ. Pass
`same_person_transport="scaled_legacy"` or
`different_person_transport="scaled_legacy"` to request it on purpose. The
returned equations record both choices. Deletions reuse the full study
same-person term; the correction drops the deleted blocks' LD scalars and uses
the retained annotation masses.

This production assembly differs from the research
`cross_trait_gram.within_trait_equations(mode="factorized_plus_residual",
same_person_mode="own_rows")` only through the joint chromosome assembly. The
difference is `s [D_R - sum_c D_c + sum_c B_c C^-1 B_c' - B C^-1 B']`, with
per-chromosome same-person matrices `D_c` and reference residual cross-moments
`B_c`. On the 338k master root it is about 0.08% of the Gram.

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
