# Chromosome LD and annotation-dependent prediction

The chromosome estimator computes SNP-pair LD **only within each chromosome**,
using the existing stochastic variant-axis reference. It then forms one
genome-wide normal system and estimates a single covariance per annotation.
It does not average separately fitted chromosome covariance estimates.

This is an opt-in approximation with its own artifact policy,
`within_chromosome_residual_profile_global_mass_v1`. The initial API requires
the same participants, genotype scaling, context basis and fixed-effect span
for reference and trait statistics. Population transfer is not implemented for
this artifact family. Full-data concordance and timing are qualification steps;
small algebraic tests do not establish calibration or biobank throughput.

## Global normalization and joint estimation

For annotation weights A[j,k], define M[k] = sum over **all chromosomes** of
A[j,k]. A chromosome's kernel contribution is

    K[c,k,qr] = sum_{j in c} A[j,k] F[q,j] F[r,j]' / M[k]

with the symmetric second orientation included when q != r, and
F[q] = (I - UU') diag(phi[q]) G. The native chromosome reference uses local
annotation masses; its unnormalized numerators are converted to the shared
global masses during joint assembly. A small chromosome therefore contributes
its share of the genome-wide annotation variance. An annotation absent on one
chromosome is omitted from that native task and embedded as zero in reduction.

Let T[c] be the within-chromosome genetic Gram, B[c] the genetic/residual cross
moments, C the common residual Gram, g[c] the genetic RHS and r the residual RHS.
The joint profiled system is

    A = sum_c {T[c] - B[c] solve(C, B[c]')}
    u = sum_c {g[c] - B[c] solve(C, r)}.

The normal system returned by `joint_chromosome_equations` reconstructs a
single shared residual coefficient vector with B = sum_c B[c]. This counts
C and r once. Simply summing T[c] and coupling that sum to B would subtract
cross-chromosome residual background without retaining the corresponding
genetic background, potentially producing negative information.

The approximation sets cross-chromosome **residual-projected** kernel products
to zero. It computes no cross-chromosome LD. Local LD makes this plausible;
population structure, relatedness and genotype/context dependence can affect
its accuracy. Concordance must be assessed on the actual cohort.

## Execution and artifacts

`GeneralizedGxENativeBEDExecutor` accepts `variant_start` and uses exactly
`annotations.shape[0]` consecutive variants. It reuses the descriptor-owned
decoder and performs two traversals of that interval. Thus a chromosome can
be read from a merged BED without copying it, or from an existing chromosome
BED using `variant_start=0`. The caller must verify chromosome membership
and the shared sample/variant identities before execution.

Optional `phenotypes` (normalized residual phenotypes) and `residual_basis`
enable fused per-SNP trait statistics from the already projected feature tiles.
The planner must receive the corresponding `num_traits` and
`num_residual_components`. These outputs add no genotype traversal and share
the costly reference across traits with exactly matching sample/design axes.
Nonmatching trait masks must not be combined as though they shared a reference.

`reduce_chromosome_result` reduces fixed LD and trait rows after execution.
`write_chromosome_moments` publishes compact arrays with content checksums and
caller-supplied provenance, refusing to overwrite an existing file.
`joint_chromosome_equations(..., expected_chromosomes=range(1, 23))` rejects
duplicate, missing, and incompatible chromosome contributions. The residual
basis must begin with the constant-one kernel. The resulting equations use
the existing rank-checked solver.

`combine_chromosome_annotations(chunk, weights, names)` reuses these compact
moments for the design `A_new = A @ weights`. An all-one column merges disjoint
MAF bins covering the panel into the unpartitioned model. This produces both
overall and partitioned estimates from the same genotype passes. It combines
unnormalized moments and recomputes global masses in each fit and deletion;
it does not average the fitted bin coefficients. It also supports nonnegative
overlapping combinations of the existing annotations.

Block deletion is post hoc: source sketches remain frozen, selected target
rows are removed, and global retained annotation masses are used. The nuisance
correction uses retained target B and full source B, symmetrized after reduction.
This is an approximate target-row jackknife, not a chromosome bootstrap or
an exact refit. No inference blocks enter native LD construction.

The native feature projection and fused information products use protected
overwrite GEMMs followed by bounded elementwise updates. Reference qualification
uses `GXELDCORE_GEMM_INTEGRITY=ON` and `GXELDCORE_GEMM_CHECKSUM=ON` explicitly;
private BLIS otherwise defaults to checksum recomputation off. Native ledgers
record independent phase audits and protected GEMM audits separately.
Singleton annotation diagonals retain the variant
summation order while moving the response-pair loop outside the variant loop.

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

The SNP prior is Lambda[j] = sum_k A[j,k] Omega[k] / M[k]. The implementation
forms only a bounded variant-by-candidate covariance tile. Each batched
operator application still has one shared genotype traversal and two large
genotype products per RHS tile. Homogeneous and annotated candidates can share
the same batch. A design can be reused across candidates without duplicating
the M-by-K matrix. Content identities are cached on immutable backing stores.

Native annotation fits using BLIS require both `GXELDCORE_GEMM_INTEGRITY=ON`
and `GXELDCORE_GEMM_CHECKSUM=ON`. Full-array qualification found intermittent
discrepancies around 1e-6 in the unchecked build, despite passing small tests;
their cause remains unresolved. The protected full-array candidate sequence
agreed with independent per-annotation products to 1.8e-16 relative error.
The API rejects the unqualified BLIS configuration before genotype preparation.
This is a qualification restriction, not a claim that the discrepancy's cause
has been fixed or that the annotation formula caused it.

The positive aggregate-covariance diagonal is an approximate preconditioner;
the covariance operator and exported weights use the exact per-SNP priors.
Convergence is checked against the actual operator. Planning, checkpoint
identity, interruption/resume, export and reloaded scoring use the regular
prediction API. Annotated designs are authenticated against the ordered
training variant/allele axis. No annotations are needed when scoring saved
posterior weights.

For the CLI, write a design with `write_annotation_design(path, design)` and
use a candidate with `operation: "annotation"`, `annotation_design` (a path
relative to the fit specification), `covariances` (K by Q by Q), and nonempty
`provenance`, alongside its `id`. The usual architecture/context and residual
specifications still apply. Candidate construction does not estimate the
annotation covariances or select hyperparameters; those must come from the
declared training/tuning procedure.

With overlap, Omega[k] is a conditional covariance contribution, not the
covariance of all variants carrying label k. For a variant set S, reconstruct
its covariance as sum_{j in S} Lambda[j]. Requiring PSD components is a
sufficient, interpretable restriction, but it excludes negative conditional
increments. In particular, an all-SNP-plus-coding PSD prior only adds coding
variance. Start with disjoint coding/noncoding bins if both enrichment and
depletion should be possible. Larger overlapping models need shrinkage,
identifiability checks, and adequate information per component.
