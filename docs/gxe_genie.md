# GxE and heterogeneous-noise estimation from summary artifacts

SUMMIT implements a one-environment, quantitative-trait method-of-moments model
with additive genetic, gene-by-environment (GxE), noise-by-environment (NxE),
and residual components. The estimator is a summary-artifact reformulation of
GENIE's projected kernel equations; GENIE itself is an individual-level method,
and this is not ordinary interaction LDSC.

## Model and exact summaries

Let `C` contain the intercept, tested environment, and user covariates, and let
`P` be its orthogonal residual maker. In implementation, a rank-revealing SVD
defines this span, so redundant covariate columns do not change `P` or its
rank. Let `G` denote centered/scaled genotype, let `e` be the centered,
unit-variance environment, and define

```text
U_j = P G_j,                 V_j = P[diag(e) G_j]
X_j = U_j sqrt(rank(P)/(U_j'U_j))
W_j = V_j sqrt(rank(P)/(V_j'V_j)).
```

The interaction is formed before projection. `P[diag(e)G]` is generally not
equal to `P[diag(e)PG]`. Variants with invalid projected variance fail QC rather
than being amplified or retained as zero columns. With annotations `a_jk >= 0`
and `M_k = sum_j a_jk`, SUMMIT's default kernels are

```text
G_k   = X diag(a_k) X' / M_k
GxE_k = W diag(a_k) W' / M_k
NxE   = P diag(e^2) P
R     = P.
```

This post-projection normalization is the partial-correlation convention used
by SUMMIT's additive LD and marginal-score equations. It is therefore the
primary `standardized` SUMMIT estimand. The explicit `genie` compatibility mode
uses HWE-scaled `G`, sets `X=PG` and `W=P[diag(e)G]`, and does not normalize
their projected norms. That matches GENIE's kernel definition, but it implies a
different per-SNP random-effect prior whenever projected column norms vary.
Neither mode is a numerical correction of the other, and manifests from the
two modes cannot be mixed.

### Why the SUMMIT convention is the default

The default was adjudicated symmetrically rather than by assuming that GENIE's
raw projected kernel was the truth. Dense calculations verified the
partial-correlation LD and marginal-score identities to approximately
`2e-15`. In 20,000 phenotype replicates generated and fitted under both kernel
conventions, each convention recovered its own expectation-level target; mean
component RMSE differed by less than `7e-4`, so the simulations did not support
an efficiency claim for either scaling. Fitting the wrong convention shifted an
expectation-level component by as much as 1.22 percentage points in that
experiment.

A separate 30-panel sweep included genotype panels correlated with an ancestry
covariate. The raw and partial-correlation additive LD matrices differed by
15.84% in relative Frobenius norm on average in the structured setting, and
cross-convention component shifts reached 1.65 percentage points. A deliberately
ancestry-determined SNP had zero projected variance and was rejected, as the
kernel contract requires. These results motivate a coherent estimand, not a
claim that post-projection normalization universally has lower sampling error:
SUMMIT uses the convention already assumed by its additive partial-score
machinery, while `genie` remains a first-class replication/sensitivity mode.

For residualized/normalized phenotype `y`, SUMMIT solves the unconstrained
system

```text
T_ij = tr(K_i K_j),       q_i = y' K_i y,       T sigma = q.
```

The phenotype-side sufficient statistics are the marginal cross-products

```text
s_G,j   = X_j' y
s_GxE,j = W_j' y
q_NxE   = sum_i e_i^2 y_i^2
q_R     = y'y.
```

The score files store `s/sqrt(rank(P))`; the exact scale is recorded in their
manifest, and the moments contract requires `q_R = rank(P)`. GWAS and GWIS
alone do not determine `q_NxE`, so this scalar is
required to distinguish heterogeneous noise from genetic interaction.

For feature families `F,H` in `{G,GxE}`, directional trace scores store

```text
ell(F<-H)[j,k] = sum_l a_lk [(F_j' H_l) / rank(P)]^2.
```

XX, XW, WX, and WW are all retained. Although the aggregate XW and WX traces
are equal in exact arithmetic, their per-SNP rows are not interchangeable.
Projected feature norms and `F_j' diag(e^2) F_j` supply the genetic-by-residual
and genetic-by-NxE trace blocks. Two scalar traces supply NxE-by-NxE and
NxE-by-residual.

## Exact deletion jackknife

GENIE deletes a SNP block from both sides of each genetic kernel. Deleting only
rows from a full-source LD-score file is not equivalent. With
`--write-gxe-jackknife`, SUMMIT stores the within-block intersections needed for

```text
S(-B) = S - rows(B) - columns(B) + intersection(B,B).
```

The fitter reconstructs each two-sided deleted trace matrix, deletes the same
block from genetic phenotype moments and kernel denominators, solves every
replicate, and reports the standard delete-block SE. Use `--njack 100` to match
the published GENIE block count, or `--njack chr` for a chromosome jackknife.
At least 100 trace probes are required by default when writing jackknife
intersections. A lower count is available only through the explicit diagnostic
override because Monte Carlo error in the within-block intersections can
materially distort the delete-block SE. Production analyses should additionally
check stability across probe counts and independent seeds.

### Reusable and parallel reference construction

`--gxe-build-cache` makes one phenotype-free, owner-only schema-v2 cache containing the
exact sample/design identity, genotype provenance, variant and annotation axes,
jackknife blocks, projected feature scales/norms, and deterministic diagonal
moments. Compact sufficient statistics independently reconstruct both NxE
traces, so edited trace metadata cannot be trusted without verification. An
uncached reference needs three genotype passes (feature norms,
source sketches, and target products); a cache-bound trace shard needs two.

`--gxe-reference-shard` assigns every randomized vector a global probe identity
using Philox and writes a non-fit-able shard. The probe generator is invariant
to probe tiling and records the SNP block size; the merger rejects shards with
different block partitions or randomization contracts. `--gxe-merge-shards`
requires disjoint, contiguous probe intervals and exact agreement in the cache hash, seed,
distribution, dtype, kernel/scaling modes, annotations, variants, and
jackknife layout. It averages raw panels and within-block intersections by
probe count, publishes the complete reference transactionally, and never
applies an XX-like storage subtraction. A merged jackknife reference with fewer
than 100 probes is permitted only with the explicit diagnostic override.
Each schema-v2 shard has a separate identity sidecar binding its exact probe
interval to the panel and jackknife hashes; duplicate numerical contributions
are rejected even if manifests are relabelled. Input files are parsed from
owner-only snapshots of the exact bytes that passed their hash checks.

The exact two-sided jackknife source sketches are spilled by SNP-deletion block
to private temporary storage. This avoids rereading every genotype block for
every deletion block: production uses a norm pass plus one source and one
combined target pass, while keeping only the global and current-block sketches
resident. The sealed Hoffman production contract uses two disjoint B50 jobs to
reproduce a monolithic B100 reference (within floating-point reduction
tolerance) without invalid chromosome sharding.

## Input contract and safety checks

- FAM, environment, covariate, and phenotype IDs are read as strings; duplicate
  IDs and incomplete ID sets are errors.
- BED magic and byte size are checked against the exact FAM/BIM dimensions
  before any genotype computation, so truncated or non-SNP-major triples fail
  immediately.
- `-9`, `NA`, `NaN`, `.`, `None`, and `null` are missing by default. Override
  with `--gxe-missing-values` only when the coding is known.
- Annotation weights must be finite and non-negative. Fractional and
  overlapping annotations are supported; empty annotations are errors.
- The same complete-case sample, environment coding, fixed-effect span,
  variants, alleles, annotations, and scaling must be used for every artifact.
  Analysis/variant fingerprints and mandatory per-file SHA-256 hashes enforce
  this for SUMMIT-generated bundles. Replacing, relabeling, or mixing one score,
  LD panel, diagonal table, or jackknife file makes fitting fail closed.
  Phenotype moments record their generating reference and bind the exact
  feature-cache SHA-256. This prevents mixing genotype content, samples,
  covariates, modes, scales, annotations, or SNP axes, while deliberately
  allowing the same marginal scores to be reused across B50/B100 trace
  checkpoints produced from that cache.
- Variants with zero/invalid additive or interaction projected variance are an
  error. They must be QC-filtered before regenerating the entire bundle; they
  are never silently retained in annotation denominators as zero columns.
- Fixed effects and the normal system use SVD rank checks. Redundant covariates
  are projected correctly rather than counted twice.
- Negative method-of-moments estimates are retained. They are statistical
  estimates, not numerical errors.
- The fitter reports singular values, rank, condition number, and solve
  residual. It stops on a materially non-positive-semidefinite trace matrix,
  which signals inadequate randomized-trace precision or inconsistent inputs.
  It also stops on a singular or excessively ill-conditioned system unless
  `--allow-ill-conditioned-gxe` is explicitly supplied; that flag does not
  permit a non-PSD trace matrix.
- Fixed-prefix generation and fitting refuse to overwrite by default. New
  bundles are staged in private directories and published with same-filesystem
  no-replace links, with hash-bound manifests last; final artifacts are
  owner-readable/writable only. Use a new output prefix for reproducible runs;
  `--gxe-overwrite` is an explicit escape hatch.
- The fitter snapshots every consumed summary artifact into a private directory
  beside the requested output (or beside the moments file for direct API calls),
  hashes and parses those same bytes, and removes the snapshots on exit. Allow
  temporary disk approximately equal to the compressed cache, four panels,
  diagonal, jackknife, and two score files; Hoffman fits therefore keep both
  outputs and snapshot workspace under the designated scratch root rather than
  node-local `/tmp`.

New generation and reusable scoring use schema v3. The fitter retains explicit
schema-v2 compatibility for already sealed, hash-bound pilot bundles, but it
rejects the older unhashed experimental manifests. After an independent audit,
`scripts/gxe/seal_legacy_bundle.py` can write new, hash-bound manifest copies
without changing any legacy artifact; low-probe jackknife migration requires
its explicit diagnostic override.

If standardized binary `e` is exactly balanced, `e^2=1`, so NxE is identical to
residual noise. Near balance can also make the system unstable. This is an
identifiability property of the model, not something ridge regularization can
repair without changing the estimand.

## PLINK 2 interaction output

Standard PLINK 2 `--glm interaction` fits the SNP main effect and interaction
jointly. Its `ADDxE` statistic is conditional on the SNP, while GENIE needs the
marginal projected GxE cross-product. SUMMIT therefore does not accept a generic
Z-only GWIS contract. Generate direct scores with `--gxe-pheno`, or provide a
file explicitly labeled `SCORE_MODE=marginal_cross_product` after independently
validating its scale. Robust-SE, logistic/Firth, extra-interaction, or differing-
sample outputs cannot be converted by a generic formula.

## Scope

The current robust path is deliberately in-sample and environment-specific.
Interaction traces depend on the joint genotype/environment distribution and
on the fixed-effect projection; ordinary ancestry-matched LD alone is not
enough. Using an external reference requires a separate fourth-moment/scaling
derivation and calibration and is not silently enabled here.

## Reuse and computational boundary

The XX/XW/WX/WW traces, projected feature scales, NxE traces, and deletion
intersections depend on the cohort, environment, fixed-effect span, variants,
and annotations, but not on the phenotype. They should therefore be computed
once per fixed design and reused. `--gxe-score-reference` implements the wide
trait path: for `T` traits observed on that same row set,
the marginal additive and interaction scores can be formed in one streamed
genotype pass using matrix products against all `T` residualized phenotypes.
Once those summary artifacts exist, fitting reads `O(MK)` values and solves a
small `(2K+2)`-dimensional system; it does not revisit individual-level
genotype or phenotype data.

On the existing 454,207-variant age-by-DBP pilot bundle, full artifact hashing,
loading, equation reconstruction, and fitting took 8.47 seconds with about
522 MiB peak RSS on the local validation host. This benchmark describes the
summary-only stage, not reference construction. Starting from raw data still
requires `O(N M V K)` trace work and `O(N M T)` marginal-score work. An
individual-level method with an equally reusable trace cache can share some of
those asymptotics; SUMMIT's durable advantage is that subsequent fits are
portable, privacy-preserving summary operations and can reuse batched marginal
scores without reloading the individual-level cohort. Conventional conditional
`ADDxE` output does not remove the need for the matched marginal score contract.

The implementation is for one quantitative environment and quantitative
traits sharing one fixed complete-case cohort, with PLINK 1 BED input for the
environment-specific reference/score generator. Wide traits are scored in one
pass, but each variance-component fit remains univariate. Multi-environment
covariance structures, PGEN streaming, and binary-trait estimators are future
extensions.

Reported `proportion` values sum to one on the fixed-effect-residualized
phenotype scale. When SUMMIT generated the phenotype moments, it also reports
`original_scale_proportion`, multiplying by the fraction of centered phenotype
sum of squares remaining after fixed-effect projection. Both scales, all
coefficients/traces/contributions, the normal matrix, eigenvalues, and any
jackknife replicates are included in the machine-readable fit JSON.

## References

- [GENIE paper, American Journal of Human Genetics (2024)](https://doi.org/10.1016/j.ajhg.2024.05.015)
- [GENIE reference implementation](https://github.com/sriramlab/GENIE)
- [PLINK 2 association-model documentation](https://www.cog-genomics.org/plink/2.0/assoc)
