# Model/code crosswalk before corrective changes

This document compares docs/reference_gxe_estimator.tex at commit
6bf009e850afd4b112bf46ee04d2cdea7acd8896 with the implementation observed
before estimator changes. Line numbers refer to that immutable starting
revision.

## Equation-level crosswalk

| Report quantity | Starting implementation | Pre-fix assessment |
|---|---|---|
| Projector, Eq. (1) | read_env_and_cov builds a rank-revealing SVD basis in gwe_ldscore.py:959-980,1198-1212; feature projection is in lines 2218-2244 | MATCH. The stored basis excludes the intercept but is centered; the separate mean subtraction completes the same projector. df_corr=N-rank(C)-1 is the full residual rank. |
| Environment scale, Eq. (2) | deterministic centering/scaling in gwe_ldscore.py:981-1001,1158-1178 | MATCH. Mean, SD, ddof, analysis moments, and design digest are recorded. |
| Phenotype scale, Eq. (3) | gwe_ldscore.py:1224-1243 and the reusable scorer | MATCH. The residual phenotype is normalized to squared norm r. |
| Projected features, Eq. (6) | gwe_ldscore.py:2224-2244,2368-2526; native analogue in gxeldcore.cpp | MATCH for both current modes. standardized is the report's post-projection standardized convention; genie is the raw/naturally scaled projected convention. The names are not the requested explicit standardized_projected/raw_projected names. |
| Kernels and covariance, Eqs. (7)-(9) | feature panels and assemble_normal_equations in inference/gxe.py:422-612 | MATCH for the four-class zero G--GxE and zero residual--NxE covariance model. Optional signed cross-kernels are not implemented. |
| Normal equation, Eqs. (10)-(14) | assemble_normal_equations, inference/gxe.py:422-612 | MATCH. Dense tests verify every matrix/RHS/trace entry. The code averages directional randomized panels without first recording a tolerance or raw orientation discrepancy at this assembly boundary. |
| Marginal scores and RHS, Eqs. (15)-(17) | gwe_ldscore.py phenotype path and gxe_score.py; aggregation in inference/gxe.py:560-578,1077-1242 | MATCH. Inputs must be marginal_cross_product; a conditional Wald statistic is rejected. |
| Directional LD panels and genetic traces, Eqs. (18)-(19) | four xx/xw/wx/ww panels, assembled in inference/gxe.py:579-605 | MATCH for an exact ordered variant set. |
| Population transfer, Eqs. (23)-(24) | transfer_reference_normal_equations, inference/gxe.py:306-420 | CORE FORMULA MATCHES. It uses N, never residual rank, in both factors and has an exact N_S=N_R test. Missing diagnostics: extrapolation when N_S>N_R, transport/moment compatibility, leverage/environment-shift summaries, and recommended reference-size regime. It unconditionally averages the returned matrix with its transpose. |
| Same-person U-statistic, Eq. (25) | gwe_ldscore.py:2668-2769 | FORMULA MATCHES and uses float64 accumulation across probe tiles. The starting code has no split-probe or biased PSD comparator, no eigenvalue diagnostic in the manifest, and no default-selection evidence. |
| NxE traces, Eqs. (26)-(29) | gwe_ldscore.py:3099-3130 and gxe_score.py:1226-1316 | MATCH. Study genetic--NxE cross-traces and low-dimensional NxE moments are computed exactly. |
| Summary interface, Eq. (30) | schema-v3 reference/moment/score bundles | MATCH for the supported strict variant set and common study cohort. |
| Variance balance | full normal equation plus q_0=r | MATCH when the system is full rank. |

## Input and estimand contracts

### Variant set

The starting external fitter already implements the prompt's strict v1
policy. It requires:

- the exact reference row count and ordered CHR/SNP/BP/A1/A2 axis for both
  marginal-score files (inference/gxe.py:2136-2171);
- all four reference panels in the exact reference order
  (inference/gxe.py:1618-1647);
- reference annotation masses recomputed from the diagonal table and equal to
  the manifest; and
- the score/reference/phenotype hashes and variant digest to agree.

It therefore fails closed rather than filtering score rows while retaining
source variants. There is no subset-composable stored reference. Adversarial
tests for annotation-concentrated and high-LD deletions were absent.

### Variant-specific study samples

Schema-v3 score rows must all contain exactly the phenotype's common N and DF
(inference/gxe.py:2145-2184 and the prepared-batch equivalent at 2604-2659).
Heterogeneous N_j/r_j is rejected. Pairwise-overlap metadata and mask-aware
estimation are not implemented and are not claimed.

### Genotype missingness

The Python decoder mean-imputes each variant within the retained cohort before
centering and feature projection (gwe_ldscore.py:2167-2216). The bounded
native feature/source/target calls instead request missing-free input. Neither
path records per-variant call rate or tests missingness association with
environment/phenotype. This does not meet the requested diagnostic contract.

### Feature conventions

The implementation is internally coherent in both standardized and genie
modes, and manifests reject cross-mode combinations. The report itself
describes their different estimands correctly. Gaps are naming/schema
explicitness and the requested dense-oracle coverage of every accepted input
conversion. Population transport deliberately supports only standardized
kernels because raw projected traces are cohort-specific.

## Solution and uncertainty

solve_normal_equations (inference/gxe.py:614-680) does not invert the matrix.
It uses SVD diagnostics and numpy.linalg.lstsq, reports singular values, rank,
condition number, eigenvalues, and solve residual, and fails rank/condition
gates by default. However, it solves the full system rather than the required
residual-eliminated system H=A-tt'/r, b=q-t, and it lacks recorded
symmetrization tolerance, per-block Cauchy--Schwarz checks, component
influence, and a distinct incompatibility status.

The starting delete-block implementation recomputes active annotation masses,
RHS terms, panel terms, and genetic--NxE traces. For population transfer it
reuses the full-reference same-person matrix in every deletion. It therefore
has exactly the approximation described in the report, not the requested
block-aware diagonal statistic. Standard errors cover only the existing
variant-block perturbation; finite-reference-person and probe uncertainty are
not propagated or coverage-calibrated.

## Multi-environment executor

Observed properties at the starting revision:

- exact common complete-case cohort required: YES (gxe_multi.py:408-470);
- three shared genotype passes at B=32/one probe tile: YES;
- feature/sketch persistence from this path: NO;
- protected entry points protected_matmul_nn/tn, float64: YES;
- exact native source and binary provenance in manifests: YES;
- feature-normalization pass loops over environments and materializes each
  projected additive/interaction block: YES;
- source panels are accumulated unprojected and projected once after the pass:
  YES;
- source and target issue approximately two wide calls per environment per
  genotype block: YES; and
- no environment tiling or packed environment GEMMs: YES.

Thus shared decoding is correct but computationally intermediate.

## Feature/sketch cache reachability audit

The historical feature cache is not dead and is not merely a compatibility
reader:

- hidden parser controls --_gxe-build-cache and --_gxe-feature-cache are
  declared in cli.py:316-329;
- _dispatch_gxe_cache calls the writer
  GenomewideEnvLDScore.write_feature_cache (cli.py:847-855);
- reference-shard construction consumes it through the estimator's
  feature_cache_path;
- scripts/gxe/hoffman/uge_cache.sh invokes the cache stage;
- scripts/gxe/hoffman/README.md documents cache, shard, and merge as a
  production deployment; and
- scripts/gxe/hoffman/deployment_config.json and hoffman_deploy.py validate
  and schedule that workflow.

The cache persists per-variant feature scales, norms, genetic--NxE
diagonals/correlations, annotations, block IDs, and design/provenance in an
NPZ. It does not store an N by M feature matrix, but it is a reachable hidden
disk-backed feature cache and violates the new non-negotiable production
contract. Existing readers are also used to validate already published
cache-bound references, so reader removal would break backward compatibility.
The appropriate change is to disable/remove all new writers and
cache-dependent construction while retaining fail-closed legacy readers for
sealed artifacts.

No normal monolithic or multi-environment reference path writes source panels
or randomized sketches. Exact jackknife NPZ files contain reduced
block-by-annotation trace statistics, not sample-by-probe sketches.

## GEMM integrity and root-cause status

The loaded conda OpenBLAS recipe uses pthreads, USE_THREAD=1,
NUM_THREADS=128, and does not explicitly pass USE_LOCKING. The native executor
calls openblas_set_num_threads(1) and enters separate OpenBLAS calls
concurrently from SUMMIT OpenMP workers (gxeldcore.cpp:395-463). Therefore the
external-concurrency hypothesis remains live.

The current ABFT implementation has eight deterministic continuous checksum
vectors, pre/post fingerprints for the decoded operand, a protected copy of
the other operand, and deterministic cache-tiled column repair. Its gaps
relative to the requested gate are:

- tolerance is a fixed relative 1e-9, not a gamma_k/norm-derived bound;
- no explicitly designated signed checksum vector;
- no post-call fingerprint that distinguishes mutation of the protected
  right operand from output corruption;
- no deterministic fault-injection API for A/B/C and multi-column faults;
- no standalone pure-C++ reproducer;
- no controlled OpenBLAS locking/target/backend matrix or second-host result;
  and
- only aggregate repair counts, not every requested call shape, affinity,
  checksum residual, and repaired index, reach the manifest.

The Stage 0 exact production-shaped repeat on the current binary made 20 2B
and 20 4B target calls with zero repairs/retries and maximum absolute error
2.2820313461124897e-6. One clean repeat does not meet the 10,000/100,000-call
backend acceptance gate and does not resolve root cause.

## Model scope not implemented

The starting model has neither the optional signed G--GxE covariance kernel
nor the signed residual--NxE covariance kernel. It also lacks the requested
validation grids for nonzero effect covariance, nonlinear/heavy-tailed
heteroskedasticity, measurement error, endogenous environments, sparse
high-leverage architectures, related samples, ancestry/environment shift,
and calibrated interval coverage.

## Post-audit disposition

| Starting discrepancy | Disposition |
|---|---|
| Full-system generic least-squares solve | Replaced by residual-eliminated binary64 SVD pseudoinverse with rank, singular values, condition, full residual, influence, symmetry, Cauchy--Schwarz, and PSD diagnostics. Rank/condition failures remain fail-closed by default. |
| Unrecorded feature-convention aliases | Added canonical `feature_convention_version=1` metadata (`standardized_projected` or `raw_projected`) to reference/moment/score outputs and strict cross-bundle validation. Legacy labels remain readable. |
| No genotype missingness evidence | Added pre-imputation call counts, environment/phenotype missingness correlations, documented thresholds, manifest status, and native-to-Python fallback when calls are missing. The response remains a warning/sensitivity requirement, not MNAR correction. |
| Unconditional population-block symmetrization | Material asymmetry is now rejected using a scale-aware tolerance; within-tolerance symmetrization and finite-probe eigenvalue/extrapolation diagnostics are recorded. |
| Per-environment feature/GEMM work | Implemented common-projector/low-rank factorization, algebraic additive moments, environment-tiled interaction moments, packed sources, and two packed target calls per genotype block/tile. Dense and independent-path tests cover continuous/binary/skewed/correlated/collinear environments, duplicate covariates, overlapping annotations, and deterministic tiling. |
| Reachable cache/shard writers | Removed hidden writer/consumer CLI controls and hard-gated historical Hoffman worker execution. Compatibility readers remain. |
| Fixed ABFT tolerance/no signed checksum | Added an alternating signed checksum and gamma-based magnitude bound; existing deterministic repair and fault-injection paths remain. |
| Full-reference same-person diagonal in jackknife | Quantified but not replaced. The synthetic audit observed median 1.62% and maximum 9.37% relative Frobenius error across deletions. Exact compact block statistics remain unimplemented. |
| Optional G--GxE and residual--NxE covariance kernels | Not implemented. No broad robustness claim is permitted. |
