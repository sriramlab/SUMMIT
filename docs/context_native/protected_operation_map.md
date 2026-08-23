# Contextual native protected-operation map (Stage 0)

> Historical notice: this document preserves the Stage 0 reconnaissance and
> placement decision. Statements about missing executors or future work describe
> that branch point and are superseded as current implementation status by the
> stable-V1 contract and release documentation.

This is a reconnaissance and placement decision, not an implementation claim.
The active source inspected for these anchors is
`src/native/gxeldcore.cpp` at the Stage 0 branch point. No contextual native
executor exists yet.

## Placement and reuse decision

Nearly all of `gxeldcore.cpp` is inside one anonymous namespace
(`src/native/gxeldcore.cpp:69-15778`). The checked GEMM machinery and the
descriptor-owned projection helper therefore cannot be linked from a new
translation unit without first refactoring legacy code. The smallest Stage 1
seam is a new, scientifically isolated `summit::context_v1` class in this same
translation unit. It may call the private execution primitives, but it must not
inherit or delegate scientific work to `DirectContext`,
`MultiEnvironmentKernel`, or `MultiEnvironmentDirectContext`.

Only after fixed-fixture differential correctness should generic machinery be
factored into `src/native/common/protected_gemm.*`,
`genotype_descriptor.*`, or `projection.*`. Such extraction requires unchanged
legacy golden tests before and after; duplicating the integrity substrate in a
second module is not an acceptable seam.

The existing lower-level files are reusable without importing legacy
scientific layouts:

- `src/native/common/genotype.hpp:31-48` defines `MailmanPackedBlock`, including
  packed calls, mean/inverse standard deviation, observations, and missing rows.
- `src/native/common/genotype.hpp:99-115` exposes packed descriptor-memory
  decode entry points, and `src/native/common/genotype.cpp:272-470` owns the BED
  mmap/cache layer.
- `src/native/common/mailman.hpp:15-303` contains generic packed multiplication
  kernels and q-panel sizing.
- `src/native/gxeldcore.cpp:6421-6449` defines and compares immutable file-state
  snapshots; `duplicate_cloexec` at lines 6451-6476 duplicates descriptors;
  `DirectContext` demonstrates validation and mmap ownership at lines
  7145-7179. These are infrastructure patterns, not permission to reuse the
  `DirectContext` scientific object.

## Active protected primitives

| Active symbol | Anchor | Reusable capability | Contextual boundary |
|---|---:|---|---|
| `dgemm_tn_checked` | `src/native/gxeldcore.cpp:4888` | TN witnesses, operand fingerprints, deterministic recomputation/repair | Call only through the future semantic dispatcher; its own scratch allocation and threshold policy do not satisfy contextual preflight. |
| `dgemm_nn_checked` | `src/native/gxeldcore.cpp:5094` | NN witnesses, immutable-input checks, deterministic recomputation/repair | Same restriction; it is not itself a semantic operation or a complete retry/fallback state machine. |
| `dgemm_tn_partitioned_columns` | `src/native/gxeldcore.cpp:6145` | Production TN routing when columns are partitioned | Candidate backend for narrow-left projection and contextual TN calls. |
| `dgemm_nn_partitioned_rows` | `src/native/gxeldcore.cpp:6204` | Production NN routing and deterministic projection update | Candidate backend for target formation and `U @ (U^T X)`. |
| `dgemm_tn_partitioned_rows` | `src/native/gxeldcore.cpp:6269` | Production TN routing when output rows are partitioned | Candidate backend for genotype-transpose and Gram calls. |
| `DirectContext::project_panel_inplace` | `src/native/gxeldcore.cpp:8661` | Applies `X <- X-U(U^T X)` with TN then NN | Algebra is reusable; the member is private to a legacy scientific class, so contextual code needs a same-TU helper or later narrow extraction. |
| `protected_matmul_tn_pair` | `src/native/gxeldcore.cpp:9803` | Sealed two-right-operand TN bridge | Python-visible bridge is suitable for tests, not production orchestration. |
| `protected_matmul_nn` | `src/native/gxeldcore.cpp:9850` | Dense protected NN bridge | Differential/testing bridge only. |
| `protected_matmul_tn` | `src/native/gxeldcore.cpp:15689` | Dense protected TN bridge | Differential/testing bridge only. |

The active integrity cutoff is conditional, not universal:
`gemm_requires_integrity_checks` at `src/native/gxeldcore.cpp:4595-4611`
requires the checksum build option and approximately one billion FLOPs
(`kCheckedGemmMinimumFlops`, line 4587). Smaller paths may use deterministic or
vendor routes. Consequently, “called a protected wrapper” does not prove that
every contextual wide operation received the required semantic integrity
coverage.

## Frozen semantic-operation routing

Every row below is a distinct stable operation ID. The future dispatcher must
attach phase, global probe range, action/annotation/context/group tile, variant
block, and attempt. Sharing the physical primitive never merges semantic IDs.

| Semantic operation | Required algebra | Candidate active primitive | Reuse qualification |
|---|---|---|---|
| `sample_probe_projection_tn` | `U^T Z` | `dgemm_tn_partitioned_columns` | Reuse after dispatcher supplies semantic coordinates and pre-admitted scratch. |
| `sample_probe_projection_nn` | `U(U^T Z)` | `dgemm_nn_partitioned_rows` | Deterministic `alpha=-1,beta=1` update is present; contextual ownership/canaries remain required. |
| `source_tn` | `G_v^T(D_q Z_P)` | `dgemm_tn_partitioned_rows` | Reuse decoder/scale only after the one sealed allele-aware affine transform is enforced. |
| `full_target_nn` | `G_v(A_k S_q)` | `dgemm_nn_partitioned_rows` or qualified Mailman | Strict-disjoint routing must not change the logical map or probe sum. |
| `action_projection_tn` | `U^T` times raw action panel | `dgemm_tn_partitioned_columns` | Must be labeled separately from sample-probe projection. |
| `action_projection_nn` | `U` times projection coefficients | `dgemm_nn_partitioned_rows` | Must preserve raw-action mass units. |
| `action_gram_tn` | `W_flat^T W_flat` | `dgemm_tn_partitioned_rows` | No dedicated protected Gram exists; a semantic dispatcher and explicit layout are mandatory. |
| `group_target_nn` | `G_v(B_g A_k S_q)` | `dgemm_nn_partitioned_rows` or qualified Mailman | Expected production route at moderate `C`; requires group execution-order evidence. |
| `group_cross_gram_tn` | `X_g,flat^T Wtilde_flat` | `dgemm_tn_partitioned_rows` | Output is the raw `M_k M_l` numerator contribution, never a normalized deletion Gram. |
| `direct_grouped_tn` | `G_v^T(D_q Wtilde_d)` (or row-scaled equivalent) | `dgemm_tn_partitioned_rows` | Differential oracle/tiny-`C` candidate; action-scaled and genotype-scaled packing need separate qualified plan IDs. |
| `same_person_target_nn` | `G_v(sqrt(A_k) Xi_v)` | `dgemm_nn_partitioned_rows` or qualified Mailman | Variant-probe U-statistic only; do not substitute sample-probe diagonals. |
| `same_person_projection_tn` | `U^T R` | `dgemm_tn_partitioned_columns` | Probe tiling must not finalize local U-statistics. |
| `same_person_projection_nn` | `U(U^T R)` | `dgemm_nn_partitioned_rows` | Same projection caveats as action panels. |
| `same_person_gram_tn` | `g_flat^T g_flat` | `dgemm_tn_partitioned_rows` | No dedicated protected Gram exists; global probe sums and Gram must survive all tiles. |
| `trait_score_tn` | `G_v^T[D_q y_l]` | `dgemm_tn_partitioned_rows` | Decode once for all admitted `q,l`; score tensor is block-local. |
| `trait_feature_projection_tn` | `U^T(D_q G_v)` | `dgemm_tn_partitioned_columns` | Must follow `P D_q G`, never `D_q P G`. |
| `trait_feature_projection_nn` | `U[U^T(D_q G_v)]` | `dgemm_nn_partitioned_rows` | Feature panels remain block-local and are reduced immediately. |

There must be no direct CBLAS/vendor call from contextual scientific code.
`dgemm_nn_raw` and `dgemm_tn_raw` at `src/native/gxeldcore.cpp:3381-3397`
show why a later source check must allow vendor symbols only in approved wrapper
code.

## Gaps that block production reuse today

The active machinery is useful but does not yet implement the contextual
fail-closed contract:

- The only explicit corruption injection is the one-shot NN diagnostic
  `_test_protected_matmul_nn_integrity_diagnostic`, implemented at
  `src/native/gxeldcore.cpp:15587-15686` and bound at lines 16004-16013. There
  is no semantic TN, projection, or Gram injection; no repeated corruption,
  repair-result corruption, fallback corruption/failure, NaN/Inf, canary,
  operand-mutation-between-attempts, runtime-mutation, or tiling-invariant
  semantic-coordinate coverage.
- The checked functions can deterministically recompute flagged output columns,
  and selected `DirectContext` callers catch `RetryableGemmInputMutation`,
  re-decode, and invoke the tiled kernel (for example lines 7426-7447 and
  7682-7699). This is not a general dispatcher-owned
  detect/retry/pre-admitted-trusted-fallback/compare/fail state machine, and it
  does not emit complete attempt-level semantic telemetry.
- Checked calls allocate vectors/snapshots in the call path (for example
  `dgemm_tn_checked` lines 4896-4927 and `dgemm_nn_checked` lines 5100-5135).
  A contextual run must instead reserve its worst-case checksum, retry, output,
  and trusted-fallback memory before decoding starts.
- GEMM and output-NUMA evidence buffers have fixed capacity 16,384
  (`src/native/gxeldcore.cpp:128` and `:146`). On saturation they evict the
  oldest record and increment `dropped_records` (lines 2808-2811 and
  2941-2945). Contextual publication requires exact preflight capacity and must
  treat any drop as fatal.
- Active generic GEMM telemetry records physical operation/dimensions
  (`GemmTelemetryRecord`, lines 2875-2905), but does not carry all contextual
  semantic coordinates or prove exact expected phase/call counts.

## Scaling and scientific-layout exclusions

Existing scientific classes are explicitly outside the reuse boundary:

- `DirectContext` begins at `src/native/gxeldcore.cpp:7075`,
  `MultiEnvironmentKernel` at line 10033, and
  `MultiEnvironmentDirectContext` at line 12950. Their shapes, passes, and
  coefficient systems are not the generalized contextual estimator.
- `DirectContext` decodes observed dosages as
  `(mean - dosage) * inverse_sd` at line 8633, whereas the dense unpack path in
  `MultiEnvironmentDirectContext` uses `(dosage - mean) * inverse_sd` at lines
  14334-14361. A contextual plan must seal allele orientation, the exact
  `dosage_minus_mean_times_inverse_scale_v1` formula, and scale/map digests; it
  must not inherit either sign convention implicitly.
- Legacy feature moments derive coordinate/variant-specific `scale_x` and
  `scale_w` in `DirectContext::compute_feature_moments_from_genotype`
  (`src/native/gxeldcore.cpp:7934-8138`) and apply them to target outputs at
  lines 8202-8217. That post-projection normalization is forbidden. Contextual
  execution applies one per-SNP affine genotype transform and then forms
  `F_q=P D_q G`, with no coordinate-specific feature scaling.
- Legacy `X/W`, `XX/XW/WX/WW`, the scalar-environment power system, the `2K`
  coefficient layout, and `[Y,EY]` trait scorer cannot be relabeled as
  generalized contextual outputs.

## All-resident verdict

An all-resident source/action schedule is structurally feasible on the target
Tabla profile, but not through an existing scientific class. At
`N=300,000`, `M=1,000,000`, `Q=4`, `K=8`, `C=80`, and `B_T=128`, the leading
fp64 objects are approximately 4.10 GB of source scores, 9.83 GB of generic raw
targets, and 24.58 GB of raw actions; a full same-person action tile is another
24.58 GB. At `B_T=1024`, source scores and raw actions grow to approximately
32.8 GB and 196.6 GB. Both schedules therefore require checked phase-lifetime
admission including decode buffers, projection scratch, concurrent reductions,
retry/fallback reserve, telemetry, allocator overhead, and explicit headroom.

The Stage 0 verdict is **same-translation-unit seam, all-resident preferred when
admitted**. Stage 1 may add only an in-memory differential contextual class.
Descriptor-owned streaming, production admission, protected state-machine
completion, and Tabla qualification remain later-stage work.
