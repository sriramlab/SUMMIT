# Contextual dense differential executor V1

Status: qualified Stage 1 differential boundary. This API is an in-memory
scientific correctness bridge with local fault detection and recovery. It is
not a production genotype reader, memory-admitted executor, or contextual
artifact publisher.

## Boundary and lifecycle

`gxeldcore.ContextualBlockExecutorV1` is additive to the legacy extension. It
accepts only native `float64`, Fortran-contiguous dense matrices (plus a
contiguous `int64` group map), copies every input into class-owned storage, and
then follows a one-shot lifecycle:

```text
constructed -> validated_and_sealed -> admitted -> running
            -> finalized -> published
```

`run()` is valid exactly once. Constructor validation rejects non-finite or
incompatible arrays, non-orthonormal fixed-effect columns, non-Rademacher
probes, noncanonical groups, invalid annotation modes, empty annotation mass,
and a variant-probe count below two. The dense input has the fixed differential
scale label `pre_scaled_dense_v1`; the executor never centers, rescales,
imputes, or changes allele orientation. It does not yet bind the input to a
canonical affine-scale-plan digest.

The constructor receives canonical annotation and group names. Pair order is
all diagonals followed by lexicographic off-diagonals; component order is
annotation-major and pair-minor. The complete generated pair and component
maps are returned as compact metadata and verified by the differential tests.

## Physical layouts and units

All protected scientific operands are column-major. A component/probe panel
uses column index `c * B + b`, so one component is a contiguous `N * B` vector
when the action Gram is evaluated. Source scores use column index `q * B + b`.
Feature construction is block-local and does not require an all-`M`
`Q x N x M` tensor: every projection applies `X - U(U^T X)` to the selected
variant panel. Selecting a full-width variant tile deliberately permits the
flattened panel to reach that same dense extent for differential testing.

The result names encode normalization:

| Field | Shape | Unit |
|---|---:|---|
| `gram` | `C x C` | fixed-probe Gram after both annotation masses and `B_T` |
| `raw_gram_numerator` | `C x C` | `M_c M_d` times `gram` |
| `same_person` | `C x C` | signed global variant-probe U-statistic |
| `group_gram_numerator` | `J x C x C` | raw grouped reference numerator |
| `annotation_masses` | `K` | sum of annotation weights |
| `group_annotation_masses` | `J x K` | group-local sum of annotation weights |
| `group_variant_counts` | `J` | retained variants per group |
| `genetic_rhs` | `C` | annotation-normalized genetic RHS |
| `genetic_traces` | `C` | annotation-normalized genetic trace |
| `genetic_residual` | `C x H` | annotation-normalized genetic/residual cross block |
| `group_rhs_unnormalized_num` | `J x C` | raw grouped trait RHS numerator |
| `group_trace_unnormalized_num` | `J x C` | raw grouped trait trace numerator |
| `group_genetic_residual_num` | `J x C x H` | raw grouped genetic/residual numerator |

The class also returns the direct action-scaled TN, direct genotype-scaled TN,
and group-restricted action results under separate diagnostic keys. The three
paths are independently accumulated. Their agreement and the reconstruction

```text
sum_g group_gram_numerator[g,c,d]
  = M_c * M_d * gram[c,d]
```

are terminal publication checks. The selected public grouped numerator is
never inferred from its shape.

The phenotype is projected and normalized so its residual sum of squares is
the residual rank. Trait reduction traverses dense variant blocks in one
logical pass. Residual-only moments use the fixed-basis low-rank identities.
This combined differential executor currently requires one phenotype and a
nonempty residual basis. Python retains ownership of population transfer,
deletion renormalization, small rank-checked solves, raw Omega unpacking,
surfaces, jackknife covariance, interpretation, immutable artifact
construction, and artifact I/O.

Strict-disjoint input is sealed as one annotation ID per variant and bypasses
generic weight and square-root lookup. Both modes still use dense target
panels; this is not a production sparse-disjoint kernel.

## Protected semantic operations

Every wide multiply is routed through the contextual dispatcher with one of
the 17 frozen operation names:

```text
sample_probe_projection_tn   sample_probe_projection_nn
source_tn                    full_target_nn
action_projection_tn         action_projection_nn
action_gram_tn               group_target_nn
group_cross_gram_tn          direct_grouped_tn
same_person_target_nn        same_person_projection_tn
same_person_projection_nn    same_person_gram_tn
trait_score_tn               trait_feature_projection_tn
trait_feature_projection_nn
```

The dispatcher calls the active protected partitioned NN/TN substrate. Every
primary output is compared with a deterministic tiled witness before the
scientific accumulator can consume it. Stage 1 fault injection targets an
operation and occurrence; detected corruption is retried and, when needed,
the verified witness is copied through the fallback buffer. An unknown or
unconsumed requested operation is terminal.

Class-local telemetry stores event class, phase, semantic operation, and
resolution plus aggregate injection/retry/repair/fallback counters. It does
not store stable tile coordinates, dimensions, leading dimensions, strides,
or a complete per-attempt record. The witness-copy fallback is not an
independently invoked, qualified one-thread production backend. This
diagnostic path does not change legacy injection or GEMM routing.

## Admission

`preflight()` reports checked unsigned-64-bit nominal byte and semantic-call
ledgers. Zero tile inputs mean the corresponding full extent. Retry, witness,
fallback, and class-local telemetry storage are physically reserved before
scientific execution. Telemetry capacity is checked against three possible
events per protected call plus fixed lifecycle/injection events; exhaustion is
fatal. Exact reported capacity is admitted and capacity-minus-one is rejected
before execution.

`required_workspace_bytes` is a checked differential reserve, not an observed
or allocator-enforced peak. It does not yet represent one exact live-range
ledger for class-owned inputs, all co-live local vectors, helper-internal
allocations, publication copies, allocator overhead, or explicit headroom.
Consequently the workspace cap test proves planner arithmetic and threshold
enforcement, not production memory admission or allocation-free execution.

The semantic call ledger always enumerates all 17 operation IDs. Publication
requires exact expected/observed call counts, no dropped class-local telemetry,
unchanged owned-input fingerprints, finite compact arrays, acceptable
projection leakage and pre-symmetry diagnostics, exact probe coverage, one
logical trait pass, exact group counts/masses, and grouped reference/trait
reconstruction. These are differential checks. The class does not claim
allocation-free descriptor execution, production memory scale, NUMA
qualification, or a stable on-disk schema.

All tile controls are operational in the dense schedule. Annotation and
context tiles split target construction, trait-feature tiles split projected
feature panels, and `group_tile` controls the outer group schedule. Full dense
assemblies remain resident in several phases, so these controls are not a
production peak-memory guarantee.

## Explicit exclusions

The class does not accept BED/BIM/FAM descriptors, call `DirectContext` or
multi-environment scientific helpers, emit legacy `X/W` layouts, expose an
`N` or `M` axis, solve the normal system, apply ridge/pseudoinverse/PSD
projection, write an artifact, or add a CLI. Existing legacy defaults and
build-info version axes remain unchanged. Published NumPy arrays have native
capsule ownership but remain writeable; the existing Python artifact boundary
defensively copies and freezes them. Native publication does not yet bind
canonical map, scale-plan, build, telemetry, or per-array digests.
