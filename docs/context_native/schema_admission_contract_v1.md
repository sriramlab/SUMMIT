# Contextual native schema, lifecycle, and admission contract V1

Status: frozen Stage 0 design, now used by the private stable-V1 descriptor
path. The older `CONTEXT_SCHEMA_VERSION=1` development NPZ families remain
separate and are never accepted by stable-V1 loaders. The stable path is not a
public CLI or target-scale production claim.

## Independent identity axes

Every stable-V1 artifact carries and validates all of these fields:

| Axis | Closed V1 value or type |
|---|---|
| Artifact family | `contextual_reference`, `contextual_trait`, or `contextual_fit` |
| Logical schema | matching `contextual_{reference,trait,fit}_v1` |
| Grouped physical encoding | `grouped_unnormalized_dense_v1` |
| Contextual native API | unsigned integer `1` |
| Reference backend | `plink_bed_descriptor_stream_stage2_v1:2` |
| Trait backend | `plink_bed_descriptor_stream_trait_v1:2` |
| Fit backend | `python_numpy_scipy_summary_fit_v1` |
| Build identity | nonempty source/binary build identifier |
| Annotation mode | `strict_disjoint_binary_v1` or `generic_nonnegative_weights_v1` |
| Numeric policy | `fp64_v1` for initial qualification |
| Genotype scaling | complete `GenotypeScalePlanV1` and its digest |

Family and logical schema must agree, but none of the other axes is inferred
from either. Unknown fields, enum values, booleans used as integers, missing
digests, and extra physical arrays fail closed. `src/summit/context/schema.py`
contains the executable identity types used by the stable adapters and
loaders.

Pair and component maps are serialized in full. Their canonical digests are
recomputed by readers and before publication; shape alone never determines
ordering. Canonical array identity is SHA-256 over the little-endian dtype
string, a zero byte, little-endian signed-int64 shape values, and contiguous
C-order payload bytes. Object arrays are forbidden.

There is no implicit migration or backend substitution. Development artifacts,
legacy backend identifiers (including reference/trait backend `:1`), and
unknown future schema/API/backend values are rejected by the current loader
and must be regenerated through a reviewed source workflow. A future format
requires a new explicit schema/backend value and loader; V1 files are not
rewritten in place.

The nominal 8-GiB member and 16-GiB total-uncompressed reader ceilings are not
writer-capacity claims. Stable publication additionally requires classic ZIP
central metadata: fewer than 65,535 members, each exact NPY size and its
conservative raw-DEFLATE bound within the audited runtime's
`zipfile.ZIP64_LIMIT`, and the cumulative local-header-plus-payload bound within
that same limit. The writer checks these conditions before touching the target
path and validates the fsynced temporary container with the stable reader
before an atomic no-replace hard-link publication. An existing regular file,
directory, symlink, or concurrent winner is rejected with `EEXIST` and is never
modified. These are format and publication gates, not memory admission.

## Scale plan

`GenotypeScalePlanV1` binds the closed policy plus retained variant order,
allele orientation/coding, centering source/formula, scale formula,
missing-value imputation, ploidy, and the canonical mean and inverse-scale
array digests. Reference and trait plans compare the full plan digest exactly.
Using the same free-text label for `G` and `2G` is not compatible.

For `pre_scaled_dense_v1`, the two affine-array digests bind the identity of
the already transformed dense differential input. For
`sealed_variant_affine_v1`, native code owns the affine arrays and applies them
exactly once during descriptor decode. Stable reference, trait, and fit V1
artifacts require the sealed policy. Both modes use one SNP scale for all
contextual coordinates and prohibit post-projection coordinate-specific
scaling.

## Artifact fields and interpretation

Reference output contains compact full Gram, signed same-person matrix,
grouped unnormalized Gram numerators, annotation/group masses and counts, map
identities, probe identities, scale identity, schema/build identity, and
complete execution telemetry. Trait output analogously contains compact full
and grouped genetic RHS/trace/genetic-residual moments plus exact residual
moments. Neither result may retain a sample or variant axis.

Raw fit fields are primary:

```text
raw_coefficients
raw_omegas
raw_rank
raw_condition_diagnostics
raw_loo_coefficients
raw_jackknife_covariance
```

Optional interpretation fields are separately named and never overwrite raw
fields. Stable `fit_v1` currently accepts the PSD fields (`psd_coefficients`
and `psd_omegas`) with method/tolerance/diagnostic metadata.
`regularized_coefficients` remains reserved schema vocabulary and is not a
stable-V1 fit member. Projection field presence must be internally consistent
on write and load.

Strict-disjoint annotations permit standalone annotation covariance outputs.
Overlapping annotations expose component values only as conditional
contributions and publish the combined total surface by default. Grouped
deletion fields are unnormalized raw numerators. Deletion is explicitly
`approximate_summary_only_v1`, retains the full same-person matrix, and must
reject any deletion that empties an annotation.

## Ownership and lifecycle

Plans and results own their array storage, make it C-contiguous and read-only,
and recursively freeze nested metadata. The native descriptor-backed reference
and trait executors follow:

```text
CONSTRUCTED -> VALIDATED_AND_SEALED -> ADMITTED
-> PHASES_RUNNING -> COMPLETE -> FINALIZED
-> SCRATCH_RELEASED -> PUBLISHED
```

`FAILED` is terminal. Publication requires exact expected call/event counts,
unchanged files and inputs, released execution scratch, completed semantic
checks, and output digests. Python requests preflight, constructs one immutable
plan, and makes one `run` call; it does not drive genotype blocks or public
GEMMs.

## Checked memory admission

All dimension products use at least unsigned 128-bit intermediates in native
code and must fit the published unsigned-64-bit ledger. Let `s_g`, `s_s`, and
`s_w` be bytes per decoded, source, and wide-panel scalar; `V` a decoded
variant block; `b`/`d` sample/variant probe tiles; `A_t`, `C_t`, `K_t`, `Q_t`,
`G_t`, `J_t`, and `V_t` admitted tiles; and `R` concurrent reduction copies.
Each candidate plan explicitly evaluates:

```text
decode                 = s_g*N*V
projected probes       = s_w*N*Q_t*b
resident sources       = s_s*M*Q*B_resident
full targets           = s_w*N*K_t*Q_t*b
resident actions       = s_w*N*A_resident*B_resident
group targets          = s_w*N*G_t*K_t*Q_t*B_T
group action tile      = s_w*N*A_t*B_T
group cross outputs    = 8*G_t*C_t*C
direct grouped scaled  = min(s_w*N*Q_t*A_t*b, s_w*N*Q_t*V)
direct grouped output  = s_w*V*Q_t*A_t*b
group accumulators     = 8*R*J_t*C_t*C_t_prime
same sketch            = s_w*N*K_t*Q_t*d
same g tile            = s_w*N*A_t*d
same persistent        = 8*N*C + 8*C*C
trait features         = s_w*N*Q_t*V_t
trait scores           = s_w*V*Q*L
compact outputs        = exact shape-derived output ledger
projection             = fixed basis plus admitted projection scratch
integrity retry        = exact protected-call retry workspace
trusted fallback       = simultaneously live trusted-backend reserve
telemetry              = event_capacity*event_record_bytes
headroom               = explicit nonnegative safety reserve
```

The plan records every allocation in one of five phase ledgers: `source`,
`action`, `grouped`, `same_person`, and `trait`. Objects shared by phases are
counted once as permanent. Objects reused across mutually exclusive phases are
counted in each phase but only the maximum phase enters peak admission:

```text
peak_bytes = permanent_bytes
           + max(source_bytes, action_bytes, grouped_bytes,
                 same_person_bytes, trait_bytes)
           + retry_bytes + trusted_fallback_bytes
           + telemetry_bytes + compact_output_bytes
           + headroom_bytes
```

No allocation may move between ledger categories after admission, and no
emergency allocation is allowed after decode begins. Observed high-water bytes
must not exceed the admitted ledger; disagreement is a terminal planning
failure. `ContextAdmissionLedgerV1` executes the checked aggregation in Python,
and the native planners bind their admitted ledgers to the sealed execution
plan.

## Checked protected-call and event admission

The semantic call ledger enumerates every `SemanticOperation` in
`src/summit/context/schema.py`, including source/action, both grouped
algorithms, same-person, and trait TN/NN/projection/Gram operations. For each
operation the planner computes calls from the exact selected tile ranges and
records zero for paths not selected. Range generation, rather than a rounded
estimate, is authoritative.

```text
protected_calls = checked_sum(call_count[operation] for every operation)

telemetry_events = protected_calls * maximum_events_per_call
                 + phase_checkpoint_events
                 + repair_fallback_events
                 + semantic_verification_events
                 + publication_events
```

Each multiply and sum is checked. Capacity must be at least this result before
descriptor access or decode. Overflow, a capacity-minus-one run, a dropped
record, or observed/expected disagreement is terminal. Event allowance covers
all attempts, detection, repair, retry, trusted fallback, witnesses,
projection leakage, Gram symmetry/finiteness, group reconstruction, file/input
checks, and publication evidence; it is not a circular buffer budget.

Descriptor passes and decoded blocks are also generated from the selected
physical schedule and checked exactly. The trait contract is one descriptor
traversal per admitted phenotype batch. A grouped plan distinguishes
contiguous sequential traversal, indexed group-major visits, and repeated
scans; nominal logical passes may not hide duplicate physical decode.

## Current qualification limits

The descriptor-backed reference and trait executors consume these ledgers and
publish bounded telemetry; stable fit remains a compact Python summary solve.
This contract does not qualify a build or dataset by itself. Release evidence
must still bind the exact build and artifact identities, exercise the complete
fault and sanitizer gates, and demonstrate whole-process RSS, NUMA, file
stability, and integrity for the admitted workload. Target-scale modeled
admission is not measured runtime evidence.
