# GxE implementation invariants

These are correctness and throughput constraints for future GxE changes. They
are not suggestions. A change may replace an implementation only if it proves
the same scientific quantity, preserves the safety boundary, and passes the
corresponding validation gate. If an invariant is intentionally changed, the
artifact schema and scientific method must identify a new convention rather
than silently reusing version 1.

## Scientific identity

1. The version-1 standardized feature definition is
   `X = scale_x * P G` and `W = scale_w * P diag(E) G`, where `P` removes the
   intercept, environment main effect, and declared covariates. Mean
   imputation and raw-genotype sample standardization occur before this
   projection. Each post-projection feature has squared norm equal to the
   residual rank. Phenotype scores are `X'Y/sqrt(rank)` and
   `W'Y/sqrt(rank)`. Projection order, normalization denominator, or the
   placement of `E` must not be changed under the same convention identifier.
2. Matched-cohort scoring may apply sealed reference scales only after exact
   genotype hashes, row order, environment/fixed-effect design, cohort, and
   variant axis have all been validated. Population-transfer scoring must
   derive study-specific feature scales and genetic-by-NxE diagonals because
   its cohort or design differs.
3. Population transport must retain separate same-individual and
   different-individual terms and the actual reference/study sample counts and
   ranks. Do not replace them with one nominal `N`, clip negative estimates,
   or project an estimated matrix to the nearest PSD matrix.
4. The residual-eliminated SVD solve is the production solver. Do not restore a
   direct inverse, conceal rank/condition failures, clip component estimates,
   or silently repair a non-PSD/asymmetric normal matrix.
5. Jackknife block count is not a reference-construction runtime parameter.
   The accepted block-local approximation stores no within-block sketch and
   must not add genotype passes as `--njack` changes. The probe count controls
   randomized trace precision; it is not a substitute for the number of
   genomic deletion blocks.

## Input and artifact identity

1. A fused multi-environment reference uses one exact common complete-case
   cohort and row order. Different masks require different reference groups;
   never fuse them by padding or trait-specific reordering.
2. BED/BIM/FAM inputs remain descriptor-owned for the entire native operation.
   Validate exact hashes, dimensions, ordered variants, inode state, and final
   state. These checks are required input identity checks, not optional GEMM
   repair guards.
3. Every published result must bind its reference hash, ordered variant digest,
   analysis fingerprint, source commit/snapshot hash, native binary hash, and
   runtime/build mode. Continue namespace-atomic no-overwrite publication and
   final artifact hashing; publication is atomic against ordinary process
   exceptions but is not crash-durable, and fused batch publication rejects
   overwrite entirely (mixed overwrite settings are a broken common contract)
   until a real backup/journal/restore transaction exists. An accepted fresh install must not contain loadable
   `.pyc`/`__pycache__` content omitted from the package identity; scrub any
   external bytecode-cache prefix and verify the source/package manifest before
   executing its bootstrap code.
4. Do not persist unsealed projected features, randomized sketches, or
   reusable feature caches in the supported production path. If a future cache
   is scientifically necessary, it requires a new threat model and an exact
   content/provenance contract before use.

## Native numerical safety boundary

1. The accepted production path requires a symbol-hidden private static
   pthread-BLIS, one immutable process-lifetime thread count, and serialized
   vendor entry. NumPy/threadpoolctl must not be able to resize or resolve that
   private BLAS. Never route a direct-reference hot product back through a
   process-shared NumPy BLAS. OpenMP-BLIS and OpenBLAS remain diagnostic or
   compatibility configurations, not accepted production GxE backends.
2. `gemm_checksum_enabled=false` is permitted only inside the qualified private
   pthread-BLIS boundary. The structural integrity contract remains enabled.
   A process-shared or OpenBLAS build must retain numerical checksum protection
   or refuse to build; do not offer an unchecked runtime override.
3. The structural context mutex, read-only prepared operands, descriptor and
   dimension validation, finite checks, and provenance hashes remain. They are
   low-overhead ownership checks and are not the removed ABFT/repair/rerun
   machinery.
4. Never make concurrent application threads enter one private BLIS instance,
   and never switch one process-global BLAS repeatedly between one and many
   threads. Independent environment groups may run concurrently only in
   separate processes with private BLAS images and disjoint physical cores.
5. Application OpenMP owns decode and deterministic non-BLAS loops; BLIS owns
   a separate fixed pthread team. Use passive OpenMP waiting with
   `GOMP_SPINCOUNT=0`. Preserve inherited `taskset` placement, select one
   hardware thread per physical core, avoid SMT sibling overlap, and keep each
   parallel group NUMA-local. Idle active-wait pools are a throughput defect
   even when they do not corrupt output.
6. An explicit NUMA membind must be applied and verified while the fresh
   process has exactly one task and before importing NumPy, pandas, or a native
   numerical module. Later code must require the immutable, PID-scoped
   in-process attestation; an inherited environment marker is not evidence.
   The policy uses `MPOL_F_STATIC_NODES`. An authenticated protected BED read
   must allocate a dedicated page-aligned anonymous mapping, apply the exact
   VMA bind before decoder first touch, decode directly into that mapping, and
   preserve it through in-place genotype standardization. Before the decoded
   block can enter a protected product, query every mapping page and require
   that every resolved page lies on the declared nodes. Verification may use
   `MPOL_MF_STRICT` without `MPOL_MF_MOVE`; it must never migrate pages or
   relabel a repaired placement as accepted evidence.
   An integrity-checked NN call must apply the same boundary to the native
   right-operand snapshot that is actually passed to vendor BLAS. Allocate a
   fresh page-aligned anonymous mapping rather than allocator-backed `new[]`,
   apply the authenticated static VMA bind before copying, exhaustively query
   every page, revalidate with strict no-move policy, and seal the mapping
   read-only before vendor entry. Accepted telemetry must retain the versioned
   per-call contract; a sampled caller buffer is not a substitute for evidence
   about this native snapshot.
   Every array returned by a protected GEMM must use the corresponding output
   boundary in an authenticated run.  Allocate a capsule-owned, page-aligned
   anonymous mapping whose logical shape and storage order are fixed before
   numerical entry, apply the exact static VMA bind before the first output
   write, and require any vendor C span to cover that exact registered output.
   After the partitioned call, integrity checks, and any deterministic repair
   have all finished, query every mapping page and repeat the strict no-move
   policy check before Python can receive the array.  Returned outputs remain
   writable; this boundary must not substitute a read-only seal for the
   post-return in-place scaling contract.  Accepted evidence must reconcile one
   monotonically ordered output-call record with every vendor or deterministic
   call-boundary record, retain the post-repair page histogram and ordered
   status digest, and report zero missing, failed, or dropped calls.  The
   existing post-vendor A/B/C page samples remain an independent gate.
   A provenance-absent legacy caller may retain its allocator-backed output,
   but its record is explicitly incomplete and cannot satisfy this contract.
   Accepted hot-call locality telemetry must reconstruct the literal A/B/C
   storage spans and query only fully contained page bases. Partially shared
   allocation-boundary pages may be retained as diagnostic evidence but must
   not decide locality acceptance. The exhaustive decoded-buffer gate and the
   bounded vendor-operand samples are independent and both remain required.
7. Acceptance arithmetic remains binary64. `--dtype float32` changes retained
   randomized probe/sketch storage only; feature moments, scales, protected
   products, reductions, and outputs remain float64. The canonical annotation
   matrix is always contiguous binary64 and defines the estimand: it is never
   rounded to the storage dtype, its masses derive from the canonical matrix,
   the native boundary verifies supplied masses against its copied matrix, and
   schema-v4 reference manifests pledge the binary64 annotation contract
   (`annotation_value_dtype`, `annotation_digest`) so legacy v3 float32-rounded
   references can never be silently identified with binary64 references. A true
   float32 compute path requires separate dense-oracle, real-data, and
   calibration acceptance.
8. Final XW/WX trace-asymmetry diagnostics must not re-enter a process-shared
   NumPy BLAS. Form the binary64 elementwise products and reduce them with
   `math.fsum`, so a T1-versus-T32 comparison does not conflate private-backend
   output accuracy with a second BLAS runtime's reduction schedule. Numerical
   reference score and diagonal tables use `%.17g` round-trip serialization so
   the artifact comparison measures the in-memory binary64 values rather than
   a lower-precision text projection. Comparison reports must retain the worst
   leaf path, reference/candidate values, signed difference, and applied
   tolerance; the global acceptance tolerance remains fail-closed.

## Throughput and memory invariants

1. A one-environment/probe-tile reference makes exactly two genotype passes:
   fused feature/source construction and paired target construction. Do not
   add passes for jackknife blocks or per-environment feature construction.
2. Multi-environment feature moments use scalar reductions plus the packed
   `[Q, E Q, E^2 Q, E^3 Q]'G` algebra. Do not materialize and project an
   `N x block` additive/interaction pair for each environment. The direct
   backend freezes this algebra, every environment projector, and the shared
   source/target layout in one persistent versioned native kernel. Feature
   postprocessing, source weight packing and context correction, projection,
   paired target scaling, and four-family score reductions remain inside that
   C++ boundary. The descriptor context also owns BED mapping/decode,
   standardization, and NumPy-compatible Philox probe generation, so no
   decoded genotype or intermediate numerical panel crosses into Python.
   Python owns plan/design initialization, evidence validation, and
   namespace-atomic publication of the final arrays. A phenotype-free direct
   single-environment reference uses this same context with environment count
   one; cache, shard, and phenotype-scoring paths remain separate.
3. Additive and GxE packed execution share the native Mailman pre/post
   primitives. Mailman is eligible only when the requested probe count is at
   most ten. Every wider direct GxE job must use dense BLIS; memory pressure may
   create dense environment/probe tiles but must never select Mailman or a
   backend-specific width-32 subdivision. A feasible B=256 one-tile plan uses
   one width-256 source product and one combined paired-target product per
   genotype block.
4. The dense target phase allocates `[S, E*S]` once, accumulates directly into
   its source half, fills the weighted half once, seals the complete mapping,
   and uses one paired product per block. Every native execution-scratch role
   (decoded genotype block, source contribution, combined target output,
   vector scratch, and per-worker Mailman arenas) holds one maximum-capacity
   allocation frozen from the admitted plan, reused with logical sub-shapes
   for terminal blocks and tiles; native-reported capacities must never
   exceed their admitted plan components, semantic call evidence is bounded
   by a frozen record maximum, and the context releases all execution-only
   scratch (release_execution_scratch) after its validated run and before
   any pandas construction or serialization. The native kernel reduces the paired output
   directly into the persistent score accumulators instead of returning four
   temporary score panels. Common covariates and genotype decodes are shared
   where the exact cohort permits it.
5. Phenotype scoring makes one genotype pass. Matched scoring uses the sealed
   scales and one fused `[Y,EY]` target product. Population-transfer scoring
   computes study-specific moments and the same fused target in the single
   decode; it must not allocate full `N x block` X and W matrices.
6. Socket parallelism is an explicit aggregate-memory tradeoff. Report both
   per-process and process-tree peaks, and enforce `h_data * slots` as the total
   Hoffman memory request. Do not use SMT siblings as if they were independent
   cores.
7. An exact multi-process topology screen must enforce the same native NUMA
   contracts as the selected single-process backend gate.  Every protected
   call must reconcile its output-evidence queue record with the corresponding
   vendor or deterministic call-boundary record, and every eligible source NN
   call must retain the exact native right-snapshot evidence.  The worker must
   prove reset/drain/status completeness; the controller must reconstruct and
   revalidate the retained records rather than trusting the worker's accepted
   flag.
8. A topology sweep stops at its first rejected or timed-out configuration.
   The report retains that failure and marks every remaining ordered
   configuration as not run; it must never silently continue with later
   scientific calls.  A controller-owned absolute `CLOCK_MONOTONIC` watchdog
   latches expiry and wakes bounded waits through a private nonblocking
   close-on-exec self-pipe; it never owns child cleanup or publication.  A
   child returned by `Popen` is registered before the next deadline check,
   TERM/KILL cleanup and diagnostic retention are bounded, and a latched
   deadline cannot replace the original worker failure.  The watchdog is
   stopped and its pipe closed before a deadline or post-run identity/summary
   error publishes a non-accepted, no-replace artifact and returns nonzero,
   including during a dry run.

## Required regression gates

Any change to these paths must pass the complete suite plus focused dense
oracles for scales, norms, NxE diagonals, X--W moments, all four LD-score
families, phenotype scores, and population transport. Guard-free runs must
record zero repairs and retries. Before a production-scale expansion, require
packed-versus-dense equivalence tests at the B=10 boundary and a dense B=256
end-to-end artifact comparison. The current architecture has a clean 10K
B=256 gate but no new N=289,111 gate. B=1024 remains a separate scientific,
uncertainty, runtime, and memory decision.

Demonstrated failures include process-global OpenBLAS ownership/thread-pool
problems and a high-thread OpenMP-BLIS execution path. The selected private
pthread-BLIS boundary removes those tested failure modes. The exact faulty
upstream statements were not localized, so future documentation must not claim
a general OpenBLAS or BLIS proof.
